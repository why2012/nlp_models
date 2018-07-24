from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import tensorflow as tf
from adversarial_net.engine import BaseModel
from adversarial_net import arguments as flags
from adversarial_net import sequences as seq
from adversarial_net import layers
from adversarial_net.utils import getLogger
from adversarial_net import osp
from adversarial_net.AdversarialDDGModel import AdversarialDDGModel, modules

logger = getLogger("model")
SOS_TAG = 1
EOS_TAG = 2
modules["SUMMARY"] = "BahdanauAttentionLoss"
modules["SUMMARY_GRUS"] = "GRUs"

class AdversarialSummaryModel(AdversarialDDGModel):
    def __init__(self, use_average=False, init_modules=AdversarialDDGModel.stepB_modules):
        init_modules.remove("T_D")
        super(AdversarialSummaryModel, self).__init__(use_average=use_average, init_modules=init_modules)
        init_modules.append("T_D")

        self.topic_discriminator_dense = seq.ClassificationModelDenseHeader(
            var_scope_name="topic_discriminator_dense",
            layer_sizes=[self.arguments["adv_cl_sequence"]["hidden_size"]] * self.arguments["adv_cl_sequence"]["num_layers"],
            input_size=self.arguments["adv_cl_sequence"]["input_size"] + self.arguments["summary"]["rnn_cell_size"] * 2,
            num_classes=self.arguments["adv_cl_sequence"]["num_classes"],
            keep_prob=self.arguments["adv_cl_sequence"]["keep_prob"])

        self.grus = seq.SummaryGRUs(var_scope_name="GRUs",
                                    state_size=self.arguments["summary"]["rnn_cell_size"],
                                    input_dim=self.arguments["lm_sequence"]["embedding_dim"],
                                    gru_keep_prob_out=self.arguments["summary"]["rnn_keep_prob_out"],
                                    build=False)

        self.atten_loss = seq.SummaryBahdanauAttentionLoss(
            var_scope_name="BahdanauAttentionLoss",
            encoder_fw_cell=self.grus.encoder_fw_cell,
            encoder_bw_cell=self.grus.encoder_bw_cell,
            decoder_cell=self.grus.decoder_cell,
            rnn_size=self.arguments["summary"]["rnn_cell_size"],
            vocab_size=self.arguments["lm_sequence"]["vocab_size"],
            num_candidate_samples=self.arguments["lm_loss"]["num_candidate_samples"],
            vocab_freqs=self.arguments["vocab_freqs"])

        self.summary_layer = seq.EvalSummaryBahdanauAttention(
            associate_var_scope_name="BahdanauAttentionLoss",
            encoder_fw_cell=self.grus.encoder_fw_cell,
            encoder_bw_cell=self.grus.encoder_bw_cell,
            decoder_cell=self.grus.decoder_cell,
            state_proj_layer=self.atten_loss.state_proj_layer,
            to_embedding_layers=self.to_embedding,
            to_embedding_layers_decoder=self.to_embedding,
            rnn_size=self.arguments["summary"]["rnn_cell_size"],
            vocab_size=self.arguments["lm_sequence"]["vocab_size"],
            decoder_type=seq.EvalSummaryBahdanauAttention.GREEDY_EMBEDDING)

    def get_summary_inputs(self, embedding, seq_length, to_embedding_fn, beam_width=10, maximum_iterations=50):
        # embedding (batch_size, time_steps, embed_size)
        batch_size = tf.shape(embedding)[0]
        # add start tag
        start_tags = tf.fill((batch_size, 1), SOS_TAG)
        # start_tags_embedding (batch_size, 1, embed_size)
        start_tags_embedding = to_embedding_fn(start_tags)
        # embedding (batch_size, 1 + time_steps, embed_size)
        embedding = tf.concat([start_tags_embedding, embedding], axis=1)
        # beam_outputs (batch_size, max_iters, beam_width)
        # final_sequence_lengths (batch_size, beam_width)
        beam_outputs, final_sequence_lengths = self.summary_layer(batch_size=batch_size, sos_tag=SOS_TAG, eos_tag=EOS_TAG,
                                                                 encoder_len=seq_length,
                                                                 encoder_embed_inputs=embedding,
                                                                 beam_width=beam_width,
                                                                 maximum_iterations=maximum_iterations)
        outputs = tf.gather_nd(
            tf.transpose(beam_outputs, [0, 2, 1]),
            tf.stack([tf.range(batch_size), tf.fill((batch_size,), 0)], axis=1)
        )
        final_sequence_lengths = tf.gather_nd(final_sequence_lengths, tf.stack([tf.range(batch_size), tf.fill((batch_size,), 0)], axis=1))
        outputs_embedding = to_embedding_fn(outputs)
        laststep_gather_indices = tf.stack([tf.range(batch_size), final_sequence_lengths - 1], axis=1)

        return outputs_embedding, final_sequence_lengths, laststep_gather_indices

    def compute_inputs_logits(self, embedding, weight_tensor, sequence_len, laststep_gather_indices, lstm_initial_state, seq2seq_fn):
        # lstm_initial_state  [LSTMTuple(c, h), ...]
        # lstm_output_tensor (None, steps, lstm_size)
        lstm_output_tensor, final_states = seq2seq_fn(embedding, lstm_initial_state, sequence_len)
        # final_output_tensor (batch_size, lstm_size)
        final_output_tensor = tf.gather_nd(lstm_output_tensor, laststep_gather_indices)
        # final_output_weights (batch_size,)
        final_output_weights = tf.gather_nd(weight_tensor, laststep_gather_indices)
        return final_output_tensor, final_output_weights, final_states

    def compute_summary_logits(self, embedding, sequence_len, laststep_gather_indices, zero_states, seq2seq_fn):
        # lstm_output_tensor (None, steps, lstm_size)
        lstm_output_tensor, final_states = seq2seq_fn(embedding, zero_states, sequence_len)
        # final_output_tensor (batch_size, lstm_size)
        final_output_tensor = tf.gather_nd(lstm_output_tensor, laststep_gather_indices)
        return final_output_tensor

    def compute_combined_cl_loss(self, output_tensor, y_tensor, tensor_weight, dense_fn):
        # logits (batch_size, n_classes)
        logits = dense_fn(output_tensor)
        cl_loss = self.classification_loss([logits, y_tensor, tensor_weight])
        return cl_loss, logits

    def compute_combined_adv_loss(self,
                                  cl_logits, cl_loss,
                                  g_s_embeddings, y_tensor, weight_tensor, g_s_seq_len, g_s_laststep_gather_indices, g_s_lstm_state,
                                  seq2seq_fn, dense_fn, eos_indicator):
        g_embedding_perturb = self.adversarial_loss(cl_loss, None, g_s_embeddings[0], only_perturb=True)
        s_embedding_perturb = self.adversarial_loss(cl_loss, None, g_s_embeddings[1], only_perturb=True)
        g_perturbed_embedding = g_embedding_perturb + g_s_embeddings[0]
        s_perturbed_embedding = s_embedding_perturb + g_s_embeddings[1]
        g_adv_output_tensor, final_output_weights, _ = self.compute_inputs_logits(g_perturbed_embedding, weight_tensor, g_s_seq_len[0], g_s_laststep_gather_indices[0], g_s_lstm_state[0], seq2seq_fn)
        s_adv_output_tensor = self.compute_summary_logits(s_perturbed_embedding, g_s_seq_len[1], g_s_laststep_gather_indices[1], g_s_lstm_state[1], seq2seq_fn)
        combined_output_tensor = tf.concat([g_adv_output_tensor, s_adv_output_tensor], axis=1)
        cl_loss, cl_logits = self.compute_combined_cl_loss(combined_output_tensor, y_tensor, final_output_weights, dense_fn)
        return cl_loss * tf.constant(self.arguments["adv_cl_loss"]["adv_reg_coeff"], name='adv_reg_coeff')

    def build_topic_discriminator(self):
        # inputs
        g_get_lstm_state, g_save_lstm_state, g_embedding, g_y_tensor, g_weight_tensor, g_eos_indicators, g_seq_length, g_laststep_gather_indices = self.get_genuing_inputs()
        # generate inputs summary
        s_embedding, s_seq_len, s_laststep_gather_indices = self.get_summary_inputs(g_embedding, g_seq_length,
                                                                                    self.to_embedding,
                                                                                    beam_width=self.arguments["summary"]["beam_width"],
                                                                                    maximum_iterations=self.arguments["summary"]["maximum_iterations"])
        # seq2seq last step outputs
        g_final_output_tensor, final_output_weights, genuing_final_states = self.compute_inputs_logits(
                                                                            g_embedding, g_weight_tensor, g_seq_length,
                                                                            g_laststep_gather_indices, g_get_lstm_state(),
                                                                            self.topic_discriminator_seq2seq)
        s_zero_states = self.topic_discriminator_seq2seq.lstm_layer.cell.zero_state(tf.shape(s_embedding)[0], s_embedding.dtype)
        s_final_output_tensor = self.compute_summary_logits(s_embedding, s_seq_len, s_laststep_gather_indices, s_zero_states, self.topic_discriminator_seq2seq)
        combined_output_tensor = tf.concat([g_final_output_tensor, s_final_output_tensor], axis=1)
        # losses
        cl_loss, cl_logits = self.compute_combined_cl_loss(combined_output_tensor, g_y_tensor, final_output_weights, self.topic_discriminator_dense)
        # adversaial losses
        combined_adv_loss = self.compute_combined_adv_loss(cl_logits, cl_loss,
                                                         [g_embedding, s_embedding], g_y_tensor,
                                                         g_weight_tensor, [g_seq_length, s_seq_len],
                                                         [g_laststep_gather_indices, s_laststep_gather_indices],
                                                         [g_get_lstm_state(), s_zero_states],
                                                         self.topic_discriminator_seq2seq, self.topic_discriminator_dense,
                                                         g_eos_indicators)
        combined_total_loss = cl_loss + combined_adv_loss
        combined_cl_acc = layers.accuracy(cl_logits, g_y_tensor, final_output_weights)
        tf.summary.scalar('combined_adv_loss', combined_adv_loss)
        tf.summary.scalar('combined_cl_acc', combined_cl_acc)
        # save_lstm_state
        with tf.control_dependencies([g_save_lstm_state(genuing_final_states)]):
            combined_total_loss = tf.identity(combined_total_loss)

        return cl_loss, combined_total_loss, combined_cl_acc

    def build_topic_discriminator_eval_graph(self):
        # inputs
        g_get_lstm_state, g_save_lstm_state, g_embedding, g_y_tensor, g_weight_tensor, g_eos_indicators, g_seq_length, g_laststep_gather_indices = self.get_genuing_inputs()
        # generate inputs summary
        s_embedding, s_seq_len, s_laststep_gather_indices = self.get_summary_inputs(g_embedding, g_seq_length,
                                                                                    self.to_embedding,
                                                                                    beam_width=self.arguments["summary"]["beam_width"],
                                                                                    maximum_iterations=self.arguments["summary"]["maximum_iterations"])
        # seq2seq last step outputs
        g_final_output_tensor, final_output_weights, genuing_final_states = self.compute_inputs_logits(
            g_embedding, g_weight_tensor, g_seq_length,
            g_laststep_gather_indices, g_get_lstm_state(),
            self.topic_discriminator_seq2seq)
        s_zero_states = self.topic_discriminator_seq2seq.lstm_layer.cell.zero_state(tf.shape(s_embedding)[0],
                                                                                    s_embedding.dtype)
        s_final_output_tensor = self.compute_summary_logits(s_embedding, s_seq_len, s_laststep_gather_indices,
                                                            s_zero_states, self.topic_discriminator_seq2seq)
        combined_output_tensor = tf.concat([g_final_output_tensor, s_final_output_tensor], axis=1)
        # losses
        cl_loss, cl_logits = self.compute_combined_cl_loss(combined_output_tensor, g_y_tensor, final_output_weights,
                                                           self.topic_discriminator_dense)
        classification_accuracy, update_op = tf.metrics.accuracy(g_y_tensor, tf.argmax(cl_logits, 1), final_output_weights)
        # save_lstm_state
        with tf.control_dependencies([g_save_lstm_state(genuing_final_states)]):
            update_op = tf.identity(update_op)
        return classification_accuracy, update_op

    def build(self, training = False, eval_cl = False, **kwargs):
        variables, savers, losses, accs, eval_graph = self.pre_build()

        if training:
            self.stepTag = "stepB"
            genuing_cl_loss, genuing_total_loss, genuing_cl_acc = self.build_topic_discriminator()
            losses["genuing_total_loss"] = genuing_total_loss
            relevent_sequences = {"EMBEDDING": self.to_embedding, "T_S": self.topic_discriminator_seq2seq,
                                  "T_D": self.topic_discriminator_dense, "ADV_LOSS": self.adversarial_loss,
                                  "SUMMARY": self.summary_layer, "SUMMARY_GRUS": self.grus}
            pretrained_sequences = {"EMBEDDING": self.to_embedding, "T_S": self.topic_discriminator_seq2seq,
                                    "SUMMARY": self.summary_layer}
            variables["genuing_total_loss"] = []
            variables["genuing_total_loss"] += relevent_sequences["EMBEDDING"].trainable_weights
            variables["genuing_total_loss"] += relevent_sequences["T_S"].trainable_weights
            variables["genuing_total_loss"] += relevent_sequences["T_D"].trainable_weights
            variables["genuing_total_loss"] += relevent_sequences["ADV_LOSS"].trainable_weights
            variables["genuing_total_loss"] += relevent_sequences["SUMMARY"].trainable_weights
            variables["genuing_total_loss"] += relevent_sequences["SUMMARY_GRUS"].trainable_weights
            accs["genuing_cl_acc"] = genuing_cl_acc
        elif eval_cl:
            self.stepTag = "eval_cl"
            relevent_sequences = {"EMBEDDING": self.to_embedding, "T_S": self.topic_discriminator_seq2seq,
                                  "T_D": self.topic_discriminator_dense, "SUMMARY": self.summary_layer,
                                  "SUMMARY_GRUS": self.grus}
            pretrained_sequences = {}
            acc_op, update_op = self.build_topic_discriminator_eval_graph()
            eval_graph["acc_op"] = acc_op
            eval_graph["update_op"] = update_op
        else:
            raise Exception("Unsupport ops")

        self.post_build(pretrained_sequences, savers, kwargs)

        self.optimize()


