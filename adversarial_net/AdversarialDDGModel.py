from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import tensorflow as tf
from adversarial_net.engine import BaseModel
from adversarial_net import arguments as flags
from adversarial_net import sequences as seq
from adversarial_net import layers
from adversarial_net.utils import getLogger
from adversarial_net.inputs import construct_language_model_input_tensor_with_state
from adversarial_net.inputs import construct_classification_model_input_tensor_with_state
from adversarial_net.inputs import construct_autoencoder_model_input_tensor_with_state
from adversarial_net import osp

logger = getLogger("adv_model")
EOS_TAG = 2
modules={"EMBEDDING": "to_embedding", "FG_S": "fake_genuing_discriminator_seq2seq", "T_S": "topic_discriminator_seq2seq",
         "FG_D": "fake_genuing_discriminator_dense", "T_D": "topic_discriminator_dense", "ADV_LOSS": "adversarial_loss",
         "CL_LOSS": "classification_loss", "SEQ_G_LSTM": "generator_lstms", "SEQ_G": "sequence_generator"}

class AdversarialDDGModel(BaseModel):
    def __init__(self, use_average = False, init_modules=modules.keys()):
        super(AdversarialDDGModel, self).__init__(use_average=use_average)
        modules_abbreviation = init_modules
        # EMBEDDING
        if "EMBEDDING" in modules_abbreviation:
            self.to_embedding = seq.EmbeddingSequence(
                vocab_size=self.arguments["lm_sequence"]["vocab_size"],
                embedding_dim=self.arguments["lm_sequence"]["embedding_dim"],
                vocab_freqs=self.arguments["vocab_freqs"],
                normalize=True,
                keep_embed_prob=self.arguments["keep_embed_prob"])
        # FG_S
        if "FG_S" in modules_abbreviation:
            self.fake_genuing_discriminator_seq2seq = seq.Seq2SeqSequence(
                var_scope_name="fake_genuing_discriminator_seq2seq",
                rnn_cell_size=self.arguments["lm_sequence"]["rnn_cell_size"],
                input_size=self.arguments["lm_sequence"]["embedding_dim"],
                rnn_num_layers=self.arguments["lm_sequence"]["rnn_num_layers"],
                lstm_keep_pro_out=self.arguments["lm_sequence"]["lstm_keep_pro_out"])
        # T_S
        if "T_S" in modules_abbreviation:
            self.topic_discriminator_seq2seq = seq.Seq2SeqSequence(
                var_scope_name="topic_discriminator_seq2seq",
                rnn_cell_size=self.arguments["lm_sequence"]["rnn_cell_size"],
                input_size=self.arguments["lm_sequence"]["embedding_dim"],
                rnn_num_layers=self.arguments["lm_sequence"]["rnn_num_layers"],
                lstm_keep_pro_out=self.arguments["lm_sequence"]["lstm_keep_pro_out"])
        # FG_D
        if "FG_D" in modules_abbreviation:
            self.fake_genuing_discriminator_dense = seq.ClassificationModelDenseHeader(
                var_scope_name="fake_genuing_discriminator_dense",
                layer_sizes=[self.arguments["adv_cl_sequence"]["hidden_size"]] * self.arguments["adv_cl_sequence"][
                    "num_layers"],
                input_size=self.arguments["adv_cl_sequence"]["input_size"],
                num_classes=self.arguments["adv_cl_sequence"]["num_classes"],
                keep_prob=self.arguments["adv_cl_sequence"]["keep_prob"])
        # T_D
        if "T_D" in modules_abbreviation:
            self.topic_discriminator_dense = seq.ClassificationModelDenseHeader(
                var_scope_name="topic_discriminator_dense",
                layer_sizes=[self.arguments["adv_cl_sequence"]["hidden_size"]] * self.arguments["adv_cl_sequence"][
                    "num_layers"],
                input_size=self.arguments["adv_cl_sequence"]["input_size"],
                num_classes=self.arguments["adv_cl_sequence"]["num_classes"],
                keep_prob=self.arguments["adv_cl_sequence"]["keep_prob"])
        # ADV_LOSS
        if "ADV_LOSS" in modules_abbreviation:
            self.adversarial_loss = seq.AdversarialLoss(perturb_norm_length=self.arguments["adv_cl_loss"]["perturb_norm_length"])
        # CL_LOSS
        if "CL_LOSS" in modules_abbreviation:
            self.classification_loss = layers.ClassificationSparseSoftmaxLoss()
        # SEQ_G_LSTM
        if "SEQ_G_LSTM" in modules_abbreviation:
            self.generator_lstms = seq.LanguageSequenceGeneratorLSTM(
                rnn_cell_size=self.arguments["lm_sequence"]["rnn_cell_size"],
                input_size=self.arguments["lm_sequence"]["embedding_dim"],
                rnn_num_layers=self.arguments["lm_sequence"]["rnn_num_layers"],
                lstm_keep_pro_out=self.arguments["lm_sequence"]["lstm_keep_pro_out"])
        # SEQ_G
        if "SEQ_G" in modules_abbreviation:
            self.sequence_generator = seq.LanguageSequenceGenerator(
                ae_lstm_cell=self.generator_lstms.ae_lstm_layer.cell,
                lm_lstm_cell=self.generator_lstms.lm_lstm_layer.cell,
                rnnOutputToEmbedding=seq.RnnOutputToEmbedding(
                    var_scope_name="sequence_generator",
                    vocab_size=self.arguments["lm_sequence"]["vocab_size"],
                    input_size=self.arguments["lm_sequence"]["rnn_cell_size"],
                    embedding_weights=self.to_embedding.embedding_layer.var))

    def genuing_inputs(self):
        logger.info("constructing classification model dataset...")
        inputs, get_lstm_state, save_lstm_state = construct_classification_model_input_tensor_with_state(**self.arguments["adv_cl_inputs"])
        logger.info("classification model dataset is constructed.")
        # X_tensor, y_tensor, weight_tensor
        # X_tensor (batch_size, steps)
        # Y_tensor (batch_size,)
        # weight_tensor (batch_size, steps)
        X_tensor, y_tensor, weight_tensor = tf.squeeze(inputs.sequences["X"], axis=-1), tf.squeeze(inputs.context["y"], axis=-1), tf.squeeze(inputs.sequences["weight"], axis=-1)
        # gather_indices
        # laststep_gather_indices [(batch_index, step_index), ...]
        laststep_gather_indices = tf.stack([tf.range(tf.shape(X_tensor)[0]), inputs.length - 1], 1)
        embedding = self.to_embedding(X_tensor)
        # eos_indicators (batch_size, steps)
        eos_indicators = tf.cast(tf.equal(X_tensor, EOS_TAG), tf.float32)
        return get_lstm_state, save_lstm_state, embedding, y_tensor, weight_tensor, eos_indicators, inputs.length, laststep_gather_indices

    def generate_synthesize_inputs(self, seq_length_tensor, batch_size_tensor = 1, topic_count = 2):
        content_initial_states = self.sequence_generator.content_states(batch_size_tensor)
        # dist_fuse_w_y (batch_size,) range(0, topic_count)
        dist_fuse_w_y = tf.round(tf.random_uniform((batch_size_tensor,), minval=0, maxval=topic_count - 1))
        # dist_fuse_w (batch_size, topic_count)
        dist_fuse_w = tf.one_hot(dist_fuse_w_y, depth=topic_count)
        distargs = []
        for i in range(topic_count):
            distargs.append({"mean": i * 10, "stddev": 1})
        topic_initial_states = self.sequence_generator.topic_states(batch_size_tensor, dist_fuse_w, distargs=distargs)
        # step_one_inputs (batch_size, input_size)
        step_one_inputs = tf.squeeze(self.to_embedding(tf.fill((batch_size_tensor, 1), EOS_TAG)), axis=1)
        # topic_outputs (batch_size, time_steps, embed_size)
        topic_outputs = self.sequence_generator(
                    content_initial_states=content_initial_states,
                    topic_initial_states=topic_initial_states,
                    step_one_inputs=step_one_inputs,
                    seq_length=seq_length_tensor,
                    keep_prob=self.arguments["lm_sequence"]["lstm_keep_pro_out"])
        # Y_tensor (batch_size,) range(0, topic_count)
        y_tensor = dist_fuse_w_y
        # weight_tensor (1, seq_length_tensor)
        weight_tensor = tf.expand_dims(tf.range(0, seq_length_tensor) / (seq_length_tensor - 1), 0)
        # weight_tensor (batch_size, seq_length_tensor)
        weight_tensor = tf.tile(weight_tensor, [batch_size_tensor, 1])
        # batch_seq_length (batch_size,)
        batch_seq_length = tf.fill((batch_size_tensor,), seq_length_tensor)
        # eos_indicators (batch_size, steps)
        eos_indicators = tf.stack([tf.fill((batch_size_tensor, seq_length_tensor - 1), 0), tf.fill((batch_size_tensor, 1), 1)], axis=1)
        return topic_outputs, y_tensor, weight_tensor, eos_indicators, batch_seq_length

    def construct_synthesize_inputs_batch_sequence_with_states(self, batch_size, unroll_steps, lstm_num_layers,
                                                               state_size, bidrec=False, topic_count=2):
        seq_length_tensor = tf.squeeze(tf.round(tf.random_uniform((1,), 200, 800)))
        # single_X (1, steps, embed_size)
        # single_y (1,)
        # single_weight (1, steps)
        # single_eos_indicators (1, steps)
        # single_seq_length(1,)
        single_X, single_y, single_weight, single_eos_indicators, single_seq_length = self.generate_synthesize_inputs(seq_length_tensor, 1, topic_count)
        # single_X (steps, embed_size)
        single_X = tf.squeeze(single_X, 0)
        # single_y ()
        single_y = tf.squeeze(single_y, 0)
        # single_weight (steps, 1)
        single_weight = tf.reshape(single_weight, (-1, 1))
        # single_eos_indicators (steps, 1)
        single_eos_indicators = tf.reshape(single_eos_indicators,(-1, 1))
        # single_seq_length()
        single_seq_length = tf.squeeze(single_seq_length, 0)
        batch, get_lstm_state, save_lstm_state = self.sequence_generator.construct_batch_sequences_with_states(
            single_seq_tensor=single_X, single_seq_topic_label=single_y, batch_size=batch_size, single_seq_length=single_seq_length,
            unroll_steps=unroll_steps, state_size=state_size, lstm_num_layers=lstm_num_layers,
            bidrec=bidrec, single_weight=single_weight, single_eos_indicators=single_eos_indicators)
        return batch, get_lstm_state, save_lstm_state

    def synthesize_inputs(self):
        batch, get_lstm_state, save_lstm_state = self.construct_synthesize_inputs_batch_sequence_with_states(
            batch_size=self.arguments["adv_cl_inputs"]["batch_size"],
            unroll_steps=self.arguments["adv_cl_inputs"]["unroll_steps"],
            lstm_num_layers=self.arguments["adv_cl_inputs"]["lstm_num_layers"],
            state_size=self.arguments["adv_cl_inputs"]["state_size"],
            bidrec=self.arguments["adv_cl_inputs"]["bidrec"],
            topic_count=self.arguments["adv_cl_inputs"]["num_classes"],
        )
        # embed_X_tensor (batch_size, steps, embed_size)
        # Y_tensor (batch_size,)
        # weight_tensor (batch_size, steps)
        embed_X_tensor, y_tensor, weight_tensor = batch.sequences["X"], batch.context["y"], tf.squeeze(batch.sequences["weight"], -1)
        # eos_indicators (batch_size, steps)
        eos_indicators = tf.squeeze(batch.sequences["eos_indicators"], -1)
        # gather_indices
        # laststep_gather_indices [(batch_index, step_index), ...]
        laststep_gather_indices = tf.stack([tf.range(tf.shape(embed_X_tensor)[0]), batch.length - 1], 1)
        return get_lstm_state, save_lstm_state, embed_X_tensor, y_tensor, weight_tensor, eos_indicators, batch.length, laststep_gather_indices

    def compute_cl_loss(self, embedding, y_tensor, weight_tensor, sequence_len, laststep_gather_indices, get_lstm_state_fn, seq2seq_fn, dense_fn):
        # lstm_initial_state  [LSTMTuple(c, h), ...]
        lstm_initial_state = get_lstm_state_fn()
        # lstm_output_tensor (None, steps, lstm_size)
        lstm_output_tensor, final_states = seq2seq_fn(embedding, lstm_initial_state, sequence_len)
        # final_output_tensor (batch_size, lstm_size)
        final_output_tensor = tf.gather_nd(lstm_output_tensor, laststep_gather_indices)
        # final_output_weights (batch_size,)
        final_output_weights = tf.gather_nd(weight_tensor, laststep_gather_indices)
        # logits (batch_size, n_classes)
        logits = dense_fn(final_output_tensor)
        cl_loss = self.classification_loss([logits, y_tensor, final_output_weights])
        return cl_loss

    def build_fake_genuing_discriminator(self):
        pass

    def build_topic_discriminator(self):
        pass

    def build(self):
        pass

    def fit(self, save_model_path=None, pretrained_model_dir=None):
        pass