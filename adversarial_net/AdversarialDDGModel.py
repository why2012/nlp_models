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

class AdversarialDDGModel(BaseModel):
    def __init__(self, use_average = False):
        super(AdversarialDDGModel, self).__init__(use_average=use_average)
        # EMBEDDING
        self.to_embedding = seq.EmbeddingSequence(
            vocab_size=self.arguments["lm_sequence"]["vocab_size"],
            embedding_dim=self.arguments["lm_sequence"]["embedding_dim"],
            vocab_freqs=self.arguments["vocab_freqs"],
            normalize=True,
            keep_embed_prob=self.arguments["keep_embed_prob"])
        # FG_S
        self.fake_genuing_discriminator_seq2seq = seq.Seq2SeqSequence(
            var_scope_name="fake_genuing_discriminator_seq2seq",
            rnn_cell_size=self.arguments["lm_sequence"]["rnn_cell_size"],
            input_size=self.arguments["lm_sequence"]["embedding_dim"],
            rnn_num_layers=self.arguments["lm_sequence"]["rnn_num_layers"],
            lstm_keep_pro_out=self.arguments["lm_sequence"]["lstm_keep_pro_out"])
        # T_S
        self.topic_discriminator_seq2seq = seq.Seq2SeqSequence(
            var_scope_name="topic_discriminator_seq2seq",
            rnn_cell_size=self.arguments["lm_sequence"]["rnn_cell_size"],
            input_size=self.arguments["lm_sequence"]["embedding_dim"],
            rnn_num_layers=self.arguments["lm_sequence"]["rnn_num_layers"],
            lstm_keep_pro_out=self.arguments["lm_sequence"]["lstm_keep_pro_out"])
        # FG_D
        self.fake_genuing_discriminator_dense = seq.ClassificationModelDenseHeader(
            var_scope_name="fake_genuing_discriminator_dense",
            layer_sizes=[self.arguments["adv_cl_sequence"]["hidden_size"]] * self.arguments["adv_cl_sequence"][
                "num_layers"],
            input_size=self.arguments["adv_cl_sequence"]["input_size"],
            num_classes=self.arguments["adv_cl_sequence"]["num_classes"],
            keep_prob=self.arguments["adv_cl_sequence"]["keep_prob"])
        # T_D
        self.topic_discriminator_dense = seq.ClassificationModelDenseHeader(
            var_scope_name="topic_discriminator_dense",
            layer_sizes=[self.arguments["adv_cl_sequence"]["hidden_size"]] * self.arguments["adv_cl_sequence"][
                "num_layers"],
            input_size=self.arguments["adv_cl_sequence"]["input_size"],
            num_classes=self.arguments["adv_cl_sequence"]["num_classes"],
            keep_prob=self.arguments["adv_cl_sequence"]["keep_prob"])
        # ADV_LOSS
        self.adversarial_loss = seq.AdversarialLoss(perturb_norm_length=self.arguments["adv_cl_loss"]["perturb_norm_length"])
        # CL_LOSS
        self.classification_loss = layers.ClassificationSparseSoftmaxLoss()
        # SEQ_G_LSTM
        self.generator_lstms = seq.LanguageSequenceGeneratorLSTM(
            rnn_cell_size=self.arguments["lm_sequence"]["rnn_cell_size"],
            input_size=self.arguments["lm_sequence"]["embedding_dim"],
            rnn_num_layers=self.arguments["lm_sequence"]["rnn_num_layers"],
            lstm_keep_pro_out=self.arguments["lm_sequence"]["lstm_keep_pro_out"])
        # SEQ_G
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
        # gather_indices
        # laststep_gather_indices [(batch_index, step_index), ...]
        laststep_gather_indices = tf.stack([tf.range(tf.shape(inputs)[0]), inputs.length - 1], 1)
        # X_tensor, y_tensor, weight_tensor
        # X_tensor (batch_size, steps)
        # Y_tensor (batch_size,)
        # weight_tensor (batch_size, steps)
        X_tensor, y_tensor, weight_tensor = tf.squeeze(inputs.sequences["X"], axis=-1), tf.squeeze(inputs.context["y"], axis=-1), tf.squeeze(inputs.sequences["weight"], axis=-1)
        embedding = self.to_embedding(X_tensor)
        return get_lstm_state, save_lstm_state, embedding, y_tensor, weight_tensor, inputs.length, laststep_gather_indices

    def generate_synthesize_inputs(self, seq_length_tensor, batch_size_tensor = 1):
        content_initial_states = self.sequence_generator.content_states(batch_size_tensor)
        # dist_fuse_w (batch_size, 2) [0, 1]
        dist_fuse_w = tf.round(tf.random_uniform((batch_size_tensor, 1), minval=0, maxval=1))
        dist_fuse_w = tf.stack([dist_fuse_w, 1 - dist_fuse_w], axis=1)
        topic_initial_states = self.sequence_generator.topic_states(batch_size_tensor, dist_fuse_w)
        # step_one_inputs (batch_size, input_size)
        step_one_inputs = tf.squeeze(self.to_embedding(tf.fill((batch_size_tensor, 1), EOS_TAG)), axis=1)
        # topic_outputs (batch_size, time_steps, embed_size)
        topic_outputs = self.sequence_generator(
                    content_initial_states=content_initial_states,
                    topic_initial_states=topic_initial_states,
                    step_one_inputs=step_one_inputs,
                    seq_length=seq_length_tensor,
                    keep_prob=self.arguments["lm_sequence"]["lstm_keep_pro_out"])
        return topic_outputs

    def construct_synthesize_inputs_batch_sequence_with_states(self):
        pass

    def synthesize_inputs(self):
        pass

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

    def build(self):
        pass

    def fit(self, save_model_path=None, pretrained_model_dir=None):
        pass