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
    stepA_modules = ["EMBEDDING", "FG_S", "FG_D", "SEQ_G_LSTM", "SEQ_G"]
    stepB_modules = ["EMBEDDING", "T_S", "T_D", "ADV_LOSS", "CL_LOSS", "SEQ_G_LSTM", "SEQ_G"]
    def __init__(self, use_average = False, init_modules=modules.keys()):
        super(AdversarialDDGModel, self).__init__(use_average=use_average)
        self._fit_kwargs = {}
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
                num_classes=1,
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
        return cl_loss, final_states

    def compute_d_logits(self, embedding,  sequence_len, get_lstm_state_fn, seq2seq_fn, dense_fn):
        # lstm_initial_state  [LSTMTuple(c, h), ...]
        lstm_initial_state = get_lstm_state_fn()
        # lstm_output_tensor (None, steps, lstm_size)
        lstm_output_tensor, final_states = seq2seq_fn(embedding, lstm_initial_state, sequence_len)
        # logits (batch_size, n_classes)
        logits = dense_fn(final_output_tensor)
        return logits, final_states

    def compute_adv_loss(self, cl_loss, embedding, y_tensor, weight_tensor, sequence_len, laststep_gather_indices, get_lstm_state_fn, seq2seq_fn, dense_fn):
        def local_compute_cl_loss(perturbed_embedding):
            return self.compute_cl_loss(perturbed_embedding, y_tensor, weight_tensor, sequence_len, laststep_gather_indices, get_lstm_state_fn, seq2seq_fn, dense_fn)
        adv_loss = self.adversarial_loss(cl_loss, local_compute_cl_loss, embedding)
        return adv_loss

    def get_genuing_inputs(self):
        if not hasattr(self, "_genuing_inputs_cache"):
            self._genuing_inputs_cache = self.genuing_inputs()
        return self._genuing_inputs_cache

    def get_synthesize_inputs(self):
        if not hasattr(self, "_synthesize_inputs_cache"):
            self._synthesize_inputs_cache = self.synthesize_inputs()
        return self._synthesize_inputs_cache

    def build_fake_genuing_discriminator(self, LAMBDA = 10):
        g_get_lstm_state, g_save_lstm_state, g_embedding, g_y_tensor, g_weight_tensor, g_eos_indicators, g_seq_length, g_laststep_gather_indices = self.get_genuing_inputs()
        s_get_lstm_state, s_save_lstm_state, s_embedding, s_y_tensor, s_weight_tensor, s_eos_indicators, s_seq_length, s_laststep_gather_indices = self.get_synthesize_inputs()
        def get_interpolated_states_fn(alpha, g_get_lstm_state, s_get_lstm_state):
            def get_states_fn():
                alpha = tf.squeeze(alpha)
                lstm_initial_state = []
                g_lstm_initial_state = g_get_lstm_state()
                s_lstm_initial_state = s_get_lstm_state()
                for g_lstm_tuple, s_lstm_tuple in zip(g_lstm_initial_state, s_lstm_initial_state):
                    g_c = g_lstm_tuple.c
                    s_c = s_lstm_tuple.c
                    g_h = g_lstm_tuple.h
                    s_h = s_lstm_tuple.h
                    c = g_c + alpha * (s_c - g_c)
                    h = g_h + alpha * (s_h - g_h)
                    lstm_initial_state.append(tf.contrib.rnn.LSTMStateTuple(c, h))
                lstm_initial_state = tuple(lstm_initial_state)
                return lstm_initial_state
            return get_states_fn
        # logits (batch_size, 1)
        genuing_logits, genuing_final_states = self.compute_d_logits(g_embedding, g_seq_length, g_get_lstm_state, self.fake_genuing_discriminator_seq2seq, self.fake_genuing_discriminator_dense)
        fake_logits, fake_final_states = self.compute_d_logits(s_embedding, s_seq_length, s_get_lstm_state, self.fake_genuing_discriminator_seq2seq, self.fake_genuing_discriminator_dense)
        # losses
        discriminator_loss = tf.reduce_mean(fake_logits) - tf.reduce_mean(genuing_logits)
        generator_loss = -tf.reduce_mean(fake_logits)
        # WGAN lipschitz-penalty
        alpha = tf.random_uniform(shape=[tf.shape(g_embedding)[0], 1, 1], minval=0., maxval=1.)
        differences = s_embedding - g_embedding
        interpolates = g_embedding + (alpha * differences)
        interpolates_cost = self.compute_d_logits(interpolates, s_seq_length,
                                                  get_interpolated_states_fn(alpha, g_get_lstm_state, s_get_lstm_state),
                                                  self.fake_genuing_discriminator_seq2seq, self.fake_genuing_discriminator_dense)
        gradients = tf.gradients(interpolates_cost, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        discriminator_loss += LAMBDA * gradient_penalty
        # save_lstm_state
        with tf.control_dependencies([g_save_lstm_state(genuing_final_states), s_save_lstm_state(fake_final_states)]):
            discriminator_loss = tf.identity(discriminator_loss)
            generator_loss = tf.identity(generator_loss)
        return discriminator_loss, generator_loss

    def build_topic_discriminator(self):
        g_get_lstm_state, g_save_lstm_state, g_embedding, g_y_tensor, g_weight_tensor, g_eos_indicators, g_seq_length, g_laststep_gather_indices = self.get_genuing_inputs()
        s_get_lstm_state, s_save_lstm_state, s_embedding, s_y_tensor, s_weight_tensor, s_eos_indicators, s_seq_length, s_laststep_gather_indices = self.get_synthesize_inputs()
        # losses
        genuing_cl_loss, genuing_final_states = self.compute_cl_loss(g_embedding, g_y_tensor, g_weight_tensor, g_seq_length,
                                                             g_laststep_gather_indices, g_get_lstm_state,
                                                             self.topic_discriminator_seq2seq, self.topic_discriminator_dense)
        fake_cl_loss, fake_final_states = self.compute_cl_loss(s_embedding, s_y_tensor, s_weight_tensor, s_seq_length,
                                                             s_laststep_gather_indices, s_get_lstm_state,
                                                             self.topic_discriminator_seq2seq, self.topic_discriminator_dense)
        genuing_adv_loss = self.compute_adv_loss(genuing_cl_loss, g_embedding, g_y_tensor, g_weight_tensor,
                                                 g_seq_length, g_laststep_gather_indices, g_get_lstm_state,
                                                 self.topic_discriminator_seq2seq, self.topic_discriminator_dense)
        fake_adv_loss = self.compute_adv_loss(fake_cl_loss, s_embedding, s_y_tensor, s_weight_tensor,
                                                 s_seq_length, s_laststep_gather_indices, s_get_lstm_state,
                                                 self.topic_discriminator_seq2seq, self.topic_discriminator_dense)
        genuing_total_loss = genuing_cl_loss + genuing_adv_loss
        fake_total_loss = fake_cl_loss + fake_adv_loss
        tf.summary.scalar('genuing_adv_loss', genuing_adv_loss)
        tf.summary.scalar('fake_adv_loss', fake_adv_loss)
        # save_lstm_state
        with tf.control_dependencies([g_save_lstm_state(genuing_final_states)]):
            genuing_cl_loss = tf.identity(genuing_cl_loss)
            genuing_total_loss = tf.identity(genuing_total_loss)
        with tf.control_dependencies([s_save_lstm_state(fake_final_states)]):
            fake_cl_loss = tf.identity(fake_cl_loss)
            fake_total_loss = tf.identity(fake_total_loss)
        return genuing_cl_loss, genuing_total_loss, fake_cl_loss, fake_total_loss

    # stepA: fake_genuing_discriminator & generator
    # stepB: topic_discriminator & generator
    def build(self, stepA = True, stepB = False):
        relevent_sequences = None
        variables = {}
        savers = {}
        self._fit_kwargs["variables"] = variables
        self._fit_kwargs["savers"] = savers
        if stepA:
            discriminator_loss, generator_loss = self.build_fake_genuing_discriminator()
            relevent_sequences = {"EMBEDDING": self.to_embedding, "FG_S": self.fake_genuing_discriminator_seq2seq,
                                  "FG_D": self.fake_genuing_discriminator_dense,
                                  "SEQ_G_LSTM": self.generator_lstms, "SEQ_G": self.sequence_generator}
            variables["discriminator_loss"] = []
            variables["generator_loss"] = []
            variables["discriminator_loss"] += relevent_sequences["FG_S"].trainable_weights
            variables["discriminator_loss"] += relevent_sequences["FG_D"].trainable_weights
            variables["generator_loss"] += relevent_sequences["SEQ_G_LSTM"].trainable_weights
            variables["generator_loss"] += relevent_sequences["SEQ_G"].trainable_weights
        elif stepB:
            genuing_cl_loss, genuing_total_loss, fake_cl_loss, fake_total_loss = self.build_topic_discriminator()
            relevent_sequences = {"EMBEDDING": self.to_embedding, "T_S": self.topic_discriminator_seq2seq,
                                  "T_D": self.topic_discriminator_dense, "ADV_LOSS": self.adversarial_loss,
                                  "SEQ_G_LSTM": self.generator_lstms, "SEQ_G": self.sequence_generator}
            variables["genuing_total_loss"] = []
            variables["fake_cl_loss"] = []
            variables["fake_total_loss"] = []
            variables["genuing_total_loss"] += relevent_sequences["EMBEDDING"].trainable_weights
            variables["genuing_total_loss"] += relevent_sequences["T_S"].trainable_weights
            variables["genuing_total_loss"] += relevent_sequences["T_D"].trainable_weights
            variables["genuing_total_loss"] += relevent_sequences["ADV_LOSS"].trainable_weights
            variables["fake_cl_loss"] += relevent_sequences["SEQ_G_LSTM"].trainable_weights
            variables["fake_cl_loss"] += relevent_sequences["SEQ_G"].trainable_weights
            variables["fake_total_loss"] += relevent_sequences["EMBEDDING"].trainable_weights
            variables["fake_total_loss"] += relevent_sequences["T_S"].trainable_weights
            variables["fake_total_loss"] += relevent_sequences["T_D"].trainable_weights
            variables["fake_total_loss"] += relevent_sequences["ADV_LOSS"].trainable_weights

        for tag, seq in relevent_sequences.items():
            pretrain_restorer = seq.pretrain_restorer
            if pretrain_restorer:
                savers[tag] = seq.pretrain_restorer

    def restore_pretrained_variables(self, sess, save_model_path, pretrain_model_pathes):
        savers = self._fit_kwargs["savers"]
        for tag, saver in savers.items():
            pretrained_model_path = pretrain_model_pathes[tag]
            self._restore_pretained_variables(sess, pretrained_model_path, variables_to_restore = None, save_model_path = osp.dirname(save_model_path), saver_for_restore = saver)

    def fit(self, save_model_path=None, pretrain_model_pathes = {}):
        with tf.Session() as sess:
            self.restore_pretrained_variables(sess, save_model_path, pretrain_model_pathes)