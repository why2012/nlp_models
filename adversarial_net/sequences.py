from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow import keras
from adversarial_net.preprocessing import WordCounter
from adversarial_net import layers

class LanguageModelSequence(object):
    def __init__(self, vocab_size, embedding_dim, vocab_freqs, rnn_cell_size, normalize=True, keep_embed_prob=1,
                 rnn_num_layers=1, lstm_keep_pro_out=1):
        self.embedding_layer = layers.Embedding(vocab_size=vocab_size, embedding_dim=embedding_dim, vocab_freqs=vocab_freqs,
                                           keep_prob=keep_embed_prob, normalize=normalize)
        self.lstm_layer = layers.LSTM(cell_size=rnn_cell_size, num_layers=rnn_num_layers, keep_prob=lstm_keep_pro_out)
        self.embedding = None

    # output, final_state(LSTMTuple, ...)
    def __call__(self, inputs, lstm_initial_state, sequence_len = None, return_embedding = False):
        if return_embedding:
            embedding = self.embedding_layer(inputs)
            return self.lstm_layer(embedding, lstm_initial_state, sequence_len), embedding
        else:
            self.embedding = self.embedding_layer(inputs)
            return self.lstm_layer(self.embedding, lstm_initial_state, sequence_len)

    @property
    def trainable_weights(self):
        return self.embedding_layer.trainable_weights + self.lstm_layer.trainable_weights

    @property
    def pretrain_weights(self):
        return [dict(zip(map(lambda x: x.op.name, self.trainable_weights), self.trainable_weights))]

    @property
    def pretrain_restorer(self):
        return []

class EmbeddingSequence(object):
    def __init__(self, vocab_size, embedding_dim, vocab_freqs, normalize=True, keep_embed_prob=1):
        self.embedding_layer = layers.Embedding(vocab_size=vocab_size, embedding_dim=embedding_dim,
                                                vocab_freqs=vocab_freqs,
                                                keep_prob=keep_embed_prob, normalize=normalize)
        self.embedding_layer.build([-1, -1])

    def __call__(self, inputs):
        # inputs (batch_size, time_steps)
        # embedding (batch_size, time_steps, embed_size)
        embedding = self.embedding_layer(inputs)
        return embedding

    @property
    def trainable_weights(self):
        return self.embedding_layer.trainable_weights

    @property
    def pretrain_weights(self):
        return [{x.op.name.split("/", 1)[1]: x for x in self.embedding_layer.trainable_weights}]

    @property
    def pretrain_restorer(self):
        restorers = []
        for pretrain_name_vars in self.pretrain_weights:
            restorers.append(tf.train.Saver(pretrain_name_vars))
        return restorers

# use this sequence as sub-net of real-fake-discriminator and topic-discriminator
class Seq2SeqSequence(object):
    def __init__(self, var_scope_name, rnn_cell_size, input_size, rnn_num_layers=1, lstm_keep_pro_out=1):
        with tf.variable_scope(var_scope_name):
            self.lstm_layer = layers.LSTM(cell_size=rnn_cell_size, num_layers=rnn_num_layers, keep_prob=lstm_keep_pro_out)
            self.lstm_layer.build(input_shape=[-1, input_size])

    def __call__(self, embedding, lstm_initial_state, sequence_len = None):
        # embedding (batch_size, time_steps, embed_size)
        # outputs (batch_size, time_steps, rnn_size)
        # final_states (batch_size, rnn_size)
        outputs, final_states = self.lstm_layer(embedding, lstm_initial_state, sequence_len)
        return outputs, final_states

    @property
    def trainable_weights(self):
        return self.lstm_layer.trainable_weights

    @property
    def pretrain_weights(self):
        return [{x.op.name.split("/", 1)[1]: x for x in self.lstm_layer.trainable_weights}]

    @property
    def pretrain_restorer(self):
        restorers = []
        for pretrain_name_vars in self.pretrain_weights:
            restorers.append(tf.train.Saver(pretrain_name_vars))
        return restorers

class AdversarialLoss(object):
    def __init__(self, perturb_norm_length):
        self.perturb_norm_length = perturb_norm_length

    def __call__(self, loss, compute_loss_fn, target):
        target_grads = tf.gradients(loss, target)[0]
        target_grads = tf.stop_gradient(target_grads)
        perturb = self.embed_scale_l2(target_grads, self.perturb_norm_length)
        return compute_loss_fn(target + perturb)

    def embed_scale_l2(self, x, norm_length):
        # alpha (None, 1, 1)
        alpha = tf.reduce_max(tf.abs(x), (1, 2), keep_dims=True) + 1e-12
        # l2_norm (None, 1, 1)
        l2_norm = alpha * tf.sqrt(tf.reduce_sum(tf.pow(x / alpha, 2), (1, 2), keep_dims=True) + 1e-6)
        x_unit = x / l2_norm
        return norm_length * x_unit

    @property
    def trainable_weights(self):
        return []

    @property
    def pretrain_weights(self):
        return []

    @property
    def pretrain_restorer(self):
        return []

class VirtualAdversarialLoss(AdversarialLoss):
    def __init__(self, perturb_norm_length, small_constant_for_finite_diff, iter_count):
        self.perturb_norm_length = perturb_norm_length
        self.small_constant_for_finite_diff = small_constant_for_finite_diff
        self.iter_count = iter_count

    def __call__(self, compute_logits_fn, logits, target, eos_indicators, sequence_length):
        batch_size_tensor = tf.shape(target)[0]
        logits = tf.stop_gradient(logits)
        laststep_gather_indices = tf.stack([tf.range(batch_size_tensor), sequence_length - 1], 1)
        # final_output_weights (None,)
        final_step_weights = tf.gather_nd(eos_indicators, laststep_gather_indices)
        # shape(embedded) = (batch_size, num_timesteps, embedding_dim)
        d = tf.random_normal(shape=tf.shape(target))
        for _ in range(self.iter_count):
            d = self.embed_scale_l2(self.mask_by_length(d, sequence_length), self.small_constant_for_finite_diff)
            d_logits = compute_logits_fn(target + d)
            kl_loss = self.kl_divergence_with_logits(logits, d_logits, final_step_weights)
            d = tf.gradients(kl_loss, d)[0]
            d = tf.stop_gradient(d)
        perturb = self.embed_scale_l2(d, self.perturb_norm_length)
        perturbed_logits = compute_logits_fn(target + perturb)
        virtual_loss = self.kl_divergence_with_logits(logits, perturbed_logits, final_step_weights)
        return virtual_loss

    def mask_by_length(self, embed, seq_length):
        embed_steps = embed.get_shape().as_list()[1]
        # Subtract 1 from length to prevent the perturbation from going on 'eos'
        mask = tf.sequence_mask(seq_length - 1, maxlen=embed_steps)
        # mask (batch, num_timesteps, 1)
        mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
        return embed * mask

    # compute kl loss, and filter out non_final_seq loss
    def kl_divergence_with_logits(self, q_logits, p_logits, weights):
        # kl = sigma(q * (logq - logp))
        # q (None, n_classes)
        q = tf.nn.softmax(q_logits)
        # kl (None, )
        kl = tf.reduce_sum(q * (tf.nn.log_softmax(q_logits) - tf.nn.log_softmax(p_logits)), -1)
        num_labels = tf.reduce_sum(weights)
        num_labels = tf.where(tf.equal(num_labels, 0.), 1., _num_labels)
        loss = tf.identity(tf.reduce_sum(weights * kl) / num_labels, name='kl_loss')
        return loss

    @property
    def trainable_weights(self):
        return []

    @property
    def pretrain_weights(self):
        return []

    @property
    def pretrain_restorer(self):
        return []

class LanguageSequenceGeneratorLSTM(object):
    def __init__(self, rnn_cell_size, input_size, rnn_num_layers=1, lstm_keep_pro_out=1):
        with tf.variable_scope("lm_lstm_layer"):
            self.lm_lstm_layer = layers.LSTM(cell_size=rnn_cell_size, num_layers=rnn_num_layers, keep_prob=lstm_keep_pro_out)
            self.lm_lstm_layer.build(input_shape=[-1, input_size])
        with tf.variable_scope("ae_lstm_layer"):
            self.ae_lstm_layer = layers.LSTM(cell_size=rnn_cell_size, num_layers=rnn_num_layers, keep_prob=lstm_keep_pro_out)
            self.ae_lstm_layer.build(input_shape=[-1, input_size])

    @property
    def trainable_weights(self):
        return self.lm_lstm_layer.trainable_weights + self.ae_lstm_layer.trainable_weights

    @property
    def pretrain_weights(self):
        return [{x.op.name.split("/", 1)[1]: x for x in self.lm_lstm_layer.trainable_weights},
                {x.op.name.split("/", 1)[1]: x for x in self.ae_lstm_layer.trainable_weights}]

    @property
    def pretrain_restorer(self):
        restorers = []
        for pretrain_name_vars in self.pretrain_weights:
            restorers.append(tf.train.Saver(pretrain_name_vars))
        return restorers

class RnnOutputToEmbedding(object):
    def __init__(self, var_scope_name, vocab_size, input_size, embedding_weights, sampler=None):
        with tf.variable_scope(var_scope_name):
            self.softmax_loss = layers.SoftmaxLoss(vocab_size=vocab_size)
            self.softmax_loss.build(([-1, input_size],))
            self.toEmbedding = layers.RnnOutputToEmbedding(vocab_size, embedding_weights, self.softmax_loss.lin_w, self.softmax_loss.lin_b, sampler)
            self.toEmbedding.build(input_shape=[-1, input_size])

    @property
    def only_logits(self):
        return self.toEmbedding.only_logits

    @only_logits.setter
    def only_logits(self, bool_val):
        self.toEmbedding.only_logits = bool_val

    def __call__(self, inputs):
        return self.toEmbedding(inputs)

    @property
    def pretrain_weights(self):
        return [{x.op.name.split("/", 1)[1]: x for x in self.softmax_loss.trainable_weights}]

    @property
    def pretrain_restorer(self):
        restorers = []
        for pretrain_name_vars in self.pretrain_weights:
            restorers.append(tf.train.Saver(pretrain_name_vars))
        return restorers

class LanguageSequenceGenerator(object):
  #                   |outputs2embedding|
  #                           ^
  #                    +-+   +-+   +-+
  #   topic_states --> | |-->| |-->| |     language_model
  #                    +-+   +-+   +-+
  #                           ^
  #                   |outputs2embedding|
  #                           ^
  #                    +-+   +-+   +-+
  # content_states --> | |-->| |-->| |     auto_encoder
  #                    +-+   +-+   +-+
  #                     ^
  #              <sos_embedding>
    def __init__(self, ae_lstm_cell, lm_lstm_cell, rnnOutputToEmbedding):
        self.ae_lstm_cell = ae_lstm_cell
        self.lm_lstm_cell = lm_lstm_cell
        ae_sszie = self.ae_lstm_cell.state_size
        lm_sszie = self.lm_lstm_cell.state_size
        self.ae_cell_state_size = ae_sszie if isinstance(ae_sszie, tuple) else (ae_sszie,)
        self.lm_cell_state_size = lm_sszie if isinstance(lm_sszie, tuple) else (lm_sszie,)
        self.toEmbedding = rnnOutputToEmbedding

    def __call__(self, content_initial_states, topic_initial_states, step_one_inputs, seq_length, keep_prob = 1., return_vocab_index = False):
        # step_one_inputs (batch_size, embed_size)
        # erase first dimension value for while-loop
        embed_size = step_one_inputs.get_shape()[1].value
        step_one_inputs_shape = tf.shape(step_one_inputs)
        step_one_inputs = tf.reshape(step_one_inputs, (step_one_inputs_shape[0], step_one_inputs_shape[1]))
        # step_one_inputs (None, embed_size)
        step_one_inputs.set_shape((None, embed_size))
        time_steps = seq_length
        time = tf.constant(0, dtype='int32', name='time')
        output_tensor_array = tf.TensorArray(tf.float32, size=time_steps)
        def step(time, output_ta_t, inputs, content_states):
            # content_outputs (batch_size, rnn_size)
            content_outputs, content_states = self.ae_lstm_cell(inputs, content_states)
            # content_outputs (batch_size, 1, rnn_size)
            content_outputs = tf.expand_dims(content_outputs, 1)
            # content_outputs (batch_size, embed_size)
            content_outputs = tf.squeeze(self.toEmbedding(content_outputs), 1)
            output_ta_t = output_ta_t.write(time, content_outputs)
            return time + 1, output_ta_t, content_outputs, content_states
        # following steps
        final_outputs = tf.while_loop(cond=lambda time, *_: time < time_steps, body=step,
                                      loop_vars=(time, output_tensor_array, step_one_inputs, content_initial_states),
                                      parallel_iterations=32, swap_memory=True)
        output_tensor_array = final_outputs[1]
        # content_outputs (time_steps, batch_size, embed_size)
        content_outputs = output_tensor_array.stack()
        # content_outputs (batch_size, time_steps, rnn_size)
        content_outputs = tf.transpose(content_outputs, [1, 0, 2])
        if keep_prob < 1.:
            content_outputs = tf.nn.dropout(content_outputs, keep_prob)
        # content_outputs_embedding (batch_size, time_steps, embed_size)
        # content_outputs_embedding = self.toEmbedding(content_outputs)
        content_outputs_embedding = content_outputs
        # topic_outputs (batch_size, time_steps, rnn_size)
        # print("--------", content_outputs_embedding)
        # print("--------", content_initial_states)
        # print("--------", topic_initial_states)
        topic_outputs, _ = tf.nn.dynamic_rnn(self.lm_lstm_cell, content_outputs_embedding, initial_state=topic_initial_states)
        if not return_vocab_index:
            # topic_outputs (batch_size, time_steps, embed_size)
            topic_outputs = self.toEmbedding(topic_outputs)
            return topic_outputs
        else:
            # topic_outputs (batch_size, time_steps)
            self.toEmbedding.only_logits = True
            topic_outputs = self.toEmbedding(topic_outputs)
            self.toEmbedding.only_logits = False
            return topic_outputs

    def content_states(self, batch_size, dist = tf.random_uniform, distargs = {"minval": -1, "maxval": 1}):
        lstm_initial_state = tuple([tf.contrib.rnn.LSTMStateTuple(dist((batch_size, c_size), **distargs),
                                                                  dist((batch_size, h_size), **distargs)) for
                                    c_size, h_size in self.lm_cell_state_size])
        return lstm_initial_state

    def topic_states(self, batch_size, dist_fuse_w, dist = tf.random_normal, distargs = [{"mean": 0, "stddev": 1}, {"mean": 10, "stddev": 1}]):
        assert isinstance(distargs[0], dict), "item of distargs must be a dict"
        assert isinstance(dist_fuse_w, tf.Tensor), "dist_fuse_w must be a tf.Tensor"
        # dist_fuse_w (batch_size, topic_count)
        assert dist_fuse_w.shape[1] == len(distargs), "dist_fuse_w.shape[1] != len(distargs)"
        # dist_fuse_w (batch_size, topic_count, 1)
        dist_fuse_w = tf.expand_dims(dist_fuse_w, 2)
        lstm_initial_state = []
        for c_size, h_size in self.lm_cell_state_size:
            c_dist_values = []
            h_dist_values = []
            for args in distargs:
                c_value = dist((batch_size, c_size), **args)
                h_value = dist((batch_size, h_size), **args)
                # (batch, 1, rnn_size)
                c_value = tf.expand_dims(c_value, 1)
                h_value = tf.expand_dims(h_value, 1)
                c_dist_values.append(c_value)
                h_dist_values.append(h_value)
            # c_dist_tensor (batch_size, topic_count, rnn_size)
            c_dist_tensor = tf.concat(c_dist_values, axis=1)
            # c_dist_tensor (batch_size, topic_count, rnn_size)
            h_dist_tensor = tf.concat(h_dist_values, axis=1)
            # c_fused_dist_tensor (batch_size, rnn_size)
            c_fused_dist_tensor = tf.reduce_sum(c_dist_tensor * dist_fuse_w, axis=1)
            # h_fused_dist_tensor (batch_size, rnn_size)
            h_fused_dist_tensor = tf.reduce_sum(h_dist_tensor * dist_fuse_w, axis=1)
            lstm_initial_state.append(tf.contrib.rnn.LSTMStateTuple(c_fused_dist_tensor, h_fused_dist_tensor))
        return tuple(lstm_initial_state)

    def construct_batch_sequences_with_states(self, single_seq_tensor, single_seq_topic_label, single_weight,
                                              single_eos_indicators, single_seq_length, batch_size, unroll_steps,
                                              state_size, lstm_num_layers, bidrec=False):
        indice_tensor = tf.Variable(0, trainable=False)
        indice_tensor = tf.assign_add(indice_tensor, 1, use_locking=True)
        # dereference indice_tensor, int32_ref -> int32
        # indice_tensor = tf.identity(indice_tensor)
        # cast int32 to string
        indice_tensor = tf.py_func(lambda x: str(x), [indice_tensor], tf.string)
        initial_states = {}
        for i in range(lstm_num_layers):
            c_state_name = "{}_lstm_c".format(i)
            h_state_name = "{}_lstm_h".format(i)
            initial_states[c_state_name] = tf.zeros(state_size)
            initial_states[h_state_name] = tf.zeros(state_size)
        if bidrec:
            for i in range(lstm_num_layers):
                c_state_name = "{}_lstm_reverse_c".format(i)
                h_state_name = "{}_lstm_reverse_h".format(i)
                initial_states[c_state_name] = tf.zeros(state_size)
                initial_states[h_state_name] = tf.zeros(state_size)
        batch = tf.contrib.training.batch_sequences_with_states(
            input_key=indice_tensor,
            input_sequences={"X": single_seq_tensor, "weight": single_weight, "eos_indicators": single_eos_indicators},
            input_context={"topic_label": single_seq_topic_label},
            input_length=single_seq_length,
            initial_states=initial_states,
            num_unroll=unroll_steps,
            batch_size=batch_size,
            allow_small_batch=False,
            num_threads=4,
            capacity=batch_size * 10,
            make_keys_unique=True,
            make_keys_unique_seed=88888)
        def get_lstm_state():
            state_names = [("{}_lstm_c".format(i), "{}_lstm_h".format(i)) for i in range(lstm_num_layers)]
            lstm_initial_state = tuple(
                [tf.contrib.rnn.LSTMStateTuple(batch.state(c_state_name), batch.state(h_state_name)) for
                 (c_state_name, h_state_name) in state_names])
            if bidrec:
                reverse_state_names = [("{}_lstm_reverse_c".format(i), "{}_lstm_reverse_h".format(i)) for i in
                                       range(lstm_num_layers)]
                reverse_lstm_initial_state = tuple(
                    [tf.contrib.rnn.LSTMStateTuple(batch.state(c_state_name), batch.state(h_state_name)) for
                     (c_state_name, h_state_name) in reverse_state_names])
                return lstm_initial_state, reverse_lstm_initial_state
            else:
                return lstm_initial_state

        def save_lstm_state(new_state, reverse_new_state=None):
            state_names = [("{}_lstm_c".format(i), "{}_lstm_h".format(i)) for i in range(lstm_num_layers)]
            save_ops = []
            for (c_state, h_state), (c_name, h_name) in zip(new_state, state_names):
                save_ops.append(batch.save_state(c_name, c_state))
                save_ops.append(batch.save_state(h_name, h_state))
            if bidrec:
                reverse_state_names = [("{}_lstm_c".format(i), "{}_lstm_h".format(i)) for i in range(lstm_num_layers)]
                for (c_state, h_state), (c_name, h_name) in zip(reverse_new_state, reverse_state_names):
                    save_ops.append(batch.save_state(c_name, c_state))
                    save_ops.append(batch.save_state(h_name, h_state))
            return tf.group(*save_ops)
        return batch, get_lstm_state, save_lstm_state

    @property
    def trainable_weights(self):
        return []

    @property
    def pretrain_weights(self):
        return [dict(zip(map(lambda x: x.op.name, self.trainable_weights), self.trainable_weights))]

    @property
    def pretrain_restorer(self):
        return []

class ClassificationModelDenseHeader(object):
    def __init__(self, layer_sizes, input_size, num_classes, keep_prob=1., var_scope_name = None):
        def build():
            self.dense_header = keras.models.Sequential(name='cl_logits')
            for i, layer_size in enumerate(layer_sizes):
                if i == 0:
                    self.dense_header.add(keras.layers.Dense(layer_size, activation='relu', input_dim=input_size))
                else:
                    self.dense_header.add(keras.layers.Dense(layer_size, activation='relu'))

                if keep_prob < 1.:
                    self.dense_header.add(keras.layers.Dropout(1. - keep_prob))
                self.dense_header.add(keras.layers.Dense(num_classes))
        if var_scope_name is None:
            build()
        else:
            with tf.variable_scope(var_scope_name):
                build()

    def __call__(self, inputs):
        return self.dense_header(inputs)

    @property
    def trainable_weights(self):
        return self.dense_header.trainable_weights

    @property
    def pretrain_weights(self):
        return [{x.op.name.split("/", 1)[1]: x for x in self.dense_header.trainable_weights}]

    @property
    def pretrain_restorer(self):
        restorers = []
        for pretrain_name_vars in self.pretrain_weights:
            restorers.append(tf.train.Saver(pretrain_name_vars))
        return restorers

