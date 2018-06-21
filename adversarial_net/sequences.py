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

class LanguageSequenceGeneratorLSTM(object):
    def __init__(self, rnn_cell_size, rnn_num_layers=1, lstm_keep_pro_out=1):
        with tf.variable_scope("lm_lstm_layer"):
            self.lm_lstm_layer = layers.LSTM(cell_size=rnn_cell_size, num_layers=rnn_num_layers, keep_prob=lstm_keep_pro_out)
        with tf.variable_scope("ae_lstm_layer"):
            self.ae_lstm_layer = layers.LSTM(cell_size=rnn_cell_size, num_layers=rnn_num_layers, keep_prob=lstm_keep_pro_out)

    @property
    def trainable_weights(self):
        return self.lm_lstm_layer.trainable_weights + self.ae_lstm_layer.trainable_weights

    @property
    def pretrain_weights(self):
        return [dict(zip(map(lambda x: x.op.name.split("/", 1)[1], self.lm_lstm_layer.trainable_weights), self.lm_lstm_layer.trainable_weights)),
                dict(zip(map(lambda x: x.op.name.split("/", 1)[1], self.ae_lstm_layer.trainable_weights), self.ae_lstm_layer.trainable_weights))]

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

    def __call__(self, content_initial_states, topic_initial_states, step_one_inputs, seq_length, keep_prob = 1.):
        time_steps = seq_length
        time = tf.constant(0, dtype='int32', name='time')
        output_tensor_array = tf.TensorArray(tf.float32)
        def step(time, output_ta_t, inputs, content_states):
            content_outputs, content_states = self.ae_lstm_cell(inputs, content_states)
            output_ta_t.write(time, content_outputs)
            return time + 1, output_ta_t, content_outputs, content_states
        final_outputs = tf.while_loop(cond=lambda time, *_: time < time_steps, body=step,
                                      loop_vars=(time, output_tensor_array, step_one_inputs, content_initial_states),
                                      parallel_iterations=32, swap_memory=True)
        output_tensor_array = final_outputs[1]
        # content_outputs (time_steps, batch_size, rnn_size)
        content_outputs = output_tensor_array.stack()
        # content_outputs (batch_size, time_steps, rnn_size)
        content_outputs = tf.transpose(content_outputs, [1, 0, 2])
        if keep_prob < 1.:
            content_outputs = tf.nn.dropout(content_outputs, keep_prob)
        # content_outputs_embedding (batch_size, time_steps, embed_size)
        content_outputs_embedding = self.toEmbedding(content_outputs)
        # topic_outputs (batch_size, time_steps, rnn_size)
        topic_outputs, _ = tf.nn.dynamic_rnn(self.lm_cell_state_size, content_outputs_embedding, initial_state=topic_initial_states)
        # topic_outputs (batch_size, time_steps, embed_size)
        topic_outputs = self.toEmbedding(topic_outputs)
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
            c_dist_tensor = tf.stack(c_dist_values, axis=1)
            # c_dist_tensor (batch_size, topic_count, rnn_size)
            h_dist_tensor = tf.stack(h_dist_values, axis=1)
            # c_fused_dist_tensor (batch_size, rnn_size)
            c_fused_dist_tensor = tf.reduce_sum(c_dist_tensor * dist_fuse_w, axis=1)
            # h_fused_dist_tensor (batch_size, rnn_size)
            h_fused_dist_tensor = tf.reduce_sum(h_dist_tensor * dist_fuse_w, axis=1)
            lstm_initial_state.append(tf.contrib.rnn.LSTMStateTuple(c_fused_dist_tensor, h_fused_dist_tensor))
        return tuple(lstm_initial_state)

    def construct_batch_sequences_with_states(self, single_seq_tensor, single_seq_content_label, single_seq_topic_label, batch_size, unroll_steps, state_size, lstm_num_layers, bidrec = False):
        indice_tensor = tf.Variable(0, trainable=False)
        indice_tensor = tf.assign_add(indice_tensor, 1, use_locking=True)
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
            input_sequences=single_seq_tensor,
            input_context={"content_label": single_seq_content_label, "topic_label": single_seq_topic_label},
            input_length=tf.shape(single_seq_tensor)[0],
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
    def __init__(self, layer_sizes, input_size, num_classes, keep_prob=1.):
        self.dense_header = keras.models.Sequential(name='cl_logits')
        for i, layer_size in enumerate(layer_sizes):
            if i == 0:
                self.dense_header.add(keras.layers.Dense(layer_size, activation='relu', input_dim=input_size))
            else:
                self.dense_header.add(keras.layers.Dense(layer_size, activation='relu'))

            if keep_prob < 1.:
                self.dense_header.add(keras.layers.Dropout(1. - keep_prob))
            self.dense_header.add(keras.layers.Dense(num_classes))

    def __call__(self, inputs):
        return self.dense_header(inputs)

    @property
    def trainable_weights(self):
        return self.dense_header.trainable_weights

    @property
    def pretrain_weights(self):
        return [dict(zip(map(lambda x: x.op.name, self.trainable_weights), self.trainable_weights))]

    @property
    def pretrain_restorer(self):
        return []

