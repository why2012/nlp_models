from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
import numpy as np
from adversarial_net.utils import getLogger

logger = getLogger("layer")

class Embedding(keras.layers.Layer):
    """Embedding layer with frequency-based normalization and dropout."""

    def __init__(self,
               vocab_size,
               embedding_dim,
               normalize=False,
               vocab_freqs=None,
               keep_prob=1.,
               lock_embedding = False,
               **kwargs):
        super(Embedding, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.normalized = normalize
        self.keep_prob = keep_prob
        self.lock_embedding = lock_embedding

        if normalize:
          assert vocab_freqs is not None
          self.vocab_freqs = tf.constant(
              vocab_freqs, dtype=tf.float32, shape=(vocab_size, 1))

    def build(self, input_shape):
        self.var = self.add_weight(
          shape=(self.vocab_size, self.embedding_dim),
          initializer=tf.random_uniform_initializer(-1., 1.),
          trainable=not self.lock_embedding,
          name='embedding')
        if self.normalized:
          self.var = self._normalize(self.var)

        super(Embedding, self).build(input_shape)

    def call(self, x):
        embedded = tf.nn.embedding_lookup(self.var, x)
        if self.keep_prob < 1.:
          shape = embedded.get_shape().as_list()
          if None in shape:
              shape = tf.shape(embedded)
          # (batch_size, time_steps, embed_size)
          if len(embedded.get_shape().as_list()) == 3:
            noise_shape = (shape[0], 1, shape[2])
          # (batch_size, embed_size)
          else:
            noise_shape = (shape[0], shape[1])

          # Use same dropout masks at each timestep with specifying noise_shape.
          # This slightly improves performance.
          # Please see https://arxiv.org/abs/1512.05287 for the theoretical
          # explanation.
          embedded = tf.nn.dropout(embedded, self.keep_prob, noise_shape=noise_shape)
        return embedded

    def _normalize(self, emb):
        weights = self.vocab_freqs / tf.reduce_sum(self.vocab_freqs)
        mean = tf.reduce_sum(weights * emb, 0, keep_dims=True)
        var = tf.reduce_sum(weights * tf.pow(emb - mean, 2.), 0, keep_dims=True)
        stddev = tf.sqrt(1e-6 + var)
        return (emb - mean) / stddev

class LSTM(keras.layers.Layer):
    """LSTM layer using dynamic_rnn.

    Exposes variables in `trainable_weights` property.
    """

    def __init__(self, cell_size, num_layers=1, keep_prob=1., forget_bias = 0.0, name = "lstm", **kwargs):
        super(LSTM, self).__init__(name=name, **kwargs)
        self.cell_size = cell_size
        self.num_layers = num_layers
        self.keep_prob = keep_prob
        self.reuse = None
        self.cell = None
        self.forget_bias = forget_bias

    def build(self, input_shape):
        super(LSTM, self).build(input_shape)
        with tf.variable_scope(self.name, reuse=self.reuse) as vs:
            cell = tf.contrib.rnn.MultiRNNCell([
              tf.contrib.rnn.BasicLSTMCell(
                  self.cell_size,
                  forget_bias=self.forget_bias,
                  reuse=tf.get_variable_scope().reuse)
              for _ in range(self.num_layers)
            ])
            cell.build(input_shape)
            if not cell.trainable_variables:
                cell(tf.random_uniform((1, input_shape[-1])), cell.zero_state(1, tf.float32))
            self.cell = cell
            if self.reuse is None:
                self._trainable_weights = vs.global_variables()
        self.reuse = True

    def __call__(self, x, initial_state, seq_length = None):
        # shape(x) = (batch_size, num_timesteps, embedding_dim)
        lstm_out, next_state = tf.nn.dynamic_rnn(self.cell, x, initial_state=initial_state, sequence_length=seq_length)
        # shape(lstm_out) = (batch_size, timesteps, cell_size)
        if self.keep_prob < 1.:
            lstm_out = tf.nn.dropout(lstm_out, self.keep_prob)
        return lstm_out, next_state

class SoftmaxLoss(keras.layers.Layer):
    """Softmax xentropy loss with candidate sampling."""

    def __init__(self,
                 vocab_size,
                 num_candidate_samples=-1,
                 vocab_freqs=None,
                 hard_mode=False,
                 use_sampler=True,
               **kwargs):
        super(SoftmaxLoss, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.num_candidate_samples = num_candidate_samples
        self.vocab_freqs = vocab_freqs
        self.hard_mode = hard_mode
        self.use_sampler = use_sampler
        self.lm_acc = None

    def build(self, input_shape):
        if self.hard_mode:
            self.dense = keras.layers.Dense(self.vocab_size, activation='softmax', input_dim=input_shape[-1])
        else:
            input_shape = input_shape[0]
            self.lin_w = self.add_weight(
              shape=(self.vocab_size, input_shape[-1]),
              name='lm_lin_w',
              initializer=keras.initializers.glorot_uniform())
            self.lin_b = self.add_weight(
              shape=(self.vocab_size,),
              name='lm_lin_b',
              initializer=keras.initializers.glorot_uniform())

        super(SoftmaxLoss, self).build(input_shape)

    def call(self, inputs):
        x, labels, weights = inputs
        labels = tf.cast(labels, tf.int64)
        labels_reshaped = tf.reshape(labels, [-1])
        inputs_reshaped = tf.reshape(x, [-1, int(x.get_shape()[2])])
        if self.hard_mode:
            logits = self.dense(inputs_reshaped)
            lm_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_reshaped, logits=logits)
            lm_loss = tf.reshape(lm_loss, [int(x.get_shape()[0]), int(x.get_shape()[1])])
        else:
            labels_reshaped = tf.expand_dims(labels_reshaped, -1)
            assert self.num_candidate_samples > -1, "self.num_candidate_samples must > -1"
            sampled = None
            if self.use_sampler:
                assert self.vocab_freqs is not None
                sampled = tf.nn.fixed_unigram_candidate_sampler(
                  true_classes=labels_reshaped,
                  num_true=1,
                  num_sampled=self.num_candidate_samples,
                  unique=True,
                  range_max=self.vocab_size,
                  unigrams=self.vocab_freqs)

            lm_loss = tf.nn.sampled_softmax_loss(
              weights=self.lin_w,
              biases=self.lin_b,
              labels=labels_reshaped,
              inputs=inputs_reshaped,
              num_sampled=self.num_candidate_samples,
              num_classes=self.vocab_size,
              sampled_values=sampled)
            lm_loss = tf.reshape(lm_loss, [int(x.get_shape()[0]), int(x.get_shape()[1])])
            logits = tf.nn.bias_add(tf.matmul(inputs_reshaped, tf.transpose(self.lin_w)), self.lin_b)

        self.lm_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=-1), labels_reshaped), tf.float32))
        lm_loss = tf.identity(tf.reduce_sum(lm_loss * weights) / num_labels(weights), name='lm_xentropy_loss')
        return lm_loss

class RnnOutputToEmbedding(keras.layers.Layer):
    # rnn_output (batch_size, seq_length, rnn_size) -> (batch_size * seq_length, rnn_size) -> (~, vocab_size) ->
    # sparse((~, vocab_size)) * embedding -> (~, embed_size) -> (batch_size, seq_length, embed_size)
    def __init__(self, vocab_size, embedding_weights, var_w = None, var_b = None, sampler = None, **kwargs):
        super(RnnOutputToEmbedding, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        # embedding_weights (vocab_size, embed_size)
        # avoid extra gradients flow
        self.embedding_weights = tf.stop_gradient(embedding_weights)
        # var_w (vocab_size, rnn_size)
        self.var_w = var_w
        # var_b (vocab_size)
        self.var_b = var_b
        # sampling logit
        self.sampler = sampler
        self.only_logits = False

    def build(self, input_shape):
        if self.var_w is None:
            self.var_w = self.add_weight(
                shape=(self.vocab_size, input_shape[-1]),
                name='var_w',
                initializer=keras.initializers.glorot_uniform())
        if self.var_b is None:
            self.var_b = self.add_weight(
                shape=(self.vocab_size,),
                name='var_b',
                initializer=keras.initializers.glorot_uniform())
        super(RnnOutputToEmbedding, self).build(input_shape)

    def call(self, rnn_outputs):
        only_logits = self.only_logits
        rnn_outputs_shape = tf.shape(rnn_outputs)
        batch_size, seq_length, rnn_size = rnn_outputs_shape[0], rnn_outputs_shape[1], rnn_outputs_shape[2]
        # rnn_output (batch_size * seq_length, rnn_size)
        rnn_outputs = tf.reshape(rnn_outputs, (-1, rnn_size))
        # vocab_logits (batch_size * seq_length, vocab_size)
        vocab_logits = tf.matmul(rnn_outputs, self.var_w, transpose_b=True)
        if self.sampler:
            vocab_logits = self.sampler(vocab_logits)
        # maximum_indices (batch_size * seq_length,)
        maximum_indices = tf.argmax(vocab_logits, -1, output_type=tf.int32)
        if not only_logits:
            indices = tf.stack([tf.range(batch_size * seq_length), maximum_indices], 1)
            indices = tf.cast(indices, tf.int64)
            # values = tf.gather_nd(vocab_logits, indices)
            values = tf.ones(shape=(tf.shape(vocab_logits)[0],))
            # sparse_vocab_logits sparse(batch_size * seq_length, vocab_size)
            sparse_vocab_logits = tf.SparseTensor(indices, values, tf.shape(vocab_logits, out_type=tf.int64))
            # embedding (batch_size * seq_length, embed_size)
            embedding = tf.sparse_tensor_dense_matmul(sparse_vocab_logits, self.embedding_weights)
            # embedding (batch_size, seq_length, embed_size)
            # print("----------", embedding)
            embedding = tf.reshape(embedding, (batch_size, seq_length, self.embedding_weights.get_shape()[-1]))
            # print("----------", embedding)
            return embedding
        else:
            vocab_logits = tf.reshape(maximum_indices, (batch_size, seq_length))
            return vocab_logits

class ClassificationSparseSoftmaxLoss(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ClassificationSparseSoftmaxLoss, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ClassificationSparseSoftmaxLoss, self).build(input_shape)

    def call(self, inputs):
        logits, labels, weights = inputs
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        return tf.identity(tf.reduce_sum(weights * loss) / num_labels(weights), name='classification_xentropy')

def accuracy(logits, labels, weights):
    if logits.get_shape().as_list()[-1] == 1:
        logits = tf.squeeze(logits)
        eq = tf.cast(tf.equal(tf.round(logits), labels), tf.float32)
        acc = tf.identity(tf.reduce_sum(weights * eq) / num_labels(weights), name='accuracy')
    else:
        assert labels.dtype in [tf.int32, tf.int64], "labels.dtype must in [int32, int64]"
        eq = tf.cast(tf.equal(tf.argmax(logits, 1, output_type=labels.dtype), labels), tf.float32)
        acc = tf.identity(tf.reduce_sum(weights * eq) / num_labels(weights), name='accuracy')
    return acc

def num_labels(weights):
    """Number of 1's in weights. Returns 1. if 0."""
    _num_labels = tf.reduce_sum(weights)
    _num_labels = tf.where(tf.equal(_num_labels, 0.), 1., _num_labels)
    return _num_labels
