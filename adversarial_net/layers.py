from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
import numpy as np

class Embedding(keras.layers.Layer):
    """Embedding layer with frequency-based normalization and dropout."""

    def __init__(self,
               vocab_size,
               embedding_dim,
               normalize=False,
               vocab_freqs=None,
               keep_prob=1.,
               **kwargs):
        super(Embedding, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.normalized = normalize
        self.keep_prob = keep_prob

        if normalize:
          assert vocab_freqs is not None
          self.vocab_freqs = tf.constant(
              vocab_freqs, dtype=tf.float32, shape=(vocab_size, 1))

    def build(self, input_shape):
        self.var = self.add_weight(
          shape=(self.vocab_size, self.embedding_dim),
          initializer=tf.random_uniform_initializer(-1., 1.),
          name='embedding')

        if self.normalized:
          self.var = self._normalize(self.var)

        super(Embedding, self).build(input_shape)

    def call(self, x):
        embedded = tf.nn.embedding_lookup(self.var, x)
        if self.keep_prob < 1.:
          shape = embedded.get_shape().as_list()

          # Use same dropout masks at each timestep with specifying noise_shape.
          # This slightly improves performance.
          # Please see https://arxiv.org/abs/1512.05287 for the theoretical
          # explanation.
          embedded = tf.nn.dropout(
              embedded, self.keep_prob, noise_shape=(shape[0], 1, shape[2]))
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

    def __init__(self, cell_size, num_layers=1, keep_prob=1., **kwargs):
        super(LSTM, self).__init__(**kwargs)
        self.cell_size = cell_size
        self.num_layers = num_layers
        self.keep_prob = keep_prob
        self.reuse = None

    def build(self, input_shape):
        super(LSTM, self).build(input_shape)

    def __call__(self, x, initial_state, seq_length = None):
        with tf.variable_scope(self.name, reuse=self.reuse) as vs:
            cell = tf.contrib.rnn.MultiRNNCell([
              tf.contrib.rnn.BasicLSTMCell(
                  self.cell_size,
                  forget_bias=0.0,
                  reuse=tf.get_variable_scope().reuse)
              for _ in range(self.num_layers)
            ])

            # shape(x) = (batch_size, num_timesteps, embedding_dim)
            lstm_out, next_state = tf.nn.dynamic_rnn(cell, x, initial_state=initial_state, sequence_length=seq_length)
            # shape(lstm_out) = (batch_size, timesteps, cell_size)

            if self.keep_prob < 1.:
                lstm_out = tf.nn.dropout(lstm_out, self.keep_prob)

            if self.reuse is None:
                self._trainable_weights = vs.global_variables()

        self.reuse = True

        return lstm_out, next_state

class SoftmaxLoss(keras.layers.Layer):
    """Softmax xentropy loss with candidate sampling."""

    def __init__(self,
               vocab_size,
               num_candidate_samples=-1,
               vocab_freqs=None,
               **kwargs):
        super(SoftmaxLoss, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.num_candidate_samples = num_candidate_samples
        self.vocab_freqs = vocab_freqs
        self.multiclass_dense_layer = keras.layers.Dense(self.vocab_size)
        self.lm_acc = None

    def build(self, input_shape):
        input_shape = input_shape[0]
        self.lin_w = self.add_weight(
          shape=(input_shape[-1], self.vocab_size),
          name='lm_lin_w',
          initializer=keras.initializers.glorot_uniform())
        self.lin_b = self.add_weight(
          shape=(self.vocab_size,),
          name='lm_lin_b',
          initializer=keras.initializers.glorot_uniform())
        self.multiclass_dense_layer.build(input_shape)

        super(SoftmaxLoss, self).build(input_shape)

    def call(self, inputs):
        x, labels, weights = inputs
        if self.num_candidate_samples > -1:
            assert self.vocab_freqs is not None
            labels = tf.cast(labels, tf.int64)
            labels_reshaped = tf.reshape(labels, [-1])
            labels_reshaped = tf.expand_dims(labels_reshaped, -1)
            sampled = tf.nn.fixed_unigram_candidate_sampler(
              true_classes=labels_reshaped,
              num_true=1,
              num_sampled=self.num_candidate_samples,
              unique=True,
              range_max=self.vocab_size,
              unigrams=self.vocab_freqs)
            inputs_reshaped = tf.reshape(x, [-1, int(x.get_shape()[2])])

            lm_loss = tf.nn.sampled_softmax_loss(
              weights=tf.transpose(self.lin_w),
              biases=self.lin_b,
              labels=labels_reshaped,
              inputs=inputs_reshaped,
              num_sampled=self.num_candidate_samples,
              num_classes=self.vocab_size,
              sampled_values=sampled)
            lm_loss = tf.reshape(lm_loss, [int(x.get_shape()[0]), int(x.get_shape()[1])])
            logits = tf.nn.bias_add(tf.matmul(inputs_reshaped, self.lin_w), self.lin_b)
        else:
            logits = self.multiclass_dense_layer(x)
            lm_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
              logits=logits, labels=labels)
            logits = tf.reshape(logits, (-1, int(logits.get_shape()[-1])))
            labels_reshaped = tf.reshape(labels, (-1, 1))

        self.lm_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=-1), labels_reshaped), tf.float32))
        lm_loss = tf.identity(tf.reduce_sum(lm_loss * weights) / _num_labels(weights), name='lm_xentropy_loss')
        return lm_loss

class ClassificationSparseSoftmaxLoss(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ClassificationSparseSoftmaxLoss, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ClassificationSparseSoftmaxLoss, self).build(input_shape)

    def call(self, inputs):
        logits, labels, weights = inputs
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        return tf.identity(tf.reduce_sum(weights * loss) / _num_labels(weights), name='classification_xentropy')

def accuracy(logits, labels, weights):
    eq = tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32)
    acc = tf.identity(tf.reduce_sum(weights * eq) / _num_labels(weights), name='accuracy')
    return acc

def _num_labels(weights):
    """Number of 1's in weights. Returns 1. if 0."""
    num_labels = tf.reduce_sum(weights)
    num_labels = tf.where(tf.equal(num_labels, 0.), 1., num_labels)
    return num_labels
