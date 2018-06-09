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
    def __call__(self, inputs, lstm_initial_state, sequence_len = None):
        self.embedding = self.embedding_layer(inputs)
        return self.lstm_layer(self.embedding, lstm_initial_state, sequence_len)

    @property
    def trainable_weights(self):
        return self.embedding_layer.trainable_weights + self.lstm_layer.trainable_weights

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

