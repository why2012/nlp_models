from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pickle
from os import path as osp
import tensorflow as tf
import multiprocessing
from adversarial_net.preprocessing import AutoPaddingInMemorySamplePool, WordCounter
from adversarial_net.preprocessing import SimpleInMemorySamplePool

def getDatasetFilePath(datapath, dataset, modelname):
    if dataset == "imdb":
        if modelname == "language_model":
            return osp.join(datapath, "imdb_lm_dataset.pickle")
        elif modelname == "classification_model":
            return osp.join(datapath, "imdb_classification_dataset.pickle")

class DataLoader(object):
    def __init__(self, base_dir, dataset):
        self.supported_dataset = ["imdb"]
        if dataset not in self.supported_dataset:
            raise Exception("Unsupported dataset: %s" % dataset)
        self.base_dir = base_dir
        self.dataset = dataset
        self.rand = np.random.RandomState(seed=8888)
        self.wordCounter = WordCounter()

    def load_data(self, include_unsup = True):
        if self.dataset == "imdb":
            training_dataset, testing_dataset, unsup_dataset = self._load_imdb(include_unsup=include_unsup)
        else:
            raise Exception("Unsupport dataset %s" % self.dataset)
        if include_unsup:
            return training_dataset, testing_dataset, unsup_dataset
        else:
            return training_dataset, testing_dataset

    def _load_imdb(self, vocab_filename="imdb_word_freqs.pickle", dataset_filename="imdb_dataset.pickle",
                   include_unsup=True):
        vocab_abspath = osp.join(self.base_dir, vocab_filename)
        dataset_abspath = osp.join(self.base_dir, dataset_filename)
        with open(vocab_abspath, "rb") as f:
            self.wordCounter.words_list = pickle.load(f)
        with open(dataset_abspath, "rb") as f:
            dataset_pack = pickle.load(f)
        training_pos_data = dataset_pack["training_pos_data"]
        training_neg_data = dataset_pack["training_neg_data"]
        testing_pos_data = dataset_pack["testing_pos_data"]
        testing_neg_data = dataset_pack["testing_neg_data"]
        unsup_data = dataset_pack["unsup_data"]

        training_pos_label = dataset_pack["training_pos_label"]
        training_neg_label = dataset_pack["training_neg_label"]
        testing_pos_label = dataset_pack["testing_pos_label"]
        testing_neg_label = dataset_pack["testing_neg_label"]

        training_data = np.concatenate([training_pos_data, training_neg_data])
        testing_data = np.concatenate([testing_pos_data, testing_neg_data])
        training_label = np.concatenate([training_pos_label, training_neg_label])
        testing_label = np.concatenate([testing_pos_label, testing_neg_label])
        if include_unsup:
            unsup_data = np.array(unsup_data)

        training_data_permutation = self.rand.permutation(len(training_data))
        testing_data_permutation = self.rand.permutation(len(testing_data))
        unsup_data_permutation = []
        if include_unsup:
            unsup_data_permutation = self.rand.permutation(len(unsup_data))

        training_data = training_data[training_data_permutation]
        testing_data = testing_data[testing_data_permutation]
        training_label = training_label[training_data_permutation]
        testing_label = testing_label[testing_data_permutation]
        if include_unsup:
            unsup_data = unsup_data[unsup_data_permutation]

        if include_unsup:
            return (training_data, training_label), (testing_data, testing_label), (unsup_data, )
        else:
            return (training_data, training_label), (testing_data, testing_label), (None, )

def construct_data_queue(data_pool, n_thread, batch_size, collection_name=tf.GraphKeys.QUEUE_RUNNERS,
                         queue_class=tf.FIFOQueue, threshold=100):
    import traceback
    def enqueue_func():
        data = []
        try:
            data = data_pool.__next__()
        except Exception as e:
            print(e)
            traceback.print_exc()
        return data
    if data_pool.get_y_in_batch:
        dtypes = [tf.int32, tf.int32]
        enqueue_tensor_name = "sample_and_label_py_func"
        queue_name = "sample_and_label_queue"
        enqueue_name = "sample_and_label_enqueue"
        dequeue_name = "sample_and_label_dequeue"
        if data_pool.y_is_sequence:
            queue_shape = [[data_pool.unroll_num], [data_pool.unroll_num]]
        else:
            queue_shape = [[data_pool.unroll_num], [1]]
    else:
        dtypes = [tf.int32]
        enqueue_tensor_name = "sample__py_func"
        queue_name = "sample_queue"
        enqueue_name = "sample_enqueue"
        dequeue_name = "sample_dequeue"
        queue_shape = [[data_pool.unroll_num]]
    if data_pool.get_sequence_len_in_batch:
        dtypes.append(tf.int32)
        queue_shape.append([1])
    enqueue_tensors = tf.py_func(enqueue_func, [], dtypes, name=enqueue_tensor_name)
    if queue_class == tf.RandomShuffleQueue:
        queue = queue_class(batch_size * threshold, dtypes=dtypes,
                            min_after_dequeue=batch_size * np.ceil(threshold / 20).astype(np.int32), shapes=queue_shape,
                            name=queue_name)
    else:
        queue = queue_class(batch_size * threshold, dtypes, name=queue_name, shapes=queue_shape)
    enqueue_op = queue.enqueue_many(enqueue_tensors, name=enqueue_name)
    dequeue_op = queue.dequeue_many(batch_size, name=dequeue_name)
    queue_runner = tf.train.QueueRunner(queue, [enqueue_op] * n_thread)
    tf.add_to_collection(collection_name, queue_runner)

    return dequeue_op

def construct_language_model_input_tensors(datapath, batch_size, unroll_steps):
    with open(osp.join(datapath, "imdb_lm_dataset.pickle"), "rb") as f:
        X_y = pickle.load(f)
    X_train = X_y["X"]
    y_train = X_y["y"]
    pool = AutoPaddingInMemorySamplePool(X_train, y=y_train, bins_count=50, batch_size=batch_size, mode="specific",
                                         y_is_sequence=True,
                                         unroll_num=unroll_steps, get_y_in_batch=True, get_sequence_len_in_batch=False)
    dequeue_op = construct_data_queue(data_pool=pool, n_thread=multiprocessing.cpu_count(), batch_size=batch_size,
                                      queue_class=tf.FIFOQueue)
    return dequeue_op

def construct_language_model_input_tensor_with_state(datapath, batch_size, unroll_steps, lstm_num_layers, state_size, dataset, bidrec = False):
    def args_fn(datapack):
        X_train = datapack["X"]
        y_train = datapack["y"]
        X_y_samples = list(zip(X_train, y_train))
        pool = SimpleInMemorySamplePool(X_y_samples, chunk_size=1)

        def get_single_example():
            # ([1, 2 * seq_len], [1])
            sample, indice = pool.__next__()
            x_sample = np.array(sample[0][0]).reshape(-1, 1)
            y_sample = np.array(sample[0][1]).reshape(-1, 1)
            assert x_sample.shape[0] == y_sample.shape[0]
            weights = np.ones((x_sample.shape[0], 1), dtype=np.float32)
            weights[-1][0] = 0  # eos tag has 0 weights
            indice = str(indice[0])
            return x_sample, y_sample, weights, indice

        X_tensor, y_tensor, weight_tensor, indice_tensor = tf.py_func(get_single_example, [],
                                                                      [tf.int32, tf.int32, tf.float32, tf.string],
                                                                      name="get_single_example")
        X_tensor.set_shape([None, 1])
        y_tensor.set_shape([None, 1])
        weight_tensor.set_shape([None, 1])
        return indice_tensor, {"X": X_tensor, "y": y_tensor, "weight": weight_tensor}, {}, tf.shape(X_tensor)[0]

    return construct_input_tensor_with_state(args_fn, datapath, batch_size, unroll_steps, lstm_num_layers, state_size, dataset, "language_model", bidrec)

def construct_classification_model_input_tensor_with_state(datapath, phase, batch_size, unroll_steps, lstm_num_layers, state_size, dataset, bidrec = False):
    if phase == "train":
        X_name = "X_train"
        y_name = "y_train"
        weight_name = "weight_train"
    elif phase == "test":
        X_name = "X_test"
        y_name = "y_test"
        weight_name = "weight_test"
    elif phase == "valid":
        X_name = "X_val"
        y_name = "y_val"
        weight_name = "weight_val"
    else:
        raise Exception("Unsupport phase %s" % phase)
    def args_fn(datapack):
        X_train = datapack[X_name]
        y_train = datapack[y_name]
        weights = datapack[weight_name]
        X_y_w_samples = list(zip(X_train, y_train, weights))
        pool = SimpleInMemorySamplePool(X_y_w_samples, chunk_size=1)

        def get_single_example():
            sample, indice = pool.__next__()
            x_sample = np.array(sample[0][0]).reshape(-1, 1)
            y_sample = np.array(sample[0][1], dtype=np.int32)
            weights_sample = np.array(sample[0][2], dtype=np.float32).reshape(-1, 1)
            indice = str(indice[0])
            return x_sample, y_sample, weights_sample, indice

        X_tensor, y_tensor, weight_tensor, indice_tensor = tf.py_func(get_single_example, [],
                                                                      [tf.int32, tf.int32, tf.float32, tf.string],
                                                                      name="get_single_example")
        X_tensor.set_shape([None, 1])
        y_tensor.set_shape([1])
        weight_tensor.set_shape([None, 1])
        return indice_tensor, {"X": X_tensor, "weight": weight_tensor}, {"y": y_tensor}, tf.shape(X_tensor)[0]

    return construct_input_tensor_with_state(args_fn, datapath, batch_size, unroll_steps, lstm_num_layers, state_size, dataset, "classification_model", bidrec)

def construct_input_tensor_with_state(args_fn, datapath, batch_size, unroll_steps, lstm_num_layers, state_size, dataset, modelname, bidrec = False):
    with open(getDatasetFilePath(datapath=datapath, dataset=dataset, modelname=modelname), "rb") as f:
        datapack = pickle.load(f)
    indice_tensor, input_sequences, input_context, input_length = args_fn(datapack)
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
        input_sequences=input_sequences,
        input_context=input_context,
        input_length=input_length,
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
            reverse_state_names = [("{}_lstm_reverse_c".format(i), "{}_lstm_reverse_h".format(i)) for i in range(lstm_num_layers)]
            reverse_lstm_initial_state = tuple(
                [tf.contrib.rnn.LSTMStateTuple(batch.state(c_state_name), batch.state(h_state_name)) for
                 (c_state_name, h_state_name) in reverse_state_names])
            return lstm_initial_state, reverse_lstm_initial_state
        else:
            return lstm_initial_state
    def save_lstm_state(new_state, reverse_new_state = None):
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
