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
        elif modelname == "autoencoder_model":
            return osp.join(datapath, "imdb_ae_dataset.pickle")
        elif modelname == "word_freqs":
            return osp.join(datapath, "imdb_word_freqs.pickle")
    elif dataset == "summary":
        if modelname == "summary_model":
            return osp.join(datapath, "summary_dataset.pickle")
        elif modelname == "word_freqs":
            return osp.join(datapath, "merged_summary_word_freqs.pickle")
    elif dataset == "truncated_summary":
        if modelname == "summary_model":
            return osp.join(datapath, "truncated_summary_dataset.pickle")
    else:
        raise Exception("No such dataset %s" % dataset)

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

    @classmethod
    def reload_word_counter(cls, vocab_abspath):
        wordCounter = WordCounter()
        with open(vocab_abspath, "rb") as f:
            wordCounter.words_list = pickle.load(f)
        return wordCounter

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

def construct_language_model_input_tensor_with_state(datapath, batch_size, unroll_steps, lstm_num_layers, state_size, dataset, bidrec = False, **kwargs):
    def args_fn(datapack):
        X_train = datapack["X"]
        y_train = datapack["y"]
        X_y_samples = list(zip(X_train, y_train))
        pool = SimpleInMemorySamplePool(X_y_samples, chunk_size=1)

        def get_single_example():
            # (([seq_len], [seq_len]), [1])
            sample, indice = pool.__next__()
            x_sample = np.array(sample[0][0]).reshape(-1, 1)
            y_sample = np.array(sample[0][1]).reshape(-1, 1)
            assert x_sample.shape[0] == y_sample.shape[0], "illegal shape"
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

def construct_autoencoder_model_input_tensor_with_state(datapath, batch_size, unroll_steps, lstm_num_layers, state_size, dataset, bidrec = False, **kwargs):
    def args_fn(datapack):
        X_train = datapack["X"]
        y_train = datapack["y"]
        weight_train = datapack["weight"]
        X_y_w_samples = list(zip(X_train, y_train, weight_train))
        pool = SimpleInMemorySamplePool(X_y_w_samples, chunk_size=1)

        def get_single_example():
            # (([2 * seq_len - 1, 1], [2 * seq_len - 1, 1], [2 * seq_len - 1, 1]), [1])
            sample, indice = pool.__next__()
            x_sample = sample[0][0]
            y_sample = sample[0][1]
            weights = sample[0][2]
            assert x_sample.shape[0] == y_sample.shape[0] == weights.shape[0], "illegal shape"
            indice = str(indice[0])
            return x_sample, y_sample, weights, indice

        X_tensor, y_tensor, weight_tensor, indice_tensor = tf.py_func(get_single_example, [],
                                                                      [tf.int32, tf.int32, tf.float32, tf.string],
                                                                      name="get_single_example")
        X_tensor.set_shape([None, 1])
        y_tensor.set_shape([None, 1])
        weight_tensor.set_shape([None, 1])
        return indice_tensor, {"X": X_tensor, "y": y_tensor, "weight": weight_tensor}, {}, tf.shape(X_tensor)[0]

    return construct_input_tensor_with_state(args_fn, datapath, batch_size, unroll_steps, lstm_num_layers, state_size, dataset, "autoencoder_model", bidrec)

def construct_classification_model_input_tensor_with_state(datapath, phase, batch_size, unroll_steps, lstm_num_layers, state_size, dataset, bidrec = False, count_examples = [0], **kwargs):
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
        y_train = np.array(datapack[y_name], dtype=np.int64)
        weights = datapack[weight_name]
        X_y_w_samples = list(zip(X_train, y_train, weights))
        pool = SimpleInMemorySamplePool(X_y_w_samples, chunk_size=1)
        count_examples[0] = len(X_y_w_samples)

        def get_single_example():
            sample, indice = pool.__next__()
            x_sample = np.array(sample[0][0]).reshape(-1, 1)
            y_sample = sample[0][1]
            weights_sample = np.array(sample[0][2], dtype=np.float32).reshape(-1, 1)
            indice = str(indice[0])
            return x_sample, y_sample, weights_sample, indice

        X_tensor, y_tensor, weight_tensor, indice_tensor = tf.py_func(get_single_example, [],
                                                                      [tf.int32, tf.int64, tf.float32, tf.string],
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

def construct_summary_model_bucket_input(datapath, dataset, modelname, batch_size, encoder_decoder_bucket_boundaries):
    def args_fn(datapack):
        training_articles = datapack["training_article"]
        training_titles = datapack["training_title"]
        X_y_samples = list(zip(training_articles, training_titles))
        pool = SimpleInMemorySamplePool(X_y_samples, chunk_size=1)
        def get_single_example():
            article_titles, doc_indices = pool.__next__()
            article, title = article_titles[0]
            # remove sos and eos tag
            encoder_input = article[1:-1]
            decoder_input = title[:-1]
            decoder_target = title[1:]
            which_bucket = -1
            for bucket_id, (s_size, t_size) in enumerate(encoder_decoder_bucket_boundaries):
                if len(article) <= s_size and len(title) <= t_size:
                    which_bucket = bucket_id
                    break
            if which_bucket == -1:
                which_bucket = len(encoder_decoder_bucket_boundaries)
            return encoder_input, decoder_input, decoder_target, which_bucket

        tensors = tf.py_func(get_single_example, [],
                          [tf.int32, tf.int32, tf.int32, tf.int32],
                          name="get_summary_single_example")
        return tensors
    with open(getDatasetFilePath(datapath=datapath, dataset=dataset, modelname=modelname), "rb") as f:
        datapack = pickle.load(f)
    encoder_input, decoder_input, decoder_target, which_bucket = args_fn(datapack)
    encoder_input = tf.reshape(encoder_input, (-1,))
    decoder_input = tf.reshape(decoder_input, (-1,))
    decoder_target = tf.reshape(decoder_target, (-1,))
    encoder_len = tf.shape(encoder_input)[0]
    decoder_len = tf.shape(decoder_input)[0]
    encoder_decode_bucket = tf.contrib.training.bucket(which_bucket=which_bucket,
                                                       tensors=[encoder_len, decoder_len, encoder_input, decoder_input, decoder_target],
                                                       num_buckets=len(encoder_decoder_bucket_boundaries),
                                                       num_threads=4, capacity=batch_size * 10, dynamic_pad=True,
                                                       allow_smaller_final_batch=False, batch_size=batch_size,
                                                       name="encoder_decoder_bucket")
    _, (encoder_len_tensor, decoder_len_tensor, encoder_input_tensor, decoder_input_tensor, decoder_target_tensor) = encoder_decode_bucket

    encoder_bucket = {"encoder_len": encoder_len_tensor, "encoder_input": encoder_input_tensor}
    decoder_bucket = {"decoder_len": decoder_len_tensor,
                      "decoder_input": decoder_input_tensor,
                      "decoder_target": decoder_target_tensor}

    return encoder_bucket, decoder_bucket

