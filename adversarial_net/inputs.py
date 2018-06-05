from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pickle
from os import path as osp
import tensorflow as tf
from adversarial_net.preprocessing import AutoPaddingInMemorySamplePool, WordCounter

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
        training_dataset = testing_dataset = unsup_dataset = None
        if self.dataset == "imdb":
            training_dataset, testing_dataset, unsup_dataset = self._load_imdb(include_unsup=include_unsup)
        return training_dataset, testing_dataset, unsup_dataset

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
            return (training_data, training_label), (testing_data, testing_label)

def construct_data_queue(data_pool, n_thread, batch_size, collection_name=tf.GraphKeys.QUEUE_RUNNERS,
                         queue_class=tf.FIFOQueue, threshold=100):
    from multiprocessing import Lock
    import traceback
    # lock = Lock()
    def enqueue_func():
        data = []
        try:
            # lock.acquire()
            data = data_pool.__next__()
        except Exception as e:
            print(e)
            traceback.print_exc()
        finally:
            # lock.release()
            1
        return data
    if data_pool.get_y_in_batch:
        dtypes = [tf.int32, tf.int32]
        enqueue_tensor_name = "sample_and_label_py_func"
        queue_name = "sample_and_label_queue"
        enqueue_name = "sample_and_label_enqueue"
        dequeue_name = "sample_and_label_dequeue"
        queue_shape = [[data_pool.unroll_num], [1]]
    else:
        dtypes = [tf.int32]
        enqueue_tensor_name = "sample__py_func"
        queue_name = "sample_queue"
        enqueue_name = "sample_enqueue"
        dequeue_name = "sample_dequeue"
        queue_shape = [[data_pool.unroll_num]]
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

