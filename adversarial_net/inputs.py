from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pickle
from os import path as osp
from preprocessing import AutoPaddingInMemorySamplePool, WordCounter

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
