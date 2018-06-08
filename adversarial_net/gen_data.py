from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import sys
sys.path.insert(0, ".")
import numpy as np
import glob
from os import path as osp
import pickle
import argparse
from sklearn.model_selection import train_test_split
from adversarial_net.preprocessing import WordCounter
from adversarial_net.inputs import DataLoader
from utils import getLogger

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--dataset", type=str)
parser.add_argument("--action", type=str)
parser.add_argument("--max_words", type=int, default=50000)
parser.add_argument("--doc_count_threshold", type=int, default=1)
parser.add_argument("--lower_case", type=bool, default=False)
parser.add_argument("--include_unk", type=bool, default=False)
parser.add_argument("--vocab_freqs_file", type=str, default=None)
parser.add_argument("--validation_rate", type=float, default=0)
parser.add_argument("--shuffle_onval", type=bool, default=True)
parser.add_argument("--no_need_start_tag", type=bool, default=True)
FLAGS = parser.parse_args()
logger = getLogger("DataGenerator")

def generate_imdb():
    wordCount = WordCounter(lower_case=FLAGS.lower_case)
    rand = np.random.RandomState(seed=8888)
    # vocab frequences
    if FLAGS.vocab_freqs_file is None:
        logger.info("generating imdb vocabulary...")
        wordCount.fit(glob.glob(osp.join(FLAGS.data_dir, "train_test_unsup/*.txt")),
                      doc_count_threshold=FLAGS.doc_count_threshold)
        logger.info("saving imdb vocabulary...")
        with open(osp.join(FLAGS.output_dir, "imdb_word_freqs.pickle"), "wb") as f:
            pickle.dump(wordCount.words_list, f)
    else:
        logger.info("loading imdb vocabulary...")
        with open(FLAGS.vocab_freqs_file, "rb") as f:
            wordCount.words_list = pickle.load(f)
    logger.info(
        "vocabulary counts: %s; most frequent words: %s" % (len(wordCount.words_list), str(wordCount.words_list[: 5])))
    # transform words
    logger.info("transforming words...")
    logger.info("transforming training-pos words...")
    training_pos_data = wordCount.transform([osp.join(FLAGS.data_dir, "train_pos.txt")], max_words=FLAGS.max_words,
                                            include_unk=FLAGS.include_unk)
    n_samples_training_pos = len(training_pos_data)
    min_seqlen_training_pos = min(map(len, training_pos_data))
    max_seqlen_training_pos = max(map(len, training_pos_data))
    logger.info(
        "total number of training_pos: %s; min_seqlen in training_pos_data: %s; max_seqlen in training_pos_data: %s" % (
        n_samples_training_pos, min_seqlen_training_pos, max_seqlen_training_pos))
    logger.info("transforming training-neg words...")
    training_neg_data = wordCount.transform([osp.join(FLAGS.data_dir, "train_neg.txt")], max_words=FLAGS.max_words,
                                            include_unk=FLAGS.include_unk)
    n_samples_training_neg = len(training_neg_data)
    min_seqlen_training_neg = min(map(len, training_neg_data))
    max_seqlen_training_neg = max(map(len, training_neg_data))
    logger.info(
        "total number of training_neg: %s; min_seqlen in training_neg_data: %s; max_seqlen in training_neg_data: %s" % (
        n_samples_training_neg, min_seqlen_training_neg, max_seqlen_training_neg))
    logger.info("transforming testing-pos words...")
    testing_pos_data = wordCount.transform([osp.join(FLAGS.data_dir, "test_pos.txt")], max_words=FLAGS.max_words,
                                           include_unk=FLAGS.include_unk)
    n_samples_testing_pos = len(testing_pos_data)
    min_seqlen_testing_pos = min(map(len, testing_pos_data))
    max_seqlen_testing_pos = max(map(len, testing_pos_data))
    logger.info(
        "total number of testing_pos: %s; min_seqlen in testing_pos_data: %s; max_seqlen in testing_pos_data: %s" % (
        n_samples_testing_pos, min_seqlen_testing_pos, max_seqlen_testing_pos))
    logger.info("transforming testing-neg words...")
    testing_neg_data = wordCount.transform([osp.join(FLAGS.data_dir, "test_neg.txt")], max_words=FLAGS.max_words,
                                           include_unk=FLAGS.include_unk)
    n_samples_testing_neg = len(testing_neg_data)
    min_seqlen_testing_neg = min(map(len, testing_neg_data))
    max_seqlen_testing_neg = max(map(len, testing_neg_data))
    logger.info(
        "total number of testing_neg: %s; min_seqlen in testing_neg_data: %s; max_seqlen in testing_neg_data: %s" % (
        n_samples_testing_neg, min_seqlen_testing_neg, max_seqlen_testing_neg))
    logger.info("transforming train_unsup words...")
    unsup_data = wordCount.transform([osp.join(FLAGS.data_dir, "train_unsup.txt")], max_words=FLAGS.max_words,
                                     include_unk=FLAGS.include_unk)
    n_samples_unsup = len(unsup_data)
    min_seqlen_unsup = min(map(len, unsup_data))
    max_seqlen_unsup = max(map(len, unsup_data))
    logger.info("total number of unsup: %s; min_seqlen in unsup_data: %s; max_seqlen in unsup_data: %s" % (
        n_samples_unsup, min_seqlen_unsup, max_seqlen_unsup))
    # [[0], [1], ...]
    training_pos_label = np.ones((len(training_pos_data), 1), dtype=np.int8)
    training_neg_label = np.zeros((len(training_neg_data), 1), dtype=np.int8)
    testing_pos_label = np.ones((len(testing_pos_data), 1), dtype=np.int8)
    testing_neg_label = np.zeros((len(testing_neg_data), 1), dtype=np.int8)
    # shuffle
    logger.info("shuffling docs...")
    rand.shuffle(training_pos_data)
    rand.shuffle(training_neg_data)
    rand.shuffle(testing_pos_data)
    rand.shuffle(testing_neg_data)
    rand.shuffle(unsup_data)
    # sample
    training_pos_sample_index = rand.choice(n_samples_training_pos, 1)[0]
    testing_pos_sample_index = rand.choice(n_samples_testing_pos, 1)[0]
    logger.info("training_pos sample: %s" % training_pos_data[training_pos_sample_index])
    logger.info("testing_pos sample: %s" % testing_pos_data[testing_pos_sample_index])
    # save
    logger.info("saving...")
    pickle_data = {"training_pos_data": training_pos_data, "training_neg_data": training_neg_data,
                   "testing_pos_data": testing_pos_data, "testing_neg_data": testing_neg_data, "unsup_data": unsup_data,
                   "training_pos_label": training_pos_label, "training_neg_label": training_neg_label,
                   "testing_pos_label": testing_pos_label, "testing_neg_label": testing_neg_label}
    with open(osp.join(FLAGS.output_dir, "imdb_dataset.pickle"), "wb") as f:
        pickle.dump(pickle_data, f)

def generate_language_training_data(no_need_start_tag = FLAGS.no_need_start_tag):
    dataLoader = DataLoader(base_dir=FLAGS.data_dir, dataset=FLAGS.dataset)
    (X_train, y_train), (X_test, y_test), (X_unsup,) = dataLoader.load_data(include_unsup=True)
    if X_unsup is not None:
        X_train_total = np.concatenate([X_train, X_test, X_unsup])
    else:
        X_train_total = np.concatenate([X_train, X_test])
    logger.info("dataset shape: %s" % X_train_total.shape)
    X = []
    y = []
    for doc in X_train_total:
        if no_need_start_tag:
            X.append(doc[1: -1])
            y.append(doc[2:])
        else:
            X.append(doc[0: -1])
            y.append(doc[1:])
    pickle_data = {"X": X, "y": y}
    rand_index = np.random.choice(len(X), 1)[0]
    logger.info("random sampled X: %s" % X[rand_index])
    logger.info("random sampled y: %s" % y[rand_index])
    with open(osp.join(FLAGS.output_dir, "imdb_lm_dataset.pickle"), "wb") as f:
        pickle.dump(pickle_data, f)

def generate_classification_data(validation_rate=FLAGS.validation_rate, shuffle_onval=FLAGS.shuffle_onval,
                                 no_need_start_tag=FLAGS.no_need_start_tag):
    dataLoader = DataLoader(base_dir=FLAGS.data_dir, dataset=FLAGS.dataset)
    (X_train, y_train), (X_test, y_test) = dataLoader.load_data(include_unsup=False)
    logger.info("training dataset shape: %s, testing dataset shape: %s" % (X_train.shape, X_test.shape))
    weight_train = []
    weight_test = []
    if no_need_start_tag:
        for i in range(X_train.shape[0]):
            X_train[i] = X_train[i][1:]
        for i in range(X_test.shape[0]):
            X_test[i] = X_test[i][1:]
    for i in range(X_train.shape[0]):
        seq_len = len(X_train[i])
        seq_weights = np.zeros(seq_len)
        if seq_len < 2:
            seq_weights[:] = 1
        else:
            seq_weights[:] = np.arange(0, seq_len) / (seq_len - 1)
        weight_train.append(seq_weights.tolist())
    for i in range(X_test.shape[0]):
        seq_len = len(X_test[i])
        seq_weights = np.zeros(seq_len)
        if seq_len < 2:
            seq_weights[:] = 1
        else:
            seq_weights[:] = np.arange(0, seq_len) / (seq_len - 1)
        weight_test.append(seq_weights.tolist())
    pickle_data = {"X_train": X_train, "y_train": y_train, "weight_train": weight_train, "X_test": X_test,
                   "y_test": y_test, "weight_test": weight_test}
    if 0 < validation_rate < 1:
        X_train, X_val, y_train, y_val, weight_train, weight_val = train_test_split(X_train, y_train, weight_train, test_size=validation_rate, shuffle=shuffle_onval)
        pickle_data["X_train"] = X_train
        pickle_data["y_train"] = y_train
        pickle_data["X_val"] = X_val
        pickle_data["y_val"] = y_val
        pickle_data["weight_train"] = weight_train
        pickle_data["weight_val"] = weight_val
    else:
        logger.info("No validation set.")
    rand_index = np.random.choice(len(X_train), 1)[0]
    logger.info("random sampled X: %s" % X_train[rand_index])
    logger.info("random sampled y: %s" % y_train[rand_index])
    logger.info("random sampled weight: %s" % weight_train[rand_index])
    rand_index = np.random.choice(len(X_test), 1)[0]
    logger.info("random sampled X: %s" % X_test[rand_index])
    logger.info("random sampled y: %s" % y_test[rand_index])
    logger.info("random sampled weight: %s" % weight_test[rand_index])
    with open(osp.join(FLAGS.output_dir, "imdb_classification_dataset.pickle"), "wb") as f:
        pickle.dump(pickle_data, f)

if __name__ == "__main__":
    if FLAGS.action == "gene_vocab":
        if FLAGS.dataset == "imdb":
            generate_imdb()
    elif FLAGS.action == "gene_lm":
        generate_language_training_data()
    elif FLAGS.action == "gene_classi":
        generate_classification_data()
    else:
        raise Exception("Unknown dataset: " + FLAGS.dataset)
