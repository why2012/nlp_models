from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import glob
from os import path as osp
import pickle
import argparse
from adversarial_net.preprocessing import WordCounter
from utils import getLogger

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--dataset", type=str)
parser.add_argument("--max_words", type=int, default=50000)
parser.add_argument("--doc_count_threshold", type=int, default=1)
parser.add_argument("--lower_case", type=bool, default=False)
parser.add_argument("--include_unk", type=bool, default=False)
parser.add_argument("--vocab_freqs_file", type=str, default=None)
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

if __name__ == "__main__":
    if FLAGS.dataset == "imdb":
        generate_imdb()
    else:
        raise Exception("Unknown dataset: " + FLAGS.dataset)
