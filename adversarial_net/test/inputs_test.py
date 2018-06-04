import sys
sys.path.insert(0, ".")
from inputs import DataLoader
from utils import getLogger
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--test_module", type=str)
FLAGS = parser.parse_args()
logger = getLogger("inputs_test")

def test_data_loader(base_dir):
    dataLoader = DataLoader(base_dir=base_dir, dataset="imdb")
    training_dataset, testing_dataset, unsup_dataset = dataLoader.load_data()
    logger.info("training_dataset shape: %s; testing_dataset shape: %s; unsup_dataset: %s" % (
    training_dataset[0].shape, testing_dataset[0].shape, unsup_dataset[0].shape))
    training_sample_index = np.random.choice(len(training_dataset[0]), 1)[0]
    testing_dataset_index = np.random.choice(len(testing_dataset[0]), 1)[0]
    unsup_dataset_index = np.random.choice(len(unsup_dataset[0]), 1)[0]
    logger.info("training sample: %s; sample label: %s" % (
    training_dataset[0][training_sample_index], training_dataset[1][training_sample_index]))
    logger.info("testing sample: %s; sample label: %s" % (
        testing_dataset[0][testing_dataset_index], testing_dataset[1][testing_dataset_index]))
    logger.info("unsup sample: %s" % (unsup_dataset[0][unsup_dataset_index],))

if __name__ == "__main__":
    if FLAGS.test_module == "test_data_loader":
        test_data_loader(base_dir="E:/kaggle/avito/imdb_testset/adversarial_net/data")
    else:
        logger.info("unknown testing module: %s" % FLAGS.test_module)