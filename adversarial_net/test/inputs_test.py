import sys
sys.path.insert(0, ".")
from adversarial_net.preprocessing import AutoPaddingInMemorySamplePool
from adversarial_net.inputs import DataLoader, construct_data_queue
from adversarial_net.utils import getLogger
import numpy as np
import argparse
import tensorflow as tf
import threading
import multiprocessing

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

def test_sample_enqueue():
    X = [[1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3], [4, 4, 4], [5,5,5,5,5,5,5]]
    y = [[1], [0], [1], [0], [2]]
    pool = AutoPaddingInMemorySamplePool(X, bins_count=2, batch_size=1, y=y, mode="specific", unroll_num=3,
                                         get_y_in_batch=True)
    dequeue_op = construct_data_queue(data_pool=pool, n_thread=10, batch_size=1, queue_class=tf.FIFOQueue)
    if pool.get_y_in_batch:
        X_tensor, y_tensor = dequeue_op
    else:
        X_tensor = dequeue_op
    gpu_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as session:
        coodinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(session, coodinator)
        for step in range(30):
            if pool.get_y_in_batch:
                batch_X, batch_y = session.run([X_tensor, y_tensor])
                print("step-%s, X: %s, y: %s" % (step, batch_X, batch_y))
            else:
                batch_X = session.run([X_tensor])
                print("step-%s, X: %s" % (step, batch_X))
        coodinator.request_stop()
        coodinator.join(threads)

def test_pool_multithread_func(pool):
    for _ in range(5):
        print(pool.__next__())

def test_pool_multithread():
    X = [[1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3], [4, 4, 4]]
    y = [[1], [0], [1], [0]]
    pool = AutoPaddingInMemorySamplePool(X, bins_count=2, batch_size=1, y=y, mode="specific", unroll_num=3)
    # threads = [threading.Thread(target=lambda x: [print(x.__next__()) for i in range(5)], args=[pool]) for i in range(4)]
    # threads = [multiprocessing.Process(target=test_pool_multithread_func, args=[pool]) for i in range(4)]
    # for thread in threads:
    #     thread.start()
    #     thread.join()
    thread_pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    thread_pool.apply_async(test_pool_multithread_func, args=[pool])
    thread_pool.close()
    thread_pool.join()

if __name__ == "__main__":
    if FLAGS.test_module == "test_data_loader":
        test_data_loader(base_dir="E:/kaggle/avito/imdb_testset/adversarial_net/data")
    elif FLAGS.test_module == "test_sample_enqueue":
        test_sample_enqueue()
    elif FLAGS.test_module == "test_pool_multithread":
        test_pool_multithread()
    else:
        logger.info("unknown testing module: %s" % FLAGS.test_module)