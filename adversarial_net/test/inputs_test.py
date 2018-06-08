import sys
sys.path.insert(0, ".")
from adversarial_net.preprocessing import AutoPaddingInMemorySamplePool
from adversarial_net.inputs import DataLoader, construct_data_queue, construct_language_model_input_tensors
from adversarial_net.inputs import construct_classification_model_input_tensor_with_state
from adversarial_net.inputs import construct_language_model_input_tensor_with_state
from adversarial_net.utils import getLogger, ArgumentsBuilder
import numpy as np
import argparse
import tensorflow as tf
import threading
import multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument("--test_module", type=str)
FLAGS, _ = parser.parse_known_args()
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
                                         get_y_in_batch=True, get_sequence_len_in_batch=True)
    dequeue_op = construct_data_queue(data_pool=pool, n_thread=1, batch_size=1, queue_class=tf.FIFOQueue)
    if pool.get_y_in_batch:
        X_tensor, y_tensor, seqlen_tensor = dequeue_op
    else:
        X_tensor, seqlen_tensor = dequeue_op
    gpu_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as session:
        coodinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(session, coodinator)
        for step in range(30):
            if pool.get_y_in_batch:
                batch_X, batch_y, seq_len = session.run([X_tensor, y_tensor, seqlen_tensor])
                print("step-%s, X: %s, y: %s, seqlen: %s" % (step, batch_X, batch_y, seq_len))
            else:
                batch_X, seq_len = session.run([X_tensor, seqlen_tensor])
                print("step-%s, X: %s, seqlen: %s" % (step, batch_X, seq_len))
        coodinator.request_stop()
        coodinator.join(threads)

def test_sample_pool_enqueue():
    datapath = "E:/kaggle/avito/imdb_testset/adversarial_net/data"
    X_tensor, y_tensor = construct_language_model_input_tensors(datapath, batch_size=1, unroll_steps=100)
    gpu_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as session:
        coodinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(session, coodinator)
        for step in range(2):
            batch_X, batch_y = session.run([X_tensor, y_tensor])
            print("step-%s, X: %s, y: %s" % (step, batch_X, batch_y))
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

def test_arguments_builder():
    sys.argv.remove("--test_module")
    sys.argv.remove("test_arguments_builder")
    argBuilder = ArgumentsBuilder()
    argBuilder.register_variable("val4")
    argBuilder.register_variable("val4", scope="model")
    argBuilder.add_argument("arg1", str, scope="model", default=1). \
        add_argument("arg2", str, scope="model", default=2). \
        add_argument("arg2", str, scope="input", default=2).\
        add_argument("arg3", str, default=3). \
        add_variable("val1", 11, scope="model").\
        add_variable("val2", 12, scope="input").\
        add_variable("val3", 13). \
        add_variable("val4", 13).add_variable("val4", scope="model", value=1)
    print("model-scope: ", argBuilder["model"])
    print("input-scope: ", argBuilder["input"])
    print("arg3", argBuilder["arg3"])
    print("val3", argBuilder["val3"])

def test_construct_language_model_input_tensor_with_state():
    batch, _, _ = construct_language_model_input_tensor_with_state(
        "E:/kaggle/avito/imdb_testset/adversarial_net/data", batch_size=1, unroll_steps=100,
        lstm_num_layers=1, state_size=10, dataset="imdb")
    print(batch.key, batch.sequences["X"], batch.sequences["y"], batch.state("0_lstm_c"), batch.length)
    with tf.Session() as sess:
        coodinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coodinator)
        for i in range(5):
            key, X, y, state = sess.run([batch.key, batch.sequences["X"], batch.sequences["y"], batch.state("0_lstm_c")])
            lens = sess.run(batch.length)
            print(key[0], X[0][:5], y[0][:5], state[0])
            print(lens)
        coodinator.request_stop()
        coodinator.join(threads)

def test_construct_classification_model_input_tensor_with_state():
    batch, _, _ = construct_classification_model_input_tensor_with_state(
        "E:/kaggle/avito/imdb_testset/adversarial_net/data", phase="train", batch_size=1, unroll_steps=100,
        lstm_num_layers=1, state_size=10, dataset="imdb")
    print(batch.key, batch.sequences["X"], batch.context["y"], batch.sequences["weight"], batch.state("0_lstm_c"), batch.length)
    with tf.Session() as sess:
        coodinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coodinator)
        for i in range(5):
            key, X, y, weight = sess.run([batch.key, batch.sequences["X"], batch.context["y"], batch.sequences["weight"]])
            lens = sess.run(batch.length)
            print(key[0], X[0][:5], y, weight[0][:5])
            print(lens)
        coodinator.request_stop()
        coodinator.join(threads)

if __name__ == "__main__":
    if FLAGS.test_module == "test_data_loader":
        test_data_loader(base_dir="E:/kaggle/avito/imdb_testset/adversarial_net/data")
    elif FLAGS.test_module == "test_sample_enqueue":
        test_sample_enqueue()
    elif FLAGS.test_module == "test_pool_multithread":
        test_pool_multithread()
    elif FLAGS.test_module == "test_arguments_builder":
        test_arguments_builder()
    elif FLAGS.test_module == "test_sample_pool_enqueue":
        test_sample_pool_enqueue()
    elif FLAGS.test_module == "test_construct_language_model_input_tensor_with_state":
        test_construct_language_model_input_tensor_with_state()
    elif FLAGS.test_module == "test_construct_classification_model_input_tensor_with_state":
        test_construct_classification_model_input_tensor_with_state()
    else:
        logger.info("unknown testing module: %s" % FLAGS.test_module)