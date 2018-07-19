import tensorflow as tf
import numpy as np

def count_nan(tensor):
    return tf.reduce_sum(tf.cast(tf.is_nan(tf.cast(tensor, tf.float32)), tf.float32))

def count_inf(tensor):
    return tf.reduce_sum(tf.cast(tf.is_inf(tf.cast(tensor, tf.float32)), tf.float32))

def count_nan_inf(tensor):
    return count_nan(tensor) + count_inf(tensor)