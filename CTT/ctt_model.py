import tensorflow as tf
from tensorflow.layers import *
import time

def CTT(Mc_placeholder, Mt_placeholder, train_mode):
    start_time = time.time()
    input = tf.concat([Mc_placeholder, Mt_placeholder], 3)
    n = conv2d(input, 64, 3, padding='same', activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(stddev=0.02))
    for i in range(16):
        nn = conv2d(n, 64, 3, padding='same', kernel_initializer=tf.random_normal_initializer(stddev=0.02), bias_initializer=None)
        nn = batch_normalization(nn, gamma_initializer =tf.random_normal_initializer(1., 0.02), training = train_mode)
        nn = tf.nn.relu(nn)
        nn = conv2d(nn, 64, 3, padding='same', kernel_initializer=tf.random_normal_initializer(stddev=0.02), bias_initializer=None)
        nn = batch_normalization(nn, gamma_initializer =tf.random_normal_initializer(1., 0.02),training = train_mode)
        nn = tf.add(n, nn)
        n = nn
    n = conv2d(n, 64, 3, padding='same', kernel_initializer=tf.random_normal_initializer(stddev=0.02), bias_initializer=None)
    n = batch_normalization(n, gamma_initializer =tf.random_normal_initializer(1., 0.02),training = train_mode)
    n = tf.add(n, M_c)

    for i in range(2):
        n = conv2d(n, 256, 3, padding='same', kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        n = tf.depth_to_space(n,2)
        n = tf.nn.relu(n)
    n = conv2d(n, 3, 1, padding='same', activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer(stddev=0.02))

    print(("build model finished: %ds" % (time.time() - start_time)))

    return n
