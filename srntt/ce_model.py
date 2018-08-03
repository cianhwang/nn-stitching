import tensorflow as tf
from tensorflow.layers import *
import time

class SRGAN:
    def __init__(self):
        pass

    def build(self, input, train_mode=None):
        start_time = time.time()
        self.layer19 = self.Layer19(input, train_mode)
        print(("build model finished: %ds" % (time.time() - start_time)))

    def Layer19(self, input, train_mode):
        n = conv2d(input, 64, 3, padding='same', activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        temp = n
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
        n = tf.add(n, temp)
        return n
