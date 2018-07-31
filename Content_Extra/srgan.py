import tensorflow as tf
from tensorflow.layers import *
import time

class SRGAN:
    def __init__(self):
        pass

    def build(self, input, train_mode=None):
        start_time = time.time()
        self.prob = self.Prob(input, train_mode)
        print(("build model finished: %ds" % (time.time() - start_time)))

    def Prob(self, input, train_mode):
        n = conv2d(input, 64, 3, padding='same', activation=tf.nn.relu)
        temp = n
        for i in range(16):
            nn = conv2d(n, 64, 3, padding='same')
            nn = batch_normalization(nn, training = train_mode)
            nn = tf.nn.relu(nn)
            nn = conv2d(n, 64, 3, padding='same')
            nn = batch_normalization(nn, training = train_mode)
            nn = tf.add(n, nn)
            n = nn
        n = conv2d(n, 64, 3, padding='same', activation=tf.nn.relu)
        n = tf.add(n, temp)

        for i in range(2):
            nn = conv2d(n, 256, 3, padding='same')
            nn = tf.depth_to_space(nn,2)
            nn = tf.nn.relu(n)
            n = nn
        n = conv2d(n, 3, (1, 1), padding='same', activation=tf.nn.tanh)

        return n