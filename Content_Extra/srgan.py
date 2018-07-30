import tensorflow as tf
from tensorflow.layers import *

class SRGAN:
    def __init__(self):
        pass

    def build(self, Input, train_mode=None):
        Input = tf.constant(Input, float)
        self.layer19 = self.Layer19(input, train_mode)
        self.prob = self.Prob(self.layer19)

    def Layer19(self, input, train_mode):
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
        n = conv2d(input, 64, 3, padding='same', activation=tf.nn.relu)
        n = tf.add(n, temp)
        return n

    def Prob(self, n):
        for i in range(2):
            n = conv2d(n, 256, 3, padding='same')
            n = tf.depth_to_space(n,2)
            n = tf.nn.relu(n)
        n = conv2d(n, 3, (1, 1), padding='same', activation=tf.nn.tanh)
