import tensorflow as tf
import numpy as np 

# a = tf.Variable(tf.random_normal([1,5,5,3]))
# b = tf.Variable(tf.random_normal([3,3,3,1]))
# c = tf.nn.conv2d(a, b, strides = [1,1,1,1], padding='SAME')

# init = tf.initialize_all_variables()
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(c))

a = tf.constant(np.arange(25).reshape([1, 5, 5, 1]), dtype=tf.float32)
b = tf.constant(np.ones([3, 3, 1, 1])/9, dtype=tf.float32)
c = tf.nn.conv2d(a, b, strides = [1,1,1,1], padding='VALID')

with tf.Session() as sess:
    d = sess.run(c)
    d


