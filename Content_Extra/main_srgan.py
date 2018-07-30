import numpy as np
import tensorflow as tf

import srgan

def test_srgan(Image_lr, Image_hr):
    assert image.shape == (160, 160, 3)

    xbatch
    ybatch

    x = tf.placeholder(float, [1, 160, 160, 3])
    y = tf.placeholder(float, [1, 160, 160, 3])
    train_mode = tf.placeholder(tf.bool)

    srgan = srgan.SRGAN()
    srgan.build(x, train_mode)
    init = tf.global_variables_initializer()
    loss = tf.losses.mean_squared_error(ybatch, y)
    train = tf.train.AdamOptimizer().minimize(loss)

    with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.5)))) as sess:
        sess.run(init)
        sess.run(train, feed_dict={x:xbatch, y:ybatch, train_mode:True})
        prob = sess.run(srgan.Prob, feed_dict = {x:xbatch, train_mode:False})

