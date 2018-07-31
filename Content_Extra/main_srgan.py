import numpy as np
import tensorflow as tf

import srgan

def test_srgan(Image_lr, Image_hr):

    xbatch = Image_lr[np.newaxis, :, :, :]
    ybatch = Image_hr[np.newaxis, :, :, :]


    x = tf.placeholder(tf.float32, [1, 40, 40, 3])
    y = tf.placeholder(tf.float32, [1, 40, 40, 3])
    train_mode = tf.placeholder(tf.bool)

    srGan = srgan.SRGAN()
    srGan.build(x, train_mode)
    init = tf.global_variables_initializer()
    loss = tf.losses.mean_squared_error(y, srGan.prob)
    optimizer = tf.train.GradientDescentOptimizer(0.001)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(1):
            sess.run(train_op, feed_dict={x:xbatch, y:ybatch, train_mode:True})
        prob = sess.run(srGan.prob, feed_dict = {x:xbatch, train_mode:False})
        

Image_hr = np.random.random([40, 40, 3])
Image_lr = np.random.random([40, 40, 3])
test_srgan(Image_lr, Image_hr)