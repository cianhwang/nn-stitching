import numpy as np
import tensorflow as tf

import srgan
import utils

def test_srgan(train_lr, train_hr, test_lr, test_hr):

    xbatch = train_lr
    ybatch = train_hr
    x_test = test_lr
    y_test = test_hr

    # with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.5)))) as sess:
    with tf.device('/cpu:0'):
        with tf.Session() as sess:    
            x = tf.placeholder(tf.float32, [65, 80, 80, 3])
            y = tf.placeholder(tf.float32, [65, 80, 80, 3])
            train_mode = tf.placeholder(tf.bool)

            srGan = srgan.SRGAN()
            srGan.build(x, train_mode)
            init = tf.global_variables_initializer()
            loss = tf.losses.mean_squared_error(y, srGan.prob)
            optimizer = tf.train.GradientDescentOptimizer(0.01)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss)

            sess.run(init)

            for epoch in range(1):
                sess.run(train_op, feed_dict={x:xbatch, y:ybatch, train_mode:True})
                if epoch %5==0:
                    print(sess.run(loss, feed_dict={x:xbatch, y:ybatch, train_mode:False}))

            prob = sess.run(srGan.prob, feed_dict = {x:xbatch, train_mode:False})
            for i in range(65):
                path = '../dataset/t' + str(i+1) + '_.bmp'
                utils.img_save(prob[i,:,:,:], path)
        

train_hr = np.zeros([65, 80, 80, 3])
train_lr = np.zeros([65, 80, 80, 3])
test_hr = np.zeros([26, 80, 80, 3])
test_lr = np.zeros([26, 80, 80, 3])

for i in range(65):
    print(i)
    path = '../dataset/91-image/t' + str(i+1) + '.bmp'
    img = utils.img_read(path)
    img = utils.img_crop(img, 80, 80)
    train_hr[i, :, :, :] = img
    img = utils.img_downsize(img)
    img = utils.img_upscale(img)
    train_lr[i, :, :, :] = img

for i in range(26):
    path = '../dataset/91-image/tt' + str(i+1) + '.bmp'
    img = utils.img_read(path)
    img = utils.img_crop(img, 80, 80)
    test_hr[i, :, :, :] = img
    img = utils.img_downsize(img)
    img = utils.img_upscale(img)
    test_lr[i, :, :, :] = img