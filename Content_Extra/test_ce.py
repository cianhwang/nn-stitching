import numpy as np
import tensorflow as tf

import ce_model
import utils

def test_ce(train_lr, train_hr, test_lr, test_hr):

    x_train = train_lr
    y_train = train_hr
    x_test = test_lr
    y_test = test_hr

    with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.5)))) as sess:
    # with tf.device('/cpu:0'):
    #     with tf.Session() as sess:    
        x = tf.placeholder(tf.float32, [None, 40, 40, 3])
        y = tf.placeholder(tf.float32, [None, 160, 160, 3])
        train_mode = tf.placeholder(tf.bool)

        srGan = ce_model.SRGAN()
        srGan.build(x, train_mode)
        loss = tf.losses.mean_squared_error(y, srGan.prob)
        optimizer = tf.train.AdamOptimizer()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        sess.run(init)

        for epoch in range(10000):
            ran=np.sort(np.random.choice(x_train.shape[0],30,replace=False))
            xbatch = x_train[ran,:,:,:]
            ybatch = y_train[ran,:,:,:]
            
            sess.run(train_op, feed_dict={x:xbatch, y:ybatch, train_mode:True})
            if epoch %20==0:
                print(sess.run(loss, feed_dict={x:xbatch, y:ybatch, train_mode:False}))

        prob = sess.run(srGan.prob, feed_dict = {x:x_train, train_mode:False})

        return prob
        

train_hr = np.zeros([65, 160, 160, 3])
train_lr = np.zeros([65, 40, 40, 3])
test_hr = np.zeros([26, 160, 160, 3])
test_lr = np.zeros([26, 40, 40, 3])

k=0
for i in range(65):

    path = '../dataset/91-image/t' + str(i+1) + '.bmp'
    img = utils.img_read(path)
    img = utils.img_crop(img, 160, 160)
    if img.shape != (160, 160, 3):
        continue
    train_hr[k, :, :, :] = img
    img = utils.img_downsize(img, 25)
    train_lr[k, :, :, :] = img
    k+=1

k=0
for i in range(26):
    path = '../dataset/91-image/tt' + str(i+1) + '.bmp'
    img = utils.img_read(path)
    img = utils.img_crop(img, 160, 160)
    if img.shape != (160, 160, 3):
        continue
    test_hr[k, :, :, :] = img
    img = utils.img_downsize(img, 25)
    test_lr[k, :, :, :] = img
    k+=1

prob = test_ce(train_lr, train_hr, test_lr, test_hr)
for i in range(10):
    path = '../dataset/91-image/t' + str(i+1) + '_.bmp'
    utils.img_save(prob[i,:,:,:], path)
