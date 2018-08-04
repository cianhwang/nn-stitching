import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import Vgg_module
import utils
import patch_match
import ce_model
import ctt_model
import vgg19
import dataload

'''------------------------start of the program-----------------------------'''

dataNum= 1000
batchSize = 16

with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.8)))) as sess:

    '''------------------------Data Load-------------------------'''
    ref, hr = dataload.dataLoader("./SRNTT1000.h5")
    M_t = np.load("autumn1000_M_t.npy")
    M_s = np.load("autumn1000_M_s.npy")
    train_hr, test_hr, train_ref, test_ref, train_Mt, test_Mt, train_Ms, test_Ms \
     = train_test_split(hr, ref, M_t, M_s, test_size=0.2)
    
    train_lr = utils.img_resize(train_hr, 25)
    train_lref = utils.img_resize(train_ref, 25)
    test_lr = utils.img_resize(test_hr, 25)
    test_lred = utils.img_resize(test_ref, 25)

    for i in range(test_ref.shape[0]):
        path = './result/ref/'+ str(i+1) + '.bmp'
        utils.img_save(test_ref[i,:,:,:], path)
        path = './result/lr/' + str(i+1) + '.bmp'
        utils.img_save(test_lr[i,:,:,:], path)
        path = './result/hr/' + str(i+1) + '.bmp'
        utils.img_save(test_hr[i,:,:,:], path)
    

    '''----------------------Net Construct-------------------------'''
    x = tf.placeholder(tf.float32, [None, 40, 40, 3])
    y = tf.placeholder(tf.float32, [None, 160, 160, 3])
    Mt_ph= tf.placeholder(tf.float32, [None, 40, 40, 256])
    Ms_ph= tf.placeholder(tf.float32, [None, 40, 40, 256])
    train_mode = tf.placeholder(tf.bool)
    Learning_rate = tf.placeholder(tf.float32, shape=[])

    ce_net = ce_model.CE(x, train_mode)
    y_pred = ctt_model.CTT(ce_net, Mt_ph, train_mode)

    vgg = vgg19.Vgg19()
    vgg.build(y)
    y_conv51 = vgg.conv5_1

    vgg2 = vgg19.Vgg19()
    vgg2.build(y_pred)
    y_pred_conv31 = vgg2.conv3_1
    y_pred_conv51 = vgg2.conv5_1

    loss = tf.norm(y-y_pred, ord=1)/(160.*160.)
    loss_total = tf.norm(y-y_pred, ord=1)/(160.*160.) + 1e-4 * tf.norm(y_conv51 - y_pred_conv51, ord=2)/512./100. \
    + 1e-4 * tf.norm(utils.Gram(y_pred_conv31*Ms_ph)-utils.Gram(Mt_ph), ord=2)/(1600.*2.*256.)**2

    optimizer = tf.train.AdamOptimizer(learning_rate=Learning_rate)    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op_pre = optimizer.minimize(loss)
        train_op = optimizer.minimize(loss_total)

    init = tf.global_variables_initializer()
    sess.run(init)
    learning_rate = 1e-4
    '''-------------------------train--------------------------'''
    for epoch in range(10000):
        ran=np.sort(np.random.choice(train_hr.shape[0],batchSize,replace=False))
        xbatch = train_lr[ran,:,:,:]
        ybatch = train_hr[ran,:,:,:]
        MtBatch = M_t[ran,:,:,:]
        MsBatch = M_s[ran,:,:,:]
        train_dict = {x:xbatch, y:ybatch, Mt_ph:MtBatch, Ms_ph:MsBatch,train_mode:True, Learning_rate: learning_rate}
        if epoch < 5:
            sess.run(train_op_pre, feed_dict = train_dict)
        else:
            sess.run(train_op, feed_dict = train_dict)

        if epoch % 50==0:
            # update learning rate
            if learning_rate > 1e-6:
                learning_rate /= 10
            print('--------------loss', sess.run(loss_total, 
            feed_dict={x:xbatch, y:ybatch, Mt_ph:MtBatch, Ms_ph:MsBatch, train_mode:False}), 
            '------------------')

        if epoch > 0 and epoch % 100 ==0:
            prediction = sess.run(y_pred, feed_dict = {x:test_lr, Mt_ph:test_Mt, train_mode:False})
            # eval_psnr = tf.image.psnr(prediction, test_hr, max_val=1.0)
            # eval_ssim = tf.image.ssim(prediction, test_hr, max_val=1.0)
            # np.save("epoch"+str(epoch)+"_psnr.npy", eval_psnr)
            # np.save("epoch"+str(epoch)+"_ssim.npy", eval_ssim)
            if epoch % 1000 == 0:
            # Calculate or Save the prediction
                for i in range(prediction.shape[0]):
                    path = './result/pred/epoch'+str(epoch)+'_' + str(i+1) + '.bmp'
                    utils.img_save(prediction[i,:,:,:], path)

