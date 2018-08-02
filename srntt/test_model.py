import numpy as np
import tensorflow as tf

import test_vgg19
import utils_ce
import patch_match
import ce_model
import ctt_model
import vgg19

# def Gram(a):
#     # a batch x height x width x features -> batch * features * features
#     a = tf.reshape(a, [10, -1, 256])
#     trans_a = tf.transpose(a, [0, 2, 1])
#     return tf.matmul(trans_a, a)

def Gram(feature_maps):
  """Computes the Gram matrix for a set of feature maps."""
  batch_size, height, width, channels = tf.unstack(tf.shape(feature_maps))
  denominator = tf.to_float(height * width)
  feature_maps = tf.reshape(
      feature_maps, tf.stack([batch_size, height * width, channels]))
  matrix = tf.matmul(feature_maps, feature_maps, adjoint_a=True)
  return matrix# / denominator

# def Load_data():
#     pass

# Img_LR, Img_Ref, Img_HR = Load_data()
train_hr = np.zeros([10, 160, 160, 3])
train_lr = np.zeros([10, 40, 40, 3])
train_ref = np.zeros([10, 160, 160, 3])

for i in range(10):
    path = '../dataset/91-image/t' + str(i+1) + '.bmp'
    img = utils_ce.img_read(path)
    img = utils_ce.img_crop(img, 160, 160)
    if img.shape != (160, 160, 3):
        continue
    train_hr[i, :, :, :] = img
    train_ref[i, :, :, :] = img
    img = utils_ce.img_downsize(img, 25)
    train_lr[i, :, :, :] = img
#np.random.shuffle(train_ref)

M_LR = test_vgg19.vgg19_pretrained(utils_ce.img_upscale(train_lr, 400))[0]
M_LRef = utils_ce.img_downsize(train_ref, 25)
M_LRef = test_vgg19.vgg19_pretrained(utils_ce.img_upscale(M_LRef, 400))[0]
M_Ref = test_vgg19.vgg19_pretrained(train_ref)[0]


M_t = np.zeros(M_LR.shape)
M_s = np.zeros(M_LR.shape)
for i in range(M_LR.shape[0]):
    M_t[i,:,:,:], M_s[i,:,:,:] = patch_match.Fun_patchMatching(M_LR[i,:,:,:], M_LRef[i,:,:,:], M_Ref[i,:,:,:])

x_train = train_lr
y_train = train_hr
Mt_train = M_t
assert M_t.shape==(10, 40, 40, 256)
print(np.max(M_s), np.max(M_t))


with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.5)))) as sess:
# with tf.device('/cpu:0'):
#     with tf.Session() as sess:    
    x = tf.placeholder(tf.float32, [None, 40, 40, 3])
    y = tf.placeholder(tf.float32, [None, 160, 160, 3])
    Mt_ph= tf.placeholder(tf.float32, [None, 40, 40, 256])
    Ms_ph= tf.placeholder(tf.float32, [None, 40, 40, 256])

    train_mode = tf.placeholder(tf.bool)

    srGan = ce_model.SRGAN()
    srGan.build(x, train_mode)
    y_pred = ctt_model.CTT(srGan.layer19, Mt_ph, train_mode)
    vgg = vgg19.Vgg19()
    vgg.build(y)
    vgg_y = vgg.conv5_1
    vgg2 = vgg19.Vgg19()
    vgg2.build(y_pred)
    vgg_ypred = vgg2.conv5_1
    vgg_ypred2 = vgg2.conv3_1

    loss = tf.norm(tf.reshape(y-y_pred, [-1]), ord=1)/1600.
    loss_total = tf.norm(tf.reshape(y-y_pred, [-1]), ord=1)/1600. + 1e-4 * tf.norm(vgg_y-vgg_ypred)/512./100. + 1e-4 * tf.norm(Gram(vgg_ypred2*Ms_ph)-Gram(Mt_ph))/(1600.*2.*256)**2
    optimizer = tf.train.AdamOptimizer(1e-4)    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)
        train_op2 = optimizer.minimize(loss_total)

    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(100):
        ran=np.sort(np.random.choice(x_train.shape[0],5,replace=False))
        xbatch = x_train[ran,:,:,:]
        ybatch = y_train[ran,:,:,:]
        MtBatch = Mt_train[ran,:,:,:]
        MsBatch = M_s[ran,:,:,:]
        if epoch < 5:
            sess.run(train_op, feed_dict={x:xbatch, y:ybatch, Mt_ph:MtBatch, Ms_ph:MsBatch, train_mode:True})
        else:
            sess.run(train_op2, feed_dict={x:xbatch, y:ybatch, Mt_ph:MtBatch, Ms_ph:MsBatch, train_mode:True})
        if epoch % 20==0:
            print(sess.run(loss_total, feed_dict={x:xbatch, y:ybatch, Mt_ph:MtBatch, Ms_ph:MsBatch, train_mode:False}))

    prediction = sess.run(y_pred, feed_dict = {x:x_train, Mt_ph:Mt_train, train_mode:False})


#'''--------------output------------------'''
for i in range(10):
    path = '../dataset/91-image/t' + str(i+1) + '_.bmp'
    utils_ce.img_save(prediction[i,:,:,:], path)

