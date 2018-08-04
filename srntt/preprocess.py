import utils
import dataload
import tensorflow as tf
import numpy as np
import Vgg_module
import patch_match

with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.8)))) as sess:

    train_ref, train_hr = dataload.dataLoader("./SRNTT1000.h5", 1000)
    train_lr = utils.img_resize(train_hr, 25)
    train_lref = utils.img_resize(train_ref, 25)

    M_t = np.zeros([1000, 40, 40, 256])
    M_s = np.zeros([1000, 40, 40, 256])
    for i in range(10):
        print("-----------------------Round", i, "----------------------")
        ran = list(range(i*100, (i+1)*100))
        M_LR = Vgg_module.vgg19_module(utils.img_resize(train_lr[ran,:,:,:], 400), sess)
        M_LRef = Vgg_module.vgg19_module(utils.img_resize(train_lref[ran,:,:,:], 400), sess)
        M_Ref = Vgg_module.vgg19_module(train_ref[ran,:,:,:], sess)
        M_t[ran,:,:,:], M_s[ran,:,:,:] = patch_match.Fun_patchMatching(M_LR, M_LRef, M_Ref, sess)

    np.save("autumn1000_M_t", M_t)
    np.save("autumn1000_M_s", M_s)
