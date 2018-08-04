import numpy as np
import tensorflow as tf

import vgg19

def vgg19_module(imageStack, sess):

    batch = np.array(imageStack)

    images = tf.placeholder("float", [None, 160, 160, 3])
    feed_dict = {images: batch}

    vgg = vgg19.Vgg19()
    with tf.name_scope("content_vgg"):
        vgg.build(images)
    conv3_1 = sess.run(vgg.conv3_1, feed_dict=feed_dict)
    return conv3_1