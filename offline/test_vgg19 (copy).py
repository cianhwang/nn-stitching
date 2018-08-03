import numpy as np
import tensorflow as tf

import vgg19
import utils

def vgg19_pretrained(image):
    assert image.shape == (160, 160, 3)

    batch = np.array(image[np.newaxis, :, :, :])

    with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.5)))) as sess:
    # with tf.device('/cpu:0'):
    #     with tf.Session() as sess:
        images = tf.placeholder("float", [1, 160, 160, 3])
        feed_dict = {images: batch}

        vgg = vgg19.Vgg19()
        with tf.name_scope("content_vgg"):
            vgg.build(images)

        conv3_1 = sess.run(vgg.conv3_1, feed_dict=feed_dict)
        conv5_1 = sess.run(vgg.conv5_1, feed_dict=feed_dict)


    return conv3_1, conv5_1

img = np.random.random([160, 160, 3])
a, b= vgg19_pretrained(img)
print(a.shape, b.shape)
