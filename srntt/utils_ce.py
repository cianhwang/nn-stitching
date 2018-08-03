import numpy as np 
from scipy import misc


def img_read(path):
    img = misc.imread(path, mode='RGB')
    return img

def img_downsize(image, rate):
    if len(image.shape) < 4:
        img = misc.imresize(image, rate, 'bicubic')
    else:
        a, b, c, d = image.shape
        img = np.zeros([a, b*rate//100, c*rate//100, d])
        for i in range(a):
            img[i, :, :, :] = misc.imresize(image[i, :, :, :], rate, 'bicubic')
    return img

def img_upscale(image, rate):
    img = img_downsize(image, rate)
    return img

def img_crop(image, x, y):
    m, n, band = image.shape
    img = image[:x, :y, :]
    return img

def img_save(image, path):
    image = np.array(image)
    img = image/np.max(image)*255.0
    img = img.astype('uint8')
    misc.imsave(path, img)
