import numpy as np 
from scipy import misc


def img_read(path):
    img = misc.imread(path, mode='RGB')/255.0  
    return img

def img_downsize(image):
    img = misc.imresize(image, 50, 'bicubic')
    return img

def img_upscale(image):
    img = misc.imresize(image, 200, 'bicubic')
    return img

def img_crop(image, x, y):
    m, n, band = image.shape
    img = image[:x, :y, :]
    return img

def img_save(image, path):
    if image.dtype !='uint8':
        img = image*255.0
        img = img.astype('uint8')
    misc.imsave(path, img)
