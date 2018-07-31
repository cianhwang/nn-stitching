import numpy as np 
from scipy import misc


def img_read(path):
    img = misc.imread(path, mode='RGB')/255.0  
    return img

def img_downsize(image):
    img = misc.imresize(image, 0.5, 'bicubic')
    return img

def img_upscale(image):
    img = misc.imresize(image, 2, 'bicubic')
    return img

def img_crop(image, x, y):
    m, n, band = image.shape
    img = image[int((m-x)/2):int((m-x)/2)+x, int((n-y)/2):int((n-y)/2)+y, :]
    return img

def img_save(image, path):
    if image.dtype !='uint8':
        img = image*255.0
        img = img.astype('uint8')
    misc.imsave(path, img)
