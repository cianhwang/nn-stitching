import numpy as np 
from scipy import misc


def img_read(path):
    img = misc.imread(path, mode='RGB')/255.0  
    return img

def img_downsize(image, rate):
    img = misc.imresize(image, rate, 'bicubic')
    return img

def img_upscale(image, rate):
    img = misc.imresize(image, rate, 'bicubic')
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
