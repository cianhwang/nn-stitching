import numpy as np 
from scipy import misc
import skimage
import skimage.io
import skimage.transform

def img_read(path):
    img = misc.imread(path, mode='RGB')
    img = img/255.0
    return img

def img_resize(image, rate):
    image = image*255.0
    image = image.astype('uint8')
    if len(image.shape) < 4:
        img = misc.imresize(image, rate, 'bicubic')
    else:
        a, b, c, d = image.shape
        img = np.zeros([a, b*rate//100, c*rate//100, d])
        for i in range(a):
            img[i, :, :, :] = misc.imresize(image[i, :, :, :], rate, 'bicubic')
    return img/255.0

def img_crop(image, x, y):
    m, n, band = image.shape
    img = image[:x, :y, :]
    return img

def img_save(image, path):
    img = image*255.0
    img = img.astype('uint8')
    misc.imsave(path, img)

def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img


# returns the top1 string
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return top1


def load_image2(path, height=None, width=None):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = img.shape[0]
        nx = img.shape[1]
    return skimage.transform.resize(img, (ny, nx))


def test():
    img = skimage.io.imread("./test_data/starry_night.jpg")
    ny = 300
    nx = img.shape[1] * ny / img.shape[0]
    img = skimage.transform.resize(img, (ny, nx))
    skimage.io.imsave("./test_data/test/output.jpg", img)


if __name__ == "__main__":
    test()
