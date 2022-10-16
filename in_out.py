from skimage.io import imshow, show, imread, imsave
import matplotlib.pyplot as plt


def read_image(path, as_gray=True):
    return imread(path, as_gray=as_gray)


def write_image(image, path):
    return imsave(path, image)
