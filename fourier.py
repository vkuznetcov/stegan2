import numpy as np
from skimage.io import imshow, show, imread
from scipy.fft import fft2, ifft2


def abs_fft_image(image):
    return np.abs(fft2(image))


def inverse_fft_image(image_complex):
    return ifft2(image_complex)


