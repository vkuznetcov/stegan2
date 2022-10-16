import numpy as np
from scipy.fft import fft2, ifft2


def get_fft_image(image):
    return fft2(image)


def get_inverse_fft_image(image_complex):
    return np.round(np.real(ifft2(image_complex)))


def get_phase_matrix(image):
    return np.angle(image)


def get_abs_matrix(image):
    return np.abs(image)


def get_complex_matrix(r, phi):
    func = np.vectorize(get_complex_number)
    return func(r, phi)


def get_complex_number(r, phi):
    im_phi = complex(0, phi)
    return r * np.exp(im_phi)
