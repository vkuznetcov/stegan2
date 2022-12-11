from math import sqrt, pi, exp

import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d


def cyclic_shift(image, p_min, p_max, p_delta):
    cyclic_shift_images = []
    items_count = int(np.round((p_max - p_min) / p_delta)) + 1
    p_current = p_min
    N1 = image.shape[0]
    N2 = image.shape[1]
    for i in range(0, items_count):
        rN1 = N1 * p_current
        rN2 = N2 * p_current
        temp_image = image.copy()
        for j in range(0, N1):
            for k in range(0, N2):
                temp_image[j][k] = image[int((j + rN1) % N1)][int((k + rN2) % N2)]

        cyclic_shift_images.append(temp_image)
        p_current += p_delta

    return np.array(cyclic_shift_images)


def rot_rest(image, p_min, p_max, p_delta):
    phi_array = np.arange(p_min, p_max + 1, p_delta)
    rot_rest_images = []
    for phi in phi_array:
        modified = ndimage.rotate(image, phi)
        modified = ndimage.rotate(modified, -phi)
        margin = (modified.shape[0] - image.shape[0]) // 2
        if margin > 0:
            modified = modified[margin:-margin, margin:-margin]
        rot_rest_images.append(modified[:512, :512])
    return np.array(rot_rest_images)


def sharpen(image, p_min, p_max, p_delta):
    params = np.arange(p_min, p_max + 1, p_delta)
    a = 1
    sharpen_images = []
    for param in params:
        window = np.ones(shape=(param, param)) / (param * param)
        smooth = convolve2d(image, window, mode='same')
        # print("min", (cw - smooth).min(), "max", (cw - smooth).max())
        c_w = image + a * (image - smooth)
        c_w[c_w > 255] = 255
        c_w[c_w < 0] = 0
        sharpen_images.append(c_w.astype(np.uint8))
    return np.array(sharpen_images)


def white_noise(image, p_min, p_max, p_delta):
    dispersions = np.arange(p_min, p_max + 1, p_delta)
    noised_images = []
    for dispersion in dispersions:
        noise = np.random.normal(size=image.shape, scale=np.sqrt(dispersion))
        noised_images.append((image + noise).astype(np.uint8))
    return np.array(noised_images)
