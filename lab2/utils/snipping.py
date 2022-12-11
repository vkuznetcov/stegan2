import numpy as np


def get_H_zone(image, modifier=0.25):
    image_shape = image.shape
    result_shape = (image_shape[0] * modifier, image_shape[1] * modifier)
    y_border_size = result_shape[1]
    x_border_size = result_shape[0]

    left_border = int(x_border_size - 1)
    upper_border = int(y_border_size - 1)

    right_border = int(image_shape[0] - x_border_size - 1 - 128)
    lower_border = int(image_shape[1] - y_border_size - 1)

    result = np.copy(image[left_border:right_border, upper_border:lower_border])
    return result


def merge_pictures_H_zone(image_source, snipped_part, modifier=0.25):
    image_shape = image_source.shape
    result_shape = (image_shape[0] * modifier, image_shape[1] * modifier)
    y_border_size = result_shape[1]
    x_border_size = result_shape[0]

    left_border = int(x_border_size - 1)
    upper_border = int(y_border_size - 1)

    right_border = int(image_shape[0] - x_border_size - 1 - 128)
    lower_border = int(image_shape[1] - y_border_size - 1)

    result = np.copy(image_source)
    result[left_border:right_border, upper_border:lower_border] = snipped_part

    return result


