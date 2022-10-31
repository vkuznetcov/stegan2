import numpy as np


def get_H_zone(image, modifier=0.25):
    image_shape = image.shape
    result_shape = (image_shape[0] * modifier, image_shape[1] * modifier)
    y_border_size = result_shape[1]
    x_border_size = result_shape[0]

    left_border = int(x_border_size - 1)
    upper_border = int(y_border_size - 1)

    right_border = int(image_shape[0] - x_border_size - 1)
    lower_border = int(image_shape[1] - y_border_size - 1)

    result = np.copy(image[left_border:right_border, upper_border:lower_border])
    return result


def merge_pictures_H_zone(image_source, snipped_part, modifier=0.25):
    # modifier - 1/4 часть от image
    image_shape = image_source.shape
    result_shape = (image_shape[0] * modifier, image_shape[1] * modifier)  # размер H (128.0, 128.0)
    y_border_size = result_shape[1]
    x_border_size = result_shape[0]

    left_border = int(x_border_size - 1)
    upper_border = int(y_border_size - 1)

    right_border = int(image_shape[0] - x_border_size - 1)
    lower_border = int(image_shape[1] - y_border_size - 1)

    result = np.copy(image_source)
    result[left_border:right_border, upper_border:lower_border] = snipped_part  # snipped_part - H зона

    return result


def merge_pictures_H_zone_parts(image, snipped_parts):
    fake_H_zone = get_H_zone(image)
    x_center, y_center = int(fake_H_zone.shape[0] / 2), int(fake_H_zone.shape[1] / 2)

    # result = np.copy(image)
    fake_H_zone[0:x_center, 0:y_center] = snipped_parts[0]
    fake_H_zone[0:x_center, y_center:image.shape[1]] = snipped_parts[1]
    fake_H_zone[x_center:image.shape[0], 0:y_center] = snipped_parts[2]
    fake_H_zone[x_center:image.shape[0], y_center:image.shape[1]] = snipped_parts[3]

    return merge_pictures_H_zone(image, fake_H_zone)
