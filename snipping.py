import numpy as np
from PIL import Image


def get_H_zone(image, modifier=0.25):

    image_shape = image.shape
    result_shape = (image_shape[0] * modifier, image_shape[1] * modifier)
    y_border_size = result_shape[1]
    x_border_size = result_shape[0]

    left_border = x_border_size - 1
    upper_border = y_border_size - 1

    right_border = image_shape[0] - x_border_size - 1
    lower_border = image_shape[1] - y_border_size - 1

    im = Image.fromarray(image, 'L')
    cropped_image = im.crop((left_border, upper_border, right_border, lower_border))
    cropped_image.save()




