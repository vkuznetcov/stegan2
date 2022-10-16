import random
import numpy as np


def generate_watermark(length, math_expectation, sigma):
    key = random.randint(0, 100)
    random.seed(key)
    result = np.zeros(length)
    for i in range(0, length, 1):
        result[i] = random.gauss(math_expectation, sigma)
    return result, key
