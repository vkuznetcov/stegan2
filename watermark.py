import random
import numpy as np


def proximity_measure(omega, omega_changed):
    nominator = np.sum(omega * omega_changed)
    denominator = (np.sqrt(np.sum(omega ** 2)) * np.sqrt(np.sum(omega_changed ** 2)))
    return nominator / denominator


def builtin_watermark(f_w, f, alpha=1):
    return (f_w - f) / (alpha * f)


def generate_watermark(length, math_expectation, sigma, key):
    random.seed(key)
    result = np.zeros(length)
    for i in range(0, length, 1):
        result[i] = random.gauss(math_expectation, sigma)
    return result, key
