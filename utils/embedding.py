from utils.fourier import get_fft_image, get_abs_matrix, get_phase_matrix, get_complex_matrix, get_inverse_fft_image
from utils.snipping import get_H_zone, merge_pictures_H_zone
from watermark import generate_watermark


def additional_embedding(f, beta, omega, alpha=1):
    return f + alpha * beta * omega


def multiplication_embedding(f, beta, omega, alpha=1):
    return f * (1 + alpha * beta * omega)
