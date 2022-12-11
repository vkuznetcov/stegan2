import numpy as np
from scipy.signal import convolve2d

from lab2.utils.consts import M, SIGMA, KEY, BETA, ALPHA
from lab2.utils.embedding import additional_embedding
from lab2.utils.fourier import get_fft_image, get_abs_matrix, get_phase_matrix, get_complex_matrix, \
    get_inverse_fft_image
from lab2.utils.snipping import get_H_zone, merge_pictures_H_zone
from lab2.utils.watermark import generate_watermark, builtin_watermark, get_rho

def get_betta(c: np.ndarray) -> np.ndarray:
    window = np.ones(shape=(9, 9)) / 81
    mo = convolve2d(c, window, mode='same', boundary='fill')
    mo_2 = np.ndarray(shape=c.shape)
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            tmp = convolve2d((c[i - 4 if i > 3 else 0:
                                i + 5 if i < (c.shape[0] - 3) else c.shape[0],
                              j - 4 if j > 3 else 0:
                              j + 5 if j < (c.shape[1] - 3) else c.shape[1]]
                              - mo[i, j]) ** 2, window, mode='same', boundary='fill')
            mo_2[i, j] = tmp[i if i < 3 else 4, j if j < 3 else 4]
    res = np.sqrt(mo_2)
    return res / res.max()


def embed(container):
    # 1. Реализовать генерацию ЦВЗ 𝛺 как псевдослучайной последовательности заданной длины из чисел,
    # распределённых по нормальному закону
    H_zone_length = int(container.shape[0] * 0.25) * int(container.shape[1] * 0.5)
    watermark, _ = generate_watermark(H_zone_length, M, SIGMA, KEY)

    # 2. Реализовать трансформацию исходного контейнера к пространству признаков
    fft_container = get_fft_image(container)
    abs_fft_container = get_abs_matrix(fft_container)
    phase_fft_container = get_phase_matrix(fft_container)

    # 3. Осуществить встраивание информации аддитивным методом встраивания.
    # Значения параметров встраивания устанавливается произвольным образом.
    H_zone = get_H_zone(abs_fft_container)
    watermark = watermark.reshape(H_zone.shape)
    H_zone_watermark = additional_embedding(H_zone, BETA, watermark, ALPHA)

    # 4. Сформировать носитель информации при помощи обратного преобразования
    # от матрицы признаков к цифровому сигналу.  Сохранить его на диск.
    merged_abs_picture = merge_pictures_H_zone(abs_fft_container, H_zone_watermark)
    complex_matrix = get_complex_matrix(merged_abs_picture, phase_fft_container)
    processed_image = get_inverse_fft_image(complex_matrix)

    return H_zone, watermark, processed_image


def embed_with_beta(container, beta=1):
    # 1. Реализовать генерацию ЦВЗ 𝛺 как псевдослучайной последовательности заданной длины из чисел,
    # распределённых по нормальному закону
    H_zone_length = int(container.shape[0] * 0.25) * int(container.shape[1] * 0.5)
    watermark, _ = generate_watermark(H_zone_length, M, SIGMA, KEY)

    # 2. Реализовать трансформацию исходного контейнера к пространству признаков
    fft_container = get_fft_image(container)
    abs_fft_container = get_abs_matrix(fft_container)
    phase_fft_container = get_phase_matrix(fft_container)

    # 3. Осуществить встраивание информации аддитивным методом встраивания.
    # Значения параметров встраивания устанавливается произвольным образом.
    H_zone = get_H_zone(abs_fft_container)

    betta = get_betta(H_zone)
    watermark = watermark.reshape(H_zone.shape)
    H_zone_watermark = additional_embedding(H_zone, beta, watermark, ALPHA)
    merged_abs_picture = merge_pictures_H_zone(abs_fft_container, H_zone_watermark)
    complex_matrix = get_complex_matrix(merged_abs_picture, phase_fft_container)
    processed_image = get_inverse_fft_image(complex_matrix)

    return H_zone, watermark, processed_image, betta


def get_rho_for_image(H_zone, watermark, processed_image, beta=BETA):
    fft_p_image = get_fft_image(processed_image)
    abs_fft_p_image = get_abs_matrix(fft_p_image)
    H_zone_p = get_H_zone(abs_fft_p_image)
    changed_watermark = builtin_watermark(H_zone_p, H_zone, ALPHA, beta)
    rho = get_rho(watermark, changed_watermark)
    return rho
