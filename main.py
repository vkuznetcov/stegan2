import cv2

from consts import ALPHA, BETA, KEY, M, SIGMA
from utils.fourier import get_fft_image, get_phase_matrix, get_abs_matrix, get_complex_matrix, get_inverse_fft_image
from utils.embedding import additional_embedding, multiplication_embedding, embed_watermark
from utils.in_out import read_image, write_image
from utils.snipping import get_H_zone, merge_pictures_H_zone, split_image_to_4_parts, merge_pictures_H_zone_parts
from watermark import generate_watermark, get_rho, builtin_watermark




def get_optimal_parameter(initial_container, target_value=0.9, max_possible_value=1.0):
    alpha = 0.01
    alphas = list()
    omega = list()
    omega_tilda = list()

    rho_max = 0
    alpha_max = 0

    while alpha < max_possible_value:

        print(f'step {alpha}')
        # 1. Get H zone from changed picture
        changed_container = embed_watermark(initial_container, M, SIGMA, alpha, BETA, KEY)
        abs_fft_recover = get_abs_matrix(get_fft_image(changed_container))
        H_zone_recover = get_H_zone(abs_fft_recover)

        # 2. Get H zone from initial picture
        abs_fft_container = get_abs_matrix(get_fft_image(initial_container))
        H_zone = get_H_zone(abs_fft_container)

        new_shape = [1, H_zone_recover.shape[0] * H_zone_recover.shape[1]]
        watermark = generate_watermark(new_shape[1], M, SIGMA, KEY)[0]
        watermark_tilda = builtin_watermark(
            H_zone_recover.reshape(new_shape[0], new_shape[1]),
            H_zone.reshape(new_shape[0], new_shape[1]),
            ALPHA)
        rho = get_rho(watermark.reshape(new_shape[0], new_shape[1]), watermark_tilda)

        if rho > rho_max:
            rho_max = rho
            alpha_max = alpha

        if rho >= target_value:
            alphas.append(alpha)
            omega.append(watermark)
            omega_tilda.append(watermark_tilda)
        alpha += 0.01

    if len(alphas) == 0:
        return alpha_max

    alpha_min = alphas[0]
    PSNR_min = cv2.PSNR(omega[0], omega_tilda[0])

    for i in range(1, len(alphas)):
        PSNR = cv2.PSNR(omega[i], omega_tilda[i])
        if PSNR_min > PSNR:
            PSNR_min = PSNR
            alpha_min = alphas[i]

    return alpha_min


# def get_rho():
#     fft_container = get_fft_image(container)
#
#     # 3. Get abs of image (+ phase)
#     abs_fft_container = get_abs_matrix(fft_container)
#
#     # 4. Snipping
#     H_zone = get_H_zone(abs_fft_container)
#
#     # 5.
#
#     new_shape = [1, H_zone.shape[0] * H_zone.shape[1]]
#
#     result_image = read_image('resource/result.png')
#     fft_recover = get_fft_image(result_image)
#     abs_fft_recover = get_abs_matrix(fft_recover)
#     H_zone_recover = get_H_zone(abs_fft_recover).reshape(new_shape[0], new_shape[1])
#
#     watermark_length = H_zone.shape[0] * H_zone.shape[1]
#     watermark = generate_watermark(watermark_length, 300, 10, KEY)[0]
#
#     H_zone = H_zone.reshape(new_shape[0], new_shape[1])
#
#     reshaped_watermark = watermark.reshape(new_shape[0], new_shape[1])
#
#     return get_rho(reshaped_watermark, builtin_watermark(H_zone_recover, H_zone, ALPHA))



def different_fragments(parts, result_image, watermark):
    fft_recover = get_fft_image(result_image)
    abs_fft_recover = get_abs_matrix(fft_recover)
    H_zone_recover = get_H_zone(abs_fft_recover)
    recover_parts = split_image_to_4_parts(H_zone_recover)

    new_shape = [1, int(H_zone_recover.shape[0] / 2) * int(H_zone_recover.shape[1] / 2)]

    for i in range(0, 4, 1):
        watermark_tilda = builtin_watermark(recover_parts[i].reshape(new_shape[0], new_shape[1]),
                                            parts[i].reshape(new_shape[0], new_shape[1]),
                                            ALPHA)
        watermark = watermark.reshape(new_shape[0], new_shape[1])
        watermark_tilda = watermark_tilda.reshape(new_shape[0], new_shape[1])
        rho = get_rho(watermark,
                      watermark_tilda)
        psnr = cv2.PSNR(watermark, watermark_tilda)
        print(f'Result {i}: Rho={rho}; PSNR={psnr}')


def get_optimal_alpha(f, abs_fft_container, phase_fft_container, watermark):

    rho = 0.0
    alpha = 0.0
    while rho < 0.9 and alpha < 10.0:
        H_zone_watermark    = additional_embedding(H_zone, BETA, watermark, ALPHA)

        merged_abs_picture  = merge_pictures_H_zone(abs_fft_container, H_zone_watermark)
        complex_matrix      = get_complex_matrix(merged_abs_picture, phase_fft_container)
        processed_image     = get_inverse_fft_image(complex_matrix)
        #write_image(processed_image, 'resource/bridge_processed_tmp.tif')

        #processed_image     = write_image(processed_image, 'resource/bridge_processed_tmp.tif')
        fft_p_image         = get_fft_image(container)
        abs_fft_p_image     = get_abs_matrix(fft_container)
        phase_fft_p_image   = get_phase_matrix(fft_container)

        H_zone_p            = get_H_zone(abs_fft_p_image)
        changed_watermark   = builtin_watermark(H_zone_p, H_zone, ALPHA)
        rho                 = get_rho(watermark, changed_watermark)
        alpha += 0.1
        print(f'RHO: {rho}')


if __name__ == '__main__':

    container = read_image('resource/bridge.tif')

    # 1. Реализовать генерацию ЦВЗ 𝛺 как псевдослучайной последовательности заданной длины из чисел,
    # распределённых по нормальному закону
    H_zone_length = int(container.shape[0] * 0.5) * int(container.shape[1] * 0.5)
    watermark, key_gen  = generate_watermark(H_zone_length, M, SIGMA, KEY)

    # 2. Реализовать трансформацию исходного контейнера к пространству признаков
    fft_container       = get_fft_image(container)
    abs_fft_container   = get_abs_matrix(fft_container)
    phase_fft_container = get_phase_matrix(fft_container)

    # 3. Осуществить встраивание информации аддитивным методом встраивания.
    # Значения параметравстраивания устанавливается произвольным образом.
    H_zone              = get_H_zone(abs_fft_container)
    watermark           = watermark.reshape(H_zone.shape)
    H_zone_watermark    = additional_embedding(H_zone, BETA, watermark, ALPHA)

    # 4. Сформировать носитель информации при помощи обратного преобразования
    # от матрицы признаков к цифровому сигналу.  Сохранить его на диск.
    merged_abs_picture  = merge_pictures_H_zone(abs_fft_container, H_zone_watermark)
    complex_matrix      = get_complex_matrix(merged_abs_picture, phase_fft_container)
    processed_image     = get_inverse_fft_image(complex_matrix)
    write_image(processed_image, 'resource/bridge_processed.png')

    # 5. Считать носитель информации из файла. Реализовать трансформацию исходного контейнера к пространству признаков
    processed_image     = read_image('resource/bridge_processed.png')
    fft_p_image         = get_fft_image(processed_image)
    abs_fft_p_image     = get_abs_matrix(fft_p_image)
    phase_fft_p_image   = get_phase_matrix(fft_p_image)

    # 6. Сформировать оценку встроенного ЦВЗ 𝛺̃неслепым методом (то есть, с использованием
    # матрицы признаков исходного контейнера); выполнить детектирование при помощи функции
    # близости 𝜌(𝛺,𝛺̃) вида (6.11).
    H_zone_p            = get_H_zone(abs_fft_p_image)
    changed_watermark   = builtin_watermark(H_zone_p, H_zone, ALPHA)
    rho                 = get_rho(watermark, changed_watermark)

    get_optimal_alpha(H_zone, abs_fft_container, phase_fft_container, watermark)
    # 7. Осуществить автоматический подбор значения параметра встраивания методом перебора
    # с целью обеспечения заданного значения функции близости 𝜌



    # # 2. Get fft of image
    # fft_container = get_fft_image(container)
    #
    # # 3. Get abs of image (+ phase)
    # abs_fft_container = get_abs_matrix(fft_container)
    #
    # # 4. Snipping
    # H_zone = get_H_zone(abs_fft_container)
    #
    # # 5.
    #
    # new_shape = [1, H_zone.shape[0] * H_zone.shape[1]]
    #
    # result_image = read_image('resource/result.png')
    # fft_recover = get_fft_image(result_image)
    # abs_fft_recover = get_abs_matrix(fft_recover)
    # H_zone_recover = get_H_zone(abs_fft_recover).reshape(new_shape[0], new_shape[1])
    #
    #
    # watermark_length = H_zone.shape[0] * H_zone.shape[1]
    # watermark = generate_watermark(watermark_length, 300, 10, KEY)[0]
    #
    # H_zone = H_zone.reshape(new_shape[0], new_shape[1])
    #
    # reshaped_watermark = watermark.reshape(new_shape[0], new_shape[1])
    #
    # prox_measure = proximity_measure(reshaped_watermark, builtin_watermark(H_zone_recover, H_zone, ALPHA))
    # print(f'Proximity measure: {prox_measure}')

    # +====================================================================================================

    # fft_container = get_fft_image(container)
    # abs_fft_container = get_abs_matrix(fft_container)
    # phase_fft_container = get_phase_matrix(fft_container)
    #
    # H_zone = get_H_zone(abs_fft_container)
    # initial_parts = split_image_to_4_parts(H_zone)
    # watermark_length = initial_parts[0].shape[0] * initial_parts[0].shape[1]
    # watermark = generate_watermark(watermark_length, 300, 10, KEY)[0]
    #
    # for i in range(0, 4, 1):
    #     initial_parts[i] = multiplication_embedding(initial_parts[i], BETA, watermark.reshape(initial_parts[i].shape[0],
    #                                                                                           initial_parts[i].shape[
    #                                                                                               1]), ALPHA)
    #
    # abs_container_with_watermark = merge_pictures_H_zone_parts(abs_fft_container, initial_parts)
    # complex_container_with_watermark = get_complex_matrix(abs_container_with_watermark, phase_fft_container)
    # result_image = get_inverse_fft_image(complex_container_with_watermark)
    # write_image(result_image, 'resource/Paul.png')
    #
    # # result_image = read_image('resource/result.png')
    # abs_fft_container = get_abs_matrix(get_fft_image(container))
    # H_zone = get_H_zone(abs_fft_container)
    # initial_parts = split_image_to_4_parts(H_zone)
    #
    # different_fragments(initial_parts, result_image, watermark)

    # alpha_result = get_optimal_parameter(container)
    # print(f'{alpha_result}')
