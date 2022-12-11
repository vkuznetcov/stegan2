import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imsave

from consts import task1consts, task2consts, task4consts, task3consts
from lab2.task import embed, get_rho_for_image, embed_with_beta
from lab2.utils.in_out import read_image
from utils.distortion import cyclic_shift, rot_rest, sharpen, white_noise

if __name__ == '__main__':
    container = read_image('resource/bridge.tif')
    H_zone, watermark, embedded_image = embed(container)

    rho = get_rho_for_image(H_zone, watermark, embedded_image)
    print(f'Original rho: {rho}')

    fig = plt.figure()
    fig.set_figheight(15)
    fig.set_figwidth(15)

    cyclic_shift_images = cyclic_shift(embedded_image, task1consts.p_min, task1consts.p_max, task1consts.delta_p)
    cyclic_shift_rhos = []
    for i in range(0, cyclic_shift_images.shape[0]):
        cyclic_shift_rhos.append(get_rho_for_image(H_zone, watermark, cyclic_shift_images[i]))
    #
    x = np.arange(0.1, 1, 0.1)
    plt.subplot(2, 2, 1)
    plt.plot(x, cyclic_shift_rhos)
    plt.title('Rhos (cyclic_shift)')
    # plt.show()

    rot_rest_images = rot_rest(embedded_image, task2consts.p_min, task2consts.p_max, task2consts.delta_p)
    rot_rest_rhos = []
    for i in range(0, rot_rest_images.shape[0]):
        rot_rest_rhos.append(get_rho_for_image(H_zone, watermark, rot_rest_images[i]))
    #
    x = np.arange(0, 43, 7)
    plt.subplot(2, 2, 2)
    plt.plot(x, rot_rest_rhos)
    plt.title('Rhos (rot_rest)')
    # plt.show()

    sharpen_images = sharpen(embedded_image, task3consts.p_min, task3consts.p_max, task3consts.delta_p)
    sharpen_rhos = []
    for i in range(0, sharpen_images.shape[0]):
        sharpen_rhos.append(get_rho_for_image(H_zone, watermark, sharpen_images[i]))
    #
    x = np.arange(3, 16, 2)
    plt.subplot(2, 2, 3)
    plt.plot(x, sharpen_rhos)
    plt.title('Rhos (sharpen)')
    # plt.show()

    noised_images = white_noise(embedded_image, task4consts.p_min, task4consts.p_max, task4consts.delta_p)
    noised_rhos = []
    for i in range(0, noised_images.shape[0]):
        noised_rhos.append(get_rho_for_image(H_zone, watermark, noised_images[i]))
    #
    x = np.arange(400, 1001, 100)
    plt.subplot(2, 2, 4)
    plt.plot(x, noised_rhos)
    plt.title('Rhos (white_noise)')
    plt.show()



    # imsave('resource/rot.tif', noised_images[6])

    # =======================================8========================================
    fig = plt.figure()
    fig.set_figheight(15)
    fig.set_figwidth(15)

    H_zone, watermark, embedded_image, beta = embed_with_beta(container)
    cyclic_shift_images = cyclic_shift(embedded_image, task1consts.p_min, task1consts.p_max, task1consts.delta_p)
    cyclic_shift_rhos = []
    for i in range(0, cyclic_shift_images.shape[0]):
        cyclic_shift_rhos.append(get_rho_for_image(H_zone, watermark, cyclic_shift_images[i], beta))
    #
    x = np.arange(0.1, 1, 0.1)
    plt.subplot(2, 2, 1)
    plt.plot(x, cyclic_shift_rhos)
    plt.title('Rhos (cyclic_shift beta MSE)')
    # plt.show()

    rot_rest_images = rot_rest(embedded_image, task2consts.p_min, task2consts.p_max, task2consts.delta_p)
    rot_rest_rhos = []
    for i in range(0, rot_rest_images.shape[0]):
        rot_rest_rhos.append(get_rho_for_image(H_zone, watermark, rot_rest_images[i], beta))
    #
    x = np.arange(0, 43, 7)
    plt.subplot(2, 2, 2)
    plt.plot(x, rot_rest_rhos)
    plt.title('Rhos (rot_rest beta MSE)')
    # plt.show()

    sharpen_images = sharpen(embedded_image, task3consts.p_min, task3consts.p_max, task3consts.delta_p)
    sharpen_rhos = []
    for i in range(0, sharpen_images.shape[0]):
        sharpen_rhos.append(get_rho_for_image(H_zone, watermark, sharpen_images[i], beta))
    #
    x = np.arange(3, 16, 2)
    plt.subplot(2, 2, 3)
    plt.plot(x, sharpen_rhos)
    plt.title('Rhos (sharpen beta MSE)')
    # plt.show()

    noised_images = white_noise(embedded_image, task4consts.p_min, task4consts.p_max, task4consts.delta_p)
    noised_rhos = []
    for i in range(0, noised_images.shape[0]):
        noised_rhos.append(get_rho_for_image(H_zone, watermark, noised_images[i], beta))
    #
    x = np.arange(400, 1001, 100)
    plt.subplot(2, 2, 4)
    plt.plot(x, noised_rhos)
    plt.title('Rhos (white_noise beta MSE)')
    plt.show()

