from utils.fourier import get_fft_image, get_abs_matrix, get_phase_matrix, get_complex_matrix, get_inverse_fft_image
from utils.snipping import get_H_zone, merge_pictures_H_zone
from watermark import generate_watermark


def additional_embedding(f, beta, omega, alpha=1):
    return f + alpha * beta * omega


def multiplication_embedding(f, beta, omega, alpha=1):
    return f * (1 + alpha * beta * omega)


def embed_watermark(image, m, sigma, alpha, beta, key):
    # 2. Get fft of image
    fft_container = get_fft_image(image)

    # 3. Get abs of image (+ phase)
    abs_fft_container = get_abs_matrix(fft_container)
    phase_fft_container = get_phase_matrix(fft_container)

    # 4. Snipping
    H_zone = get_H_zone(abs_fft_container)

    # 5. Get watermark
    watermark_length = H_zone.shape[0] * H_zone.shape[1]
    # нормальное распределение
    watermark = generate_watermark(watermark_length, m, sigma, key)[0].reshape(H_zone.shape[0], H_zone.shape[1])

    # 6. Embedding (аддитивное)
    H_zone_abs_container_with_watermark = additional_embedding(H_zone, beta, watermark, alpha)

    # 7. Merge pictures
    abs_container_with_watermark = merge_pictures_H_zone(abs_fft_container, H_zone_abs_container_with_watermark)

    # 8. Recover complex matrix
    complex_container_with_watermark = get_complex_matrix(abs_container_with_watermark, phase_fft_container)

    # 9. ifft
    return get_inverse_fft_image(complex_container_with_watermark)
