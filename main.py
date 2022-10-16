from utils.fourier import get_fft_image, get_phase_matrix, get_abs_matrix, get_complex_matrix, get_inverse_fft_image
from utils.embedding import additional_embedding
from utils.in_out import read_image, write_image
from utils.snipping import get_H_zone, merge_pictures_H_zone
from watermark import generate_watermark


if __name__ == '__main__':
     # 1. Get image
     container = read_image('resource/bridge.tif')

     # 2. Get fft of image
     fft_container = get_fft_image(container)

     # 3. Get abs of image (+ phase)
     abs_fft_container = get_abs_matrix(fft_container)
     phase_fft_container = get_phase_matrix(fft_container)

     # 4. Snipping
     H_zone = get_H_zone(abs_fft_container)
     # For Pavel !!!
     # split_image_to_4_parts...

     # 5. Get watermark
     watermark_length = H_zone.shape[0] * H_zone.shape[1]
     watermark = generate_watermark(watermark_length, 10, 36)[0].reshape(H_zone.shape[0], H_zone.shape[1])

     # 6. Embedding
     H_zone_abs_container_with_watermark = additional_embedding(H_zone, 1, watermark, 10)

     # 7. Merge pictures
     abs_container_with_watermark = merge_pictures_H_zone(abs_fft_container, H_zone_abs_container_with_watermark)

     # 8. Recover complex matrix
     complex_container_with_watermark = get_complex_matrix(abs_container_with_watermark, phase_fft_container)

     # 9. ifft
     result_image = get_inverse_fft_image(complex_container_with_watermark).astype('uint8')

     # 10. Save
     write_image(result_image, 'resource/result.tif')


