from in_out import read_image, write_image
from snipping import get_H_zone

if __name__ == '__main__':
    image = read_image('resource/bridge.tif')
    H_zone = get_H_zone(image)
    write_image('resource/bridge2.tif', H_zone)
    a = 12
