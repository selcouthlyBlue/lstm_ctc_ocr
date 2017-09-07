import os
import random
from multiprocessing import Pool

from captcha.image import ImageCaptcha


def rand_rgb():
    return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)


# 10+26+26
char_set = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
imgDir = None
numProcess = 12


def gen_rand():
    buf = ""
    max_len = random.randint(4, 6)
    for i in range(max_len):
        buf += random.choice(char_set)
    return buf


def generate_image(ind):
    global imgDir
    captcha = ImageCaptcha(fonts=['./fonts/Ubuntu-M.ttf'])
    theChars = gen_rand()
    img_name = '{:08d}'.format(ind) + '_' + theChars + '.png'
    img_path = imgDir + '/' + img_name
    captcha.write(theChars, img_path)
    print(img_path)


def run(num, path):
    global imgDir
    imgDir = path
    if not os.path.exists(path):
        os.makedirs(path)
    with Pool(processes=numProcess) as pool:
        pool.map(generate_image, range(num))


if __name__ == '__main__':
    run(64 * 2000, './data/train')
    run(500, './data/val')
