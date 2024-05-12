import os.path

from config.const import *
from captcha.image import ImageCaptcha
from uuid import uuid4
import random


def generate_captcha(path=TRAIN_PATH, generation_num=GENERATION_NUM, length=0):
    print(F"开始生成 ： {TRAIN_PATH}")
    captcha = ImageCaptcha(width=(40 if length == 0 else 40 * length))

    if not os.path.exists(path):
        os.makedirs(path)

    if length > 0:
        for _ in range(generation_num):
            content = random.sample(BASE_CHART, k=4)
            # 生成验证码
            captcha_image = captcha.generate_image(content)
            captcha_image.save(os.path.join(path, '{}_{}.png'.format("".join(content), uuid4())))
    else:
        for (index, char) in enumerate(BASE_CHART):
            for _ in range(generation_num):
                # 生成验证码
                captcha_image = captcha.generate_image(char)
                captcha_image.save(os.path.join(path, '{}_{}.png'.format(char, uuid4())))

    print("数据生成完成")


def generate_test_captcha():
    generate_captcha(TEST_PATH, 10)


def generate_train_captcha():
    generate_captcha(TRAIN_PATH, GENERATION_NUM)


def generate_val_captcha():
    generate_captcha(VAL_PATH, 100, length=4)


if __name__ == '__main__':
    # generate_train_captcha()
    # generate_test_captcha()
    generate_val_captcha()
