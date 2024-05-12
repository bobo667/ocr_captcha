import torch
from PIL import Image

from config.const import *


# 文本转换成张量
def text_to_vec(text):
    vectors = torch.zeros(len(text), len(BASE_CHART))

    for i, char in enumerate(text):
        if char in BASE_CHART:
            index = BASE_CHART.index(char)
            vectors[i, index] = 1
        else:
            raise ValueError(F'character {char} not in BASE_CHART')
    return vectors


# 张量转换为文本
def vec_to_text(vec):
    text_label = ""
    for row in vec:
        index = torch.argmax(row).item()
        char = BASE_CHART[index]
        text_label += char

    return text_label


def img_convert(img):
    return Image.open(img).convert("RGB")
