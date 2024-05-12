import os.path

BASE_CHART = "1234567890zxcvbnmasdfghjklqwertyuiopZXCVBNMASDFGHJKLQWERTYUIOP"
# 单个验证码生成次数
GENERATION_NUM = 1000
BASE_PATH = "F:/xiangmu/Python/ocr_captcha"
# 保存路径
TRAIN_PATH = os.path.join(BASE_PATH, "data/train/img/")
# 测试集路径
TEST_PATH = os.path.join(BASE_PATH, "data/test/img/")
# 验证集路径
VAL_PATH = os.path.join(BASE_PATH, "data/val/img/")
# tensorboard日志路径
TENSORBOARD_PATH = os.path.join(BASE_PATH, "log/tensorboard")
# 模型保存路径
TRAIN_MODEL_PATH = os.path.join(BASE_PATH, "model/")
# 模型命名
TRAIN_MODEL_NAME = "ocr_captcha_model.pth"
