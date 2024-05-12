import os.path

from config.const import *
from utils.captcha_dataset import CaptchaDataset
from net.captcha_net import CaptchaNet
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(TENSORBOARD_PATH)


def train(epoch=100,
          batch_size=16,
          num_workers=1):
    # 定义训练集
    train_dataset = CaptchaDataset(TRAIN_PATH)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # 定义测试集
    test_dataset = CaptchaDataset(TEST_PATH)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # 定义驱动
    drive = "cuda" if torch.cuda.is_available() else "cpu"

    print(F"当前驱动为 {drive}")

    net = CaptchaNet()
    net.to(drive)

    # 定义损失函数
    criterion_fn = torch.nn.CrossEntropyLoss()
    criterion_fn.to(drive)

    # 定义优化器
    # 1e -2 = 1 * (10)^(-2) = 1 / 100 = 0.01
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)

    if not os.path.exists(TRAIN_MODEL_PATH):
        os.makedirs(TRAIN_MODEL_PATH)

    # 加载已经训练好的模型
    model_all_path = load_model_dict(net)

    train_step_count = 0
    test_step_count = 0

    for epoch in range(epoch):
        print(F"---------- 第{epoch + 1}轮训练开始 -------------")
        # 开启训练模式
        net.train()

        for img, lab in train_data_loader:
            imgs = img.to(drive)
            labs = lab.to(drive)

            output = net(imgs)
            # 计算损失度
            loss = criterion_fn(output, labs)
            # 梯度清0
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 使用优化器进行优化
            optimizer.step()

            train_step_count += 1

            if train_step_count % 100 == 0:
                print(F"训练次数 {train_step_count} ，当前Loss {loss.item()}")
                writer.add_scalar("train_loss", loss.item(), train_step_count)

        loss_count = 0
        accuracy_count = 0
        for imgs, labs in test_data_loader:
            imgs = imgs.to(drive)
            labs = labs.to(drive)
            output = net(imgs)
            loss = criterion_fn(output, labs)
            loss_count += loss.item()
            accuracy = (output.argmax(dim=1) == labs.argmax(dim=1)).sum()
            accuracy_count += accuracy

        test_step_count += 1
        correctness = round((accuracy_count.item() / len(test_dataset)) * 100, 2)
        print(F"第 {epoch} 轮训练测试结果：正确率为 {correctness}%")
        writer.add_scalar("test_result", correctness, test_step_count)

        # 保存模型
        torch.save(net.state_dict(), model_all_path)

    writer.close()


def load_model_dict(net: CaptchaNet):
    model_all_path = os.path.join(TRAIN_MODEL_PATH, TRAIN_MODEL_NAME)
    if os.path.exists(model_all_path):
        model_dict = torch.load(model_all_path)
        net.load_state_dict(model_dict)
    return model_all_path


if __name__ == '__main__':
    train(10)
