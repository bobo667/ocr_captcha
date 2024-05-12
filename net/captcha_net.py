from torch import nn
from torch.utils.data import DataLoader
from utils.captcha_dataset import CaptchaDataset
from config.const import *


class CaptchaNet(nn.Module):

    def __init__(self):
        super(CaptchaNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5 * 7 * 64, 1024),
            nn.ReLU(),
            nn.Linear(1024, len(BASE_CHART))
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    dataset = CaptchaDataset(TRAIN_PATH)
    net = CaptchaNet()
    img = dataset.__getitem__(0)[0].view((1, 3, 40, 60))
    print(net(img).shape)
