import os

from torch.utils.data import Dataset
from torchvision.transforms import transforms

from config.const import *
from utils.comm import text_to_vec, img_convert


class CaptchaDataset(Dataset):

    def __init__(self, image_path):
        self.img_list = os.listdir(image_path)
        self.label_list = [img.split("_")[0] for img in self.img_list]
        self.img_list = [os.path.join(image_path, img_name) for img_name in self.img_list]

        self.run = transforms.Compose([
            img_convert,
            transforms.RandomRotation(0.1),
            transforms.RandomAffine(0.1),
            transforms.Resize((40, 60)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = self.run(img_path)

        lab = self.label_list[index]
        lab = text_to_vec(lab).view(1, -1)[0]

        return img, lab

    def __len__(self):
        return len(self.img_list)


if __name__ == '__main__':
    data = CaptchaDataset(TRAIN_PATH)
    print(data.__getitem__(0))
