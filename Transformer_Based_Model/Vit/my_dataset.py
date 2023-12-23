from PIL import Image
import torch
from torch.utils.data import Dataset

class My_Dataset(Dataset):
    def __init__(self,image_path,image_label,transform=None):
        self.images_path = image_path
        self.images_label = image_label
        self.transform=transform

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_label[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label