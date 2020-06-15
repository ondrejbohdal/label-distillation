import os
import os.path
import pickle

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage


class K49Dataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.data = self.data.unsqueeze(1)
        self.target = torch.from_numpy(target).long()
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        y = self.target[index]
        img = ToPILImage()(img)

        if self.transform:
            img = self.transform(img)

        return img, y

    def __len__(self):
        return len(self.data)
