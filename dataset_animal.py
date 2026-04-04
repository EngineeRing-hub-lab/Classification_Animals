import os.path
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Compose, Resize, ToTensor


class AnimalDataset(Dataset):
    def __init__(self, root, train = None, transform = None):
        self.root = root
        self.transform = transform
        if train:
            mode = "train"
        else:
            mode = "test"
        self.images_path = []
        self.labels = []
        path_mode = os.path.join(root, mode)
        self.catigories = [f for f in sorted(os.listdir(path_mode))]
        print(self.catigories)
        for i, catigory in enumerate(self.catigories):
            folder_path = os.path.join(path_mode, catigory)
            for img in os.listdir(folder_path):
                self.images_path.append(os.path.join(folder_path, img))
                self.labels.append(i)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        img_path = self.images_path[item]
        img = Image.open(img_path).convert("RGB")
        label = self.labels[item]
        if self.transform:
            img = self.transform(img)
        return img, label

if __name__ == "__main__":
    dataset = AnimalDataset(root="animals_v2/animals", train=True)

