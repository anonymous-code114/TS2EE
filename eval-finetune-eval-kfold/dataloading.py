import torch.nn.functional as F
import torch
import sys
from torch.utils.data import Dataset
import os
torch.manual_seed(3407)
sys.path.append('../')
import re


class TensorTransforms_simple:
    def resize(self, x):
        # x = x.float()
        x = x.unsqueeze(0)
        if x.size(-1) > 360 or x.size(-2) > 360:
            x = F.interpolate(x, size=(360, 360), mode='area')
        # x = x.half()
        return x.squeeze(0)

    def augment(self, x):
        x = self.resize(x)
        #x = torch.cat([x_layer for x_layer in x], dim=-1)
        #x = x.unsqueeze(0)
        x = x.float()
        return x

    def __call__(self, x):
        return self.augment(x)


class CustomDataset(Dataset):
    def __init__(self, directory1, directory2=None, transform=None, small_size=None):
        self.directory1 = directory1
        self.directory2 = directory2
        self.transform = transform
        self.images = []
        self.labels = []
        self.small_size = small_size

        # Load all images and labels
        for filename in os.listdir(directory1):
            if filename.endswith('.pt'):
                self.images.append(os.path.join(directory1, filename))
                # Assuming the first character of the filename is the label
                match = re.match(r'(-?\d+)_', filename)
                if match:
                    self.labels.append(int(match.group(1)))
        if directory2:
            for filename in os.listdir(directory2):
                if filename.endswith('.pt'):
                    self.images.append(os.path.join(directory2, filename))
                    # Assuming the first character of the filename is the label
                    match = re.match(r'(-?\d+)_', filename)
                    if match:
                        self.labels.append(int(match.group(1)))

        unique_labels = sorted(set(self.labels))
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        self.labels = [label_mapping[label] for label in self.labels]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = torch.load(image_path)
        image = image.float()
        if self.small_size is not None:
            image = image.unsqueeze(0)
            image = F.interpolate(image, size=(self.small_size[1], self.small_size[2]), mode='area')
            image = image.squeeze(0)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def all_labels(self):
        return self.labels


def get_small(directory1, directory2):
    image_size = [1, 10000, 10000]
    for filename in os.listdir(directory1):
        if filename.endswith('.pt'):
            image = torch.load((os.path.join(directory1, filename)))
            if image.size(0) * image.size(1) * image.size(2) < image_size[0] * image_size[1] * image_size[2]:
                image_size = [image.size(0), image.size(1), image.size(2)]
    if directory2:
        for filename in os.listdir(directory2):
            if filename.endswith('.pt'):
                image = torch.load((os.path.join(directory2, filename)))
                if image.size(0) * image.size(1) * image.size(2) < image_size[0] * image_size[1] * image_size[2]:
                    image_size = [image.size(0), image.size(1), image.size(2)]
    return image_size
