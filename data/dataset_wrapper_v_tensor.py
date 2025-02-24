import os
import random
import re
from typing import Any, Callable, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
# from data_aug.gaussian_blur import GaussianBlur
from torchvision import datasets
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
torch.manual_seed(3407)
# torch.set_printoptions(profile="full")
np.random.seed(0)


class TensorTransforms:
    def __init__(self, jitter_std=0.1, scaling_factor=1.1, permutation_segments=5, masking_percentage=0.1,
                 small_size=None):
        self.random_state = random.random()
        self.jitter_std = jitter_std
        self.scaling_factor = scaling_factor
        self.permutation_segments = permutation_segments
        self.masking_percentage = masking_percentage
        self.prob_squeeze_time = 0.5
        self.prob_jitter = 0.5
        self.prob_scaling = 0.5
        self.prob_permutation = 0.5
        self.prob_masking = 0.5
        self.prob_flip = 0.5
        self.size = small_size

    def jitter(self, x):
        noise = torch.normal(0, self.jitter_std, size=x.size())
        return x + noise

    def resize(self, x):
        # x = x.float()
        x = x.unsqueeze(0)
        if x.size(-1) > 360 or x.size(-2) > 360:
             x = F.interpolate(x, size=(360, 360), mode='area')
        # x = x.half()
        return x.squeeze(0)

    def squeeze_time(self, x):
        # x = x.float()
        x = x.unsqueeze(0)
        x = F.interpolate(x, size=(x.size(2), int(x.size(3) * random.uniform(0.9, 1.1))), mode='area')
        # x = x.half()
        return x.squeeze(0)

    def scaling(self, x):
        num_sequences_to_scale = int(x.size(2) * 0.1)
        indices_to_scale = np.random.choice(x.size(2), num_sequences_to_scale, replace=False)

        for i in indices_to_scale:
            scaling = torch.FloatTensor([np.random.uniform(self.scaling_factor * 0.8, self.scaling_factor)])
            x[:, :, i] *= scaling

        return x

    def permutation(self, x):
        random_integer = random.randint(1, self.permutation_segments)
        segments = torch.chunk(x, random_integer, dim=2)
        permuted_indices = torch.randperm(random_integer)
        permuted_segments = [segments[i] for i in permuted_indices]
        return torch.cat(permuted_segments, dim=2)

    def masking(self, x):

        mask = torch.bernoulli(torch.full((x.size(0), 1, x.size(2)), 1 - self.masking_percentage))
        mask = mask.expand_as(x)
        # mask_check = mask.cpu().numpy()
        return x * mask

    def flip(self, x):
        return torch.flip(x, dims=[2])

    def augment(self, x):
        random.seed(1919810)
        x = self.resize(x)
        if random.random() < self.prob_squeeze_time:
            x = self.squeeze_time(x)
        # if random.random() < self.prob_jitter:
        #     x = self.jitter(x)
        if random.random() < self.prob_scaling:
            x = self.scaling(x)
        if random.random() < self.prob_permutation:
            x = self.permutation(x)
        if random.random() < self.prob_masking:
            x = self.masking(x)
        #x = torch.cat([x_layer for x_layer in x], dim=-1)
        #x = x.unsqueeze(0)
        if random.random() < self.prob_flip:
            x = self.flip(x)
        return x

    def __call__(self, x):
        return self.augment(x)


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
                    match = re.match(r'(-?\d+)_', filename)
                    if match:
                        self.labels.append(int(match.group(1)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = torch.load(image_path)
        image = image.float()
        #image = image.unsqueeze(0)
        #image = F.interpolate(image, size=(self.small_size[1], self.small_size[2]), mode='area')
        # image = F.interpolate(image, size=(224*2, 224*2), mode='area')
        #image = image.squeeze(0)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class DataSetWrapper(object):

    def __init__(self, batch_size, name, num_workers):
        self.name = name
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_data_loaders(self):
        data_augment = self._get_simclr_pipeline_transform()
        small_size = get_small('./dataset/Univariate_ts_tensor/' + self.name + "/train",
                               './dataset/Univariate_ts_tensor/' + self.name + "/test")

        # train_dataset = datasets.STL10('./data', split='train+unlabeled', download=True,
        # transform=SimCLRDataTransform(data_augment))

        # train_dataset = datasets.CIFAR10(root='./data/CIFAR10', train=True,
        #                                  download=True, transform=SimCLRTransform(data_augment))
        train_dataset = CustomDataset('./dataset/Univariate_ts_tensor/' + self.name + "/train",
                                      './dataset/Univariate_ts_tensor/' + self.name + "/test",
                                      small_size=small_size,
                                      transform=SimCLRTransform(data_augment))

        # if self.name == 'FDA':
        #     train_dataset = CustomDataset('./dataset/pic_FDA_train_tensor', transform=SimCLRTransform(data_augment))
        # elif self.name == 'Sleep':
        #     train_dataset = CustomDataset('./data/pic_Sleep_train_tensor_n', transform=SimCLRTransform(data_augment))
        # elif self.name == 'FDB':
        #     train_dataset = CustomDataset('./data/pic_FDB_train_TF-C', transform=SimCLRTransform(data_augment))
        # elif self.name == 'Epilepsy':
        #     train_dataset = CustomDataset('./data/pic_Epilepsy_train_TF-C', transform=SimCLRTransform(data_augment))
        # else:
        #     train_dataset = datasets.CIFAR10(root='./data/CIFAR10', train=True,
        #                                      download=True, transform=SimCLRTransform(data_augment))

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                  drop_last=False)
        return train_loader
        # train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        # return train_loader, valid_loader

    def get_data_loaders_fine_tune(self):
        data_augment = self._get_simclr_pipeline_transform()

        # train_dataset = datasets.STL10('./data', split='train+unlabeled', download=True,
        # transform=SimCLRDataTransform(data_augment))

        # train_dataset = datasets.CIFAR10(root='./data/CIFAR10', train=True,
        #                                  download=True, transform=SimCLRTransform(data_augment))
        if self.name == 'FDA':
            train_dataset = CustomDataset('./dataset/pic_FDA_train_tensor', transform=SimCLRTransform(data_augment))
        elif self.name == 'Sleep':
            train_dataset = CustomDataset('./data/pic_Sleep_train_tensor_n', transform=SimCLRTransform(data_augment))
        elif self.name == 'FDB':
            train_dataset = CustomDataset('./data/pic_FDB_train_TF-C', transform=SimCLRTransform(data_augment))
        elif self.name == 'Epilepsy':
            train_dataset = CustomDataset('./data/pic_Epilepsy_train_TF-C', transform=SimCLRTransform(data_augment))
        else:
            train_dataset = datasets.CIFAR10(root='./data/CIFAR10', train=True,
                                             download=True, transform=SimCLRTransform(data_augment))

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                  drop_last=False)
        return train_loader
        # train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        # return train_loader, valid_loader

    def _get_simclr_pipeline_transform(self):
        # # get a set of data augmentation transformations as described in the SimCLR paper.
        # color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        # data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=self.input_shape),
        #                                       transforms.RandomHorizontalFlip(p=0.1),
        #                                       transforms.RandomApply([color_jitter], p=0.5),
        #                                       # transforms.RandomVerticalFlip(p=0.1),
        #                                       # transforms.RandomGrayscale(p=0.2),
        #                                       # GaussianBlur(kernel_size=int(0.1 * self.input_shape[0])),
        #                                       transforms.ToTensor()])
        return TensorTransforms()

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)
        return train_loader, valid_loader


class SimCLRTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj
