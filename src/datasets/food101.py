import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from omegaconf import OmegaConf, DictConfig
import functools
from typing import Union, Optional, Literal



def _load_food101(download: bool = True, data_path: Optional[str] = None) -> tuple[Dataset, Dataset]:
    from torchvision.datasets import Food101

    data_path = './data' if data_path is None else data_path

    # mean and std from 
    normalize_mean = [0.533, 0.442, 0.344]
    normalize_std = [0.274, 0.264, 0.267]

    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std),

    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std),

    ])

    train_data = Food101(
        root=data_path,
        split='train',
        download=download,
        transform=transform_train,
    )

    test_data = Food101(
        root=data_path,
        split='test',
        download=download,
        transform=transform_test,
    )

    return train_data, test_data
