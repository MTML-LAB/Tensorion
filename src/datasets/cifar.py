import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from omegaconf import OmegaConf, DictConfig
from typing import Union, Optional, Literal


def _load_CIFAR10(
    download: bool = True, data_path: Optional[str] = None
) -> tuple[Dataset, Dataset]:
    from torchvision.datasets import CIFAR10

    data_path = "./data" if data_path is None else data_path
    normalize_mean = (0.4914, 0.4822, 0.4465)
    normalize_std = (0.2023, 0.1994, 0.2010)

    transform_train = transforms.Compose(
        [
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize(32),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
        ]
    )

    train_data = CIFAR10(
        root=data_path,
        train=True,
        download=download,
        transform=transform_train,
    )

    test_data = CIFAR10(
        root=data_path,
        train=False,
        download=download,
        transform=transform_test,
    )

    return train_data, test_data


def _load_CIFAR100(download: bool = True, data_path: Optional[str] = None) -> tuple[Dataset, Dataset]:
    from torchvision.datasets import CIFAR100

    data_path = './data' if data_path is None else data_path

    # mean and std from
    # https://github.com/weiaicunzai/pytorch-cifar100/blob/master/conf/global_settings.py
    normalize_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    normalize_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    transform_train = transforms.Compose(
        [
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
        ]
    )

    transform_test = transforms.Compose(
        [
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
        ]
    )

    train_data = CIFAR100(
        root=data_path,
        train=True,
        download=download,
        transform=transform_train,
    )

    test_data = CIFAR100(
        root=data_path,
        train=False,
        download=download,
        transform=transform_test,
    )

    return train_data, test_data
