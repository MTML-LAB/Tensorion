import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import numpy as np
import random

from omegaconf import OmegaConf, DictConfig
import re
from typing import Union, Optional, Literal

from src.datasets.hf_imagenet import _load_imaginet


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


def get_datasets(
    dataset_name: Literal["CIFAR10", "CIFAR100"],
    download: bool = True,
    data_path: Optional[str] = None,
) -> tuple[Dataset, Dataset]:
    dataset_loader = {
        "CIFAR10": _load_CIFAR10,
        "CIFAR100": _load_CIFAR100,
        "Imaginet": _load_imaginet,
        "FOOD101": _load_food101,
    }

    return dataset_loader[dataset_name](download, data_path)


def _set_worker_seed(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(
    train_ds: Dataset,
    test_ds: Dataset,
    train_bs: int,
    test_bs: int,
    num_workers: int = 0,
    ddp_sampler: bool = False,
    size: Optional[int] = None,
    rank: Optional[int] = None,
    g: Optional[torch.Generator] = None,
):
    train_sampler, test_sampler = None, None
    if ddp_sampler:
        from torch.utils.data.distributed import DistributedSampler

        train_sampler = DistributedSampler(
            train_ds, size, rank=rank, shuffle=True, drop_last=True
        )
        test_sampler = DistributedSampler(
            test_ds, size, rank=rank, shuffle=False, drop_last=False
        )

    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_ds,
        sampler=train_sampler,
        batch_size=train_bs,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        worker_init_fn=_set_worker_seed,
        generator=g,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        sampler=test_sampler,
        batch_size=test_bs,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def load_from_config(
    cfg: DictConfig, random_gen: torch.Generator
) -> tuple[Dataset, Dataset, DataLoader, DataLoader]:
    """
    :param cfg: Omegaconf config
    :return: train dataset, test dataset, train dataloader, test dataloader
    """

    data_cfg = cfg.dataset

    datasets_kwargs = {
        "dataset_name": data_cfg.dataset_name,
        "download": data_cfg.download,
        "data_path": data_cfg.data_path,
    }

    dataloaders_kwargs = {
        "train_bs": cfg.loader_params.train_bs,
        "test_bs": cfg.loader_params.test_bs,
        "num_workers": cfg.loader_params.num_workers,
    }

    train_ds, test_ds = get_datasets(**datasets_kwargs)

    train_loader, test_loader = get_dataloader(
        train_ds=train_ds,
        test_ds=test_ds,
        persistent_workers=True,
        ddp_sampler=False,
        g=random_gen,
        **dataloaders_kwargs
    )

    return train_ds, test_ds, train_loader, test_loader
