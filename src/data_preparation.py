import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import numpy as np
import random

from omegaconf import OmegaConf
import re
from typing import Union, Optional, Literal


def _load_CIFAR10(download: bool = True) -> tuple[Dataset, Dataset]:
    from torchvision.datasets import CIFAR10

    transform_train = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = CIFAR10(
        root='./data',
        train=True,
        download=download,
        transform=transform_train,
    )

    test_data = CIFAR10(
        root='./data',
        train=False,
        download=download,
        transform=transform_test,
    )

    return train_data, test_data


def _load_CIFAR100(download: bool = True) -> tuple[Dataset, Dataset]:
    raise NotImplementedError


def get_datasets(dataset_name: Literal['CIFAR10', 'CIFAR100'], download: bool = True) -> tuple[Dataset, Dataset]:
    dataset_loader = {
        'CIFAR10': _load_CIFAR10,
        'CIFAR100': _load_CIFAR100
    }

    return dataset_loader[dataset_name](download)


def _set_worker_seed(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(
        train_ds: Dataset, test_ds: Dataset, 
        train_bs: int, test_bs: int, 
        num_workers: int = 0, persistent_workers: bool = False, 
        ddp_sampler: bool = False, size: Optional[int] = None, rank: Optional[int] = None,
        g: Optional[torch.Generator] = None, 
        ):
    
    train_sampler, test_sampler = None, None
    if ddp_sampler:
        from torch.utils.data.distributed import DistributedSampler

        train_sampler = DistributedSampler(train_ds, size, rank=rank, shuffle=True, drop_last=True)
        test_sampler = DistributedSampler(test_ds, size, rank=rank, shuffle=False, drop_last=False)

    train_loader = DataLoader(train_ds,
                        sampler=train_sampler,
                        batch_size=train_bs,
                        num_workers=num_workers,
                        persistent_workers=persistent_workers,
                        worker_init_fn=_set_worker_seed,
                        generator=g,
                        pin_memory=True)
    
    test_loader = DataLoader(test_ds,
                        sampler=test_sampler,
                        batch_size=test_bs,
                        num_workers=num_workers,  
                        persistent_workers=persistent_workers,
                        pin_memory=True)

    return train_loader, test_loader
