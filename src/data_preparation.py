import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import numpy as np
import random

from omegaconf import OmegaConf, DictConfig
import functools
from typing import Union, Optional, Literal

from src.datasets.hf_imagenet import _load_imagenet
from src.datasets.cifar import _load_CIFAR10, _load_CIFAR100
from src.datasets.food101 import _load_food101


def get_datasets(
    dataset_name: Literal[
        "CIFAR10",
        "CIFAR100",
        "ImageNet",
        "FOOD101",
    ],
    download: bool = True,
    data_path: Optional[str] = None,
    hf_name: Optional[str] = None,
) -> tuple[Dataset, Dataset]:

    dataset_loader = {
        "CIFAR10": _load_CIFAR10,
        "CIFAR100": _load_CIFAR100,
        "FOOD101": _load_food101,
        "ImageNet": functools.partial(_load_imagenet, hf_name=hf_name)
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
    hf_name = data_cfg.get("hf_name")
    if hf_name is not None:
        datasets_kwargs["hf_name"] = hf_name

    dataloaders_kwargs = {
        "train_bs": cfg.loader_params.train_bs,
        "test_bs": cfg.loader_params.test_bs,
        "num_workers": cfg.loader_params.num_workers,
    }

    train_ds, test_ds = get_datasets(**datasets_kwargs)

    train_loader, test_loader = get_dataloader(
        train_ds=train_ds,
        test_ds=test_ds,
        ddp_sampler=False,
        g=random_gen,
        **dataloaders_kwargs
    )

    return train_ds, test_ds, train_loader, test_loader
