from typing import Optional

from torch.utils.data import Dataset

from omegaconf import DictConfig

from src.data_transforms import build_split_transform


__all__ = ['_load_CIFAR10', '_load_CIFAR100']


def _load_CIFAR10(
    download: bool = True,
    data_path: Optional[str] = None,
    transforms: Optional[DictConfig] = None,
    **_unused,
) -> tuple[Dataset, Dataset]:
    from torchvision.datasets import CIFAR10

    data_path = "./data" if data_path is None else data_path
    transform_train = build_split_transform(transforms, "train")
    transform_test = build_split_transform(transforms, "test")

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


def _load_CIFAR100(
    download: bool = True,
    data_path: Optional[str] = None,
    transforms: Optional[DictConfig] = None,
    **_unused,
) -> tuple[Dataset, Dataset]:
    from torchvision.datasets import CIFAR100

    data_path = "./data" if data_path is None else data_path

    transform_train = build_split_transform(transforms, "train")
    transform_test = build_split_transform(transforms, "test")

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
