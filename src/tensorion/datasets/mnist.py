from typing import Optional

from torch.utils.data import Dataset, Subset

from omegaconf import DictConfig

from ..data_transforms import build_split_transform


__all__ = ["_load_mnist"]


def _apply_subset(dataset: Dataset, subset_size: Optional[int]) -> Dataset:
    if subset_size is None:
        return dataset

    subset_size = int(subset_size)
    if subset_size < 1:
        raise ValueError("subset_size must be >= 1")

    subset_size = min(subset_size, len(dataset))
    return Subset(dataset, range(subset_size))


def _load_mnist(
    download: bool = True,
    data_path: Optional[str] = None,
    transforms: Optional[DictConfig] = None,
    train_subset: Optional[int] = None,
    test_subset: Optional[int] = None,
    **_unused,
) -> tuple[Dataset, Dataset]:
    from torchvision.datasets import MNIST

    data_path = "./data" if data_path is None else data_path

    transform_train = build_split_transform(transforms, "train")
    transform_test = build_split_transform(transforms, "test")

    train_data = MNIST(
        root=data_path,
        train=True,
        download=download,
        transform=transform_train,
    )

    test_data = MNIST(
        root=data_path,
        train=False,
        download=download,
        transform=transform_test,
    )

    train_data = _apply_subset(train_data, train_subset)
    test_data = _apply_subset(test_data, test_subset)

    return train_data, test_data
