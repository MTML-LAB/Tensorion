from typing import Optional
from torch.utils.data import Dataset
from omegaconf import DictConfig
from tensorion.data_transforms import build_split_transform


__all__ = ["_load_food101"]


def _load_food101(
    download: bool = True,
    data_path: Optional[str] = None,
    transforms: Optional[DictConfig] = None,
    **_unused,
) -> tuple[Dataset, Dataset]:
    from torchvision.datasets import Food101

    data_path = "./data" if data_path is None else data_path

    transform_train = build_split_transform(transforms, "train")
    transform_test = build_split_transform(transforms, "test")

    train_data = Food101(
        root=data_path,
        split="train",
        download=download,
        transform=transform_train,
    )

    test_data = Food101(
        root=data_path,
        split="test",
        download=download,
        transform=transform_test,
    )

    return train_data, test_data
