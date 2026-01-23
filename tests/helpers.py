from typing import Sequence, Tuple

import torch
from torch.utils.data import Dataset


class DummyImageDataset(Dataset):
    def __init__(
        self,
        length: int = 16,
        image_shape: Sequence[int] = (3, 32, 32),
        num_classes: int = 10,
    ) -> None:
        self.length = int(length)
        self.image_shape = tuple(int(x) for x in image_shape)
        self.num_classes = int(num_classes)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        image = torch.full(self.image_shape, float(idx), dtype=torch.float32)
        label = idx % self.num_classes
        return image, label


def build_dummy_datasets(
    train_size: int = 16,
    test_size: int = 8,
    image_shape: Sequence[int] = (3, 32, 32),
    num_classes: int = 10,
) -> Tuple[Dataset, Dataset]:
    train_ds = DummyImageDataset(
        length=train_size, image_shape=image_shape, num_classes=num_classes
    )
    test_ds = DummyImageDataset(
        length=test_size, image_shape=image_shape, num_classes=num_classes
    )
    return train_ds, test_ds
