from itertools import product
from typing import Optional, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from omegaconf import DictConfig

from ..data_transforms import build_split_transform


__all__ = ["_load_barcoded_mnist"]

CORNERS_10 = (
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (0, 6),
    (0, 7),
    (0, 8),
    (0, 9),
)


def _get_rng(seed: Optional[int]) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(seed))


def _load_split_mnist(
    data_path: str,
    download: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from torchvision.datasets import MNIST

    train_set = MNIST(root=data_path, train=True, download=download)

    mnist_x = train_set.data.cpu().numpy()
    mnist_y = train_set.targets.cpu().numpy()
    permutation = rng.permutation(len(mnist_x))

    mnist_x = mnist_x[permutation]
    mnist_y = mnist_y[permutation]

    tr_va_split = 50_000

    mnist_x_tr = mnist_x[:tr_va_split]
    mnist_y_tr = mnist_y[:tr_va_split]
    mnist_x_va = mnist_x[tr_va_split:]
    mnist_y_va = mnist_y[tr_va_split:]

    return mnist_x_tr, mnist_y_tr, mnist_x_va, mnist_y_va


def make_barcoded_mnist(
    corners: Sequence[tuple[int, int]] = CORNERS_10,
    samples_per_class_tr: int = 5,
    samples_per_class_val: int = 1,
    with_mnist: bool = False,
    normalize: bool = True,
    data_path: str = "./data",
    download: bool = True,
    seed: Optional[int] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Creates a barcoded variant of the MNIST dataset.

    Creates 10 * 2^len(corners) classes with samples_per_class samples.

    With default parameters, 10 corners, gives
        10 * 2^10 = 10240 classes
        5 * 10240 = 51200 training samples
        1 * 10240 = 10240 validation samples

    Uses a 50k/10k split for training and validation.
    """
    rng = _get_rng(seed)
    mnist_x_tr, mnist_y_tr, mnist_x_va, mnist_y_va = _load_split_mnist(
        data_path=data_path,
        download=download,
        rng=rng,
    )
    digits = np.unique(mnist_y_tr)
    max_value = mnist_x_tr.max()
    barcodes = product([0, 1], repeat=len(corners))

    assert len(np.unique(mnist_y_tr)) == 10
    assert len(np.unique(mnist_y_va)) == 10

    def apply_barcode(barcode, x):
        img = np.copy(x)
        for binary, corner in zip(barcode, corners):
            if binary == 1:
                img[corner[0]][corner[1]] = max_value
        return img

    def make_new_random_image(candidates, barcode):
        i = rng.integers(len(candidates))
        return apply_barcode(barcode, candidates[i])

    xs_tr = []
    ys_tr = []
    xs_va = []
    ys_va = []
    class_idx = 0
    for barcode in barcodes:
        for digit in digits:
            digit_samples_tr = mnist_x_tr[mnist_y_tr == digit]
            digit_samples_va = mnist_x_va[mnist_y_va == digit]
            for j in range(samples_per_class_tr):
                xs_tr.append(make_new_random_image(digit_samples_tr, barcode))
                ys_tr.append(class_idx)
            for j in range(samples_per_class_val):
                xs_va.append(make_new_random_image(digit_samples_va, barcode))
                ys_va.append(class_idx)
            class_idx += 1

    X_tr: Tensor = torch.from_numpy(np.stack(xs_tr)).to(torch.float32)
    y_tr: Tensor = torch.from_numpy(np.array(ys_tr)).to(torch.long)
    X_va: Tensor = torch.from_numpy(np.stack(xs_va)).to(torch.float32)
    y_va: Tensor = torch.from_numpy(np.array(ys_va)).to(torch.long)

    if with_mnist:
        X_tr = torch.cat([torch.from_numpy(mnist_x_tr).to(torch.float32), X_tr], dim=0)
        y_tr = torch.cat(
            [torch.from_numpy(mnist_y_tr).to(torch.long), y_tr + 10], dim=0
        )
        X_va = torch.cat([torch.from_numpy(mnist_x_va).to(torch.float32), X_va], dim=0)
        y_va = torch.cat(
            [torch.from_numpy(mnist_y_va).to(torch.long), y_va + 10], dim=0
        )

    if normalize:

        def safe_normalize(to_normalize, means, stds):
            to_normalize[:, stds > 0] = to_normalize[:, stds > 0] - means[stds > 0]
            to_normalize[:, stds > 0] = to_normalize[:, stds > 0] / stds[stds > 0]
            return to_normalize

        means = X_tr.mean(dim=0)
        stds = X_tr.std(dim=0)
        X_tr = safe_normalize(X_tr, means, stds)
        X_va = safe_normalize(X_va, means, stds)

    X_tr = X_tr.unsqueeze(1)
    X_va = X_va.unsqueeze(1)

    return X_tr, y_tr, X_va, y_va


class _BarcodedTensorDataset(Dataset):
    def __init__(self, images: Tensor, labels: Tensor, transform=None):
        if images.ndim != 4:
            raise ValueError("images must have shape (N, C, H, W)")
        if images.shape[0] != labels.shape[0]:
            raise ValueError("images and labels must have the same length")
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


def _load_barcoded_mnist(
    download: bool = True,
    data_path: Optional[str] = None,
    transforms: Optional[DictConfig] = None,
    corners: Sequence[tuple[int, int]] = CORNERS_10,
    samples_per_class_tr: int = 5,
    samples_per_class_val: int = 1,
    with_mnist: bool = False,
    normalize: bool = True,
    seed: Optional[int] = None,
    **_unused,
) -> tuple[Dataset, Dataset]:
    data_path = "./data" if data_path is None else data_path

    transform_train = build_split_transform(transforms, "train")
    transform_test = build_split_transform(transforms, "test")

    X_tr, y_tr, X_va, y_va = make_barcoded_mnist(
        corners=corners,
        samples_per_class_tr=samples_per_class_tr,
        samples_per_class_val=samples_per_class_val,
        with_mnist=with_mnist,
        normalize=normalize,
        data_path=data_path,
        download=download,
        seed=seed,
    )

    train_data = _BarcodedTensorDataset(X_tr, y_tr, transform=transform_train)
    test_data = _BarcodedTensorDataset(X_va, y_va, transform=transform_test)

    return train_data, test_data
