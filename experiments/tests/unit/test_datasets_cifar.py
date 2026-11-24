import pytest
import torch
from torch.utils.data import TensorDataset

from tensorion_experiments.datasets import cifar


def test_apply_subset_returns_original_for_none():
    dataset = TensorDataset(torch.zeros(5, 1))

    subset = cifar._apply_subset(dataset, None)

    assert subset is dataset


def test_apply_subset_limits_size():
    dataset = TensorDataset(torch.zeros(10, 1))

    subset = cifar._apply_subset(dataset, 4)

    assert len(subset) == 4


def test_apply_subset_rejects_non_positive_size():
    dataset = TensorDataset(torch.zeros(5, 1))

    with pytest.raises(ValueError, match="subset_size must be >= 1"):
        cifar._apply_subset(dataset, 0)
