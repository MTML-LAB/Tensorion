import pytest
import torch
from omegaconf import OmegaConf

from tensorion_experiments import data_preparation
from tests.helpers import DummyImageDataset


def test_load_from_config_builds_train_and_test_loaders():
    cfg = OmegaConf.create(
        {
            "dataset": {
                "_target_": "tests.helpers.build_dummy_datasets",
                "train_size": 10,
                "test_size": 6,
                "image_shape": [3, 32, 32],
                "num_classes": 10,
            },
            "dataloader": {
                "train": {
                    "_target_": "torch.utils.data.DataLoader",
                    "batch_size": 4,
                    "shuffle": True,
                    "num_workers": 0,
                },
                "test": {
                    "_target_": "torch.utils.data.DataLoader",
                    "batch_size": 3,
                    "shuffle": False,
                    "num_workers": 0,
                },
            },
            "seed": 123,
        }
    )

    train_ds, test_ds, train_loader, test_loader = data_preparation.load_from_config(cfg)

    assert len(train_ds) == 10
    assert len(test_ds) == 6
    assert train_loader.batch_size == 4
    assert test_loader.batch_size == 3

    images, labels = next(iter(train_loader))
    assert images.shape == (4, 3, 32, 32)
    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64


def test_train_dataloader_is_deterministic_with_seed():
    dataset = DummyImageDataset(length=12, num_classes=100)
    cfg = OmegaConf.create(
        {
            "seed": 7,
            "dataloader": {
                "train": {
                    "_target_": "torch.utils.data.DataLoader",
                    "batch_size": 3,
                    "shuffle": True,
                    "num_workers": 0,
                }
            },
        }
    )

    loader_a = data_preparation.get_dataloader(cfg, dataset, split="train")
    loader_b = data_preparation.get_dataloader(cfg, dataset, split="train")

    order_a = [int(label) for _, labels in loader_a for label in labels]
    order_b = [int(label) for _, labels in loader_b for label in labels]

    assert order_a == order_b


def test_get_dataloader_requires_dataset():
    cfg = OmegaConf.create(
        {
            "dataloader": {
                "train": {
                    "_target_": "torch.utils.data.DataLoader",
                    "batch_size": 2,
                    "shuffle": False,
                    "num_workers": 0,
                }
            }
        }
    )

    with pytest.raises(ValueError, match="Dataset for split 'train' is required"):
        data_preparation.get_dataloader(cfg, None, split="train")


def test_get_dataloader_missing_split_config_raises():
    cfg = OmegaConf.create(
        {
            "dataloader": {
                "train": {
                    "_target_": "torch.utils.data.DataLoader",
                    "batch_size": 2,
                    "shuffle": False,
                    "num_workers": 0,
                }
            }
        }
    )
    dataset = DummyImageDataset(length=2, num_classes=2)

    with pytest.raises(ValueError, match="Missing dataloader config for split 'test'"):
        data_preparation.get_dataloader(cfg, dataset, split="test")


def test_get_datasets_requires_dataset_config():
    cfg = OmegaConf.create({})

    with pytest.raises(ValueError, match="Missing required config key 'dataset'"):
        data_preparation.get_datasets(cfg)
