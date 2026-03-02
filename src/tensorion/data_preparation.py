import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


def _set_worker_seed(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _require_cfg(cfg: DictConfig, key: str, context: str):
    """Require a config key, safely handling potential Hydra interpolation errors.

    First tries to get the value using OmegaConf.select(). If that fails due to
    unresolved interpolations, tries to get the raw value using _get_node().
    """
    if OmegaConf.is_missing(cfg, key):
        raise ValueError(f"Missing required config key '{key}' in {context}")

    value = None
    try:
        # Try to resolve the value (works if Hydra is running and interpolations are resolved)
        value = OmegaConf.select(cfg, key)
    except Exception:
        # If resolution fails (e.g., due to unresolved Hydra interpolations),
        # try to get the raw value using _get_node()
        try:
            if hasattr(cfg, "_get_node"):
                node = cfg._get_node(key)
                if node is not None:
                    value = node._value() if hasattr(node, "_value") else node
        except Exception:
            pass

    if value is None:
        raise ValueError(f"Missing required config key '{key}' in {context}")

    return value


def _build_generator(cfg: DictConfig) -> Optional[torch.Generator]:
    seed = cfg.get("seed")
    if seed is None:
        return None

    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def get_datasets(cfg: DictConfig) -> dict[str, Dataset]:
    dataset_cfg = _require_cfg(cfg, "dataset", "root config")
    datasets = instantiate(dataset_cfg)

    if isinstance(datasets, tuple):
        if len(datasets) != 2:
            raise ValueError(
                "Dataset loader must return (train, test) or a dict of splits"
            )
        datasets = {"train": datasets[0], "test": datasets[1]}

    if not isinstance(datasets, dict):
        raise ValueError("Dataset loader must return a dict of split datasets")

    for split, dataset in datasets.items():
        if dataset is None or not isinstance(dataset, Dataset):
            raise ValueError(f"Dataset split '{split}' is missing or invalid")

    return datasets


def get_dataloader(
    cfg: DictConfig, dataset: Dataset, split: str = "train"
) -> DataLoader:
    if dataset is None:
        raise ValueError(f"Dataset for split '{split}' is required")

    dataloader_cfg = _require_cfg(cfg, "dataloader", "root config")
    if split not in dataloader_cfg:
        raise ValueError(f"Missing dataloader config for split '{split}'")

    split_cfg = dataloader_cfg[split]
    if "_target_" not in split_cfg:
        raise ValueError(f"Dataloader config for split '{split}' must define _target_")

    num_workers = split_cfg.get("num_workers", 0)
    persistent_workers = num_workers > 0

    sampler_cfg = split_cfg.get("sampler")
    sampler = instantiate(sampler_cfg, dataset=dataset) if sampler_cfg else None

    extra_kwargs = {
        "dataset": dataset,
        "num_workers": num_workers,
        "persistent_workers": persistent_workers,
        "pin_memory": split_cfg.get("pin_memory", False),
    }

    if sampler is not None:
        extra_kwargs["sampler"] = sampler
        if "shuffle" in split_cfg:
            extra_kwargs["shuffle"] = False

    if split == "train":
        extra_kwargs.setdefault("worker_init_fn", _set_worker_seed)
        generator = _build_generator(cfg)
        if generator is not None:
            extra_kwargs.setdefault("generator", generator)

    return instantiate(split_cfg, **extra_kwargs)


def load_from_config(
    cfg: DictConfig,
) -> tuple[Dataset, Dataset, DataLoader, DataLoader]:
    datasets = get_datasets(cfg)
    if "train" not in datasets or "test" not in datasets:
        raise ValueError("Both train and test datasets are required")

    train_ds = datasets["train"]
    test_ds = datasets["test"]

    train_loader = get_dataloader(cfg, train_ds, split="train")
    test_loader = get_dataloader(cfg, test_ds, split="test")

    return train_ds, test_ds, train_loader, test_loader
