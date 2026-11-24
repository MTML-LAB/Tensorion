import os
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from tensorion_experiments import run_experiment

pytestmark = pytest.mark.integration


@pytest.mark.integration
def _cifar10_available(data_dir: Path) -> bool:
    return (data_dir / "cifar-10-batches-py").exists()


@pytest.mark.integration
def test_resnet18_cifar10_training_short(tmp_path, monkeypatch):
    data_dir = Path(
        os.getenv(
            "TENSORION_DATA_DIR",
            Path(__file__).resolve().parents[2] / "data",
        )
    ).resolve()
    allow_download = os.getenv("TENSORION_TEST_DOWNLOAD", "0") == "1"

    if not _cifar10_available(data_dir) and not allow_download:
        pytest.skip(
            "CIFAR10 not found. Set TENSORION_TEST_DOWNLOAD=1 or "
            "place data under TENSORION_DATA_DIR."
        )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(
            config_name="config.yaml",
            overrides=[
                "dataset=cifar10",
                "model=resnet18",
                "num_epochs=1",
                "save_steps=1",
                "dataloader.train.batch_size=8",
                "dataloader.test.batch_size=8",
                "dataloader.train.num_workers=0",
                "dataloader.test.num_workers=0",
                "dataloader.train.pin_memory=false",
                "dataloader.test.pin_memory=false",
                "dataset.train_subset=64",
                "dataset.test_subset=64",
                "comet.enabled=false",
                f"save_dir={tmp_path.as_posix()}",
            ],
        )

    cfg.dataset.data_path = data_dir.as_posix()
    cfg.dataset.download = allow_download

    run_experiment.run(cfg)

    checkpoint_dir = tmp_path / "checkpoints" / "epoch-1"
    assert (checkpoint_dir / "model.pt").exists()
    assert (checkpoint_dir / "optimizer.pt").exists()
    assert (checkpoint_dir / "scheduler.pt").exists()

    train_loss = np.load(tmp_path / "train_loss.npy")
    accuracy = np.load(tmp_path / "accuracy.npy")
    assert train_loss.size >= 1
    assert accuracy.size >= 1
    assert np.isfinite(train_loss).all()
