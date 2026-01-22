import math

import torch
from torch import nn
from torch.utils.data import DataLoader

from tensorion import experiment_logger
from tensorion import train as train_module
from tests.helpers import DummyImageDataset


class CaptureLogger(experiment_logger.ExperimentLogger):
    enabled = True

    def __init__(self) -> None:
        self.metrics = []

    def log_metrics(self, metrics, step=None, epoch=None) -> None:
        self.metrics.append((step, epoch, metrics))


def test_train_loop_logs_metrics_and_returns_losses():
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
    train_ds = DummyImageDataset(length=8, num_classes=10)
    test_ds = DummyImageDataset(length=4, num_classes=10)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    logger = CaptureLogger()

    train_losses, test_losses, accs, lrs = train_module.train_loop(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device="cpu",
        num_epochs=1,
        scheduler_step="epoch",
        save_steps=-1,
        save_path=None,
        experiment_logger=logger,
    )

    assert len(train_losses) == 1
    assert len(test_losses) == 1
    assert len(accs) == 1
    assert len(lrs) == 1
    assert math.isfinite(train_losses[0])
    assert math.isfinite(test_losses[0])
    assert logger.metrics[0][2].get("test/accuracy") is not None
    assert logger.metrics[1][2].get("train/loss") is not None
