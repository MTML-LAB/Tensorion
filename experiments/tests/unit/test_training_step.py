import math

import torch
from torch import nn
from torch.utils.data import DataLoader

from tensorion_experiments import train as train_module
from tests.helpers import DummyImageDataset


def test_train_epoch_updates_params_and_scheduler():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
    dataset = DummyImageDataset(length=8, num_classes=10)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    initial_weight = model[1].weight.detach().clone()
    initial_lr = scheduler.get_last_lr()[0]

    loss, num_samples, _elapsed = train_module.train_epoch(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=loader,
        loss_fn=criterion,
        device="cpu",
        scheduler_step="step",
    )

    assert math.isfinite(loss)
    assert num_samples == len(dataset)
    assert not torch.allclose(initial_weight, model[1].weight.detach())
    assert model[1].weight.grad is not None
    assert scheduler.get_last_lr()[0] != initial_lr


def test_format_lrs_multiple_groups():
    formatted = train_module._format_lrs([0.1, 0.01])

    assert formatted == {"lr/group_0": 0.1, "lr/group_1": 0.01}
