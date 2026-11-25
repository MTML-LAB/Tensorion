import math

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tensorion_experiments import train as train_module


def test_test_returns_accuracy_and_loss():
    inputs = torch.randn(4, 5)
    labels = torch.zeros(4, dtype=torch.long)
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=2, shuffle=False)

    model = nn.Linear(5, 2)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()

    accuracy, loss = train_module.test(
        model=model,
        test_loader=loader,
        loss_fn=nn.CrossEntropyLoss(),
        device="cpu",
    )

    assert accuracy == 1.0
    assert math.isfinite(loss)
