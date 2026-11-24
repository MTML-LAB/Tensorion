"""Train a small CNN on random data with Tensorion. Runs on CPU in a few seconds.

    python examples/quickstart.py
"""

import torch
from torch import nn

from tensorion import Tensorion

torch.manual_seed(0)
model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 32, 3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(32, 10),
)

# Conv kernels and the linear layer get the Tensorion update, biases get AdamW.
optimizer = Tensorion(model.parameters(), lr=1e-2, unfolding="maxrank")

x, y = torch.randn(256, 3, 16, 16), torch.randint(0, 10, (256,))
for step in range(50):
    optimizer.zero_grad()
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()
    if step % 10 == 0 or step == 49:
        print(f"step {step:2d}  loss {loss.item():.3f}")
