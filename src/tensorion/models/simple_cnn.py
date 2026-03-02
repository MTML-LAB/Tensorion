from collections.abc import Sequence
from typing import Optional

import torch
from hydra.utils import instantiate
from torch import nn

__all__ = ["SimpleCNN", "_load_simple_cnn"]


class SimpleCNN(nn.Module):
    def __init__(self, layers: Sequence[nn.Module]) -> None:
        super().__init__()
        if not layers:
            raise ValueError("simple_cnn requires at least one layer")
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load_simple_cnn(
    layers: Optional[Sequence] = None,
    in_channels: Optional[int] = None,
    num_classes: Optional[int] = None,
    **_unused,
) -> nn.Module:
    if in_channels is None:
        raise ValueError("simple_cnn requires 'in_channels'")
    if num_classes is None:
        raise ValueError("simple_cnn requires 'num_classes'")

    if layers is None:
        raise ValueError("simple_cnn requires 'layers'")

    layer_modules = instantiate(layers)
    if not isinstance(layer_modules, Sequence) or not layer_modules:
        raise ValueError("simple_cnn layers must be a non-empty sequence")
    for idx, layer in enumerate(layer_modules):
        if not isinstance(layer, nn.Module):
            raise ValueError(
                f"simple_cnn.layers[{idx}] did not instantiate to nn.Module"
            )

    return SimpleCNN(layer_modules)
