from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from torch import nn

from tensorion_experiments.models.resnet import ResNet18
from tensorion_experiments.run_experiment import build_optimizer

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def _optimizer_cfg(*overrides):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIGS)):
        return compose(config_name="config.yaml", overrides=list(overrides)).optimizer


def _step(model, optimizer):
    model(torch.randn(2, 3, 32, 32)).sum().backward()
    optimizer.step()


def test_paper_routing_puts_every_conv_kernel_on_tensorion():
    model = ResNet18(num_classes=10)
    optimizer = build_optimizer(_optimizer_cfg(), model)
    _step(model, optimizer)

    conv_weights = {id(m.weight) for m in model.modules() if isinstance(m, nn.Conv2d)}
    assert any(m.kernel_size == (1, 1) for m in model.modules() if isinstance(m, nn.Conv2d))
    for name, p in model.named_parameters():
        on_tensorion = "momentum_buffer" in optimizer.state[p]
        assert on_tensorion == (id(p) in conv_weights), name


def test_without_tensorion_modules_the_order_decides():
    model = ResNet18(num_classes=10)
    optimizer = build_optimizer(_optimizer_cfg("~optimizer.tensorion_modules", "+optimizer.min_order=3"), model)
    _step(model, optimizer)
    for name, p in model.named_parameters():
        order = sum(n > 1 for n in p.shape)
        assert ("momentum_buffer" in optimizer.state[p]) == (order >= 3), name


def test_per_layer_unfolding_still_works_with_routing():
    model = ResNet18(num_classes=10)
    optimizer = build_optimizer(_optimizer_cfg("optimizer.unfolding={conv1.weight:[0,1]}"), model)
    _step(model, optimizer)


def test_unknown_modules_are_reported():
    with pytest.raises(ValueError, match="match no module"):
        build_optimizer(_optimizer_cfg("optimizer.tensorion_modules=[Conv9d]"), ResNet18(num_classes=10))


@pytest.mark.parametrize("name", ["adamw", "sgd", "shampoo"])
def test_other_optimizers_are_built_on_named_parameters(name):
    model = ResNet18(num_classes=10)
    _step(model, build_optimizer(_optimizer_cfg(f"optimizer={name}"), model))
