from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


def test_simple_cnn_from_config():
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(
            config_name="config.yaml",
            overrides=[
                "model=simple_cnn",
                "dataset=cifar10",
                "comet.enabled=false",
            ],
        )

    model = instantiate(cfg.model)
    inputs = torch.randn(2, cfg.model.in_channels, 28, 28)
    outputs = model(inputs)

    assert outputs.shape == (2, cfg.model.num_classes)
    assert model.net[0].weight.shape == (32, cfg.model.in_channels, 5, 5)
    assert model.net[3].weight.shape == (100, 32, 5, 5)
    assert model.net[7].weight.shape == (cfg.model.num_classes, 100 * 7 * 7)
