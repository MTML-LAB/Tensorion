import torch

from tensorion.models import resnet


def test_resnet18_forward_shape():
    model = resnet.ResNet18(num_classes=10)
    inputs = torch.randn(2, 3, 32, 32)
    outputs = model(inputs)

    assert outputs.shape == (2, 10)
