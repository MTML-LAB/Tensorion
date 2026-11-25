import pytest

from tensorion_experiments import data_transforms


class DummyImage:
    def __init__(self) -> None:
        self.called_with = None

    def convert(self, mode: str):
        self.called_with = mode
        return f"converted-{mode}"


def test_convert_to_rgb_calls_convert():
    image = DummyImage()
    converter = data_transforms.ConvertToRGB()

    result = converter(image)

    assert image.called_with == "RGB"
    assert result == "converted-RGB"


def test_build_split_transform_returns_config_entry():
    transforms_cfg = {"train": object(), "test": object()}

    assert (
        data_transforms.build_split_transform(transforms_cfg, "train")
        is transforms_cfg["train"]
    )


def test_build_split_transform_missing_split_raises():
    transforms_cfg = {"train": object()}

    with pytest.raises(ValueError, match="Missing transforms config for split 'test'"):
        data_transforms.build_split_transform(transforms_cfg, "test")
