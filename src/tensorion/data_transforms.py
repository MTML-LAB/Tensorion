from typing import Any


class ConvertToRGB:
    def __call__(self, img: Any):
        return img.convert("RGB")


def build_split_transform(transforms_cfg, split: str):
    if transforms_cfg is None or split not in transforms_cfg:
        raise ValueError(f"Missing transforms config for split '{split}'")

    return transforms_cfg[split]
