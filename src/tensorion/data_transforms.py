<<<<<<< HEAD
from typing import Any
=======
from typing import Any, Union
from hydra.utils import instantiate
>>>>>>> origin/class_imbalance

import torch
import numpy as np
import PIL.Image


class ConvertToRGB:
    def __call__(self, img: Union[PIL.Image.Image, np.ndarray, torch.Tensor]):
        if isinstance(img, torch.Tensor):
            return img.repeat(3, 1, 1) if img.shape[0] == 1 else img
        if isinstance(img, np.ndarray):
            return np.repeat(img, 3, axis=0) if img.shape[0] == 1 else img
        
        return img.convert("RGB")


def build_split_transform(transforms_cfg, split: str):
    if transforms_cfg is None or split not in transforms_cfg:
        raise ValueError(f"Missing transforms config for split '{split}'")

    return transforms_cfg[split]
