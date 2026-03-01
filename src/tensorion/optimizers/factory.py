"""Factory functions for creating optimizers with proper parameter splitting."""

import torch
from torch.optim import Optimizer
from typing import Any

from tensorion.optimizers import tensorion, tensorion_sgd
from tensorion.optimizers.tensorion_st_hosvd import STHoSVDTensorion
from tensorion.optimizers.tensorion_st_hosvd_sgd import STHoSVDTensorionSGD


def create_tensorion_optimizer(
    params,
    lr: float = 1e-3,
    wd: float = 5e-3,
    momentum: float = 0.95,
    nesterov: bool = False,
    min_dim: int = 3,
    **kwargs
) -> Optimizer:
    """Create a Tensorion optimizer with automatic parameter splitting."""
    muon_params = []
    adamw_params = []

    for param in params:
        if param.ndim >= min_dim:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    return tensorion.Tensorion(
        params,
        muon_params=muon_params,
        adamw_params=adamw_params,
        lr=lr,
        wd=wd,
        momentum=momentum,
        nesterov=nesterov,
        **kwargs
    )


def create_tensorion_sgd_optimizer(
    params,
    lr: float = 1e-3,
    wd: float = 5e-3,
    momentum: float = 0.95,
    nesterov: bool = False,
    min_dim: int = 3,
    **kwargs
) -> Optimizer:
    """Create a TensorionSGD optimizer with automatic parameter splitting."""
    muon_params = []
    sgd_params = []

    for param in params:
        if param.ndim >= min_dim:
            muon_params.append(param)
        else:
            sgd_params.append(param)

    return tensorion_sgd.TensorionSGD(
        params,
        muon_params=muon_params,
        sgd_params=sgd_params,
        lr=lr,
        wd=wd,
        momentum=momentum,
        nesterov=nesterov,
        **kwargs
    )


def create_st_hosvd_tensorion_optimizer(
    params,
    lr: float = 1e-3,
    wd: float = 5e-3,
    momentum: float = 0.95,
    nesterov: bool = False,
    min_dim: int = 3,
    ranks: int = 16,
    nuclear: bool = False,
    adjust_lr: bool = True,
    **kwargs
) -> Optimizer:
    """Create a STHoSVDTensorion optimizer with automatic parameter splitting."""
    muon_params = []
    adamw_params = []

    for param in params:
        if param.ndim >= min_dim:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    return STHoSVDTensorion(
        params,
        muon_params=muon_params,
        adamw_params=adamw_params,
        lr=lr,
        wd=wd,
        momentum=momentum,
        nesterov=nesterov,
        ranks=ranks,
        nuclear=nuclear,
        adjust_lr=adjust_lr,
        **kwargs
    )


def create_st_hosvd_tensorion_sgd_optimizer(
    params,
    lr: float = 1e-3,
    wd: float = 5e-3,
    momentum: float = 0.95,
    nesterov: bool = False,
    min_dim: int = 3,
    ranks: int = 16,
    nuclear: bool = False,
    adjust_lr: bool = True,
    **kwargs
) -> Optimizer:
    """Create a STHoSVDTensorionSGD optimizer with automatic parameter splitting."""
    muon_params = []
    sgd_params = []

    for param in params:
        if param.ndim >= min_dim:
            muon_params.append(param)
        else:
            sgd_params.append(param)

    return STHoSVDTensorionSGD(
        params,
        muon_params=muon_params,
        sgd_params=sgd_params,
        lr=lr,
        wd=wd,
        momentum=momentum,
        nesterov=nesterov,
        ranks=ranks,
        nuclear=nuclear,
        adjust_lr=adjust_lr,
        **kwargs
    )
