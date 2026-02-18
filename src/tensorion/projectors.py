import math
import numpy as np
import torch
from typing import Optional, Union, Callable, Any, Literal

from .tensor_utils import fold, unfold, matricization, un_matricization
from .loggers import (
    MetricsCallback,
    MetricsCallbackFactory,
    projectors_metrics_collector_factory,
)

Tensor = torch.Tensor
BaseProjector = Callable[[Tensor, Optional[int]], Tensor]
ModeStrategy = Callable[[list[int]], int]
Projector = Callable[[Tensor], Union[Tensor, tuple[Tensor, float]]]

# <-- Base projectors -->


def unfold_by_idx(x: Tensor, idx: Union[int, list[int], None]) -> Tensor:
    """
    :param x (Tensor): Tensor with shape (n_1, ..., n_d)
    :param idx: 
        If `None`, function requires only 2D Matrix `x`, returns `x`
        If `int`, function performs unfolding by mode `idx`
        If `list[int]`, function performs matricization by modes `idx`
    """

    if idx is None:
        X = x
    elif isinstance(idx, int):
        X = unfold(x, idx)
    elif isinstance(idx, list):
        X = matricization(x, idx)
    else:
        raise ValueError(f"Unexpected {idx=}")
    
    return X


def fold_by_idx(X: Tensor, original_shape: tuple[int], idx: Union[int, list[int], None]) -> Tensor:
    """
    :param X (Tensor): Unfolding or matricization of tensor `x`
    :param original_shape: Shape of the original tensor `x`
    :param idx: 
        If `None`, function requires only 2D Matrix `x`, returns `x`
        If `int`, function performs unfolding by mode `idx`
        If `list[int]`, function performs matricization by modes `idx`
    """

    if idx is None:
        x = X
    elif isinstance(idx, int):
        x = fold(X, original_shape, idx)
    elif isinstance(idx, list):
        x = un_matricization(X, original_shape, idx)
    else:
        raise ValueError(f"Unexpected {idx=}")
    
    return x


def project_spectral_ball(x: Tensor, idx: Union[int, list[int], None]) -> Tensor:
    if (idx is None) and (x.ndim > 2):
        raise ValueError(f"idx may be None only for 2D x\n{x.shape=}\t{idx=}")
    
    X = unfold_by_idx(x, idx)
    
    U, S, Vt = torch.linalg.svd(X, full_matrices=False)
    S = torch.clip(S, min=0.0, max=1.0)
    X = U @ torch.diag(S) @ Vt

    x_proj = fold_by_idx(X, x.shape, idx)
    
    assert x_proj.shape == x.shape
    return x_proj


def project_spectral_sphere(x: Tensor, idx: Optional[int]) -> Tensor:
    if (idx is None) and (x.ndim > 2):
        raise ValueError(f"idx may be None only for 2D x\n{x.shape=}\t{idx=}")
    
    X = unfold_by_idx(x, idx)

    U, S, Vt = torch.linalg.svd(X, full_matrices=False)
    X = U @ Vt

    x_proj = fold_by_idx(X, x.shape, idx)

    assert x_proj.shape == x.shape
    return x_proj


def project_ns(x: Tensor, idx: int, num_ns_iterations: int = 2):
    X = unfold_by_idx(x, idx)

    transpose = X.shape[0] > X.shape[1]
    if transpose:
        X.transpose_(0, 1)

    # m x n, m <= n

    X.div_(torch.linalg.norm(X, ord='fro'))
    m = X.shape[0]
    I = torch.eye(m, device=X.device, dtype=X.dtype)

    for _ in range(num_ns_iterations):
        X = (3 * I - X @ X.T) @ (X / 2)

    if transpose:
        X.transpose_(0, 1)
    
    x_proj = fold_by_idx(X, x.shape, idx)

    assert x_proj.shape == x.shape
    return x_proj

# <!- Base projectors -!>

# <-- Projectors on intersection -->


def dykstra(
    x0: Tensor,
    projector: BaseProjector,
    num_sets: int,
    max_iter: int = 1000,
    tol: float = 1e-6,
    metrics_callback_factory: Optional[MetricsCallbackFactory] = None,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    """
    Dykstra's algorithm

    Return:
      - 'projection': projection on intersection (Tensor)
      - 'iterations': number of iterations (int)
      - 'converged': bool
      - 'history': dict with metrics history
    """
    if device is None:
        device = x0.device

    x0 = x0.to(device)
    x = x0.clone()
    p = [torch.zeros_like(x, device=device) for _ in range(num_sets)]

    if metrics_callback_factory is None:
        metrics_callback, history = projectors_metrics_collector_factory(
            x0=x0, metrics=["sigvals_bounds", "dot", "residual"]
        )
        history["delta_cycle"] = []
    else:
        metrics_callback, history = metrics_callback_factory(x0)
        history["delta_cycle"] = []

    global_step = 0
    for it in range(1, max_iter + 1):
        x_prev_cycle = x.clone()
        for i in range(num_sets):
            # Dykstra step
            y = x + p[i]
            z = projector(y, i)
            p[i] = y - z
            x = z

            # metrics
            if metrics_callback is not None:
                metrics_callback(x, i, global_step)
            global_step += 1

        delta = torch.linalg.norm(x - x_prev_cycle).item()
        history["delta_cycle"].append(delta)
        if delta <= tol:
            return {
                "projection": x,
                "iterations": it,
                "converged": True,
                "history": history,
            }

    return {
        "projection": x,
        "iterations": max_iter,
        "converged": False,
        "history": history,
    }


def alternating_projectors(
    x0: Tensor,
    projector: BaseProjector,
    num_sets: int,
    max_iter: int = 1000,
    tol: float = 1e-6,
    metrics_callback_factory: Optional[MetricsCallbackFactory] = None,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    """
    POCS algorithm

    Return:
      - 'projection': projection on intersection (Tensor)
      - 'iterations': number of iterations (int)
      - 'converged': bool
      - 'history': dict with metrics history
    """
    if device is None:
        device = x0.device
    x0 = x0.to(device)
    x = x0.clone()

    if metrics_callback_factory is None:
        metrics_callback, history = projectors_metrics_collector_factory(
            x0=x0, metrics=["sigvals_bounds", "dot", "residual"]
        )
        history["delta_cycle"] = []
    else:
        metrics_callback, history = metrics_callback_factory(x0)
        history["delta_cycle"] = []

    global_step = 0
    for it in range(1, max_iter + 1):
        x_prev_cycle = x.clone()
        for i in range(num_sets):
            x = projector(x, i)
            if metrics_callback is not None:
                metrics_callback(x, i, global_step)
            global_step += 1

        delta = torch.linalg.norm(x - x_prev_cycle).item()
        history["delta_cycle"].append(delta)
        if delta <= tol:
            return {
                "projection": x,
                "iterations": it,
                "converged": True,
                "history": history,
            }

    return {
        "projection": x,
        "iterations": max_iter,
        "converged": False,
        "history": history,
    }


# <-- Projectors on intersection -->

# <-- Projectors by modes -->


def single_step_projection(
    x0: Tensor,
    projector: BaseProjector,
    mode_strategy: ModeStrategy,
) -> dict[str, Any]:
    """
    Single step heuristic algorithm

    Return:
      - 'projection': projection on intersection (Tensor)
      - 'iterations': number of iterations (int), always == 1
      - 'converged': bool, always == 1
      - 'history': dict with metrics history
    """

    x_dims = list(x0.shape)
    mode = mode_strategy(x_dims)
    proj = projector(x0, mode)

    unf_shape = (x0.shape[mode], math.prod(x0.shape) // x0.shape[mode])
    irms = math.sqrt(max(unf_shape))

    return {
        "projection": proj,
        "iterations": 1,
        "converged": True,
        "history": dict(),
        "irms": irms,
    }


def dual_projection(
    x0: Tensor,
    projector: BaseProjector,
) -> dict[str, Any]:
    """
    Accurate algorithm from "Dual norm is all you need" idea

    Return:
      - 'projection': projection on intersection (Tensor)
      - 'iterations': number of iterations (int), always == 1
      - 'converged': bool, always == 1
      - 'history': dict with metrics history
    """
    
    mode = 0
    max_nuc_norm = 0.0
    for idx in range(x0.dim()):
        X = unfold(x0, idx)
        norm = torch.linalg.norm(X, ord='nuc')

        if norm > max_nuc_norm:
            mode = idx
            max_nuc_norm = norm

    proj = projector(x0, mode)

    unf_shape = (x0.shape[mode], math.prod(x0.shape) // x0.shape[mode])
    irms = math.sqrt(max(unf_shape))

    return {
        "projection": proj,
        "iterations": 1,
        "converged": True,
        "history": dict(),
        "irms": irms,
    }


def unfold_projection(
    x0: Tensor,
    projector: BaseProjector,
    mode_idxs: Union[int, list[int]]
) -> dict[str, Any]:
    """
    Orthogonalize the `mode_idxs` unfolding

    Return:
      - 'projection': projection on intersection (Tensor)
      - 'iterations': number of iterations (int), always == 1
      - 'converged': bool, always == 1
      - 'history': dict with metrics history
    """
    
    X_unf = unfold_by_idx(x0, mode_idxs)
    proj_unf = projector(X_unf, None)
    proj = fold_by_idx(proj_unf, x0.shape, mode_idxs)
    
    irms = math.sqrt(max(X_unf.shape))

    return {
        "projection": proj,
        "iterations": 1,
        "converged": True,
        "history": dict(),
        "irms": irms,
    }


# <!- Projectors by modes -!>


# <-- Projectors factories -->


def build_single_step_projection(projector: BaseProjector, mode_strategy: Union[ModeStrategy, Literal['min', 'max']], requires_irms: bool = False) -> Projector:
    if mode_strategy == 'min':
        _strategy = lambda shape: int(np.argmin(shape))
    elif mode_strategy == 'max':
        _strategy = lambda shape: int(np.argmax(shape))
    elif callable(mode_strategy):
        _strategy = mode_strategy
    else:
        raise ValueError(f"Incorrect {mode_strategy=}. Expected Callable[[list[int]], int] or Literal['min', 'max']")
    
    def _single_step_projection(x0: Tensor) -> Union[Tensor, tuple[Tensor, float]]:
        res = single_step_projection(x0, projector, _strategy)
        proj = res["projection"]

        if requires_irms:
            irms = res["irms"]

            return proj, irms

        return proj

    return _single_step_projection


def build_dual_projection(projector: BaseProjector, requires_irms: bool = False) -> Projector:
    def _dual_projector(x0: Tensor) -> Union[Tensor, tuple[Tensor, float]]:
        res = dual_projection(x0, projector)
        proj = res["projection"]

        if requires_irms:
            irms = res["irms"]

            return proj, irms

        return proj

    return _dual_projector


def build_unfold_projection(projector: BaseProjector, mode_idxs: Union[int, list[int]], requires_irms: bool = False) -> Projector:
    def _unfold_projector(x0: Tensor) -> Union[Tensor, tuple[Tensor, float]]:
        res = unfold_projection(x0, projector, mode_idxs)
        proj = res["projection"]

        if requires_irms:
            irms = res["irms"]

            return proj, irms

        return proj

    return _unfold_projector

