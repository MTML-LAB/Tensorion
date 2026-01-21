import numpy as np
import torch
from typing import Optional, Union, Callable, Any

from tensor_utils import fold, unfold
from loggers import (
    MetricsCallback,
    MetricsCallbackFactory,
    projectors_metrics_collector_factory,
)

Tensor = torch.Tensor

# base projectors


def project_spectral_ball(x: Tensor, idx: int) -> Tensor:
    X = unfold(x, idx)
    U, S, Vt = torch.linalg.svd(X, full_matrices=False)
    S = torch.clip(S, min=0.0, max=1.0)
    X = U @ torch.diag(S) @ Vt

    x_proj = fold(X, x.shape, idx)
    assert x_proj.shape == x.shape
    return x_proj


def project_spectral_sphere(x: Tensor, idx: int) -> Tensor:
    X = unfold(x, idx)
    U, S, Vt = torch.linalg.svd(X, full_matrices=False)
    X = U @ Vt

    x_proj = fold(X, x.shape, idx)
    assert x_proj.shape == x.shape
    return x_proj


def project_ns(x: Tensor, idx: int, num_ns_iterations: int = 2):
    X = unfold(x, idx)
    transpose = X.shape[0] > X.shape[1]
    if transpose:
        X.transpose_(0, 1)

    # m x n, m <= n
    X /= torch.linalg.norm(X, ord=2)  # TODO: speedup
    m = X.shape[0]
    I = torch.eye(m, device=X.device, dtype=X.dtype)

    for _ in range(num_ns_iterations):
        X = (3 * I - X @ X.T) @ (X / 2)  # TODO: speedup

    x_proj = fold(X, x.shape, idx)
    assert x_proj.shape == x.shape
    return x_proj


# intersection projectors


def dykstra(
    x0: Tensor,
    projector: Callable[[Tensor, int], Tensor],
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
    projector: Callable[[Tensor, int], Tensor],
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


def single_step_projection(
    x0: Tensor,
    projector: Callable[[Tensor, int], Tensor],
    num_sets: int,
    max_iter: int = 1000,
    tol: float = 1e-6,
    metrics_callback_factory: Optional[MetricsCallbackFactory] = None,
    device: Optional[torch.device] = None,
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
    dims_idxs = np.argsort(x_dims)
    # assert x_dims[dims_idxs[0]] < x_dims[dims_idxs[1]], f"{x_dims=}\t{dims_idxs=}"

    mode = dims_idxs[0]
    proj = projector(x0, mode)

    return {
        "projection": proj,
        "iterations": 1,
        "converged": True,
        "history": dict(),
    }
