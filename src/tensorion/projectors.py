import numpy as np
import torch
import operator
import math
from typing import Optional, Union, Callable, Any, Dict

from itertools import chain
from tensorion.tensor_utils import fold, unfold, partitions, invert_permutation
from tensorion.optimizers.utils.cans_svd import cans_svd, cans_polar_iteration
from tensorion.loggers import (
    MetricsCallbackFactory,
    projectors_metrics_collector_factory,
)

Tensor = torch.Tensor

# base projectors


def project_spectral_ball(
    x: Tensor,
    idx: int,
    *,
    use_cans_svd: bool = False,
    cans_svd_kwargs: Optional[Dict[str, Any]] = None,
) -> Tensor:
    X = unfold(x, idx)
    if use_cans_svd:
        U, S, Vt = cans_svd(X, **(cans_svd_kwargs or {}))
    else:
        U, S, Vt = torch.linalg.svd(X, full_matrices=False)
    S = torch.clip(S, min=0.0, max=1.0)
    X = U @ torch.diag(S) @ Vt

    x_proj = fold(X, x.shape, idx)
    assert x_proj.shape == x.shape
    return x_proj


def project_spectral_sphere(
    x: Tensor,
    idx: int,
    *,
    use_cans_svd: bool = False,
    cans_svd_kwargs: Optional[Dict[str, Any]] = None,
) -> Tensor:
    X = unfold(x, idx)
    if use_cans_svd:
        X = cans_polar_iteration(X, **(cans_svd_kwargs or {}))
    else:
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
    eye = torch.eye(m, device=X.device, dtype=X.dtype)

    for _ in range(num_ns_iterations):
        X = (3 * eye - X @ X.T) @ (X / 2)  # TODO: speedup

    if transpose:
        X.transpose_(0, 1)
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


def st_hosvd_truncation(
    x0: Tensor,
    *,
    ranks: Union[list[int], int],  # currently only supports fixed rank (fix later)
    use_cans_svd: bool = False,
    cans_svd_kwargs: Optional[Dict[str, Any]] = None,
) -> Tensor:

    x_dims = list(x0.shape)
    dims_idxs = np.argsort(x_dims)
    if isinstance(ranks, int):
        ranks = [ranks] * len(x_dims)
    else:
        ranks = operator.itemgetter(*dims_idxs)(ranks)

    mode = dims_idxs[0]
    x = x0
    factors = []

    for mode, rank in zip(dims_idxs, ranks):
        original_shape = x.shape
        x_unfolded = unfold(x, mode)
        rank = min(rank, x_unfolded.shape[0])

        if use_cans_svd:
            u_t, *_ = cans_svd(x_unfolded, **(cans_svd_kwargs or {}))
        else:
            u_t, *_ = torch.linalg.svd(x_unfolded, full_matrices=False)

        orthoprojector = u_t[:, :rank]
        x = orthoprojector.T @ x_unfolded
        factors.append(orthoprojector)

        # Compute new shape after projection: replace original_shape[mode] with rank
        new_shape = list(original_shape)
        new_shape[mode] = rank
        new_shape = tuple(new_shape)
        x = fold(x, new_shape, mode)

    return x, factors


def st_hosvd_projection(
    x: Tensor,
    *,
    use_cans_svd: bool = False,
    cans_svd_kwargs: Optional[Dict[str, Any]] = None,
) -> Tensor:

    reshapes = partitions(x.dim())
    best_val = float("-inf")

    for reshape in reshapes:
        perm = [*chain.from_iterable(reshape)]
        x = x.permute(*perm)

        perm_shape = x.shape
        prod_shape = math.prod([x.shape[i] for i in reshape[0]])

        x_reshaped = x.reshape(prod_shape, -1)

        u, s, vt = cans_svd(x_reshaped, **(cans_svd_kwargs or {}))
        nuclear_norm = s.sum()

        inv_order = invert_permutation(perm)

        x = x.reshape(*perm_shape)
        x = x.permute(*inv_order)

        if nuclear_norm > best_val:
            best_val = nuclear_norm
            # cans_svd returns (U, s, V.T), so v is already V.T
            # For polar projection (spectral sphere), we want U @ V.T = u @ v
            x_orth = u @ vt
            x_orth = x_orth.reshape(*perm_shape)
            x_orth = x_orth.permute(*inv_order)

    return {
        "projection": x_orth,
        "iterations": 1,
        "converged": True,
        "history": dict(),
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
