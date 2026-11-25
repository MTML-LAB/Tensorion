"""Cheap estimates of the nuclear norm of a matrix, without an SVD.

The online unfolding strategies need to know which matricization of an update
has the largest nuclear norm. They need a ranking, not accurate values, so an
estimate that only uses matrix products is enough. Two estimates are provided.

**Polar factor and trace** (:func:`nuclear_norm_polar`). For the polar factor
``P = U V^T`` of ``M = U S V^T``, ``<M, P> = tr(S)`` is the nuclear norm. The
optimizer computes ``P`` for the unfolding it uses anyway, so the factor of the
winning candidate is reused as the update. With an approximate orthogonalizer
the value is the inner product of the momentum with the update that would
actually be applied.

**Stochastic Lanczos quadrature** (:func:`nuclear_norm_bounds`). For
``A = M M^T`` the nuclear norm is ``tr(sqrt(A))``, estimated with

* *Lanczos quadrature*: for a vector ``v``, ``k`` Lanczos steps on ``A`` give a
  ``k x k`` tridiagonal matrix ``T`` with ``v^T sqrt(A) v ~ |v|^2 e_1^T sqrt(T) e_1``
  (a Gauss quadrature rule),
* *two-sided rules*: the even derivatives of ``sqrt`` are negative and the odd
  ones positive, so the Gauss rule overestimates and the Gauss-Radau rule with a
  node at 0 underestimates (Golub and Meurant, *Matrices, Moments and Quadrature
  with Applications*, 2010). Momentum matrices have many singular values near
  zero, where a single rule is biased by tens of percent. The two rules bracket
  the value and their mean is used,
* *deflation*: the trace over a sketched dominant subspace is computed directly
  and random probes are used only for the rest, as in Hutch++ (Meyer et al.,
  2021), which removes most of the probe noise for low-rank gradients.

Everything runs on the device of the input. Nothing is moved to the CPU, and
the only operations besides matrix products are ``eigvalsh``, ``cholesky`` and
triangular or square solves of tiny matrices (at most ``sketch x sketch`` or
``(steps + 1) x (steps + 1)``), in float32. The large products can run in a
reduced precision (``dtype=torch.bfloat16``). One number per candidate is read
back to decide which split wins.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Optional

import torch

Tensor = torch.Tensor

__all__ = ["nuclear_norm", "nuclear_norm_bounds", "nuclear_norm_exact", "nuclear_norm_polar"]

# Below this size of the smaller side the Gram matrix is cheaper than the estimate.
_GRAM_BELOW = 128


def nuclear_norm_exact(M: Tensor) -> float:
    """Nuclear norm by a full SVD."""
    return torch.linalg.svdvals(M.float()).sum().item()


def nuclear_norm_polar(M: Tensor, orthogonalize: Callable[[Tensor], Tensor]) -> tuple[Tensor, Tensor]:
    """``<M, P>`` for ``P = orthogonalize(M)``, and ``P`` itself.

    With an exact polar factor this is the nuclear norm. The value is a 0-dim
    float32 tensor on the device of ``M``.
    """
    P = orthogonalize(M)
    return (M * P).sum(dtype=torch.float32), P


def _nuclear_norm_gram(M: Tensor) -> Tensor:
    """Nuclear norm of a wide matrix from its small Gram matrix ``M M^T``.

    The singular values of ``M`` are the square roots of the eigenvalues of
    ``M M^T``, which is only ``rows x rows``. For a ``3 x N`` unfolding this is
    one matrix product and a ``3 x 3`` eigenvalue problem. Forming the Gram
    matrix in float32 costs accuracy for singular values far below the largest
    one, at most about 1% of the nuclear norm for the sizes used here. A
    reduced precision would cost much more, so this product stays in float32.
    """
    M = M.float()
    return torch.linalg.eigvalsh(M @ M.mT).clamp_min(0).sqrt().sum()


@lru_cache(maxsize=128)
def _probes(n: int, count: int, device: torch.device) -> Tensor:
    """A fixed ``n x count`` Rademacher matrix.

    The probes are the same at every call, so the estimate is a deterministic
    function of the matrix: training is reproducible and the user's random
    number stream is left alone.
    """
    generator = torch.Generator().manual_seed(n * 7919 + count)
    signs = torch.randint(0, 2, (n, count), generator=generator, dtype=torch.float32)
    return (2 * signs - 1).to(device)


def _orthonormalize(Y: Tensor) -> Tensor:
    """An orthonormal basis of the range of ``Y`` by Cholesky QR (no Householder QR).

    A small ridge keeps the Cholesky factorization defined when ``Y`` is rank
    deficient. Directions that ``Y`` barely contains come out with a norm
    below 1, so they are simply counted less in the deflation.
    """
    gram = Y.mT @ Y
    ridge = 1e-5 * gram.diagonal().mean().clamp_min(1e-30)
    L, _ = torch.linalg.cholesky_ex(gram + ridge * torch.eye(gram.shape[0], device=Y.device, dtype=Y.dtype))
    return torch.linalg.solve_triangular(L, Y.mT, upper=False).mT


def _tridiagonal(diagonal: Tensor, off_diagonal: Tensor) -> Tensor:
    return torch.diag_embed(diagonal) + torch.diag_embed(off_diagonal, 1) + torch.diag_embed(off_diagonal, -1)


def _sqrt_quadrature(diagonal: Tensor, off_diagonal: Tensor) -> Tensor:
    """``e_1^T sqrt(T) e_1`` for a batch of tridiagonal matrices ``T``, from eigenvalues only.

    The Gauss weights are the squared first components of the eigenvectors of
    ``T``. With the eigenvalues ``theta`` of ``T`` and ``mu`` of ``T`` without
    its first row and column, they are

        w_i = prod_j (theta_i - mu_j) / prod_{j != i} (theta_i - theta_j).

    The two spectra interlace, so every factor of the product, paired as below,
    lies in [0, 1]. Clamping the factors makes the formula safe when Ritz values
    coincide, including a breakdown of the Lanczos recurrence. No eigenvectors
    are needed, and ``eigvalsh`` of tiny matrices runs on every backend.
    """
    theta = torch.linalg.eigvalsh(_tridiagonal(diagonal, off_diagonal))  # (batch, k), ascending
    k = theta.shape[-1]
    if k == 1:
        return theta[:, 0].clamp_min(0).sqrt()
    mu = torch.linalg.eigvalsh(_tridiagonal(diagonal[:, 1:], off_diagonal[:, 1:]))  # (batch, k - 1)
    # theta_1 <= mu_1 <= theta_2 <= ... <= mu_{k-1} <= theta_k.
    # Pair mu_j with theta_j for j < i and with theta_{j+1} for j >= i.
    i = torch.arange(k, device=theta.device)
    j = torch.arange(k - 1, device=theta.device)
    partner = torch.where(j[None, :] < i[:, None], j[None, :], j[None, :] + 1)  # (k, k - 1)
    numerator = (theta[:, :, None] - mu[:, None, :]).abs()
    denominator = (theta[:, :, None] - theta[:, partner]).abs()
    factors = (numerator / denominator.clamp_min(1e-30)).clamp(0, 1)
    factors = torch.where(denominator > 0, factors, torch.ones_like(factors))
    weights = factors.prod(-1)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-30)
    return (weights * theta.clamp_min(0).sqrt()).sum(-1)


def _two_sided_quadrature(alpha: Tensor, beta: Tensor) -> tuple[Tensor, Tensor]:
    """Gauss-Radau (lower) and Gauss (upper) values of ``e_1^T sqrt(T) e_1``."""
    upper = _sqrt_quadrature(alpha, beta[:, :-1])
    # Gauss-Radau: append a row whose diagonal entry makes 0 an eigenvalue.
    rhs = torch.zeros(*alpha.shape, 1, device=alpha.device, dtype=alpha.dtype)
    rhs[:, -1, 0] = beta[:, -1] ** 2
    delta = torch.linalg.solve_ex(_tridiagonal(alpha, beta[:, :-1]), rhs).result[:, -1, 0]
    # A singular T gives a non-finite entry, which eigvalsh would reject.
    delta = torch.nan_to_num(delta, nan=0.0, posinf=1e6, neginf=0.0).clamp(0, 1e6)
    lower = _sqrt_quadrature(torch.cat([alpha, delta[:, None]], 1), beta)
    # A singular T (breakdown of the recurrence) or rounding can spoil the lower rule.
    lower = torch.where(torch.isfinite(lower), torch.minimum(lower, upper), upper)
    return lower, upper


def _bounds(M: Tensor, probes: int, sketch: int, steps: int, dtype: Optional[torch.dtype]) -> Tensor:
    """Lower and upper estimate as a tensor ``[lower, upper]`` on the device of ``M``."""
    if M.shape[0] > M.shape[1]:
        M = M.mT
    rows = M.shape[0]
    if rows < max(_GRAM_BELOW, 2 * steps, sketch + 1):
        return _nuclear_norm_gram(M).expand(2)
    scale = torch.linalg.vector_norm(M, dtype=torch.float32)
    inverse_scale = torch.where(scale > 0, 1 / scale, torch.zeros_like(scale))
    work = M if dtype is None or M.dtype == dtype else M.to(dtype)

    def apply(V: Tensor) -> Tensor:
        """``(M M^T / scale^2) V``, with the large products in the working precision.

        The scale is divided out after each product, so neither a tiny nor a huge ``M`` leaves float32.
        """
        inner = (work.mT @ V.to(work.dtype)).float() * inverse_scale
        return (work @ inner.to(work.dtype)).float() * inverse_scale

    Z = _probes(rows, sketch + probes, M.device)
    Q = _orthonormalize(apply(Z[:, :sketch]))
    G = Z[:, sketch:]
    G = G - Q @ (Q.mT @ G)
    start = torch.cat([Q, G], 1)
    weights = (start * start).sum(0)  # about 1 for the sketch, |g|^2 for the probes
    weights = torch.cat([weights[:sketch], weights[sketch:] / probes])

    # Lanczos on M M^T, started from every column at once. The vectors are small and stay in float32.
    Q = start / torch.linalg.vector_norm(start, dim=0).clamp_min(1e-30)
    Q_prev = torch.zeros_like(Q)
    beta = Q.new_zeros(Q.shape[1])
    alphas, betas = [], []
    for _ in range(steps):
        W = apply(Q) - beta * Q_prev
        alpha = (W * Q).sum(0)
        W = W - alpha * Q
        beta = torch.linalg.vector_norm(W, dim=0)
        alphas.append(alpha)
        betas.append(beta)
        Q_prev, Q = Q, W / beta.clamp_min(1e-12)

    alpha = torch.nan_to_num(torch.stack(alphas, -1), nan=0.0, posinf=0.0, neginf=0.0)
    beta = torch.nan_to_num(torch.stack(betas, -1), nan=0.0, posinf=0.0, neginf=0.0)
    lower, upper = _two_sided_quadrature(alpha, beta)
    return torch.stack([(lower * weights).sum(), (upper * weights).sum()]) * scale


def nuclear_norm_bounds(
    M: Tensor, probes: int = 16, sketch: int = 16, steps: int = 8, dtype: Optional[torch.dtype] = None
) -> tuple[float, float]:
    """Estimated lower and upper values of the nuclear norm of ``M``.

    They are bounds up to the noise of the random probes. For matrices whose
    smaller side is below 128 the value is computed from the small Gram matrix
    instead, and both values coincide.

    Args:
        M: a matrix.
        probes: number of random probes for the part outside the sketch.
        sketch: dimension of the dominant subspace that is handled without probes.
        steps: Lanczos steps per start vector.
        dtype: precision of the products with ``M``, e.g. ``torch.bfloat16``.
            ``None`` uses the precision of ``M``.
    """
    lower, upper = _bounds(M, probes, sketch, steps, dtype).tolist()
    return lower, upper


def nuclear_norm(
    M: Tensor, probes: int = 16, sketch: int = 16, steps: int = 8, dtype: Optional[torch.dtype] = None
) -> float:
    """Estimate of the nuclear norm of ``M``: the mean of :func:`nuclear_norm_bounds`."""
    lower, upper = nuclear_norm_bounds(M, probes, sketch, steps, dtype)
    return (lower + upper) / 2
