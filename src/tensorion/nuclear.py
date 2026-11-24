"""Cheap estimates of the nuclear norm of a matrix, without an SVD.

The online unfolding strategy needs to know which matricization of an update
has the largest nuclear norm. It needs a ranking, not accurate values, so an
estimate that only uses matrix products is enough.

For a matrix ``M`` with ``A = M M^T``, the nuclear norm is ``tr(sqrt(A))``.
:func:`nuclear_norm_bounds` estimates this trace by stochastic Lanczos
quadrature with three ingredients:

* **Lanczos quadrature.** For a vector ``v``, ``k`` Lanczos steps on ``A``
  give a ``k x k`` tridiagonal matrix ``T`` with
  ``v^T sqrt(A) v ~ |v|^2 e_1^T sqrt(T) e_1`` (a Gauss quadrature rule).
* **Two-sided rules.** The even derivatives of ``sqrt`` are negative and the odd
  ones positive, so the Gauss rule overestimates and the Gauss-Radau rule with
  a node at 0 underestimates (Golub and Meurant, *Matrices, Moments and
  Quadrature with Applications*, 2010). Momentum matrices have many singular
  values near zero, where ``sqrt`` is steep and a single rule is biased by
  tens of percent. The two rules bracket the value and their mean is used.
* **Deflation.** The trace over a sketched dominant subspace is computed
  directly and random probes are used only for the rest, as in Hutch++
  (Meyer et al., 2021). This removes most of the probe noise for the
  low-rank matrices that small-batch gradients produce.

The cost is ``2 * steps * (sketch + probes) + 2 * sketch`` products of ``M``
or ``M^T`` with a vector, batched into ``2 * steps + 2`` matrix products.
"""

from __future__ import annotations

from functools import lru_cache

import torch

Tensor = torch.Tensor

__all__ = ["nuclear_norm", "nuclear_norm_bounds", "nuclear_norm_exact"]

# Below this size of the smaller side the Gram matrix is cheaper than the estimate.
_GRAM_BELOW = 128


def nuclear_norm_exact(M: Tensor) -> float:
    """Nuclear norm by a full SVD."""
    return torch.linalg.svdvals(M.float()).sum().item()


def _nuclear_norm_gram(M: Tensor) -> float:
    """Nuclear norm of a wide matrix from its small Gram matrix ``M M^T``.

    The singular values of ``M`` are the square roots of the eigenvalues of
    ``M M^T``, which is only ``rows x rows``. For a ``3 x N`` unfolding this is
    one matrix product and a ``3 x 3`` eigenvalue problem. Forming the Gram
    matrix in float32 costs accuracy for singular values far below the largest
    one, at most about 1% of the nuclear norm for the sizes used here.
    """
    gram = (M @ M.mT).cpu().double()
    return torch.linalg.eigvalsh(gram).clamp_min(0).sqrt().sum().item()


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


def _lanczos(M: Tensor, V: Tensor, steps: int) -> tuple[Tensor, Tensor]:
    """Lanczos on ``M M^T``, started from every column of ``V`` at once.

    Returns the diagonals ``(columns, steps)`` and off-diagonals
    ``(columns, steps)`` of the tridiagonal matrices, on the CPU in float64.
    """
    Q = V / torch.linalg.vector_norm(V, dim=0).clamp_min(1e-30)
    Q_prev = torch.zeros_like(Q)
    beta = Q.new_zeros(Q.shape[1])
    alphas, betas = [], []
    for _ in range(steps):
        W = M @ (M.mT @ Q) - beta * Q_prev
        alpha = (W * Q).sum(0)
        W = W - alpha * Q
        beta = torch.linalg.vector_norm(W, dim=0)
        alphas.append(alpha)
        betas.append(beta)
        Q_prev, Q = Q, W / beta.clamp_min(1e-12)
    return torch.stack(alphas, -1).cpu().double(), torch.stack(betas, -1).cpu().double()


def _sqrt_quadrature(diagonal: Tensor, off_diagonal: Tensor) -> Tensor:
    """``e_1^T sqrt(T) e_1`` for a batch of tridiagonal matrices ``T``."""
    T = torch.diag_embed(diagonal) + torch.diag_embed(off_diagonal, 1) + torch.diag_embed(off_diagonal, -1)
    eigenvalues, eigenvectors = torch.linalg.eigh(T)
    return (eigenvectors[:, 0, :] ** 2 * eigenvalues.clamp_min(0).sqrt()).sum(-1)


def _two_sided_quadrature(alpha: Tensor, beta: Tensor) -> tuple[Tensor, Tensor]:
    """Gauss-Radau (lower) and Gauss (upper) values of ``e_1^T sqrt(T) e_1``."""
    upper = _sqrt_quadrature(alpha, beta[:, :-1])
    # Gauss-Radau: append a row whose diagonal entry makes 0 an eigenvalue.
    T = torch.diag_embed(alpha) + torch.diag_embed(beta[:, :-1], 1) + torch.diag_embed(beta[:, :-1], -1)
    rhs = torch.zeros(*alpha.shape, 1, dtype=alpha.dtype)
    rhs[:, -1, 0] = beta[:, -1] ** 2
    # T is positive definite up to rounding. lstsq also covers a breakdown of the recurrence.
    delta = torch.linalg.lstsq(T, rhs).solution[:, -1, 0]
    lower = _sqrt_quadrature(torch.cat([alpha, delta[:, None]], 1), beta)
    # Rounding can swap the two for a start vector that is almost an eigenvector.
    return torch.minimum(lower, upper), upper


def nuclear_norm_bounds(M: Tensor, probes: int = 16, sketch: int = 16, steps: int = 8) -> tuple[float, float]:
    """Estimated lower and upper values of the nuclear norm of ``M``.

    They are bounds up to the noise of the random probes. For matrices whose
    smaller side is below 128 the value is computed from the small Gram matrix
    instead, and both values coincide.

    Args:
        M: a matrix.
        probes: number of random probes for the part outside the sketch.
        sketch: dimension of the dominant subspace that is handled without probes.
        steps: Lanczos steps per start vector.
    """
    M = M.float()
    if M.shape[0] > M.shape[1]:
        M = M.mT
    rows = M.shape[0]
    if rows < max(_GRAM_BELOW, 2 * steps, sketch + 1):
        value = _nuclear_norm_gram(M)
        return value, value
    scale = torch.linalg.matrix_norm(M).item()
    if not scale > 0:
        return 0.0, 0.0
    M = M / scale

    Z = _probes(rows, sketch + probes, M.device)
    Q, _ = torch.linalg.qr(M @ (M.mT @ Z[:, :sketch]))
    G = Z[:, sketch:]
    G = G - Q @ (Q.mT @ G)
    start = torch.cat([Q, G], 1)

    lower, upper = _two_sided_quadrature(*_lanczos(M, start, steps))
    weights = (start * start).sum(0).cpu().double()  # 1 for the sketch, |g|^2 for the probes
    weights[sketch:] /= probes
    return (lower * weights).sum().item() * scale, (upper * weights).sum().item() * scale


def nuclear_norm(M: Tensor, probes: int = 16, sketch: int = 16, steps: int = 8) -> float:
    """Estimate of the nuclear norm of ``M``: the mean of :func:`nuclear_norm_bounds`."""
    lower, upper = nuclear_norm_bounds(M, probes, sketch, steps)
    return (lower + upper) / 2
