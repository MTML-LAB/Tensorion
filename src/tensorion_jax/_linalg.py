"""Orthogonalizers and the SVD-free nuclear norm estimate, in JAX.

These mirror :mod:`tensorion.orthogonalize` and :mod:`tensorion.nuclear`.
Everything is jit-compatible: all branches depend on shapes only.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Callable, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array
Orthogonalizer = Callable[..., Array]

_EPS = 1e-7


def matricize(x: Array, row_modes: tuple[int, ...]) -> Array:
    """Flatten the axes in ``row_modes`` into rows and all other axes into columns."""
    perm = list(row_modes) + [m for m in range(x.ndim) if m not in row_modes]
    rows = math.prod(x.shape[m] for m in row_modes)
    return jnp.transpose(x, perm).reshape(rows, -1)


def unmatricize(m: Array, shape: Sequence[int], row_modes: tuple[int, ...]) -> Array:
    """Inverse of :func:`matricize`."""
    perm = list(row_modes) + [a for a in range(len(shape)) if a not in row_modes]
    inverse = [perm.index(a) for a in range(len(shape))]
    return jnp.transpose(m.reshape([shape[p] for p in perm]), inverse)


# --- orthogonalizers ---------------------------------------------------------


def svd(G: Array, steps: int = 0) -> Array:
    """Exact polar factor ``U V^T``. ``steps`` is ignored."""
    U, _, Vh = jnp.linalg.svd(G, full_matrices=False)
    return U @ Vh


def _odd_polynomial_iteration(G: Array, coefficients, scale: Array) -> Array:
    """Apply ``X <- a X + b (X X^T) X + c (X X^T)^2 X`` for each ``(a, b, c)``."""
    transpose = G.shape[0] > G.shape[1]
    X = G.T if transpose else G
    X = X / scale
    for a, b, c in coefficients:
        A = X @ X.T
        B = b * A + c * (A @ A) if c else b * A
        X = a * X + B @ X
    return X.T if transpose else X


def newton_schulz(G: Array, steps: int = 5) -> Array:
    """Quintic Newton-Schulz iteration with the coefficients used in Muon."""
    return _odd_polynomial_iteration(G, [(3.4445, -4.7750, 2.0315)] * steps, jnp.linalg.norm(G) + _EPS)


_POLAR_EXPRESS_COEFFICIENTS = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]
_POLAR_EXPRESS_COEFFICIENTS = [
    (a / 1.01, b / 1.01**3, c / 1.01**5) for a, b, c in _POLAR_EXPRESS_COEFFICIENTS[:-1]
] + _POLAR_EXPRESS_COEFFICIENTS[-1:]


def polar_express(G: Array, steps: int = 5) -> Array:
    """Polar Express iteration (Amsel et al., 2025, arXiv:2505.16932)."""
    table = _POLAR_EXPRESS_COEFFICIENTS
    coefficients = table[:steps] + table[-1:] * max(0, steps - len(table))
    return _odd_polynomial_iteration(G, coefficients, jnp.linalg.norm(G) * 1.01 + _EPS)


@lru_cache(maxsize=None)
def _cans_coefficients(steps: int, lower: float) -> tuple[tuple[float, float, float], ...]:
    coefficients = []
    a, b = lower, 1.0
    for _ in range(steps):
        s = a * a + a * b + b * b
        e = math.sqrt(s / 3)
        denominator = 2 * e**3 + a * a * b + b * b * a
        alpha = 2 / denominator
        err = (2 * e**3 - a * a * b - b * b * a) / denominator
        coefficients.append((alpha * s, -alpha, 0.0))
        a, b = 1 - err, 1 + err
    return tuple(coefficients)


def cans(G: Array, steps: int = 7, lower: float = 9e-4) -> Array:
    """CANS iteration with degree-3 polynomials (Grishina et al., 2025, arXiv:2506.10935)."""
    X = G.T if G.shape[0] > G.shape[1] else G
    A = X @ X.T
    scale = jnp.linalg.norm(A @ A) ** (1 / 4) + _EPS
    return _odd_polynomial_iteration(G, _cans_coefficients(steps, lower), scale)


ORTHOGONALIZERS: dict[str, Orthogonalizer] = {
    "svd": svd,
    "newton_schulz": newton_schulz,
    "polar_express": polar_express,
    "cans": cans,
}


def resolve_orthogonalizer(spec: Union[str, Orthogonalizer]) -> Orthogonalizer:
    if isinstance(spec, str):
        if spec not in ORTHOGONALIZERS:
            raise ValueError(f"Unknown orthogonalizer {spec!r}. Choose one of {sorted(ORTHOGONALIZERS)} or pass a callable.")
        return ORTHOGONALIZERS[spec]
    if callable(spec):
        return spec
    raise TypeError("orthogonalizer must be a string or a callable f(matrix, steps) -> matrix.")


# --- nuclear norm ------------------------------------------------------------

# Below this size of the smaller side the Gram matrix is cheaper than the estimate.
_GRAM_BELOW = 128


def nuclear_norm_exact(M: Array) -> tuple[Array, Array]:
    """Nuclear norm by a full SVD, returned as a (lower, upper) pair."""
    # full_matrices=False matters even without U and V: the default would allocate a
    # square matrix of the larger side, which is huge for a 3 x N unfolding.
    value = jnp.linalg.svd(M, full_matrices=False, compute_uv=False).sum()
    return value, value


@lru_cache(maxsize=128)
def _probes(n: int, count: int) -> np.ndarray:
    """A fixed ``n x count`` Rademacher matrix, so that the estimate is deterministic."""
    signs = np.random.default_rng(n * 7919 + count).integers(0, 2, size=(n, count))
    return (2 * signs - 1).astype(np.float32)


def _tridiagonal(diagonal: Array, off_diagonal: Array) -> Array:
    return jax.vmap(lambda a, b: jnp.diag(a) + jnp.diag(b, 1) + jnp.diag(b, -1))(diagonal, off_diagonal)


def _sqrt_quadrature(diagonal: Array, off_diagonal: Array) -> Array:
    """``e_1^T sqrt(T) e_1`` for a batch of tridiagonal matrices ``T``."""
    eigenvalues, eigenvectors = jnp.linalg.eigh(_tridiagonal(diagonal, off_diagonal))
    return (eigenvectors[:, 0, :] ** 2 * jnp.sqrt(jnp.clip(eigenvalues, 0))).sum(-1)


def _nuclear_norm_gram(M: Array) -> tuple[Array, Array]:
    """Nuclear norm of a wide matrix from its small Gram matrix ``M M^T``.

    The singular values of ``M`` are the square roots of the eigenvalues of
    ``M M^T``, which is only ``rows x rows``: one matrix product and a small
    eigenvalue problem, with no factorization of ``M`` itself.
    """
    value = jnp.sqrt(jnp.clip(jnp.linalg.eigvalsh(M @ M.T), 0)).sum()
    return value, value


def nuclear_norm_bounds(M: Array, probes: int = 16, sketch: int = 16, steps: int = 8) -> tuple[Array, Array]:
    """Estimated lower and upper values of the nuclear norm of ``M``, without an SVD.

    ``tr(sqrt(M M^T))`` by stochastic Lanczos quadrature: the Gauss rule
    overestimates and the Gauss-Radau rule with a node at 0 underestimates, and
    Hutch++ deflation handles a sketched dominant subspace without probes.
    See :mod:`tensorion.nuclear` for the details. For matrices whose smaller
    side is below 128 the value is computed from the small Gram matrix instead.
    """
    if M.shape[0] > M.shape[1]:
        M = M.T
    rows = M.shape[0]
    if rows < max(_GRAM_BELOW, 2 * steps, sketch + 1):
        return _nuclear_norm_gram(M)
    scale = jnp.linalg.norm(M)
    M = M / jnp.where(scale > 0, scale, 1.0)

    Z = jnp.asarray(_probes(rows, sketch + probes), dtype=M.dtype)
    Q, _ = jnp.linalg.qr(M @ (M.T @ Z[:, :sketch]))
    G = Z[:, sketch:]
    G = G - Q @ (Q.T @ G)
    start = jnp.concatenate([Q, G], axis=1)
    weights = (start * start).sum(0)
    weights = weights.at[sketch:].divide(probes)

    # Lanczos on M M^T, started from every column at once.
    Q = start / jnp.maximum(jnp.linalg.norm(start, axis=0), 1e-30)
    Q_prev = jnp.zeros_like(Q)
    beta = jnp.zeros(Q.shape[1], dtype=M.dtype)
    alphas, betas = [], []
    for _ in range(steps):
        W = M @ (M.T @ Q) - beta * Q_prev
        alpha = (W * Q).sum(0)
        W = W - alpha * Q
        beta = jnp.linalg.norm(W, axis=0)
        alphas.append(alpha)
        betas.append(beta)
        Q_prev, Q = Q, W / jnp.maximum(beta, 1e-12)
    alpha = jnp.stack(alphas, -1)
    beta = jnp.stack(betas, -1)

    upper = _sqrt_quadrature(alpha, beta[:, :-1])
    # Gauss-Radau: append a row whose diagonal entry makes 0 an eigenvalue.
    T = _tridiagonal(alpha, beta[:, :-1])
    rhs = jnp.zeros_like(alpha).at[:, -1].set(beta[:, -1] ** 2)
    delta = jnp.linalg.solve(T, rhs[..., None])[:, -1, 0]
    lower = _sqrt_quadrature(jnp.concatenate([alpha, delta[:, None]], 1), beta)
    # A singular T (breakdown of the recurrence) or rounding can spoil the lower rule.
    lower = jnp.where(jnp.isfinite(lower), jnp.minimum(lower, upper), upper)
    return (lower * weights).sum() * scale, (upper * weights).sum() * scale
