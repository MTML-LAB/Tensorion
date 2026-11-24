"""Approximations of the polar factor ``U V^T`` of a matrix ``G = U S V^T``.

Every orthogonalizer has the signature ``f(G: Tensor, steps: int) -> Tensor``,
returns a matrix of the same shape and dtype as ``G`` and never modifies ``G``.
Iterative methods run in float32 internally.

* ``"svd"``            -- exact, via ``torch.linalg.svd`` (default, used in the paper).
* ``"newton_schulz"``  -- quintic Newton-Schulz iteration from Muon.
* ``"polar_express"``  -- Polar Express (Amsel et al., 2025).
* ``"cans"``           -- CANS, Chebyshev-accelerated Newton-Schulz with
  degree-3 polynomials (Grishina et al., 2025).
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Callable, Sequence, Union

import torch

Tensor = torch.Tensor
Orthogonalizer = Callable[[Tensor, int], Tensor]
OrthogonalizerSpec = Union[str, Orthogonalizer]

__all__ = [
    "svd",
    "newton_schulz",
    "polar_express",
    "cans",
    "ORTHOGONALIZERS",
    "resolve_orthogonalizer",
]

_EPS = 1e-7


def svd(G: Tensor, steps: int = 0) -> Tensor:
    """Exact polar factor ``U V^T``. ``steps`` is ignored."""
    U, _, Vh = torch.linalg.svd(G.float(), full_matrices=False)
    return (U @ Vh).to(G.dtype)


def _odd_polynomial_iteration(G: Tensor, coefficients: Sequence[tuple[float, float, float]], scale: Tensor) -> Tensor:
    """Apply ``X <- a X + b (X X^T) X + c (X X^T)^2 X`` for each ``(a, b, c)``."""
    X = G.float()
    transpose = X.shape[0] > X.shape[1]
    if transpose:
        X = X.mT
    X = X / scale
    for a, b, c in coefficients:
        A = X @ X.mT
        B = b * A + c * (A @ A) if c else b * A
        X = a * X + B @ X
    if transpose:
        X = X.mT
    return X.to(G.dtype)


def newton_schulz(G: Tensor, steps: int = 5) -> Tensor:
    """Quintic Newton-Schulz iteration with the coefficients used in Muon.

    It converges fast but only roughly: singular values end up in about ``[0.7, 1.2]``.
    """
    scale = torch.linalg.matrix_norm(G.float()) + _EPS
    return _odd_polynomial_iteration(G, [(3.4445, -4.7750, 2.0315)] * steps, scale)


# Coefficients from "The Polar Express: Optimal Matrix Sign Methods and Their
# Application to the Muon Algorithm" (Amsel et al., 2025, arXiv:2505.16932),
# computed for a lower singular value bound of 1e-3, with the 1.01 safety
# factor folded into all but the last polynomial.
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


def polar_express(G: Tensor, steps: int = 5) -> Tensor:
    """Polar Express iteration. Steps beyond the tabulated eight reuse the last polynomial."""
    table = _POLAR_EXPRESS_COEFFICIENTS
    coefficients = table[:steps] + table[-1:] * max(0, steps - len(table))
    scale = torch.linalg.matrix_norm(G.float()) * 1.01 + _EPS
    return _odd_polynomial_iteration(G, coefficients, scale)


@lru_cache(maxsize=None)
def _cans_coefficients(steps: int, lower: float) -> tuple[tuple[float, float, float], ...]:
    """Optimal degree-3 odd polynomials on ``[lower, upper]``, composed ``steps`` times.

    Each step maps ``[lower, upper]`` into ``[1 - err, 1 + err]``, which becomes
    the interval for the next step (explicit formula from the CANS paper).
    """
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


def cans(G: Tensor, steps: int = 5, lower: float = 1e-3) -> Tensor:
    """CANS iteration with degree-3 polynomials.

    ``lower`` is the assumed lower bound on the normalized singular values.
    The matrix is normalized by ``||G G^T G||_F^(1/3)``, which upper-bounds the
    spectral norm more tightly than the Frobenius norm.
    """
    X = G.float()
    if X.shape[0] > X.shape[1]:
        X = X.mT
    scale = torch.linalg.matrix_norm((X @ X.mT) @ X) ** (1 / 3) + _EPS
    return _odd_polynomial_iteration(G, _cans_coefficients(steps, lower), scale)


ORTHOGONALIZERS: dict[str, Orthogonalizer] = {
    "svd": svd,
    "newton_schulz": newton_schulz,
    "polar_express": polar_express,
    "cans": cans,
}


def resolve_orthogonalizer(spec: OrthogonalizerSpec) -> Orthogonalizer:
    """Turn an orthogonalizer spec (a name or a callable) into a function."""
    if isinstance(spec, str):
        try:
            return ORTHOGONALIZERS[spec]
        except KeyError:
            raise ValueError(
                f"Unknown orthogonalizer {spec!r}. Choose one of {sorted(ORTHOGONALIZERS)} or pass a callable."
            ) from None
    if callable(spec):
        return spec
    raise TypeError("orthogonalizer must be a string or a callable f(matrix, steps) -> matrix.")
