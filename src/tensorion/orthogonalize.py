"""Approximations of the polar factor ``U V^T`` of a matrix ``G = U S V^T``.

Every orthogonalizer has the signature ``f(G: Tensor, steps: int = <default>) -> Tensor``,
returns a matrix of the same shape and dtype as ``G`` and never modifies ``G``.
Iterative methods run in the precision of ``G`` (float32, bfloat16 or float16),
so passing a bfloat16 matrix runs every product in bfloat16, as Muon does.
Norms are accumulated in float32. Default step counts use about
15 matrix multiplications for every iterative method.

* ``"newton_schulz"``  -- quintic Newton-Schulz iteration from Muon (default, used in the paper).
* ``"svd"``            -- exact, via ``torch.linalg.svd``.
* ``"polar_express"``  -- Polar Express (Amsel et al., 2025, arXiv:2505.16932).
* ``"cans"``           -- CANS, Chebyshev-accelerated Newton-Schulz with degree-3
  polynomials (Grishina, Smirnov and Rakhuba, 2025, arXiv:2506.10935).
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
    X = _working(G)
    transpose = X.shape[0] > X.shape[1]
    if transpose:
        X = X.mT
    X = X / scale.to(X.dtype)
    for a, b, c in coefficients:
        A = X @ X.mT
        B = b * A + c * (A @ A) if c else b * A
        X = a * X + B @ X
    if transpose:
        X = X.mT
    return X.to(G.dtype)


_WORKING_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def _working(G: Tensor) -> Tensor:
    """``G`` itself if its precision can be used for the iteration, else a float32 copy."""
    return G if G.dtype in _WORKING_DTYPES else G.float()


def _norm(G: Tensor) -> Tensor:
    """Frobenius norm accumulated in float32, without a float32 copy of ``G``."""
    return torch.linalg.vector_norm(G, dtype=torch.float32)


def newton_schulz(G: Tensor, steps: int = 5) -> Tensor:
    """Quintic Newton-Schulz iteration with the coefficients used in Muon.

    It converges fast but only roughly: singular values end up in about ``[0.7, 1.2]``.
    """
    return _odd_polynomial_iteration(G, [(3.4445, -4.7750, 2.0315)] * steps, _norm(G) + _EPS)


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
    return _odd_polynomial_iteration(G, coefficients, _norm(G) * 1.01 + _EPS)


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


def cans(G: Tensor, steps: int = 7, lower: float = 9e-4) -> Tensor:
    """CANS iteration with degree-3 polynomials (2 matmuls per step).

    ``lower`` is the assumed lower bound on the normalized singular values.
    The defaults are the Muon setting of the CANS paper: 7 steps, which map
    singular values in ``[9e-4, 1]`` into about ``[0.7, 1.3]``. Use more steps
    for a full orthogonalization (10 steps converge for ``lower=1e-3``).
    The matrix is normalized by Gelfand's bound ``||(G G^T)^2||_F^(1/4)`` on
    the spectral norm, as in the paper.
    """
    X = _working(G)
    if X.shape[0] > X.shape[1]:
        X = X.mT
    # Normalize first so that the fourth power stays in range in a reduced precision.
    frobenius = _norm(X) + _EPS
    X = X / frobenius.to(X.dtype)
    A = X @ X.mT
    scale = (_norm(A @ A) ** (1 / 4)) * frobenius + _EPS
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
