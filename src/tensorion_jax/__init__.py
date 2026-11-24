"""Tensorion for JAX, as optax gradient transformations (experimental).

    import optax, tensorion_jax

    optimizer = tensorion_jax.tensorion(learning_rate=1e-3)

Requires ``jax`` and ``optax`` (``pip install "tensorion[jax]"``).
"""

from ._linalg import (
    ORTHOGONALIZERS,
    cans,
    matricize,
    newton_schulz,
    nuclear_norm_bounds,
    nuclear_norm_exact,
    polar_express,
    svd,
    unmatricize,
)
from ._optimizer import ScaleByTensorionState, offline_dimension_numbers, scale_by_tensorion, tensorion
from ._unfolding import candidates, offline_rows, two_way_splits

__all__ = [
    "tensorion",
    "scale_by_tensorion",
    "ScaleByTensorionState",
    "offline_dimension_numbers",
    "offline_rows",
    "candidates",
    "two_way_splits",
    "matricize",
    "unmatricize",
    "ORTHOGONALIZERS",
    "svd",
    "newton_schulz",
    "polar_express",
    "cans",
    "nuclear_norm_bounds",
    "nuclear_norm_exact",
]
