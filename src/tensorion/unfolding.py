"""Tensor unfoldings (matricizations) and strategies for choosing one.

A tensor ``x`` of shape ``(n_0, ..., n_{d-1})`` is turned into a matrix by
choosing a set of *row modes*: the axes in that set are flattened into rows,
the remaining axes into columns. The mode-``k`` unfolding is the special case
``row_modes = (k,)``.

An *unfolding strategy* decides which row modes to use for a given update
tensor. Built-in strategies:

* ``"min"``     -- unfold along the smallest axis (default, used in the paper).
* ``"max"``     -- unfold along the largest axis.
* ``"maxrank"`` -- the split of the axes into two groups whose matricization
  has the largest possible rank, i.e. maximizes ``min(rows, cols)``.
* ``"dual"``    -- the split whose matricization has the largest nuclear norm
  for the current update (recomputed every step, costs one SVD per split).
* ``int``       -- a fixed mode-``k`` unfolding (negative indices allowed).
* ``tuple`` / ``list`` of ints -- a fixed set of row modes.
* a callable ``f(update: Tensor) -> int | Sequence[int]``.
"""

from __future__ import annotations

import itertools
import math
from typing import Callable, Sequence, Union

import torch

Tensor = torch.Tensor
RowModes = tuple[int, ...]
UnfoldingSpec = Union[str, int, Sequence[int], Callable[[Tensor], Union[int, Sequence[int]]]]

__all__ = [
    "matricize",
    "unmatricize",
    "two_way_splits",
    "resolve_unfolding",
    "STRATEGIES",
]


def _permutation(ndim: int, row_modes: RowModes) -> list[int]:
    return list(row_modes) + [m for m in range(ndim) if m not in row_modes]


def matricize(x: Tensor, row_modes: RowModes) -> Tensor:
    """Flatten the axes in ``row_modes`` into rows and all other axes into columns."""
    perm = _permutation(x.ndim, row_modes)
    rows = math.prod(x.shape[m] for m in row_modes)
    return x.permute(*perm).reshape(rows, -1)


def unmatricize(m: Tensor, shape: Sequence[int], row_modes: RowModes) -> Tensor:
    """Inverse of :func:`matricize`."""
    perm = _permutation(len(shape), row_modes)
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return m.reshape(*(shape[p] for p in perm)).permute(*inverse)


def two_way_splits(ndim: int) -> list[RowModes]:
    """All splits of ``range(ndim)`` into two non-empty groups, each split listed once.

    Each split is returned as the group that is used for rows. A split and its
    transpose give the same polar factor, so only one of them is listed.
    """
    modes = range(ndim)
    splits: list[RowModes] = []
    for k in range(1, ndim // 2 + 1):
        for rows in itertools.combinations(modes, k):
            if 2 * k == ndim and 0 not in rows:
                continue
            splits.append(tuple(rows))
    return splits


def _min_mode(x: Tensor) -> RowModes:
    return (min(range(x.ndim), key=lambda m: x.shape[m]),)


def _max_mode(x: Tensor) -> RowModes:
    return (max(range(x.ndim), key=lambda m: x.shape[m]),)


def _maxrank(x: Tensor) -> RowModes:
    numel = x.numel()

    def rank_bound(rows: RowModes) -> int:
        r = math.prod(x.shape[m] for m in rows)
        return min(r, numel // r)

    return max(two_way_splits(x.ndim), key=rank_bound)


def _dual(x: Tensor) -> RowModes:
    def nuclear_norm(rows: RowModes) -> float:
        return torch.linalg.svdvals(matricize(x, rows).float()).sum().item()

    return max(two_way_splits(x.ndim), key=nuclear_norm)


STRATEGIES: dict[str, Callable[[Tensor], RowModes]] = {
    "min": _min_mode,
    "max": _max_mode,
    "maxrank": _maxrank,
    "dual": _dual,
}


def _normalize(row_modes: Union[int, Sequence[int]], ndim: int) -> RowModes:
    if isinstance(row_modes, int):
        row_modes = (row_modes,)
    row_modes = tuple(int(m) for m in row_modes)
    if any(not -ndim <= m < ndim for m in row_modes):
        raise ValueError(f"Unfolding modes {row_modes} are out of range for a {ndim}-D tensor.")
    modes = tuple(m % ndim for m in row_modes)
    if len(set(modes)) != len(modes):
        raise ValueError(f"Unfolding modes {tuple(row_modes)} contain duplicates.")
    if not 0 < len(modes) < ndim:
        raise ValueError(
            f"Unfolding modes {tuple(row_modes)} must leave at least one row and one column axis "
            f"of a {ndim}-D tensor."
        )
    return modes


def resolve_unfolding(spec: UnfoldingSpec) -> Callable[[Tensor], RowModes]:
    """Turn an unfolding spec into a function ``update -> row_modes``."""
    if isinstance(spec, str):
        try:
            return STRATEGIES[spec]
        except KeyError:
            raise ValueError(
                f"Unknown unfolding strategy {spec!r}. Choose one of {sorted(STRATEGIES)}, "
                "an int, a sequence of ints or a callable."
            ) from None
    if isinstance(spec, bool):
        raise TypeError("unfolding must be a string, an int, a sequence of ints or a callable.")
    if isinstance(spec, int) or (isinstance(spec, Sequence) and all(isinstance(m, int) for m in spec)):
        fixed = spec
        return lambda x: _normalize(fixed, x.ndim)
    if callable(spec):
        fn = spec
        return lambda x: _normalize(fn(x), x.ndim)
    raise TypeError("unfolding must be a string, an int, a sequence of ints or a callable.")
