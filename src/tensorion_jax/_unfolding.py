"""Unfoldings of a tensor and the per-layer unfolding spec (shape logic only, no arrays).

This mirrors :mod:`tensorion.unfolding` and is kept free of PyTorch so that the
JAX implementation can be used, or moved elsewhere, on its own.
"""

from __future__ import annotations

import fnmatch
import itertools
import math
from functools import lru_cache
from typing import Mapping, Sequence, Union

RowModes = tuple[int, ...]
LayerUnfolding = Union[str, int, Sequence[int]]
UnfoldingSpec = Union[str, Mapping[str, LayerUnfolding]]
STRATEGIES = ("offline", "online", "online_exact")
DEFAULT = "offline"


def two_way_splits(ndim: int) -> list[RowModes]:
    """All splits of ``range(ndim)`` into two non-empty groups, each split listed once."""
    splits: list[RowModes] = []
    for k in range(1, ndim // 2 + 1):
        for rows in itertools.combinations(range(ndim), k):
            if 2 * k == ndim and 0 not in rows:
                continue
            splits.append(tuple(rows))
    return splits


@lru_cache(maxsize=None)
def candidates(shape: tuple[int, ...]) -> tuple[tuple[RowModes, int], ...]:
    """The distinct two-way splits of ``shape`` with their rank bounds, largest rank first.

    Splits that differ only in axes of size 1 give the same matrix, so only the
    first of them is kept. A split with a single row or column has the smallest
    possible nuclear norm, so it is dropped when there is another one.
    """
    numel = math.prod(shape)
    real = frozenset(m for m, n in enumerate(shape) if n > 1)
    seen = set()
    found = []
    for rows in two_way_splits(len(shape)):
        part = frozenset(rows) & real
        key = frozenset((part, real - part))
        if key in seen:
            continue
        seen.add(key)
        r = math.prod(shape[m] for m in rows)
        found.append((rows, min(r, numel // r)))
    found.sort(key=lambda c: -c[1])
    return tuple(c for i, c in enumerate(found) if i == 0 or c[1] > 1)


def offline_rows(shape: tuple[int, ...]) -> RowModes:
    """The split of the axes whose matricization has the largest possible rank."""
    return candidates(tuple(shape))[0][0]


def normalize(row_modes: Union[int, Sequence[int]], ndim: int) -> RowModes:
    if isinstance(row_modes, int):
        row_modes = (row_modes,)
    row_modes = tuple(int(m) for m in row_modes)
    if any(not -ndim <= m < ndim for m in row_modes):
        raise ValueError(f"unfolding axes {row_modes} are out of range for a {ndim}-D tensor")
    modes = tuple(m % ndim for m in row_modes)
    if len(set(modes)) != len(modes):
        raise ValueError(f"unfolding axes {row_modes} contain duplicates")
    if not 0 < len(modes) < ndim:
        raise ValueError(f"unfolding axes {row_modes} must leave at least one row and one column axis of a {ndim}-D tensor")
    return modes


def _is_axes(spec: object) -> bool:
    if isinstance(spec, bool):
        return False
    if isinstance(spec, int):
        return True
    return isinstance(spec, Sequence) and not isinstance(spec, str) and all(
        isinstance(m, int) and not isinstance(m, bool) for m in spec
    )


def _check_layer(spec: object, where: str) -> None:
    if isinstance(spec, str):
        if spec not in STRATEGIES:
            raise ValueError(f"{where}: unknown unfolding strategy {spec!r}, choose one of {list(STRATEGIES)}")
    elif not _is_axes(spec):
        raise TypeError(f"{where}: expected {list(STRATEGIES)}, an axis or a tuple of axes, got {spec!r}")


def check_unfolding(spec: object) -> None:
    """Validate an ``unfolding`` argument without looking at the parameters."""
    if isinstance(spec, str):
        _check_layer(spec, "unfolding")
    elif isinstance(spec, Mapping):
        for key, value in spec.items():
            if not isinstance(key, str):
                raise TypeError(f"unfolding keys must be parameter paths or patterns, got {key!r}")
            _check_layer(value, f"unfolding[{key!r}]")
    elif _is_axes(spec):
        raise TypeError(
            "A fixed unfolding depends on the layer shape, so give it per parameter, "
            "e.g. unfolding={'conv1/kernel': (0, 1)}."
        )
    else:
        raise TypeError(f"unfolding must be one of {list(STRATEGIES)} or a mapping from parameter paths, got {spec!r}")


def select(spec: UnfoldingSpec, path: str) -> tuple[LayerUnfolding, str | None]:
    """Pick the unfolding for the parameter at ``path``.

    For a mapping, an exact path wins, then the first matching ``fnmatch``
    pattern in insertion order. Unmatched parameters use ``"offline"``.
    Returns the spec and the key it came from (``None`` if not from a key).
    """
    if isinstance(spec, str):
        return spec, None
    if path in spec:
        return spec[path], path
    for key, value in spec.items():
        if fnmatch.fnmatchcase(path, key):
            return value, key
    return DEFAULT, None
