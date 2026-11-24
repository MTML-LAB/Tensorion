"""Tensor unfoldings (matricizations) and how to choose one per layer.

A tensor ``x`` of shape ``(n_0, ..., n_{d-1})`` is turned into a matrix by
choosing a set of *row modes*: the axes in that set are flattened into rows,
the remaining axes into columns. The mode-``k`` unfolding is the special case
``row_modes = (k,)``.

The unfolding of a layer is one of

* ``"offline"`` -- the split of the axes into two groups whose matricization
  has the largest possible rank, i.e. maximizes ``min(rows, cols)``. It depends
  only on the shape, so it is fixed per layer. This is the default.
* ``"online"``  -- the split whose matricization has the largest nuclear norm
  for the current update. It is recomputed every step and costs one SVD per split.
* an ``int`` -- the mode-``k`` unfolding of that layer (negative indices allowed).
* a tuple / list of ints -- the row modes of that layer.

A fixed unfolding only makes sense for a particular layer shape, so the
optimizer takes it per parameter: ``unfolding`` is either a strategy name for
all layers, or a mapping from parameter names (or ``fnmatch`` patterns such as
``"layer*.conv2.weight"``) to one of the specs above. See :func:`select`.
"""

from __future__ import annotations

import fnmatch
import itertools
import math
from functools import lru_cache, partial
from typing import Callable, Mapping, Sequence, Union

import torch

Tensor = torch.Tensor
RowModes = tuple[int, ...]
LayerUnfolding = Union[str, int, Sequence[int]]
UnfoldingSpec = Union[str, Mapping[str, LayerUnfolding]]
DEFAULT = "offline"

__all__ = [
    "matricize",
    "unmatricize",
    "two_way_splits",
    "STRATEGIES",
    "check_unfolding",
    "select",
    "resolve",
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


@lru_cache(maxsize=None)
def _offline_for_shape(shape: tuple[int, ...]) -> RowModes:
    numel = math.prod(shape)

    def rank_bound(rows: RowModes) -> int:
        r = math.prod(shape[m] for m in rows)
        return min(r, numel // r)

    return max(two_way_splits(len(shape)), key=rank_bound)


def _offline(x: Tensor) -> RowModes:
    return _offline_for_shape(tuple(x.shape))


def _online(x: Tensor) -> RowModes:
    def nuclear_norm(rows: RowModes) -> float:
        return torch.linalg.svdvals(matricize(x, rows).float()).sum().item()

    return max(two_way_splits(x.ndim), key=nuclear_norm)


STRATEGIES: dict[str, Callable[[Tensor], RowModes]] = {
    "offline": _offline,
    "online": _online,
}


def _normalize(row_modes: Union[int, Sequence[int]], ndim: int) -> RowModes:
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


def _fixed(modes: RowModes, x: Tensor) -> RowModes:
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
            raise ValueError(f"{where}: unknown unfolding strategy {spec!r}, choose one of {sorted(STRATEGIES)}")
    elif not _is_axes(spec):
        raise TypeError(f"{where}: expected {sorted(STRATEGIES)}, an axis or a tuple of axes, got {spec!r}")


def check_unfolding(spec: object) -> None:
    """Validate an ``unfolding`` argument without looking at the parameters."""
    if isinstance(spec, str):
        _check_layer(spec, "unfolding")
    elif isinstance(spec, Mapping):
        for key, value in spec.items():
            if not isinstance(key, str):
                raise TypeError(f"unfolding keys must be parameter names or patterns, got {key!r}")
            _check_layer(value, f"unfolding[{key!r}]")
    elif _is_axes(spec):
        raise TypeError(
            "A fixed unfolding depends on the layer shape, so give it per parameter, e.g. "
            f"unfolding={{'conv1.weight': {tuple(spec) if not isinstance(spec, int) else spec!r}}} "
            "with model.named_parameters()."
        )
    else:
        raise TypeError(f"unfolding must be one of {sorted(STRATEGIES)} or a mapping from parameter names, got {spec!r}")


def select(spec: UnfoldingSpec, name: str | None) -> tuple[LayerUnfolding, str | None]:
    """Pick the unfolding for the parameter called ``name``.

    For a mapping, an exact name wins, then the first matching ``fnmatch``
    pattern in insertion order. Unmatched parameters use ``"offline"``.
    Returns the spec and the key it came from (``None`` if not from a key).
    """
    if isinstance(spec, str):
        return spec, None
    if name is not None:
        if name in spec:
            return spec[name], name
        for key, value in spec.items():
            if fnmatch.fnmatchcase(name, key):
                return value, key
    return DEFAULT, None


def resolve(spec: LayerUnfolding, ndim: int) -> Callable[[Tensor], RowModes]:
    """Turn a single layer's spec into a function ``update -> row_modes``, checked against ``ndim``."""
    if isinstance(spec, str):
        return STRATEGIES[spec]
    return partial(_fixed, _normalize(spec, ndim))
