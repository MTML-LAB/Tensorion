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
  for the current update, recomputed every step. The nuclear norms are
  estimated without an SVD (see :mod:`tensorion.nuclear`), and most splits are
  ruled out by a bound before anything is computed (see :func:`select_online`).
* ``"online_exact"`` -- the same choice with exact nuclear norms from SVDs,
  as in the ablation of the paper. Slower.
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

from .nuclear import nuclear_norm_bounds, nuclear_norm_exact

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
    "select_online",
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
def _candidates(shape: tuple[int, ...]) -> tuple[tuple[RowModes, int], ...]:
    """The distinct two-way splits of ``shape`` with their rank bounds, largest rank first.

    Splits that differ only in axes of size 1 give the same matrix, so only the
    first of them is kept. A split with a single row or column has the smallest
    possible nuclear norm, so it is dropped when there is another one. A 1x1
    convolution is left with a single unfolding.
    """
    numel = math.prod(shape)
    real = frozenset(m for m, n in enumerate(shape) if n > 1)
    seen = set()
    candidates = []
    for rows in two_way_splits(len(shape)):
        part = frozenset(rows) & real
        key = frozenset((part, real - part))
        if key in seen:
            continue
        seen.add(key)
        r = math.prod(shape[m] for m in rows)
        candidates.append((rows, min(r, numel // r)))
    candidates.sort(key=lambda c: -c[1])
    return tuple(c for i, c in enumerate(candidates) if i == 0 or c[1] > 1)


def _offline(x: Tensor) -> RowModes:
    return _candidates(tuple(x.shape))[0][0]


def select_online(x: Tensor, bounds: Callable[[Tensor], tuple[float, float]] = nuclear_norm_bounds) -> RowModes:
    """The split of ``x`` with the largest nuclear norm.

    ``bounds(matrix)`` returns a lower and an upper value of the nuclear norm
    of a matricization, and their mean ranks the splits.

    The Frobenius norm is the same for every matricization and
    ``|M|_* <= sqrt(rank) |M|_F``. Splits are therefore visited from the largest
    rank bound down (the offline choice first), and the search stops as soon as
    this bound of the next split is below the lower value of the best one so
    far. The remaining splits are never formed.
    """
    candidates = _candidates(tuple(x.shape))
    best = candidates[0][0]
    if len(candidates) == 1:
        return best
    frobenius = torch.linalg.vector_norm(x.float()).item()
    best_value = best_lower = -1.0
    for rows, rank in candidates:
        if math.sqrt(rank) * frobenius <= best_lower:
            break
        lower, upper = bounds(matricize(x, rows))
        value = (lower + upper) / 2
        if value > best_value:
            best, best_value, best_lower = rows, value, lower
    return best


def _exact_bounds(M: Tensor) -> tuple[float, float]:
    value = nuclear_norm_exact(M)
    return value, value


STRATEGIES: dict[str, Callable[[Tensor], RowModes]] = {
    "offline": _offline,
    "online": select_online,
    "online_exact": partial(select_online, bounds=_exact_bounds),
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
