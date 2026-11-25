"""Tensorion as optax gradient transformations."""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
import optax
from jax import lax

from . import _unfolding
from ._linalg import (
    Orthogonalizer,
    matricize,
    nuclear_norm_bounds,
    nuclear_norm_exact,
    resolve_orthogonalizer,
    unmatricize,
)
from ._unfolding import RowModes, UnfoldingSpec

Array = jax.Array

_ADJUST_LR_FNS: dict[Optional[str], Callable[[int, int], float]] = {
    "match_rms_adamw": lambda rows, cols: 0.2 * math.sqrt(max(rows, cols)),
    "original": lambda rows, cols: math.sqrt(max(1.0, rows / cols)),
    None: lambda rows, cols: 1.0,
}
_BOUNDS = {"online": nuclear_norm_bounds, "online_exact": nuclear_norm_exact}


class ScaleByTensorionState(NamedTuple):
    """State of :func:`scale_by_tensorion`."""

    count: Array  # number of updates so far
    momentum: Any  # momentum buffers, same structure as the parameters
    choice: Any  # per parameter: index of the split in use (online unfoldings)


def _order(shape) -> int:
    """The number of axes larger than 1."""
    return sum(n > 1 for n in shape)


def _is_unfoldable(shape) -> bool:
    """Without two axes larger than 1 the parameter is a vector."""
    return _order(shape) >= 2


class _Plan(NamedTuple):
    """How one parameter is unfolded: the splits to choose from and how to choose."""

    splits: tuple[RowModes, ...]
    ranks: tuple[int, ...]
    bounds: Optional[Callable[[Array], tuple[Array, Array]]]  # None: always the first split


def _path(key_path) -> str:
    return jax.tree_util.keystr(key_path, simple=True, separator="/")


def _plans(tree: Any, unfolding: UnfoldingSpec) -> tuple[list[_Plan], Any]:
    """Resolve the unfolding of every leaf of ``tree``, checked against its shape."""
    leaves, treedef = jax.tree_util.tree_flatten_with_path(tree)
    matched = set()
    plans = []
    for key_path, leaf in leaves:
        path = _path(key_path)
        shape = tuple(leaf.shape)
        if not _is_unfoldable(shape):
            raise ValueError(
                f"Parameter {path!r} of shape {shape} cannot get the Tensorion update. "
                "Only parameters with 2 or more axes larger than 1 can."
            )
        spec, key = _unfolding.select(unfolding, path)
        if key is not None:
            matched.add(key)
        if spec == "offline":
            plans.append(_Plan((_unfolding.offline_rows(shape),), (0,), None))
        elif isinstance(spec, str):
            splits, ranks = zip(*_unfolding.candidates(shape))
            plans.append(_Plan(splits, ranks, _BOUNDS[spec] if len(splits) > 1 else None))
        else:
            try:
                plans.append(_Plan((_unfolding.normalize(spec, len(shape)),), (0,), None))
            except ValueError as e:
                raise ValueError(f"Parameter {path!r} of shape {shape}: {e}") from None
    if isinstance(unfolding, Mapping):
        unmatched = [k for k in unfolding if k not in matched]
        if unmatched:
            paths = [_path(key_path) for key_path, _ in leaves]
            raise ValueError(
                f"unfolding keys {unmatched} match no parameter that gets the Tensorion update. "
                f"The parameter paths are {paths}."
            )
    return plans, treedef


def _select_online(x: Array, plan: _Plan) -> Array:
    """Index of the split of ``x`` with the largest (estimated) nuclear norm.

    Splits are visited from the largest rank bound down. Since
    ``|M|_* <= sqrt(rank) |M|_F`` and the Frobenius norm is the same for every
    split, the search stops once this bound is below the lower value of the
    best split so far. Under ``jit`` the skipped splits are not computed.
    """
    frobenius = jnp.linalg.norm(x)
    best = jnp.zeros([], jnp.int32)
    best_value = best_lower = jnp.asarray(-1.0, x.dtype)
    active = jnp.asarray(True)
    for i, (rows, rank) in enumerate(zip(plan.splits, plan.ranks)):
        active = active & (math.sqrt(rank) * frobenius > best_lower)

        def evaluate(rows=rows):
            lower, upper = plan.bounds(matricize(x, rows))
            return lower.astype(x.dtype), ((lower + upper) / 2).astype(x.dtype)

        def skip():
            return jnp.asarray(-jnp.inf, x.dtype), jnp.asarray(-jnp.inf, x.dtype)

        lower, value = lax.cond(active, evaluate, skip)
        better = value > best_value
        best = jnp.where(better, i, best).astype(jnp.int32)
        best_value = jnp.where(better, value, best_value)
        best_lower = jnp.where(better, lower, best_lower)
    return best


def scale_by_tensorion(
    momentum: float = 0.95,
    nesterov: bool = False,
    unfolding: UnfoldingSpec = "offline",
    online_every: int = 10,
    orthogonalizer: Union[str, Orthogonalizer] = "newton_schulz",
    ns_steps: Optional[int] = None,
    adjust_lr_fn: Optional[str] = "match_rms_adamw",
    nuclear: bool = False,
) -> optax.GradientTransformation:
    """Replace every update by the orthogonalized, folded-back momentum.

    Every parameter this transformation sees must have at least 2 axes larger
    than 1 (axes of size 1 are ignored when a tensor is unfolded). Use
    :func:`tensorion` to route the other parameters to a fallback optimizer.

    Args:
        momentum: momentum coefficient.
        nesterov: use Nesterov momentum.
        unfolding: ``"offline"``, ``"online"``, ``"online_exact"``, or a mapping
            from parameter paths (keys joined by ``/``, e.g. ``"conv1/kernel"``) or
            ``fnmatch`` patterns to a strategy, an axis or a tuple of axes.
            Unmatched parameters use ``"offline"``.
        online_every: with an online unfolding, choose the split every this many steps.
        orthogonalizer: ``"newton_schulz"``, ``"polar_express"``, ``"cans"``, ``"svd"``
            or a callable ``f(matrix, steps) -> matrix``.
        ns_steps: iterations of iterative orthogonalizers, ``None`` for each method's default.
        adjust_lr_fn: scaling by the unfolded shape: ``"match_rms_adamw"``,
            ``"original"`` or ``None``.
        nuclear: scale the update by the nuclear norm of the unfolded momentum.
    """
    _unfolding.check_unfolding(unfolding)
    orthogonalize = resolve_orthogonalizer(orthogonalizer)
    if adjust_lr_fn not in _ADJUST_LR_FNS:
        raise ValueError(f"adjust_lr_fn must be one of {list(_ADJUST_LR_FNS)}, got {adjust_lr_fn!r}")
    if not (isinstance(online_every, int) and online_every >= 1):
        raise ValueError(f"online_every must be a positive integer, got {online_every!r}")
    adjust = _ADJUST_LR_FNS[adjust_lr_fn]

    def direction(rows: RowModes) -> Callable[[Array], Array]:
        def apply(m: Array) -> Array:
            G = matricize(m, rows)
            U = orthogonalize(G) if ns_steps is None else orthogonalize(G, ns_steps)
            if nuclear:
                U = U * (G * U).sum()
            return unmatricize(U, m.shape, rows) * adjust(*G.shape)

        return apply

    def init_fn(params):
        _plans(params, unfolding)  # fail early on a bad unfolding
        return ScaleByTensorionState(
            count=jnp.zeros([], jnp.int32),
            momentum=jax.tree.map(jnp.zeros_like, params),
            choice=jax.tree.map(lambda p: jnp.zeros([], jnp.int32), params),
        )

    def update_fn(updates, state, params=None):
        del params
        plans, treedef = _plans(updates, unfolding)
        new_updates, new_momentum, new_choice = [], [], []
        for g, buf, choice, plan in zip(
            treedef.flatten_up_to(updates),
            treedef.flatten_up_to(state.momentum),
            treedef.flatten_up_to(state.choice),
            plans,
        ):
            buf = momentum * buf + g
            m = g + momentum * buf if nesterov else buf
            if plan.bounds is None:
                update = direction(plan.splits[0])(m)
            else:
                if online_every == 1:
                    choice = _select_online(m, plan)
                else:
                    choice = lax.cond(
                        state.count % online_every == 0,
                        lambda m=m, plan=plan: _select_online(m, plan),
                        lambda choice=choice: choice,
                    )
                update = lax.switch(choice, [direction(rows) for rows in plan.splits], m)
            new_updates.append(update)
            new_momentum.append(buf)
            new_choice.append(choice)
        new_state = ScaleByTensorionState(
            count=optax.safe_increment(state.count),
            momentum=treedef.unflatten(new_momentum),
            choice=treedef.unflatten(new_choice),
        )
        return treedef.unflatten(new_updates), new_state

    return optax.GradientTransformation(init_fn, update_fn)


def tensorion(
    learning_rate: optax.ScalarOrSchedule = 1e-3,
    weight_decay: float = 0.1,
    momentum: float = 0.95,
    nesterov: bool = False,
    unfolding: UnfoldingSpec = "offline",
    online_every: int = 10,
    orthogonalizer: Union[str, Orthogonalizer] = "newton_schulz",
    ns_steps: Optional[int] = None,
    adjust_lr_fn: Optional[str] = "match_rms_adamw",
    nuclear: bool = False,
    min_order: int = 2,
    fallback: str = "adamw",
    adamw_b1: float = 0.9,
    adamw_b2: float = 0.95,
    adamw_eps: float = 1e-8,
    sgd_momentum: float = 0.95,
    param_labels: Optional[Union[Any, Callable[[Any], Any]]] = None,
) -> optax.GradientTransformation:
    """Muon for matrix and tensor layers, with a fallback for everything else.

    Parameters of order at least ``min_order`` get the Tensorion update (see
    :func:`scale_by_tensorion`), the others AdamW or SGD. The order is the number
    of axes larger than 1: a ``1 x 1`` convolution kernel has order 2, and a
    class token ``(1, 1, d)`` has order 1 and always uses the fallback. The arguments and their defaults are
    those of the PyTorch ``tensorion.Tensorion``.

    Args:
        learning_rate: a scalar or an optax schedule.
        weight_decay: decoupled weight decay for Tensorion and AdamW parameters,
            coupled (L2) for SGD parameters.
        min_order: parameters of at least this order get the Tensorion update. ``3`` keeps
            every matrix, including ``1 x 1`` convolutions, on the fallback.
        fallback: ``"adamw"`` or ``"sgd"``.
        adamw_b1, adamw_b2, adamw_eps: AdamW fallback hyperparameters.
        sgd_momentum: SGD fallback momentum.
        param_labels: optional pytree (or function of the parameters) with the label
            ``"tensorion"`` or ``"fallback"`` per parameter, overriding ``min_order``.

    Example::

        optimizer = tensorion_jax.tensorion(1e-3, unfolding={"conv1/kernel": (0, 1)})
        state = optimizer.init(params)
        updates, state = optimizer.update(grads, state, params)
        params = optax.apply_updates(params, updates)
    """
    if fallback == "adamw":
        fallback_tx = optax.adamw(learning_rate, b1=adamw_b1, b2=adamw_b2, eps=adamw_eps, weight_decay=weight_decay)
    elif fallback == "sgd":
        fallback_tx = optax.chain(
            optax.add_decayed_weights(weight_decay) if weight_decay else optax.identity(),
            optax.trace(decay=sgd_momentum) if sgd_momentum else optax.identity(),
            optax.scale_by_learning_rate(learning_rate),
        )
    else:
        raise ValueError(f"fallback must be 'adamw' or 'sgd', got {fallback!r}")

    tensorion_tx = optax.chain(
        scale_by_tensorion(
            momentum=momentum,
            nesterov=nesterov,
            unfolding=unfolding,
            online_every=online_every,
            orthogonalizer=orthogonalizer,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
            nuclear=nuclear,
        ),
        optax.add_decayed_weights(weight_decay),
        optax.scale_by_learning_rate(learning_rate),
    )

    if param_labels is None:

        def param_labels(params):
            def label(p):
                return "tensorion" if _order(jnp.shape(p)) >= max(2, min_order) else "fallback"

            return jax.tree.map(label, params)

    return optax.multi_transform({"tensorion": tensorion_tx, "fallback": fallback_tx}, param_labels)


def offline_dimension_numbers(params: Any, min_order: int = 2) -> Any:
    """The offline unfolding of every parameter, as ``optax.contrib.MuonDimensionNumbers``.

    This lets the stock optax Muon use the offline Tensorion unfolding::

        optax.contrib.muon(1e-3, muon_weight_dimension_numbers=offline_dimension_numbers)

    Parameters of order below ``min_order`` get ``None``, which Muon
    hands to its Adam fallback.
    """

    def numbers(p):
        if _order(jnp.shape(p)) < max(2, min_order):
            return None
        rows = _unfolding.offline_rows(tuple(jnp.shape(p)))
        columns = tuple(a for a in range(jnp.ndim(p)) if a not in rows)
        return optax.contrib.MuonDimensionNumbers(reduction_axis=rows, output_axis=columns)

    return jax.tree.map(numbers, params)
