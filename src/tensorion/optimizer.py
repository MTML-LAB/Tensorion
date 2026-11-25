"""The Tensorion optimizer: Muon for matrices and tensors."""

from __future__ import annotations

import math
from fnmatch import fnmatchcase
from functools import partial
from typing import Any, Callable, Iterable, Mapping, Optional, Union

import torch

from .orthogonalize import OrthogonalizerSpec, resolve_orthogonalizer
from . import unfolding as _unfolding
from .nuclear import nuclear_norm_bounds
from .unfolding import UnfoldingSpec, matricize, unmatricize

Tensor = torch.Tensor

__all__ = ["Tensorion"]

_ADJUST_LR_FNS: dict[Optional[str], Callable[[int, int], float]] = {
    # Match the update RMS of AdamW so that AdamW learning rates transfer (Moonlight).
    "match_rms_adamw": lambda rows, cols: 0.2 * math.sqrt(max(rows, cols)),
    # The scaling of the original Muon implementation.
    "original": lambda rows, cols: math.sqrt(max(1.0, rows / cols)),
    None: lambda rows, cols: 1.0,
}
_FALLBACKS = ("adamw", "sgd")


class Tensorion(torch.optim.Optimizer):
    r"""Muon for matrix *and* tensor layers.

    For every parameter of order at least ``min_order``, Tensorion keeps
    a momentum buffer, unfolds it into a matrix (see ``unfolding``), replaces
    that matrix with its polar factor :math:`UV^\top` (see ``orthogonalizer``),
    folds it back and takes a step. For a 2-D weight this is exactly Muon.
    All other parameters (biases, norms, and anything you route there) are
    updated by a fallback optimizer, AdamW by default.

    The optimizer only sees tensors, so it cannot tell an output head or an
    embedding table from any other matrix. Keep such layers on the fallback
    with ``exclude``, or build the optimizer with :meth:`from_model`, which
    routes by module type. :meth:`routing` shows the result.

    The order of a parameter is the number of its axes larger than 1. Axes of
    size 1 carry no information and are ignored throughout: a ``1 x 1``
    convolution kernel has order 2 and gets exactly the Muon update of the
    matrix it is, and a class token of shape ``(1, 1, d)`` has order 1 and
    always goes to the fallback optimizer.

    Args:
        params: parameters or parameter groups, as for any ``torch.optim`` optimizer.
            Pass ``model.named_parameters()`` to set the unfolding per layer by name.
            A group may set ``use_muon`` (or ``use_tensorion``) to ``True``/``False`` to
            force the update rule; otherwise it is chosen by the order and ``min_order``.
            Any other argument below can also be overridden per group. Groups written
            for Muon's ``MuonWithAuxAdam`` (keys ``use_muon``, ``betas``, ``eps``) work as is.
        lr: learning rate.
        weight_decay: decoupled weight decay for Tensorion and AdamW parameters,
            coupled (L2) for SGD parameters.
        momentum: momentum of the Tensorion update.
        nesterov: use Nesterov momentum for the Tensorion update.
        unfolding: how a tensor is turned into a matrix. Either a strategy for every layer,
            ``"offline"`` (the split of axes with the largest rank, fixed by the shape) or
            ``"online"`` (the split with the largest nuclear norm, chosen every step), or a
            mapping from parameter names or ``fnmatch`` patterns to a strategy, an axis or
            a tuple of axes for that layer. Unmatched layers use ``"offline"``. Each layer's
            unfolding is checked against its shape when the optimizer is built.
            ``"online"`` estimates the nuclear norms by Lanczos quadrature,
            ``"online_polar"`` takes them from the polar factors and reuses the factor
            of the winner as the update, ``"online_exact"`` uses SVDs.
            See :mod:`tensorion.unfolding`.
        online_every: with an online unfolding, choose the split again only every this
            many steps and reuse it in between. The momentum changes slowly, so this
            costs little and divides the overhead of the selection. ``1`` selects at
            every step, as in the paper.
        orthogonalizer: ``"newton_schulz"``, ``"polar_express"``, ``"cans"``, ``"svd"``
            (exact) or a callable ``f(matrix, steps) -> matrix``.
            See :mod:`tensorion.orthogonalize`.
        matmul_dtype: precision of the large matrix products of the orthogonalizer and of
            the online selection: ``None`` (the precision of the parameter), or e.g.
            ``torch.bfloat16`` or ``"bfloat16"`` as in Muon. The momentum and the
            parameters keep their own precision.
        ns_steps: number of iterations for iterative orthogonalizers. ``None`` uses each
            method's default (Newton-Schulz 5, Polar Express 5, CANS 7, about 15 matmuls each).
        adjust_lr_fn: how the learning rate is scaled by the unfolded shape
            ``(rows, cols)``: ``"match_rms_adamw"`` (``0.2 * sqrt(max(rows, cols))``),
            ``"original"`` (``sqrt(max(1, rows / cols))``) or ``None`` (no scaling).
        nuclear: scale the update by :math:`\langle G, UV^\top\rangle`, the nuclear norm
            of the unfolded update.
        min_order: parameters of at least this order (number of axes larger than 1) get
            the Tensorion update. The default ``2`` covers matrices and tensors. ``3`` keeps
            every matrix, including ``1 x 1`` convolutions, on the fallback optimizer.
        exclude: parameter names or ``fnmatch`` patterns (a string or a list) that always
            use the fallback, whatever their order, e.g. ``["fc.*", "*embed*"]`` for a
            head and embeddings. Needs ``model.named_parameters()``.
        fallback: ``"adamw"`` or ``"sgd"``, the update for all other parameters.
        adamw_betas, adamw_eps: AdamW hyperparameters for the fallback.
        sgd_momentum: SGD momentum for the fallback.

    Example::

        optimizer = Tensorion(model.parameters(), lr=1e-3)

        # A fixed unfolding for some layers, "online" for others, "offline" for the rest.
        optimizer = Tensorion(
            model.named_parameters(),
            unfolding={"conv1.weight": (0, 1), "layer4.*.weight": "online"},
        )
    """

    def __init__(
        self,
        params: Iterable[Union[Tensor, dict[str, Any]]],
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = False,
        unfolding: UnfoldingSpec = "offline",
        online_every: int = 10,
        orthogonalizer: OrthogonalizerSpec = "newton_schulz",
        ns_steps: Optional[int] = None,
        matmul_dtype: Optional[Union[torch.dtype, str]] = None,
        adjust_lr_fn: Optional[str] = "match_rms_adamw",
        nuclear: bool = False,
        min_order: int = 2,
        exclude: Union[str, Iterable[str]] = (),
        fallback: str = "adamw",
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        sgd_momentum: float = 0.95,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            unfolding=unfolding,
            online_every=online_every,
            orthogonalizer=orthogonalizer,
            ns_steps=ns_steps,
            matmul_dtype=matmul_dtype,
            adjust_lr_fn=adjust_lr_fn,
            nuclear=nuclear,
            min_order=min_order,
            exclude=exclude,
            fallback=fallback,
            adamw_betas=tuple(adamw_betas),
            adamw_eps=adamw_eps,
            sgd_momentum=sgd_momentum,
            use_tensorion=None,
        )
        self._unfoldings: dict[Tensor, Callable[[Tensor], tuple[int, ...]]] = {}
        self._online: dict[Tensor, str] = {}
        self._matched_keys: dict[int, set[str]] = {}
        self._excluded: set[Tensor] = set()
        self._matched_exclusions: set[str] = set()
        super().__init__(params, defaults)
        self._check_all_keys_matched()

    @classmethod
    def from_model(
        cls,
        model: torch.nn.Module,
        modules: tuple[type, ...] = (torch.nn.Linear, torch.nn.modules.conv._ConvNd, torch.nn.MultiheadAttention),
        head: Union[str, Iterable[str], None] = "auto",
        **kwargs: Any,
    ) -> "Tensorion":
        """Build the optimizer for ``model``, routing by module type instead of by shape.

        The weights of ``modules`` (by default linear layers, convolutions and attention
        projections) get the Tensorion update. Everything else uses the fallback:
        embedding tables, positional embeddings and other bare parameters, norms and
        biases. A weight shared with a module outside ``modules``, such as an output
        layer tied to an embedding, uses the fallback as well.

        Args:
            model: the model to optimize.
            modules: module classes whose weights get the Tensorion update.
            head: the output layer, which is kept on the fallback. ``"auto"`` takes the
                last of ``modules`` in ``model.named_modules()`` if it is a linear layer,
                ``None`` keeps no head apart, and a module name or pattern (or a list)
                such as ``"lm_head"`` names it explicitly.
            **kwargs: arguments of the constructor. ``exclude`` adds parameter-name
                patterns on top of this routing.

        Check the result with :meth:`routing`.
        """
        attention = ("in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight")
        chosen: dict[int, str] = {}  # id(weight) -> name of the module
        other: set[int] = set()
        for name, module in model.named_modules():
            if isinstance(module, modules):
                keys = attention if isinstance(module, torch.nn.MultiheadAttention) else ("weight",)
                for key in keys:
                    weight = getattr(module, key, None)
                    if isinstance(weight, Tensor) and cls._is_unfoldable(weight):
                        chosen[id(weight)] = name
            else:
                other.update(id(p) for p in module._parameters.values() if p is not None)
        if head == "auto":
            last = next(reversed(chosen.values()), None)
            patterns = [last] if last is not None and isinstance(model.get_submodule(last), torch.nn.Linear) else []
        else:
            patterns = [] if head is None else [head] if isinstance(head, str) else list(head)
            unmatched = [k for k in patterns if not any(fnmatchcase(name, k) for name in chosen.values())]
            if unmatched:
                raise ValueError(f"head {unmatched} matches no module that would get the Tensorion update.")
        chosen = {i: n for i, n in chosen.items() if i not in other and not any(fnmatchcase(n, k) for k in patterns)}
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        groups = [
            {"params": [(n, p) for n, p in named if id(p) in chosen], "use_tensorion": True},
            {"params": [(n, p) for n, p in named if id(p) not in chosen], "use_tensorion": False},
        ]
        return cls([g for g in groups if g["params"]], **kwargs)

    def routing(self) -> dict[str, str]:
        """The update rule of every parameter: ``"tensorion"``, ``"adamw"`` or ``"sgd"``.

        Keys are parameter names if the optimizer was given named parameters, and
        ``"group.index"`` otherwise.
        """
        result = {}
        for g, group in enumerate(self.param_groups):
            names = group.get("param_names")
            for i, p in enumerate(group["params"]):
                key = names[i] if names is not None else f"{g}.{i}"
                result[key] = "tensorion" if self._uses_tensorion(p, group) else group["fallback"]
        return result

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        # Accept the group keys of Muon's `MuonWithAuxAdam`.
        for alias, key in (("use_muon", "use_tensorion"), ("betas", "adamw_betas"), ("eps", "adamw_eps")):
            if alias in param_group:
                param_group.setdefault(key, param_group.pop(alias))
        super().add_param_group(param_group)
        group = self.param_groups[-1]

        # Fail at construction time rather than at the first step.
        _unfolding.check_unfolding(group["unfolding"])
        resolve_orthogonalizer(group["orthogonalizer"])
        if group["adjust_lr_fn"] not in _ADJUST_LR_FNS:
            raise ValueError(f"adjust_lr_fn must be one of {list(_ADJUST_LR_FNS)}, got {group['adjust_lr_fn']!r}")
        dtype = group["matmul_dtype"]
        if isinstance(dtype, str):
            if not isinstance(getattr(torch, dtype, None), torch.dtype):
                raise ValueError(f"matmul_dtype {dtype!r} is not a torch dtype")
            dtype = group["matmul_dtype"] = getattr(torch, dtype)
        if dtype is not None and not (isinstance(dtype, torch.dtype) and dtype.is_floating_point):
            raise ValueError(f"matmul_dtype must be a floating point torch dtype or None, got {dtype!r}")
        if not (isinstance(group["online_every"], int) and group["online_every"] >= 1):
            raise ValueError(f"online_every must be a positive integer, got {group['online_every']!r}")
        if group["fallback"] not in _FALLBACKS:
            raise ValueError(f"fallback must be one of {_FALLBACKS}, got {group['fallback']!r}")
        self._resolve_exclusions(group)
        if group["use_tensorion"]:
            for p in group["params"]:
                if p in self._excluded:
                    continue
                if not self._is_unfoldable(p):
                    raise ValueError(
                        f"A parameter of shape {tuple(p.shape)} cannot get the Tensorion update. "
                        "Only parameters with 2 or more axes larger than 1 can."
                    )
        self._resolve_unfoldings(group)

    def _resolve_exclusions(self, group: dict[str, Any]) -> None:
        """Mark the parameters of ``group`` whose names match its ``exclude`` patterns."""
        patterns = group.get("exclude", ())
        patterns = group["exclude"] = (patterns,) if isinstance(patterns, str) else tuple(patterns)
        if not patterns:
            return
        if not all(isinstance(k, str) for k in patterns):
            raise TypeError(f"exclude must be parameter names or patterns, got {patterns!r}")
        names = group.get("param_names")
        if names is None:
            raise ValueError(
                "exclude is given by parameter name, so the optimizer needs the names: "
                "pass model.named_parameters() (or named parameters in each group)."
            )
        for name, p in zip(names, group["params"]):
            keys = [k for k in patterns if fnmatchcase(name, k)]
            if keys:
                self._excluded.add(p)
                self._matched_exclusions.update(keys)

    def _resolve_unfoldings(self, group: dict[str, Any]) -> None:
        """Fix the unfolding of every Tensorion parameter in ``group``, checked against its shape."""
        spec = group["unfolding"]
        names = group.get("param_names")
        if isinstance(spec, Mapping) and names is None:
            raise ValueError(
                "unfolding is given per parameter name, so the optimizer needs the names: "
                "pass model.named_parameters() (or named parameters in each group)."
            )
        matched = self._matched_keys.setdefault(id(spec), set())
        for i, p in enumerate(group["params"]):
            if not self._uses_tensorion(p, group):
                continue
            name = names[i] if names is not None else None
            layer_spec, key = _unfolding.select(spec, name)
            if key is not None:
                matched.add(key)
            if isinstance(layer_spec, str) and layer_spec.startswith("online"):
                self._online[p] = layer_spec
            try:
                self._unfoldings[p] = _unfolding.resolve(layer_spec, p.ndim)
            except ValueError as e:
                raise ValueError(f"Parameter {name!r} of shape {tuple(p.shape)}: {e}") from None

    def _check_all_keys_matched(self) -> None:
        unmatched = sorted({k for g in self.param_groups for k in g["exclude"]} - self._matched_exclusions)
        if unmatched:
            raise ValueError(f"exclude patterns {unmatched} match no parameter (check model.named_parameters()).")
        for group in self.param_groups:
            spec = group["unfolding"]
            if isinstance(spec, Mapping):
                unmatched = [k for k in spec if k not in self._matched_keys.get(id(spec), set())]
                if unmatched:
                    raise ValueError(
                        f"unfolding keys {unmatched} match no parameter that gets the Tensorion update "
                        "(check the names in model.named_parameters(), min_order and exclude)."
                    )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self._unfoldings.clear()
        self._online.clear()
        self._matched_keys.clear()
        self._excluded.clear()
        for group in self.param_groups:
            self._resolve_exclusions(group)
            self._resolve_unfoldings(group)

    @staticmethod
    def _order(p: Tensor) -> int:
        """The number of axes larger than 1."""
        return sum(n > 1 for n in p.shape)

    @classmethod
    def _is_unfoldable(cls, p: Tensor) -> bool:
        return cls._order(p) >= 2

    def _uses_tensorion(self, p: Tensor, group: dict[str, Any]) -> bool:
        if p in self._excluded:
            return False
        flag = group["use_tensorion"]
        if flag is None:
            return self._order(p) >= max(2, group["min_order"])
        return bool(flag)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None):
        """Perform a single optimization step.

        Args:
            closure: a closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            orthogonalize = resolve_orthogonalizer(group["orthogonalizer"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                if self._uses_tensorion(p, group):
                    self._tensorion_step(p, group, self._unfoldings[p], orthogonalize)
                elif group["fallback"] == "adamw":
                    self._adamw_step(p, group)
                else:
                    self._sgd_step(p, group)

        return loss

    def _tensorion_step(self, p: Tensor, group: dict[str, Any], unfold, orthogonalize) -> None:
        g = p.grad
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(g)
        buf = state["momentum_buffer"]
        buf.mul_(group["momentum"]).add_(g)
        g = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf

        dtype = group["matmul_dtype"]
        steps = group["ns_steps"]
        orth = orthogonalize if steps is None else (lambda matrix: orthogonalize(matrix, steps))

        G = U = None
        if p not in self._online:
            row_modes = unfold(g)
        elif group["online_every"] == 1:
            row_modes, G, U = self._select_online(p, g, unfold, orth, dtype)
        else:
            # Choose the split every `online_every` steps and reuse it in between.
            age = state.get("unfolding_age", 0)
            if age % group["online_every"] == 0:
                state["row_modes"], G, U = self._select_online(p, g, unfold, orth, dtype)
            state["unfolding_age"] = age + 1
            row_modes = state["row_modes"]
        if G is None:
            G = matricize(g, row_modes)
            if dtype is not None:
                G = G.to(dtype)
        if U is None:
            U = orth(G)
        if group["nuclear"]:
            U = U * (G * U).sum()
        update = unmatricize(U, g.shape, row_modes)

        lr = group["lr"]
        adjusted_lr = lr * _ADJUST_LR_FNS[group["adjust_lr_fn"]](*G.shape)
        p.mul_(1 - lr * group["weight_decay"])
        p.add_(update, alpha=-adjusted_lr)

    def _select_online(self, p: Tensor, g: Tensor, unfold, orth, dtype):
        """Row modes of the best split, and its matricization and factor if they were computed."""
        kind = self._online[p]
        if kind == "online_polar":
            return _unfolding.select_online_polar(g, orth, dtype)
        if kind == "online" and dtype is not None:
            return _unfolding.select_online(g, bounds=partial(nuclear_norm_bounds, dtype=dtype)), None, None
        return unfold(g), None, None

    def _adamw_step(self, p: Tensor, group: dict[str, Any]) -> None:
        g = p.grad
        state = self.state[p]
        if "step" not in state:
            state["step"] = 0
            state["moment1"] = torch.zeros_like(g)
            state["moment2"] = torch.zeros_like(g)
        state["step"] += 1
        step = state["step"]
        beta1, beta2 = group["adamw_betas"]
        state["moment1"].lerp_(g, 1 - beta1)
        state["moment2"].lerp_(g.square(), 1 - beta2)

        update = state["moment1"] / (group["adamw_eps"] + state["moment2"].sqrt())
        scale = (1 - beta1**step) / (1 - beta2**step) ** 0.5
        lr = group["lr"]
        p.mul_(1 - lr * group["weight_decay"])
        p.add_(update, alpha=-lr / scale)

    def _sgd_step(self, p: Tensor, group: dict[str, Any]) -> None:
        g = p.grad
        state = self.state[p]
        if "step" not in state:
            state["step"] = 0
            state["momentum"] = torch.zeros_like(g)
        state["step"] += 1

        if group["weight_decay"]:
            g = g.add(p, alpha=group["weight_decay"])
        if group["sgd_momentum"]:
            state["momentum"].mul_(group["sgd_momentum"]).add_(g)
            g = state["momentum"]
        p.add_(g, alpha=-group["lr"])
