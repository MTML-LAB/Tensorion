"""The Tensorion optimizer: Muon for matrices and tensors."""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Mapping, Optional, Union

import torch

from .orthogonalize import OrthogonalizerSpec, resolve_orthogonalizer
from . import unfolding as _unfolding
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

    For every parameter with at least ``min_ndim`` dimensions, Tensorion keeps
    a momentum buffer, unfolds it into a matrix (see ``unfolding``), replaces
    that matrix with its polar factor :math:`UV^\top` (see ``orthogonalizer``),
    folds it back and takes a step. For a 2-D weight this is exactly Muon.
    All other parameters (biases, norms, and anything you route there) are
    updated by a fallback optimizer, AdamW by default.

    Args:
        params: parameters or parameter groups, as for any ``torch.optim`` optimizer.
            Pass ``model.named_parameters()`` to set the unfolding per layer by name.
            A group may set ``use_muon`` (or ``use_tensorion``) to ``True``/``False`` to
            force the update rule; otherwise it is chosen by ``p.ndim >= min_ndim``.
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
            ``"online"`` estimates the nuclear norms without an SVD, ``"online_exact"``
            uses SVDs. See :mod:`tensorion.unfolding`.
        online_every: with an online unfolding, choose the split again only every this
            many steps and reuse it in between. The momentum changes slowly, so this
            costs little and divides the overhead of the selection. ``1`` selects at
            every step, as in the paper.
        orthogonalizer: ``"newton_schulz"``, ``"polar_express"``, ``"cans"``, ``"svd"``
            (exact) or a callable ``f(matrix, steps) -> matrix``.
            See :mod:`tensorion.orthogonalize`.
        ns_steps: number of iterations for iterative orthogonalizers. ``None`` uses each
            method's default (Newton-Schulz 5, Polar Express 5, CANS 7, about 15 matmuls each).
        adjust_lr_fn: how the learning rate is scaled by the unfolded shape
            ``(rows, cols)``: ``"match_rms_adamw"`` (``0.2 * sqrt(max(rows, cols))``),
            ``"original"`` (``sqrt(max(1, rows / cols))``) or ``None`` (no scaling).
        nuclear: scale the update by :math:`\langle G, UV^\top\rangle`, the nuclear norm
            of the unfolded update.
        min_ndim: parameters with at least this many dimensions get the Tensorion update.
            Use ``3`` to keep 2-D layers on the fallback optimizer.
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
        adjust_lr_fn: Optional[str] = "match_rms_adamw",
        nuclear: bool = False,
        min_ndim: int = 2,
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
            adjust_lr_fn=adjust_lr_fn,
            nuclear=nuclear,
            min_ndim=min_ndim,
            fallback=fallback,
            adamw_betas=tuple(adamw_betas),
            adamw_eps=adamw_eps,
            sgd_momentum=sgd_momentum,
            use_tensorion=None,
        )
        self._unfoldings: dict[Tensor, Callable[[Tensor], tuple[int, ...]]] = {}
        self._online: set[Tensor] = set()
        self._matched_keys: dict[int, set[str]] = {}
        super().__init__(params, defaults)
        self._check_all_keys_matched()

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
        if not (isinstance(group["online_every"], int) and group["online_every"] >= 1):
            raise ValueError(f"online_every must be a positive integer, got {group['online_every']!r}")
        if group["fallback"] not in _FALLBACKS:
            raise ValueError(f"fallback must be one of {_FALLBACKS}, got {group['fallback']!r}")
        if group["use_tensorion"]:
            for p in group["params"]:
                if p.ndim < 2:
                    raise ValueError(
                        f"A parameter of shape {tuple(p.shape)} cannot get the Tensorion update. "
                        "Only parameters with 2 or more dimensions can."
                    )
        self._resolve_unfoldings(group)

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
                self._online.add(p)
            try:
                self._unfoldings[p] = _unfolding.resolve(layer_spec, p.ndim)
            except ValueError as e:
                raise ValueError(f"Parameter {name!r} of shape {tuple(p.shape)}: {e}") from None

    def _check_all_keys_matched(self) -> None:
        for group in self.param_groups:
            spec = group["unfolding"]
            if isinstance(spec, Mapping):
                unmatched = [k for k in spec if k not in self._matched_keys.get(id(spec), set())]
                if unmatched:
                    raise ValueError(
                        f"unfolding keys {unmatched} match no parameter that gets the Tensorion update "
                        "(check the names in model.named_parameters() and min_ndim)."
                    )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self._unfoldings.clear()
        self._online.clear()
        self._matched_keys.clear()
        for group in self.param_groups:
            self._resolve_unfoldings(group)

    @staticmethod
    def _uses_tensorion(p: Tensor, group: dict[str, Any]) -> bool:
        flag = group["use_tensorion"]
        return p.ndim >= group["min_ndim"] if flag is None else bool(flag)

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

        every = group["online_every"]
        if every > 1 and p in self._online:
            # Reuse the split chosen at the last selection step.
            age = state.get("unfolding_age", 0)
            if age % every == 0:
                state["row_modes"] = unfold(g)
            state["unfolding_age"] = age + 1
            row_modes = state["row_modes"]
        else:
            row_modes = unfold(g)
        G = matricize(g, row_modes)
        steps = group["ns_steps"]
        U = orthogonalize(G) if steps is None else orthogonalize(G, steps)
        if group["nuclear"]:
            U = U * (G * U).sum()
        update = unmatricize(U, g.shape, row_modes)

        lr = group["lr"]
        adjusted_lr = lr * _ADJUST_LR_FNS[group["adjust_lr_fn"]](*G.shape)
        p.mul_(1 - lr * group["weight_decay"])
        p.add_(update, alpha=-adjusted_lr)

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
