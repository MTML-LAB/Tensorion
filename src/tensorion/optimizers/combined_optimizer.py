from typing import Iterable, Sequence, Tuple, Any

import torch
from torch.optim import Optimizer
from hydra.utils import instantiate


__all__ = ["CombinedOptimizer"]


def _looks_like_named_param(item: Any) -> bool:
    return (
        isinstance(item, tuple)
        and len(item) == 2
        and isinstance(item[0], str)
        and hasattr(item[1], "requires_grad")
    )


def _resolve_named_parameters(
    params: Iterable[Any] | None,
    named_parameters: Iterable[Tuple[str, torch.Tensor]] | None,
) -> list[Tuple[str, torch.Tensor]]:
    if named_parameters is not None:
        return list(named_parameters)

    if params is None:
        raise ValueError(
            "CombinedOptimizer requires named_parameters or params as (name, param) tuples."
        )

    params_list = list(params)
    if params_list and _looks_like_named_param(params_list[0]):
        return list(params_list)

    raise ValueError("CombinedOptimizer requires named_parameters to select by name.")


def _build_matcher_re(patterns: Sequence[str]):
    import re

    compiled = [re.compile(p) for p in patterns]

    def _match(name: str) -> bool:
        return any(p.search(name) for p in compiled)

    return _match


def _build_matcher_glob(patterns: Sequence[str]):
    from fnmatch import fnmatchcase

    def _match(name: str) -> bool:
        return any(fnmatchcase(name, p) for p in patterns)

    return _match


def _build_matcher(match_type: str, patterns: Sequence[str]):
    if match_type == "regex":
        return _build_matcher_re(patterns)

    if match_type == "glob":
        return _build_matcher_glob(patterns)

    raise ValueError(f"Unknown selectors.match_type: {match_type}")


def _partition_parameters(
    named_params: Sequence[Tuple[str, torch.Tensor]],
    selectors: dict[str, Any],
) -> tuple[list[Tuple[str, torch.Tensor]], list[Tuple[str, torch.Tensor]]]:
    include = list(selectors.get("include", []))
    exclude = list(selectors.get("exclude", []))
    match_type = selectors.get("match_type", "regex")

    if not include:
        raise ValueError("selectors.include must contain at least one pattern.")

    include_match = _build_matcher(match_type, include)
    exclude_match = _build_matcher(match_type, exclude) if exclude else None

    selected: list[Tuple[str, torch.Tensor]] = []
    remaining: list[Tuple[str, torch.Tensor]] = []
    seen: dict[int, str] = {}

    for name, param in named_params:
        if not param.requires_grad:
            continue

        param_id = id(param)
        if param_id in seen:
            raise ValueError(
                "Parameter appears multiple times in named_parameters: "
                f"{seen[param_id]} and {name}."
            )

        is_selected = include_match(name) and not (
            exclude_match(name) if exclude_match is not None else False
        )
        if is_selected:
            selected.append((name, param))
        else:
            remaining.append((name, param))
        seen[param_id] = name

    if not selected:
        raise ValueError("selectors.include did not match any trainable parameters.")
    if not remaining:
        raise ValueError(
            "selectors.include matched all trainable parameters; "
            "use a single optimizer instead."
        )

    return selected, remaining


class CombinedOptimizer(Optimizer):
    def __init__(
        self,
        params: Iterable[Any],
        optimizer1: dict[str, Any],
        optimizer2: dict[str, Any],
        selectors: dict[str, Any],
        named_parameters: Iterable[Tuple[str, torch.Tensor]] | None = None,
    ) -> None:
        named_params = _resolve_named_parameters(params, named_parameters)
        selected, remaining = _partition_parameters(named_params, selectors)

        selected_params = [param for _name, param in selected]
        remaining_params = [param for _name, param in remaining]
        all_params = selected_params + remaining_params
        if not all_params:
            raise ValueError("No trainable parameters found for optimization.")

        super().__init__(all_params, defaults={})

        self.optimizer1 = instantiate(optimizer1, params=selected_params)
        self.optimizer2 = instantiate(optimizer2, params=remaining_params)
        self.param_groups = self.optimizer1.param_groups + self.optimizer2.param_groups

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.optimizer1.zero_grad(set_to_none=set_to_none)
        self.optimizer2.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.optimizer1.step()
        self.optimizer2.step()
        return loss

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizer1": self.optimizer1.state_dict(),
            "optimizer2": self.optimizer2.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if "optimizer1" not in state_dict or "optimizer2" not in state_dict:
            raise ValueError("CombinedOptimizer state_dict is missing optimizer keys.")

        self.optimizer1.load_state_dict(state_dict["optimizer1"])
        self.optimizer2.load_state_dict(state_dict["optimizer2"])
        self.param_groups = self.optimizer1.param_groups + self.optimizer2.param_groups
