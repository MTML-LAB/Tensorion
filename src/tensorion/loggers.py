import torch
from torch import nn
from typing import Optional, Callable, Any

from tensorion import tensor_utils

Tensor = torch.Tensor
MetricsCallback = Optional[Callable[[Tensor, int, int], None]]
MetricsCallbackFactory = Callable[[Tensor], tuple[MetricsCallback, dict[str, Any]]]


def _init_history(x0: Tensor, metrics: Optional[list[str]]) -> dict[str, Any]:
    num_dims = x0.dim()

    history: dict[str, Any] = {
        "max_sigvals": [[] for _ in range(num_dims)],
        "min_sigvals": [[] for _ in range(num_dims)],
        "dot": [],
        "res_norm": [],
        "delta_cycle": [],
    }

    return history


def _calc_min_sigvals(
    x: Tensor, i: int, step: int, x0: Tensor, history: dict[str, Any]
) -> None:
    raise NotImplementedError("_calc_min_sigvals not implemented")


def _calc_max_sigvals(
    x: Tensor, i: int, step: int, x0: Tensor, history: dict[str, Any]
) -> None:
    raise NotImplementedError("_calc_max_sigvals not implemented")


def _calc_sigvals_bounds(
    x: Tensor, i: int, step: int, x0: Tensor, history: dict[str, Any]
) -> None:
    unfoldings_sigvals = tensor_utils.get_unfoldings_sigvals(x)

    for idx, sigvals in enumerate(unfoldings_sigvals):
        max_sigval = sigvals[0]
        min_sigval = sigvals[-1]

        history["max_sigvals"][idx].append(max_sigval)
        history["min_sigvals"][idx].append(min_sigval)


def _calc_dot(
    x: Tensor, i: int, step: int, x0: Tensor, history: dict[str, Any]
) -> None:
    dot = (x * x0).sum().item()
    history["dot"].append(dot)


def _calc_residual(
    x: Tensor, i: int, step: int, x0: Tensor, history: dict[str, Any]
) -> None:
    res_norm = (((x - x0) ** 2).sum() ** 0.5).item()
    history["res_norm"].append(res_norm)


def projectors_metrics_collector_factory(
    x0: Tensor, metrics: Optional[list[str]] = None
) -> tuple[Callable[[Tensor, int, int], None], dict[str, Any]]:
    """
    Return (metrics_callback, history)
    history:
      - sigma_min_history: list (len = x0.dim()) of lists
      - sigma_max_history: list (len = x0.dim()) of lists
      - dot_history: list of floats
      - res_norm_history: list of floats
    """
    metric_collectors_funcs = {
        "min_sigvals": _calc_min_sigvals,
        "max_sigvals": _calc_max_sigvals,
        "sigvals_bounds": _calc_sigvals_bounds,
        "dot": _calc_dot,
        "residual": _calc_residual,
    }

    metrics = list() if metrics is None else metrics
    metric_collectors = [metric_collectors_funcs[metric] for metric in metrics]

    history = _init_history(x0=x0, metrics=metrics)

    def metrics_callback(x: Tensor, i: int, step: int) -> None:
        # mutate history in closure
        for collector in metric_collectors:
            collector(x, i, step, x0, history)

    return metrics_callback, history


class RandomGradLogger:
    def __init__(self, model: nn.Module, log_proba: float, seed: int = 42) -> None:
        self.model = model
        self.log_proba = log_proba

        self.g = torch.Generator()
        self.g.manual_seed(seed)

        self.grad_logs: list[tuple[int, str, torch.Tensor]] = []

    @torch.no_grad()
    def sample_log(self, global_step: int = 0):
        for name, param in self.model.named_parameters():
            if (param.requires_grad == False) or (param.grad is None):
                continue

            if torch.rand((), generator=self.g).item() >= self.log_proba:
                continue

            grad = param.grad.detach().cpu()
            self.grad_logs.append((global_step, name, grad))

    @torch.no_grad()
    def save_logs(self, path: str):
        torch.save(self.grad_logs, path)
