"""How good and how fast is the SVD-free online unfolding selection?

Compares ``unfolding="online"`` (estimated nuclear norms) with
``unfolding="online_exact"`` (SVDs) on momentum tensors of the CIFAR ResNet-18
used in the experiments and on synthetic tensors. Run from ``experiments/``:

    python benchmarks/online_selection.py
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tensorion_experiments.models.resnet import ResNet18  # noqa: E402

from tensorion import Tensorion
from tensorion.nuclear import nuclear_norm, nuclear_norm_exact
from tensorion.unfolding import STRATEGIES, _candidates, matricize, two_way_splits


def resnet_momentum(batch: int, steps: int = 6, beta: float = 0.95) -> dict[str, torch.Tensor]:
    """Momentum buffers of the conv layers after a few SGD steps on random CIFAR-sized batches."""
    torch.manual_seed(0)
    model = ResNet18(num_classes=10)
    buffers = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.ndim == 4}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    for _ in range(steps):
        optimizer.zero_grad()
        x, y = torch.randn(batch, 3, 32, 32), torch.randint(0, 10, (batch,))
        torch.nn.functional.cross_entropy(model(x), y).backward()
        optimizer.step()
        for n, p in model.named_parameters():
            if n in buffers:
                buffers[n].mul_(beta).add_(p.grad)
    return {f"{n} (batch {batch})": b for n, b in buffers.items()}


def synthetic() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(1)
    tensors = {"gaussian (256, 256, 3, 3)": torch.randn(256, 256, 3, 3, generator=g)}
    # Low rank in one particular matricization, so that another split wins.
    for rows in [(0,), (0, 2), (0, 3)]:
        shape = (256, 128, 3, 3)
        r = 1
        for m in rows:
            r *= shape[m]
        low = torch.randn(r, 4, generator=g) @ torch.randn(4, 256 * 128 * 9 // r, generator=g)
        perm = list(rows) + [m for m in range(4) if m not in rows]
        inverse = [perm.index(m) for m in range(4)]
        x = low.reshape(*(shape[m] for m in perm)).permute(*inverse)
        tensors[f"rank 4 in split {rows}"] = x + 0.01 * torch.randn(shape, generator=g)
    return tensors


def timed(fn, *args, repeat: int = 3):
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        out = fn(*args)
        best = min(best, time.perf_counter() - start)
    return out, best


def main() -> None:
    tensors = {**resnet_momentum(128), **resnet_momentum(16), **synthetic()}

    errors = []
    for x in tensors.values():
        for rows, _ in _candidates(tuple(x.shape)):
            M = matricize(x, rows)
            if min(M.shape) >= 128:  # smaller matrices are handled exactly
                errors.append(abs(nuclear_norm(M) / nuclear_norm_exact(M) - 1))
    errors = torch.tensor(errors)
    print(f"estimate vs SVD on {len(errors)} matrices: mean error {errors.mean():.2%}, max {errors.max():.2%}\n")

    print(f"{'tensor':42s} {'shape':18s} {'exact':8s} {'online':8s} {'loss':>6s} {'exact ms':>9s} {'online ms':>9s}")
    total_exact = total_online = total_naive = 0.0
    same = 0
    worst = 0.0
    for name, x in tensors.items():
        true = {rows: nuclear_norm_exact(matricize(x, rows)) for rows in two_way_splits(x.ndim)}
        _, t_naive = timed(lambda: [nuclear_norm_exact(matricize(x, rows)) for rows in two_way_splits(x.ndim)], repeat=1)
        exact, t_exact = timed(STRATEGIES["online_exact"], x)
        online, t_online = timed(STRATEGIES["online"], x)
        best = max(true.values())
        assert true[exact] >= best * (1 - 1e-5), name  # pruning is lossless
        loss = max(0.0, 1 - true[online] / best)
        same += loss < 1e-5
        worst = max(worst, loss)
        total_naive += t_naive
        total_exact += t_exact
        total_online += t_online
        print(f"{name:42s} {str(tuple(x.shape)):18s} {str(exact):8s} {str(online):8s} {loss:6.2%} {1e3 * t_exact:9.1f} {1e3 * t_online:9.1f}")
    print(f"\nbest split found for {same}/{len(tensors)} tensors, largest loss in nuclear norm {worst:.2%}")
    print(f"total time: SVD of every split {total_naive:.2f}s, online_exact {total_exact:.2f}s, online {total_online:.2f}s")
    step_times()


def step_times(steps: int = 20) -> None:
    """Time of ``optimizer.step()`` on the ResNet-18 (conv layers on Tensorion), per strategy."""
    print(f"\nmean time of optimizer.step() over {steps} steps, ResNet-18:")
    baseline = None
    for label, kwargs in [
        ("offline", dict(unfolding="offline")),
        ("online_exact", dict(unfolding="online_exact")),
        ("online", dict(unfolding="online")),
        ("online, online_every=10", dict(unfolding="online", online_every=10)),
        ("online, online_every=100", dict(unfolding="online", online_every=100)),
    ]:
        torch.manual_seed(0)
        model = ResNet18(num_classes=10)
        optimizer = Tensorion(model.parameters(), min_ndim=3, **kwargs)
        x, y = torch.randn(32, 3, 32, 32), torch.randint(0, 10, (32,))
        elapsed = 0.0
        for _ in range(steps):
            optimizer.zero_grad()
            torch.nn.functional.cross_entropy(model(x), y).backward()
            start = time.perf_counter()
            optimizer.step()
            elapsed += time.perf_counter() - start
        mean = elapsed / steps
        baseline = baseline or mean
        print(f"  {label:26s} {1e3 * mean:7.1f} ms   {mean / baseline - 1:+7.0%} over offline")


if __name__ == "__main__":
    main()
