"""Online unfolding variants on a GPU: selection quality, step time, memory.

Compares offline, online (Lanczos quadrature), online_polar (polar factor and trace)
and online_exact (SVD) on the conv layers of a ResNet, with the large matrix products
in float32, bfloat16 and float16. Run from ``experiments/``:

    python benchmarks/online_variants.py --model ResNet34

PyTorch warns about an operation that falls back to the CPU only once per process, so
this script does not report fallbacks. On Intel GPUs, run a single strategy with
``PYTORCH_DEBUG_XPU_FALLBACK=1`` to list them.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from offline_vs_online import backend, pick_device, synchronize  # noqa: E402
from tensorion import Tensorion  # noqa: E402
from tensorion.nuclear import nuclear_norm_exact  # noqa: E402
from tensorion.unfolding import _candidates, matricize  # noqa: E402
from tensorion_experiments.models import resnet  # noqa: E402


def conv_routing(model):
    conv = {id(m.weight) for m in model.modules() if isinstance(m, torch.nn.Conv2d)}
    return [
        {"params": [p for p in model.parameters() if id(p) in conv], "use_muon": True},
        {"params": [p for p in model.parameters() if id(p) not in conv], "use_muon": False},
    ]


def make(model_name, classes, device, **kwargs):
    torch.manual_seed(0)
    model = getattr(resnet, model_name)(num_classes=classes).to(device)
    return model, Tensorion(conv_routing(model), **kwargs)


def batch(device, size, classes):
    return torch.randn(size, 3, 32, 32, device=device), torch.randint(0, classes, (size,), device=device)


def backward(model, x, y):
    model.zero_grad()
    torch.nn.functional.cross_entropy(model(x), y).backward()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ResNet34")
    parser.add_argument("--classes", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()
    device = pick_device()
    print(f"device {device}, torch {torch.__version__}, {args.model}, batch {args.batch_size}")

    # --- 1. Which split does each variant pick, against the exact best?
    model, optimizer = make(args.model, args.classes, device, unfolding="offline")
    for _ in range(6):
        backward(model, *batch(device, args.batch_size, args.classes))
        optimizer.step()
    momenta = [optimizer.state[p]["momentum_buffer"].clone() for p in optimizer.param_groups[0]["params"]]
    exact = []
    for m in momenta:
        cpu = m.float().cpu()
        exact.append({rows: nuclear_norm_exact(matricize(cpu, rows)) for rows, _ in _candidates(tuple(m.shape))})
    multi = sum(len(e) > 1 for e in exact)
    print(f"\n1) selection on {len(momenta)} conv momenta ({multi} with more than one unfolding)")
    print(f"   {'strategy':14s} {'precision':10s} {'best found':>10s} {'mean loss':>10s} {'max loss':>9s} {'time, all layers':>17s}")
    for strategy in ["offline", "online", "online_polar", "online_exact"]:
        for dtype in [None, torch.bfloat16, torch.float16]:
            if strategy in ("offline", "online_exact") and dtype is not None:
                continue
            probe = Tensorion([torch.nn.Parameter(m.clone()) for m in momenta], unfolding=strategy, online_every=1, matmul_dtype=dtype)
            params = probe.param_groups[0]["params"]

            def select():
                rows = []
                for p, m in zip(params, momenta):
                    if p in probe._online:
                        rows.append(probe._select_online(p, m, probe._unfoldings[p], probe_orth, dtype)[0])
                    else:
                        rows.append(probe._unfoldings[p](m))
                return rows

            from tensorion.orthogonalize import newton_schulz as probe_orth

            select()
            synchronize(device)
            start = time.perf_counter()
            rows = select()
            synchronize(device)
            elapsed = time.perf_counter() - start
            losses = torch.tensor([max(0.0, 1 - e[r] / max(e.values())) for e, r in zip(exact, rows)])
            name = "float32" if dtype is None else str(dtype).replace("torch.", "")
            print(f"   {strategy:14s} {name:10s} {(losses < 1e-5).sum().item():7d}/{len(rows):<2d} {losses.mean().item():10.3%} {losses.max().item():9.2%} {1e3 * elapsed:14.1f} ms")

    # --- 2. Time of optimizer.step() and peak memory.
    print(f"\n2) optimizer.step(), mean over {args.steps} steps after 2 warm-up steps")
    print(f"   {'strategy':14s} {'every':>5s} {'precision':10s} {'step':>9s} {'vs offline':>10s} {'peak memory':>12s}")
    baseline = {}
    configs = []
    for dtype in [None, torch.bfloat16, torch.float16]:
        configs.append(("offline", 1, dtype))
        for strategy in ["online", "online_polar"]:
            for every in (1, 10):
                configs.append((strategy, every, dtype))
    configs.append(("online_exact", 1, None))
    configs.append(("online_exact", 10, None))
    x, y = batch(device, args.batch_size, args.classes)
    for strategy, every, dtype in configs:
        model, optimizer = make(args.model, args.classes, device, unfolding=strategy, online_every=every, matmul_dtype=dtype)
        if backend(device) is not None:
            backend(device).reset_peak_memory_stats()
        elapsed = 0.0
        for step in range(-2, args.steps):
            backward(model, x, y)
            synchronize(device)
            start = time.perf_counter()
            optimizer.step()
            synchronize(device)
            if step >= 0:
                elapsed += time.perf_counter() - start
        mean = elapsed / args.steps
        name = "float32" if dtype is None else str(dtype).replace("torch.", "")
        baseline.setdefault(name, mean)
        peak = backend(device).max_memory_allocated() / 2**30 if backend(device) is not None else float("nan")
        print(f"   {strategy:14s} {every:5d} {name:10s} {1e3 * mean:7.1f} ms {mean / baseline[name] - 1:+10.0%} {peak:9.2f} GiB")


if __name__ == "__main__":
    main()
