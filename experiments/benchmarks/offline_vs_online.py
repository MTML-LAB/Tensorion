"""Offline against online unfolding in a short training run: accuracy, loss, time, memory.

Model, data, transforms, scheduler and optimizer come from the experiment configs.
This script only adds device selection (CUDA, Intel XPU, Apple MPS) and the
recording of epoch time and peak memory. Run from ``experiments/``:

    python benchmarks/offline_vs_online.py --unfolding online --routing paper --out result.json

``--routing paper`` puts every convolution kernel on Tensorion (the config default),
``--routing order2`` routes by ``min_order=2``, which adds the linear head.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tensorion_experiments import data_preparation  # noqa: E402
from tensorion_experiments.config_resolvers import register_resolvers  # noqa: E402
from tensorion_experiments.run_experiment import build_optimizer, set_global_seed  # noqa: E402


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def backend(device: torch.device):
    return {"cuda": torch.cuda, "xpu": getattr(torch, "xpu", None)}.get(device.type)


def synchronize(device: torch.device) -> None:
    if backend(device) is not None:
        backend(device).synchronize()


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss = correct = count = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss += criterion(logits, y).item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        count += len(y)
    return loss / count, correct / count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unfolding", default="offline")
    parser.add_argument("--online-every", type=int, default=10)
    parser.add_argument("--routing", choices=["paper", "order2"], default="paper")
    parser.add_argument("--matmul-dtype", default=None, help="e.g. bfloat16 for the large matrix products")
    parser.add_argument("--model", default="ResNet34")
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", default="./data")
    parser.add_argument("--max-steps", type=int, default=0, help="stop every epoch early (for a quick check)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    overrides = [
        f"dataset={args.dataset}",
        f"dataset.data_path={args.data}",
        f"model._target_=tensorion_experiments.models.resnet.{args.model}",
        "optimizer=tensorion",
        f"optimizer.unfolding={args.unfolding}",
        f"optimizer.online_every={args.online_every}",
        f"optimizer.lr={args.lr}",
        f"optimizer.weight_decay={args.weight_decay}",
        "scheduler=onecyclelr",
        f"scheduler.scheduler_kwargs.max_lr={args.lr}",
        f"num_epochs={args.epochs}",
        f"seed={args.seed}",
        f"dataloader.train.batch_size={args.batch_size}",
        "dataloader.train.pin_memory=false",
        "dataloader.test.pin_memory=false",
        "comet.enabled=false",
    ]
    if args.matmul_dtype:
        overrides.append(f"+optimizer.matmul_dtype={args.matmul_dtype}")
    if args.routing == "order2":
        overrides += ["~optimizer.tensorion_modules", "+optimizer.min_order=2"]
    register_resolvers()
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="config.yaml", overrides=overrides)

    device = pick_device()
    set_global_seed(cfg.seed)
    model = instantiate(cfg.model).to(device)
    _, _, train_loader, test_loader = data_preparation.load_from_config(cfg)
    optimizer = build_optimizer(cfg.optimizer, model)
    scheduler = instantiate(cfg.scheduler.scheduler_kwargs, optimizer=optimizer, steps_per_epoch=len(train_loader))
    criterion = instantiate(cfg.criterion)

    on_tensorion = sum(1 for g in optimizer.param_groups for p in g["params"] if optimizer._uses_tensorion(p, g))
    result = {
        "args": vars(args),
        "device": str(device),
        "torch": torch.__version__,
        "parameters_on_tensorion": on_tensorion,
        "epochs": [],
    }
    print(f"{device=} unfolding={args.unfolding} routing={args.routing} tensorion parameters={on_tensorion}", flush=True)
    if backend(device) is not None:
        backend(device).reset_peak_memory_stats()

    for epoch in range(args.epochs):
        model.train()
        loss_sum = count = 0.0
        step_seconds = 0.0
        synchronize(device)
        epoch_start = time.perf_counter()
        for step, (x, y) in enumerate(train_loader):
            if args.max_steps and step >= args.max_steps:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            synchronize(device)
            start = time.perf_counter()
            optimizer.step()
            synchronize(device)
            step_seconds += time.perf_counter() - start
            scheduler.step()
            loss_sum += loss.item() * len(y)
            count += len(y)
        synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_start
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        row = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / count,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "epoch_seconds": epoch_seconds,
            "optimizer_seconds": step_seconds,
            "peak_memory_gib": backend(device).max_memory_allocated() / 2**30 if backend(device) is not None else None,
        }
        result["epochs"].append(row)
        print(json.dumps(row), flush=True)
        Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
