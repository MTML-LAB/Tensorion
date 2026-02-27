import time
import torch
from torch import nn
from torch.utils.data import DataLoader

import os
from tensorion.loggers import RandomGradLogger
from typing import Optional, Literal, Callable, Any
import pathlib

from tensorion.experiment_logger import ExperimentLogger


@torch.no_grad()
def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    checkpoint_path: str | pathlib.Path,
    epoch: int,
) -> None:
    """
    Save checkpoint artifacts into:
      {checkpoint_path}/epoch-{epoch}/
        - model.pt
        - optimizer.pt
        - scheduler.pt   (only if scheduler is not None and has state_dict)
    """
    epoch_dir = pathlib.Path(checkpoint_path) / f"epoch-{epoch}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    model_file = epoch_dir / "model.pt"
    optim_file = epoch_dir / "optimizer.pt"
    sched_file = epoch_dir / "scheduler.pt"

    # Model
    torch.save(model.state_dict(), model_file)

    # Optimizer
    torch.save(optimizer.state_dict(), optim_file)

    # Scheduler (optional)
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        torch.save(scheduler.state_dict(), sched_file)


@torch.no_grad()
def test(
    model: nn.Module,
    test_loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str,
) -> tuple:
    model.eval()
    epoch_loss = torch.zeros((1,), device=device)
    epoch_accuracy = torch.zeros((1,), device=device)
    num_samples = 0

    for inp, labels in test_loader:
        bs = labels.shape[0]
        inp = inp.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        preds = model(inp)

        epoch_loss += loss_fn(preds, labels) * bs

        preds = torch.argmax(preds, dim=-1)
        epoch_accuracy += (preds == labels).float().sum()

        num_samples += bs

    epoch_loss = epoch_loss.item() / num_samples
    epoch_accuracy = epoch_accuracy.item() / num_samples

    return epoch_accuracy, epoch_loss


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    train_loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str,
    verbose_iters: int = -1,
    epoch: int = 0,
    scheduler_step: Literal["step", "epoch"] = "epoch",
    log_grads: bool = False,
) -> tuple[float, int, float]:
    """
    :return: epoch loss(float), num samples(int), elapsed time(float)
    """

    model.train()
    epoch_loss = torch.zeros((1,), device=device)
    num_samples = 0
    global_step = 0

    logger = RandomGradLogger(model, log_proba=1e-4) if log_grads else None

    start_time = time.perf_counter()
    for batch_idx, (inp, labels) in enumerate(train_loader):
        bs = labels.shape[0]
        inp = inp.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        preds = model(inp)
        loss = loss_fn(preds, labels)
        loss.backward()

        if logger is not None:
            logger.sample_log(global_step)

        optimizer.step()
        if scheduler_step == "step":
            scheduler.step()

        epoch_loss = epoch_loss + loss.detach() * bs
        num_samples += bs
        global_step += 1

        if (verbose_iters > 0) and ((batch_idx + 1) % verbose_iters == 0):
            curr_loss = epoch_loss.item() / num_samples
            num_batches = len(train_loader)

            print(f"{batch_idx:4}/{num_batches:4}:\tloss{curr_loss:.5f}", flush=True)

    if scheduler_step == "epoch":
        scheduler.step()
    epoch_loss = epoch_loss.item() / num_samples

    if logger is not None:
        os.makedirs("sample_data", exist_ok=True)
        logger.save_logs(path=f"sample_data/gradients_from_{epoch}.pt")

    elapsed = time.perf_counter() - start_time
    return epoch_loss, num_samples, elapsed


def train_loop(
    model,
    train_loader,
    test_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    num_epochs,
    scheduler_step: Literal["step", "epoch"] = "epoch",
    save_steps: int = -1,
    save_path: Optional[str] = None,
    experiment_logger: Optional[ExperimentLogger] = None,
):

    train_losses = []
    test_losses = []
    accs = []
    lr_diffs = []
    best_acc = float("-inf")
    best_epoch = None
    last_checkpoint_dir = None

    acc, test_loss = test(
        model=model, test_loader=test_loader, loss_fn=criterion, device=device
    )
    lr = scheduler.get_last_lr()

    print(
        f"Epoch {0:2} ||  Accuracy = {acc:.4f} || test_loss = {test_loss:.6f} || lr = {lr}",
        flush=True,
    )

    if experiment_logger is not None:
        experiment_logger.log_metrics(
            {
                "test/accuracy": acc,
                "test/loss": test_loss,
                **_format_lrs(lr),
            },
            step=0,
            epoch=0,
        )

    for epoch in range(1, num_epochs + 1):
        train_loss, num_samples, elapsed = train_epoch(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            loss_fn=criterion,
            device=device,
            scheduler=scheduler,
            epoch=epoch,
            scheduler_step=scheduler_step,
        )
        lr = scheduler.get_last_lr()

        acc, test_loss = test(
            model=model, test_loader=test_loader, loss_fn=criterion, device=device
        )

        print(
            f"Epoch {epoch:2} ||  Accuracy = {acc:.4f} || test_loss = {test_loss:.6f} || train_loss = {train_loss:.6f} || lr = {lr}",
            flush=True,
        )

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        accs.append(acc)
        lr_diffs.append(lr)

        if experiment_logger is not None:
            metrics = {
                "train/loss": train_loss,
                "test/loss": test_loss,
                "test/accuracy": acc,
                **_format_lrs(lr),
            }
            if elapsed > 0:
                metrics["train/samples_per_sec"] = num_samples / elapsed
            experiment_logger.log_metrics(metrics, step=epoch, epoch=epoch)

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch

        if (save_steps > 0) and (epoch % save_steps == 0):
            assert save_path is not None

            save_checkpoint(model, optimizer, scheduler, save_path, epoch)
            last_checkpoint_dir = pathlib.Path(save_path) / f"epoch-{epoch}"

    if experiment_logger is not None and save_steps > 0 and save_path is not None:
        if last_checkpoint_dir is not None:
            experiment_logger.log_asset_folder(
                str(last_checkpoint_dir), name="checkpoint-last"
            )
        if best_epoch is not None:
            best_dir = pathlib.Path(save_path) / f"epoch-{best_epoch}"
            if best_dir.exists():
                experiment_logger.log_asset_folder(
                    str(best_dir), name="checkpoint-best"
                )

    return (train_losses, test_losses, accs, lr_diffs)


def _format_lrs(lrs: list[float]) -> dict[str, float]:
    if len(lrs) == 1:
        return {"lr": lrs[0]}
    return {f"lr/group_{idx}": lr for idx, lr in enumerate(lrs)}
