import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import random

from omegaconf import OmegaConf
import re
from src.loggers import RandomGradLogger
from typing import Union, Optional, Literal, Callable


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
) -> float:
    model.train()
    epoch_loss = torch.zeros((1,), device=device)
    num_samples = 0
    gs = 0

    logger = RandomGradLogger(model, log_proba=1e-4)

    for batch_idx, (inp, labels) in enumerate(train_loader):
        bs = labels.shape[0]
        inp = inp.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        preds = model(inp)
        loss = loss_fn(preds, labels)
        loss.backward()

        logger.sample_log(gs)

        optimizer.step()

        epoch_loss = epoch_loss + loss.detach() * bs
        num_samples += bs
        gs += 1

        if (verbose_iters > 0) and ((batch_idx + 1) % verbose_iters == 0):
            curr_loss = epoch_loss.item() // num_samples
            num_batches = len(train_loader)

            print(f"{batch_idx:4}/{num_batches:4}:\tloss{curr_loss:.5f}", flush=True)

    scheduler.step()
    epoch_loss = epoch_loss.item() / num_samples
    logger.save_logs(path=f"sample_data/gradients_from_{epoch}.pt")
    return epoch_loss


def train_loop(
    model,
    train_loader,
    test_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    num_epochs,
):
    train_losses = []
    test_losses = []
    accs = []
    lr_diffs = []

    acc, test_loss = test(
        model=model, test_loader=test_loader, loss_fn=criterion, device=device
    )
    lr = scheduler.get_last_lr()

    print(
        f"Epoch {0:2} ||  Accuracy = {acc:.4f} || test_loss = {test_loss:.6f} || lr = {lr}",
        flush=True,
    )

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            loss_fn=criterion,
            device=device,
            scheduler=scheduler,
            epoch=epoch,
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

    return (train_losses, test_losses, accs, lr_diffs)
