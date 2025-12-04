import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

import numpy as np
import random
import os

from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
import re
from typing import Union, Optional, Literal, Callable, Any

from models import resnet, vit
from optimizers import tensorion

import data_preparation
from train import train_loop


def set_global_seed(seed: int) -> torch.Generator:
    """
    Set global seed for reproducibility.
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.use_deterministic_algorithms(True)

    g = torch.Generator()
    g.manual_seed(seed)

    return g


def save_data(cfg: DictConfig, train_results):
    train_losses, test_losses, accs, lrs = train_results

    dir_name = cfg.save_dir
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    np.save(f"{dir_name}/train_loss", train_losses)
    np.save(f"{dir_name}/test_loss", test_losses)
    np.save(f"{dir_name}/accuracy", accs)
    np.save(f"{dir_name}/lr", lrs)


@hydra.main(config_path="../configs", config_name="config.yaml", version_base="1.3")
def run(cfg):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"{device=}")
    g = set_global_seed(cfg.seed)

    model = instantiate(cfg.model).to(device)
    train_ds, test_ds, train_loader, test_loader = data_preparation.load_from_config(
        cfg, random_gen=g
    )
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    criterion = instantiate(cfg.criterion)

    num_epochs = cfg.num_epochs
    train_results = train_loop(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=num_epochs,
        scheduler=scheduler,
    )

    save_data(cfg, train_results)


if __name__ == "__main__":
    run()
