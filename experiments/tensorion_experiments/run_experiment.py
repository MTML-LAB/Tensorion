import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

import numpy as np
import random
import os

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
import re
from typing import Union, Optional, Literal, Callable, Any

from .models import resnet, vit
from .schedulers import WarmupCosineAnnealingWarmRestarts

from . import data_preparation
from .train import train_loop
from .experiment_logger import build_experiment_logger


def set_global_seed(seed: int) -> torch.Generator:
    """
    Set global seed for reproducibility.
    """

    torch.manual_seed(seed)
    if torch.cuda.is_available():
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
    os.makedirs(dir_name, exist_ok=True)

    np.save(f"{dir_name}/train_loss", train_losses)
    np.save(f"{dir_name}/test_loss", test_losses)
    np.save(f"{dir_name}/accuracy", accs)
    np.save(f"{dir_name}/lr", lrs)


def save_config_snapshot(cfg: DictConfig) -> str:
    os.makedirs(cfg.save_dir, exist_ok=True)
    config_path = os.path.join(cfg.save_dir, "config.yaml")
    OmegaConf.save(cfg, config_path)
    return config_path


def build_optimizer(optimizer_cfg: DictConfig, model: nn.Module) -> Optimizer:
    """Instantiate the optimizer on the named parameters of ``model``.

    If the config has ``tensorion_modules`` (a list of module class names, e.g.
    ``[Conv2d]``), the weights of those modules get the Tensorion update and every
    other parameter the fallback. This states the routing explicitly instead of
    leaving it to the shapes of the parameters.
    """
    module_names = optimizer_cfg.get("tensorion_modules")
    optimizer_cfg = OmegaConf.masked_copy(optimizer_cfg, [k for k in optimizer_cfg if k != "tensorion_modules"])
    factory = instantiate(optimizer_cfg, _partial_=True)
    named = list(model.named_parameters())
    if module_names is None:
        return factory(params=named)

    known = {type(m).__name__ for m in model.modules()}
    missing = [name for name in module_names if name not in known]
    if len(missing) == len(module_names):
        raise ValueError(f"tensorion_modules {list(module_names)} match no module of the model (it has {sorted(known)}).")
    chosen = {
        id(m.weight)
        for m in model.modules()
        if type(m).__name__ in module_names
        and getattr(m, "weight", None) is not None
        # A weight with fewer than two axes larger than 1 is a vector and has no unfolding.
        and sum(n > 1 for n in m.weight.shape) >= 2
    }
    groups = [
        {"params": [(n, p) for n, p in named if id(p) in chosen], "use_muon": True},
        {"params": [(n, p) for n, p in named if id(p) not in chosen], "use_muon": False},
    ]
    return factory(params=[g for g in groups if g["params"]])


def run(cfg: DictConfig):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"{device=}")
    set_global_seed(cfg.seed)

    model = instantiate(cfg.model).to(device)
    train_ds, test_ds, train_loader, test_loader = data_preparation.load_from_config(cfg)
    optimizer = build_optimizer(cfg.optimizer, model)

    scheduler_kwargs = {"optimizer": optimizer}
    if "steps_per_epoch" in cfg.scheduler.scheduler_kwargs.keys():
        scheduler_kwargs["steps_per_epoch"] = len(train_loader)

    scheduler = instantiate(cfg.scheduler.scheduler_kwargs, **scheduler_kwargs)
    
    scheduler_step = cfg.scheduler.get('scheduler_step', 'epoch')

    save_path = cfg.save_dir + '/checkpoints/'
    save_steps = cfg.get('save_steps', -1)

    criterion = instantiate(cfg.criterion)

    experiment_logger = build_experiment_logger(cfg)
    config_snapshot_path = save_config_snapshot(cfg)
    experiment_logger.log_config(cfg, config_path=config_snapshot_path)

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
        scheduler_step=scheduler_step,
        save_steps=save_steps,
        save_path=save_path,
        experiment_logger=experiment_logger,
    )

    save_data(cfg, train_results)
    for asset_name in ["train_loss.npy", "test_loss.npy", "accuracy.npy", "lr.npy"]:
        asset_path = os.path.join(cfg.save_dir, asset_name)
        if os.path.exists(asset_path):
            experiment_logger.log_asset(asset_path)
    experiment_logger.end()
