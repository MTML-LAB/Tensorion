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
from .optimizers import tensorion, tensorion_sgd
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


def get_optim_and_kwargs(model, optim_name: str, optimizer_kwargs: dict) -> tuple[type[Optimizer], dict[str, Any]]:
    if optim_name == 'AdamW': 
        optimizer = torch.optim.AdamW
        kwargs =  {
            'weight_decay': 5e-3,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
        }
    elif optim_name == 'SGD': 
        optimizer = torch.optim.SGD
        kwargs =  {
            'weight_decay': 5e-3,
            'momentum': 0.95,
        }
    elif optim_name == 'Tensorion':
        optimizer = tensorion.Tensorion
        min_dim: int = optimizer_kwargs.get('min_dim', 3)

        muon_params = []
        adamw_params = []

        for name, param in model.named_parameters():
            if param.ndim >= min_dim:
                muon_params.append(param)
            else:
                adamw_params.append(param)

        kwargs = {
            'muon_params': muon_params,
            'adamw_params': adamw_params,
            'nesterov': False,
            'momentum': 0.95,
            'wd': 5e-3,
            'adamw_betas': (0.9, 0.999),
            'adamw_eps': 1e-8
        }
    elif optim_name == 'TensorionSGD':
        optimizer = tensorion_sgd.TensorionSGD

        muon_params = []
        sgd_params = []

        for name, param in model.named_parameters():
            if param.ndim > 2:
                muon_params.append(param)
            else:
                sgd_params.append(param)

        kwargs = {
            'muon_params': muon_params,
            'sgd_params': sgd_params,
            'nesterov': False,
            'momentum': 0.95,
            'wd': 5e-3,
            'sgd_momentum': 0.95,
        }
    else:
        raise ValueError("Incorrect optimizer name")
    
    kwargs.update(optimizer_kwargs)
    return optimizer, kwargs


def get_scheduler_and_kwargs(scheduler_name: str, scheduler_kwargs: dict) -> tuple[type[LRScheduler], dict[str, Any]]:
    schedulers = {
        'CosineAnnealingWarmRestarts': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        'WarmupCosineAnnealingWarmRestarts': WarmupCosineAnnealingWarmRestarts.WarmupCosineAnnealingWarmRestarts,
        'CosineAnnealingLR': torch.optim.lr_scheduler.CosineAnnealingLR,
        'ExponentialLR': torch.optim.lr_scheduler.ExponentialLR,
        'StepLR': torch.optim.lr_scheduler.StepLR,
        'MultiStepLR': torch.optim.lr_scheduler.MultiStepLR,
    }

    if scheduler_name in schedulers:
        scheduler = schedulers[scheduler_name]
        kwargs = {}
    else:
        raise ValueError("Incorrect scheduler name")

    kwargs.update(scheduler_kwargs)
    return scheduler, kwargs


def load_model(model_name: Literal['ResNet18', 'ResNet34', 'VIT'], num_classes: int, device) -> nn.Module:
    if model_name == 'ResNet18':
        model = resnet.ResNet18(num_classes=num_classes)
    elif model_name == 'ResNet34':
        model = resnet.ResNet34(num_classes=num_classes)
    elif model_name == 'VIT':
        model = vit.VIT_0(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    model.to(device)
    return model


def load_optim_from_config(cfg: DictConfig, model: nn.Module) -> Optimizer:
    optimizer_cfg = cfg.get('optimizer', {})
    optimizer_name = optimizer_cfg['optimizer_name']

    optimizer_kwargs = dict(optimizer_cfg)
    del optimizer_kwargs['optimizer_name']
    print(f"{optimizer_kwargs=}")

    optimizer_cls, optimizer_kwargs = get_optim_and_kwargs(model, optim_name=optimizer_name, optimizer_kwargs=optimizer_kwargs)
    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)

    return optimizer


def load_scheduler_from_config(cfg: DictConfig, optimizer: Optimizer):
    scheduler_cfg = cfg.get('scheduler', {})
    scheduler_name = scheduler_cfg['scheduler_name']
    
    scheduler_kwargs = dict(scheduler_cfg)
    del scheduler_kwargs['scheduler_name']
    print(f"{scheduler_kwargs=}")

    scheduler_cls, scheduler_kwargs = get_scheduler_and_kwargs(scheduler_name, scheduler_kwargs=scheduler_kwargs)
    scheduler = scheduler_cls(optimizer, **scheduler_kwargs)

    return scheduler


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


def run(cfg: DictConfig):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"{device=}")
    set_global_seed(cfg.seed)

    model = instantiate(cfg.model).to(device)
    train_ds, test_ds, train_loader, test_loader = data_preparation.load_from_config(cfg)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())

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
