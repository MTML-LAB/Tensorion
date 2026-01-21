import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

import numpy as np
import random
import os

from omegaconf import OmegaConf, DictConfig
import re
from typing import Union, Optional, Literal, Callable, Any

from models import resnet
from optimizers import tensorion, tensorion_sgd

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

        muon_params = []
        adamw_params = []

        for name, param in model.named_parameters():
            if param.ndim > 2:
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


def load_model(model_name: Literal['ResNet18', 'ResNet34'], num_classes: int, device) -> nn.Module:
    if model_name == 'ResNet18':
        model = resnet.ResNet18(num_classes=num_classes)
    elif model_name == 'ResNet34':
        model = resnet.ResNet34(num_classes=num_classes)
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
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    np.save(f'{dir_name}/train_loss', train_losses)
    np.save(f'{dir_name}/test_loss', test_losses)
    np.save(f'{dir_name}/accuracy', accs)
    np.save(f'{dir_name}/lr', lrs)


def run(cfg_path: str):
    cfg: DictConfig = OmegaConf.load(cfg_path)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"{device=}")
    g = set_global_seed(cfg.get('seed', 42))

    model = load_model(model_name=cfg.model_name, num_classes=cfg.num_classes, device=device)
    train_ds, test_ds, train_loader, test_loader = data_preparation.load_from_config(cfg, random_gen=g)
    optimizer = load_optim_from_config(cfg, model)
    scheduler = load_scheduler_from_config(cfg, optimizer)

    criterion = torch.nn.CrossEntropyLoss()

    num_epochs = cfg.num_epochs
    train_results = train_loop(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=num_epochs,
        scheduler=scheduler
    )
    
    save_data(cfg, train_results)
