import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import random
import os

from omegaconf import OmegaConf, DictConfig
import re
from typing import Union, Optional, Literal, Callable

from models import resnet
from optimizers import tensorion

from data_preparation import get_datasets, get_dataloader
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

    # Для Dataloader
    g = torch.Generator()
    g.manual_seed(seed)
    
    return g


def get_optim_and_kwargs(model, optim_name: str, optimizer_kwargs: dict) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
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
    else:
        raise ValueError("Incorrect optimizer name")
    
    kwargs.update(optimizer_kwargs)
    return optimizer, kwargs


def get_scheduler_and_kwargs(scheduler_name: str, scheduler_kwargs: dict):
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


def load_model(num_classes: int, device) -> nn.Module:
    model = resnet.ResNet34(num_classes=num_classes)
    model.to(device)

    return model


def run(cfg_path: str):
    cfg: DictConfig = OmegaConf.load(cfg_path)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"{device=}")
    g = set_global_seed(cfg.get('seed', 42))
    model = load_model(num_classes=cfg.num_classes, device=device)

    train_ds, test_ds = get_datasets(dataset_name=cfg.dataset_name, download=True)
    train_loader, test_loader = get_dataloader(
        train_ds=train_ds, test_ds=test_ds,
        train_bs=32, test_bs=1250,
        num_workers=4,
        persistent_workers=True,
        ddp_sampler=False,
        g=g
    )

    optimizer_cls, optimizer_kwargs = get_optim_and_kwargs(model, optim_name=optim_name, optimizer_kwargs=optimizer_kwargs)
    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)

    scheduler_cls, scheduler_kwargs = get_scheduler_and_kwargs(scheduler_name, scheduler_kwargs=scheduler_kwargs)
    scheduler = scheduler_cls(optimizer, **scheduler_kwargs)

    print(f"{optimizer_kwargs=}\n\n{scheduler_kwargs=}\n")

    criterion = torch.nn.CrossEntropyLoss()

    train_losses, test_losses, accs, lrs = train_loop(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=num_epochs,
        scheduler=scheduler
    )
    
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    np.save(f'{dir_name}/train_loss', train_losses)
    np.save(f'{dir_name}/test_loss', test_losses)
    np.save(f'{dir_name}/accuracy', accs)
    np.save(f'{dir_name}/lr', lrs)

