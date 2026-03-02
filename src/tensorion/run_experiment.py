import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

import numpy as np
import random
import os

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from typing import Literal, Any

from tensorion.models import resnet, vit
from tensorion.optimizers import (
    tensorion,
    tensorion_sgd,
    tensorion_st_hosvd,
    tensorion_st_hosvd_sgd,
)
from tensorion.schedulers import WarmupCosineAnnealingWarmRestarts

from tensorion import data_preparation
from tensorion.train import train_loop
from tensorion.experiment_logger import build_experiment_logger


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


def get_optim_and_kwargs(
    model, optim_name: str, optimizer_kwargs: dict
) -> tuple[type[Optimizer], dict[str, Any]]:
    if optim_name == "AdamW":
        optimizer = torch.optim.AdamW
        kwargs = {
            "weight_decay": 5e-3,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
        }
    elif optim_name == "SGD":
        optimizer = torch.optim.SGD
        kwargs = {
            "weight_decay": 5e-3,
            "momentum": 0.95,
        }
    elif optim_name == "Tensorion":
        optimizer = tensorion.Tensorion
        min_dim: int = optimizer_kwargs.get("min_dim", 3)

        muon_params = []
        adamw_params = []

        for name, param in model.named_parameters():
            if param.ndim >= min_dim:
                muon_params.append(param)
            else:
                adamw_params.append(param)

        kwargs = {
            "muon_params": muon_params,
            "adamw_params": adamw_params,
            "nesterov": False,
            "momentum": 0.95,
            "wd": 5e-3,
            "adamw_betas": (0.9, 0.999),
            "adamw_eps": 1e-8,
        }
    elif optim_name == "TensorionSGD":
        optimizer = tensorion_sgd.TensorionSGD

        muon_params = []
        sgd_params = []

        for name, param in model.named_parameters():
            if param.ndim > 2:
                muon_params.append(param)
            else:
                sgd_params.append(param)

        kwargs = {
            "muon_params": muon_params,
            "sgd_params": sgd_params,
            "nesterov": False,
            "momentum": 0.95,
            "wd": 5e-3,
            "sgd_momentum": 0.95,
        }
    elif optim_name == "STHoSVDTensorion":
        optimizer = tensorion_st_hosvd.STHoSVDTensorion
        min_dim: int = optimizer_kwargs.get("min_dim", 2)

        muon_params = []
        adamw_params = []

        for name, param in model.named_parameters():
            if param.ndim >= min_dim:
                muon_params.append(param)
            else:
                adamw_params.append(param)

        kwargs = {
            "muon_params": muon_params,
            "adamw_params": adamw_params,
            "nesterov": False,
            "momentum": 0.95,
            "wd": 5e-3,
            "adamw_betas": (0.9, 0.999),
            "adamw_eps": 1e-8,
            "ranks": optimizer_kwargs.get("ranks", 8),
            "nuclear": optimizer_kwargs.get("nuclear", False),
            "adjust_lr": optimizer_kwargs.get("adjust_lr", True),
        }
    elif optim_name == "STHoSVDTensorionSGD":
        optimizer = tensorion_st_hosvd_sgd.STHoSVDTensorionSGD
        min_dim: int = optimizer_kwargs.get("min_dim", 2)

        muon_params = []
        sgd_params = []

        for name, param in model.named_parameters():
            if param.ndim >= min_dim:
                muon_params.append(param)
            else:
                sgd_params.append(param)

        kwargs = {
            "muon_params": muon_params,
            "sgd_params": sgd_params,
            "nesterov": False,
            "momentum": 0.95,
            "wd": 5e-3,
            "sgd_momentum": 0.95,
            "ranks": optimizer_kwargs.get("ranks", 8),
            "nuclear": optimizer_kwargs.get("nuclear", False),
            "adjust_lr": optimizer_kwargs.get("adjust_lr", True),
        }
    else:
        raise ValueError("Incorrect optimizer name")

    kwargs.update(optimizer_kwargs)
    return optimizer, kwargs


def get_scheduler_and_kwargs(
    scheduler_name: str, scheduler_kwargs: dict
) -> tuple[type[LRScheduler], dict[str, Any]]:
    schedulers = {
        "CosineAnnealingWarmRestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        "WarmupCosineAnnealingWarmRestarts": WarmupCosineAnnealingWarmRestarts.WarmupCosineAnnealingWarmRestarts,
        "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
        "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR,
        "StepLR": torch.optim.lr_scheduler.StepLR,
        "MultiStepLR": torch.optim.lr_scheduler.MultiStepLR,
    }

    if scheduler_name in schedulers:
        scheduler = schedulers[scheduler_name]
        kwargs = {}
    else:
        raise ValueError("Incorrect scheduler name")

    kwargs.update(scheduler_kwargs)
    return scheduler, kwargs


# Note: load_model is deprecated. Models are now instantiated via Hydra's instantiate(cfg.model)
# This function is kept for backward compatibility but is not used in the main pipeline.
def load_model(
    model_name: Literal["ResNet18", "ResNet34", "VIT"], num_classes: int, device
) -> nn.Module:
    """Deprecated: Use Hydra's instantiate(cfg.model) instead."""
    if model_name == "ResNet18":
        model = resnet.ResNet18(num_classes=num_classes)
    elif model_name == "ResNet34":
        model = resnet.ResNet34(num_classes=num_classes)
    elif model_name == "VIT":
        model = vit.VIT_0(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.to(device)
    return model


def load_optim_from_config(cfg: DictConfig, model: nn.Module) -> Optimizer:
    optimizer_cfg = cfg.get("optimizer", {})
    optimizer_name = optimizer_cfg["optimizer_name"]

    optimizer_kwargs = dict(optimizer_cfg)
    del optimizer_kwargs["optimizer_name"]
    print(f"{optimizer_kwargs=}")

    optimizer_cls, optimizer_kwargs = get_optim_and_kwargs(
        model, optim_name=optimizer_name, optimizer_kwargs=optimizer_kwargs
    )
    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)

    return optimizer


def load_scheduler_from_config(cfg: DictConfig, optimizer: Optimizer):
    scheduler_cfg = cfg.get("scheduler", {})
    scheduler_name = scheduler_cfg["scheduler_name"]

    scheduler_kwargs = dict(scheduler_cfg)
    del scheduler_kwargs["scheduler_name"]
    print(f"{scheduler_kwargs=}")

    scheduler_cls, scheduler_kwargs = get_scheduler_and_kwargs(
        scheduler_name, scheduler_kwargs=scheduler_kwargs
    )
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
    """Save a snapshot of the config to the results directory."""
    save_dir = str(cfg.save_dir)
    print(f"[DEBUG] save_config_snapshot: save_dir={save_dir}", flush=True)
    try:
        os.makedirs(save_dir, exist_ok=True)
        if not os.path.exists(save_dir):
            raise RuntimeError(f"Results directory does not exist: {save_dir}")
        config_path = os.path.join(save_dir, "config.yaml")
        OmegaConf.save(cfg, config_path)
        print(f"[DEBUG] Config snapshot saved to: {config_path}", flush=True)
        return config_path
    except Exception as e:
        print(f"[ERROR] Failed to save config snapshot: {e}", flush=True)
        import traceback

        traceback.print_exc()
        raise


def run(cfg: DictConfig):
    import sys

    print("[DEBUG] run_experiment.run() called", flush=True)
    print("[DEBUG] run_experiment.run() called", file=sys.stderr, flush=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"{device=}", flush=True)
    set_global_seed(cfg.seed)

    # Handle image size mismatch between dataset and model
    dataset_img_size = cfg.dataset.get("img_size", None)

    # Check if model uses create_timm_model and needs img_size
    if cfg.model.get("_target_") == "tensorion.models.vit.create_timm_model":
        from tensorion.models.vit import get_default_img_size

        model_name = cfg.model.get("model_name", "")
        model_default_size = get_default_img_size(model_name)

        # Determine final img_size to use
        final_img_size = None
        if "img_size" in cfg.model:
            final_img_size = cfg.model["img_size"]
        elif model_default_size is not None:
            final_img_size = model_default_size
        elif dataset_img_size is not None:
            final_img_size = dataset_img_size

        # If sizes don't match, use model's required size and update transforms
        if final_img_size is not None:
            if dataset_img_size is not None and dataset_img_size != final_img_size:
                print(
                    f"Info: Dataset img_size ({dataset_img_size}) != model required size ({final_img_size}). "
                    f"Automatically adding Resize({final_img_size}) to transforms."
                )
                # Add Resize to transforms config if available
                if "transforms" in cfg.dataset:
                    transforms_cfg = cfg.dataset.transforms
                    # Make config mutable to modify transforms
                    OmegaConf.set_struct(transforms_cfg, False)
                    # Add Resize at the beginning of both train and test transforms
                    for split in ["train", "test"]:
                        if (
                            split in transforms_cfg
                            and "transforms" in transforms_cfg[split]
                        ):
                            # Check if Resize already exists
                            has_resize = any(
                                str(t.get("_target_", "")).endswith("Resize")
                                for t in transforms_cfg[split].transforms
                            )
                            if not has_resize:
                                # Make transforms list mutable
                                transforms_list = list(transforms_cfg[split].transforms)
                                transforms_list.insert(
                                    0,
                                    OmegaConf.create(
                                        {
                                            "_target_": "torchvision.transforms.Resize",
                                            "size": final_img_size,
                                        }
                                    ),
                                )
                                transforms_cfg[split].transforms = transforms_list
                    OmegaConf.set_struct(transforms_cfg, True)

            # Make model config mutable to add img_size
            OmegaConf.set_struct(cfg.model, False)
            cfg.model["img_size"] = final_img_size
            OmegaConf.set_struct(cfg.model, True)

    model = instantiate(cfg.model).to(device)
    train_ds, test_ds, train_loader, test_loader = data_preparation.load_from_config(
        cfg
    )
    optimizer = instantiate(cfg.optimizer, params=model.parameters())

    scheduler_kwargs = {"optimizer": optimizer}
    if "steps_per_epoch" in cfg.scheduler.scheduler_kwargs.keys():
        scheduler_kwargs["steps_per_epoch"] = len(train_loader)

    scheduler = instantiate(cfg.scheduler.scheduler_kwargs, **scheduler_kwargs)

    scheduler_step = cfg.scheduler.get("scheduler_step", "epoch")

    # Resolve save_dir - it may contain Hydra interpolations
    # During Hydra multirun, the working directory is changed to the job subdirectory
    # and hydra.runtime.output_dir should be available
    save_dir = None
    cwd = os.getcwd()
    print(f"[DEBUG] Current working directory: {cwd}", flush=True)

    # First, try to get output_dir from hydra runtime (most reliable during Hydra runs)
    output_dir = None
    try:
        if hasattr(cfg, "hydra") and hasattr(cfg.hydra, "runtime"):
            output_dir = getattr(cfg.hydra.runtime, "output_dir", None)
            if output_dir:
                print(
                    f"[DEBUG] Found hydra.runtime.output_dir: {output_dir}", flush=True
                )
    except Exception as e:
        print(f"[DEBUG] Could not access hydra.runtime.output_dir: {e}", flush=True)

    # If we have output_dir, use it
    if output_dir:
        save_dir = os.path.join(str(output_dir), "results")
        print(
            f"[DEBUG] Resolved save_dir from hydra.runtime.output_dir: {save_dir}",
            flush=True,
        )
    else:
        # Fallback: try to resolve from config (may work if Hydra already resolved it)
        try:
            save_dir_raw = str(cfg.get("save_dir", ""))
            if save_dir_raw and not save_dir_raw.startswith("${"):
                save_dir = save_dir_raw
                print(f"[DEBUG] Resolved save_dir from config: {save_dir}", flush=True)
        except Exception as e:
            print(f"[DEBUG] Could not resolve save_dir from config: {e}", flush=True)

        # If still not resolved, infer from current directory
        if not save_dir or save_dir.startswith("${"):
            # Check if we're in a multirun subdirectory
            if "multirun" in cwd:
                # We're in a multirun subdirectory, use it as the base
                save_dir = os.path.join(cwd, "results")
                print(
                    f"[DEBUG] Inferred save_dir from current directory (multirun): {save_dir}",
                    flush=True,
                )
            else:
                # Not in multirun - this shouldn't happen during normal Hydra runs
                # But for debugging, create a local results directory
                save_dir = os.path.join(cwd, "results")
                print(
                    f"[WARNING] Not in multirun directory, using current directory for results: {save_dir}",
                    flush=True,
                )

    # Update cfg.save_dir with resolved value
    OmegaConf.set_struct(cfg, False)
    cfg.save_dir = save_dir
    OmegaConf.set_struct(cfg, True)

    # Create the results directory immediately to ensure it exists
    print(f"[DEBUG] Creating results directory: {save_dir}", flush=True)
    try:
        os.makedirs(save_dir, exist_ok=True)
        if os.path.exists(save_dir):
            print(f"[DEBUG] SUCCESS: Results directory created: {save_dir}", flush=True)
        else:
            print(
                f"[ERROR] FAILED: Results directory does not exist after creation attempt: {save_dir}",
                flush=True,
            )
            raise RuntimeError(f"Failed to create results directory: {save_dir}")
    except Exception as e:
        print(f"[ERROR] Exception while creating results directory: {e}", flush=True)
        import traceback

        traceback.print_exc()
        raise

    save_path = save_dir + "/checkpoints/"
    save_steps = cfg.get("save_steps", -1)

    criterion = instantiate(cfg.criterion)

    experiment_logger = build_experiment_logger(cfg)
    print(
        f"[DEBUG] Experiment logger created: type={type(experiment_logger)}, enabled={getattr(experiment_logger, 'enabled', False)}",
        flush=True,
    )
    if hasattr(experiment_logger, "reason"):
        print(f"[DEBUG] Logger reason: {experiment_logger.reason}", flush=True)
    config_snapshot_path = save_config_snapshot(cfg)
    experiment_logger.log_config(cfg, config_path=config_snapshot_path)

    # Log model, dataset, optimizer, and scheduler information
    if experiment_logger:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        experiment_logger.log_other("model/total_parameters", total_params)
        experiment_logger.log_other("model/trainable_parameters", trainable_params)
        experiment_logger.log_other("model/model_type", str(type(model).__name__))

        # Log dataset information
        experiment_logger.log_other("dataset/train_size", len(train_ds))
        experiment_logger.log_other("dataset/test_size", len(test_ds))
        experiment_logger.log_other("dataset/batch_size", train_loader.batch_size)
        experiment_logger.log_other("dataset/num_workers", train_loader.num_workers)

        # Log optimizer information
        experiment_logger.log_other("optimizer/type", str(type(optimizer).__name__))
        if hasattr(optimizer, "defaults"):
            for key, value in optimizer.defaults.items():
                if isinstance(value, (int, float, str, bool)):
                    experiment_logger.log_other(f"optimizer/{key}", value)

        # Log scheduler information
        experiment_logger.log_other("scheduler/type", str(type(scheduler).__name__))

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
