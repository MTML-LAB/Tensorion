import datetime as dt
import json
import os
import socket
import subprocess
import warnings
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn

from omegaconf import DictConfig, OmegaConf


class ExperimentLogger:
    enabled = False
    track_weight_norms: bool

    def __init__(self, track_weight_norms: bool = False) -> None:
        self.track_weight_norms = track_weight_norms

    def log_parameters(self, params: dict[str, Any]) -> None:
        return None

    def log_weight_norms(
        self,
        model: nn.Module,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ):
        return None

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> None:
        return None

    def log_asset(self, path: str, name: Optional[str] = None) -> None:
        return None

    def log_asset_folder(self, path: str, name: Optional[str] = None) -> None:
        return None

    def set_name(self, name: str) -> None:
        return None

    def add_tags(self, tags: list[str]) -> None:
        return None

    def log_other(self, key: str, value: Any) -> None:
        return None

    def log_code(self, folder: Optional[str] = None) -> None:
        return None

    def log_config(self, cfg: DictConfig, config_path: Optional[str] = None) -> None:
        params = flatten_config(cfg)
        self.log_parameters(params)
        if config_path is not None and os.path.exists(config_path):
            self.log_asset(config_path, name="config.yaml")

    def end(self) -> None:
        return None

    def __bool__(self) -> bool:
        return self.enabled


class NullExperimentLogger(ExperimentLogger):
    def __init__(self, reason: Optional[str] = None, **base_kwargs) -> None:
        super().__init__(**base_kwargs)
        self.reason = reason


class CometExperimentLogger(ExperimentLogger):
    enabled = True

    def __init__(self, experiment: Any, **base_kwargs) -> None:
        super().__init__(**base_kwargs)
        self._experiment = experiment

    def log_parameters(self, params: dict[str, Any]) -> None:
        self._experiment.log_parameters(params)

    @torch.no_grad()
    def log_weight_norms(
        self,
        model: nn.Module,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ):
        if not self.track_weight_norms:
            return
        
        norms = {
            f'weight/{name}': param.norm().item()
            for name, param in model.named_parameters()
            if param.requires_grad 
        }
        self._experiment.log_metrics(norms, step=step, epoch=epoch)


    def log_metrics(
        self,
        metrics: dict[str, float],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> None:
        self._experiment.log_metrics(metrics, step=step, epoch=epoch)

    def log_asset(self, path: str, name: Optional[str] = None) -> None:
        if name is None:
            self._experiment.log_asset(path)
            return None
        self._experiment.log_asset(path, file_name=name)

    def log_asset_folder(self, path: str, name: Optional[str] = None) -> None:
        folder = Path(path)
        if not folder.exists():
            return None
        for file_path in folder.rglob("*"):
            if not file_path.is_file():
                continue
            asset_name = None
            if name is not None:
                rel = file_path.relative_to(folder).as_posix()
                asset_name = f"{name}/{rel}"
            self.log_asset(str(file_path), name=asset_name)

    def set_name(self, name: str) -> None:
        self._experiment.set_name(name)

    def add_tags(self, tags: list[str]) -> None:
        if tags:
            self._experiment.add_tags(tags)

    def log_other(self, key: str, value: Any) -> None:
        self._experiment.log_other(key, value)

    def log_code(self, folder: Optional[str] = None) -> None:
        if folder is None:
            self._experiment.log_code()
            return None
        self._experiment.log_code(folder=folder)

    def end(self) -> None:
        self._experiment.end()


def build_experiment_logger(cfg: DictConfig) -> ExperimentLogger:
    comet_cfg = cfg.get("comet")
    if comet_cfg is None or not comet_cfg.get("enabled", False):
        return NullExperimentLogger("comet.disabled")

    try:
        from comet_ml import Experiment, OfflineExperiment
    except Exception as exc:  # pragma: no cover - optional dependency
        warnings.warn(f"Comet ML import failed: {exc}")
        return NullExperimentLogger("comet.import_failed")

    api_key = comet_cfg.get("api_key") or os.getenv("COMET_API_KEY")
    workspace = comet_cfg.get("workspace") or os.getenv("COMET_WORKSPACE")
    project_name = comet_cfg.get("project_name") or os.getenv("COMET_PROJECT_NAME")
    offline = bool(comet_cfg.get("offline", False))

    if not api_key and not offline:
        warnings.warn("Comet enabled but no API key set; disable or enable offline mode.")
        return NullExperimentLogger("comet.no_api_key")

    experiment_cls = OfflineExperiment if (offline and not api_key) else Experiment

    exp_kwargs: dict[str, Any] = {
        "project_name": project_name,
        "workspace": workspace,
        "auto_param_logging": False,
        "auto_metric_logging": False,
        "auto_output_logging": False,
    }
    offline_dir = comet_cfg.get("offline_directory")
    if offline_dir:
        exp_kwargs["offline_directory"] = offline_dir
    if api_key:
        exp_kwargs["api_key"] = api_key

    experiment = experiment_cls(**{k: v for k, v in exp_kwargs.items() if v is not None})

    logger_kwargs = {
        'track_weight_norms': comet_cfg.get('track_weight_norms', False)
    }
    logger = CometExperimentLogger(experiment, **logger_kwargs)
    _configure_experiment(logger, cfg, comet_cfg, offline=offline)
    
    return logger


def flatten_config(cfg: DictConfig) -> dict[str, Any]:
    try:
        data = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        data = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(data, dict):
        return {}
    return _flatten_dict(data)


def _flatten_dict(
    data: dict[str, Any], parent_key: str = "", sep: str = "."
) -> dict[str, Any]:
    items: dict[str, Any] = {}
    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
        if _is_sensitive_key(new_key):
            continue
        if isinstance(value, dict):
            items.update(_flatten_dict(value, new_key, sep=sep))
        else:
            items[new_key] = _sanitize_value(value)
    return items


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return json.dumps(value, default=str, ensure_ascii=True)
    return str(value)


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(token in lowered for token in ["api_key", "secret", "token"])


def _configure_experiment(
    logger: ExperimentLogger, cfg: DictConfig, comet_cfg: DictConfig, offline: bool
) -> None:
    name = comet_cfg.get("experiment_name") or _default_experiment_name(cfg)
    if name:
        logger.set_name(str(name))

    tags = comet_cfg.get("tags")
    if tags is None:
        tags = _default_tags(cfg)
    if isinstance(tags, str):
        tags = [tags]
    if tags:
        logger.add_tags([str(tag) for tag in tags])

    logger.log_other("host", socket.gethostname())
    logger.log_other("start_time_utc", dt.datetime.utcnow().isoformat())
    save_dir = _safe_select(cfg, "save_dir")
    if save_dir is not None:
        logger.log_other("save_dir", str(save_dir))
    seed = _safe_select(cfg, "seed")
    if seed is not None:
        logger.log_other("seed", seed)

    git_commit = _get_git_commit()
    if git_commit:
        logger.log_other("git_commit", git_commit)
    if offline:
        logger.log_other("comet.offline", True)

    if comet_cfg.get("log_code", False):
        code_dir = comet_cfg.get("code_dir") or os.getcwd()
        logger.log_code(folder=str(code_dir))


def _default_experiment_name(cfg: DictConfig) -> Optional[str]:
    save_dir = _safe_select(cfg, "save_dir")
    if save_dir:
        return Path(str(save_dir)).name
    return None


def _default_tags(cfg: DictConfig) -> list[str]:
    tags = []
    for key in [
        "model._target_",
        "optimizer._target_",
        "dataset._target_",
        "scheduler._target_",
    ]:
        value = _safe_select(cfg, key)
        if value:
            tags.append(str(value).split(".")[-1])
    return tags


def _safe_select(cfg: DictConfig, key: str) -> Optional[Any]:
    try:
        return OmegaConf.select(cfg, key)
    except Exception:
        return None


def _get_git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None
