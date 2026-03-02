import datetime as dt
import json
import os
import socket
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf


class ExperimentLogger:
    enabled = False

    def log_parameters(self, params: dict[str, Any]) -> None:
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
    def __init__(self, reason: Optional[str] = None) -> None:
        self.reason = reason


class CometExperimentLogger(ExperimentLogger):
    enabled = True

    def __init__(self, experiment: Any) -> None:
        self._experiment = experiment

    def log_parameters(self, params: dict[str, Any]) -> None:
        self._experiment.log_parameters(params)

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
    # Use both print and sys.stderr to ensure output is visible even if stdout is redirected
    comet_cfg = cfg.get("comet")
    msg = f"[DEBUG] CometML config check: comet_cfg={'present' if comet_cfg else 'None'}, enabled={comet_cfg.get('enabled') if comet_cfg else None}"
    print(msg, flush=True)
    print(msg, file=sys.stderr, flush=True)
    if comet_cfg is None or not comet_cfg.get("enabled", False):
        msg = (
            "[DEBUG] CometML disabled or not configured. Returning NullExperimentLogger"
        )
        print(msg, flush=True)
        print(msg, file=sys.stderr, flush=True)
        return NullExperimentLogger("comet.disabled")

    try:
        from comet_ml import Experiment, OfflineExperiment

        print("[DEBUG] CometML imported successfully", flush=True)
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[DEBUG] Comet ML import failed: {exc}", flush=True)
        warnings.warn(f"Comet ML import failed: {exc}")
        return NullExperimentLogger("comet.import_failed")

    # Try to get API key from config or environment variable
    # If not provided, CometML's Experiment class will automatically read from ~/.comet.config
    api_key = comet_cfg.get("api_key")
    if not api_key or (isinstance(api_key, str) and api_key.strip() == ""):
        api_key = os.getenv("COMET_API_KEY")

    workspace = comet_cfg.get("workspace") or os.getenv("COMET_WORKSPACE")
    project_name = comet_cfg.get("project_name") or os.getenv("COMET_PROJECT_NAME")
    # Default to ONLINE mode unless explicitly requested otherwise
    offline = bool(comet_cfg.get("offline", False))

    # Derive a resolved experiment name once, to use both for Comet init
    # and for subsequent configuration.
    # At actual Hydra runtime, accessing comet_cfg.get("experiment_name")
    # returns the fully resolved string (e.g. "opt=sgd_model=resnet18_ds=mnist"),
    # so we prefer that over any manual interpolation handling.
    experiment_name_resolved: Optional[str] = None
    try:
        raw_name = comet_cfg.get("experiment_name", None)
        if raw_name:
            experiment_name_resolved = str(raw_name)
    except Exception as e:
        print(
            f"[DEBUG] Could not derive experiment_name from comet config via get(): {e}",
            flush=True,
        )
        experiment_name_resolved = None

    # Fallback: use default experiment name if nothing resolved
    if not experiment_name_resolved:
        try:
            experiment_name_resolved = _default_experiment_name(cfg)
        except Exception as e:
            print(f"[DEBUG] Could not derive default experiment name: {e}", flush=True)
            experiment_name_resolved = None

    # Get offline directory from config or use default
    offline_directory = None
    try:
        # Try to get raw value without resolving Hydra interpolations
        if "offline_directory" in comet_cfg:
            offline_node = comet_cfg._get_node("offline_directory")
            if offline_node is not None:
                offline_directory = str(
                    offline_node._value()
                    if hasattr(offline_node, "_value")
                    else offline_node
                )
    except Exception:
        # Fallback: try direct access (may trigger resolution, but we catch it)
        try:
            offline_directory = comet_cfg.get("offline_directory")
        except Exception:
            pass

    if offline_directory:
        offline_directory = str(offline_directory)
        # Resolve ${save_dir} if present - try to resolve save_dir dynamically from config
        # Note: save_dir may be pre-resolved in run_experiment.py, but we handle both cases here
        if "${save_dir}" in offline_directory or "$save_dir" in offline_directory:
            save_dir_resolved = None
            try:
                # Use _safe_get_config_value to avoid triggering interpolation errors
                save_dir_raw = _safe_get_config_value(cfg, "save_dir", "")
                if save_dir_raw:
                    save_dir_resolved = str(save_dir_raw)
                    # Check if it's still unresolved (contains interpolation pattern)
                    if save_dir_resolved.startswith("${"):
                        save_dir_resolved = None
                    else:
                        # Successfully resolved
                        offline_directory = offline_directory.replace(
                            "${save_dir}", save_dir_resolved
                        )
                        offline_directory = offline_directory.replace(
                            "$save_dir", save_dir_resolved
                        )
            except Exception as e:
                print(f"[DEBUG] Could not resolve save_dir directly: {e}", flush=True)
                save_dir_resolved = None

            # If still not resolved, try to infer from config structure
            if not save_dir_resolved or save_dir_resolved.startswith("${"):
                try:
                    # Try to get output_dir from hydra runtime if available (only during actual runs)
                    output_dir = None
                    if hasattr(cfg, "hydra") and hasattr(cfg.hydra, "runtime"):
                        output_dir = getattr(cfg.hydra.runtime, "output_dir", None)

                    if output_dir:
                        # Build save_dir from inferred output_dir
                        save_dir_resolved = os.path.join(str(output_dir), "results")
                        offline_directory = offline_directory.replace(
                            "${save_dir}", save_dir_resolved
                        )
                        offline_directory = offline_directory.replace(
                            "$save_dir", save_dir_resolved
                        )
                    else:
                        # Last resort: use current working directory structure
                        # This assumes we're in a multirun subdirectory
                        cwd = os.getcwd()
                        if "multirun" in cwd:
                            # We're likely in a multirun subdirectory, use it as base
                            save_dir_resolved = os.path.join(cwd, "results")
                            offline_directory = offline_directory.replace(
                                "${save_dir}", save_dir_resolved
                            )
                            offline_directory = offline_directory.replace(
                                "$save_dir", save_dir_resolved
                            )
                            print(
                                f"[DEBUG] Inferred save_dir from current directory: {save_dir_resolved}",
                                flush=True,
                            )
                        else:
                            print(
                                f"[WARNING] Could not resolve save_dir, offline_directory may contain unresolved variables: {offline_directory}",
                                flush=True,
                            )
                except Exception as e:
                    print(f"[DEBUG] Could not infer save_dir: {e}", flush=True)
                    print(
                        f"[WARNING] offline_directory may contain unresolved variables: {offline_directory}",
                        flush=True,
                    )

        # Resolve Hydra runtime choice variables (e.g., ${hydra:runtime.choices.*})
        offline_directory = _resolve_hydra_vars(offline_directory, cfg)
        # Resolve any path variables
        offline_directory = os.path.expanduser(offline_directory)
        offline_directory = os.path.expandvars(offline_directory)
        # Create directory if it doesn't exist (including parent directories)
        # Also ensure the parent 'results' directory exists if we're creating comet_offline inside it
        if offline_directory:
            parent_dir = os.path.dirname(offline_directory)
            if parent_dir and not os.path.exists(parent_dir):
                print(
                    f"[DEBUG] Creating parent directory for offline logs: {parent_dir}",
                    flush=True,
                )
                os.makedirs(parent_dir, exist_ok=True)
            os.makedirs(offline_directory, exist_ok=True)
            print(
                f"[DEBUG] Offline directory created/verified: {offline_directory}",
                flush=True,
            )

    print(
        f"[DEBUG] CometML setup: offline={offline}, api_key={'***' if api_key else 'None'}, workspace={workspace}, project={project_name}",
        flush=True,
    )
    if offline:
        print(
            f"[INFO] CometML running in OFFLINE mode. Logs will be saved to: {offline_directory or 'default CometML offline directory'}",
            flush=True,
        )

    # For offline mode, API key is optional
    # For online mode, API key is required
    if not offline and not api_key:
        raise ValueError(
            "CometML API key is required for online mode but not found. "
            "Please provide it via: "
            "1. comet.api_key in config, "
            "2. COMET_API_KEY environment variable, or "
            "3. ~/.comet.config file. "
            "Alternatively, set comet.offline=true to use offline mode."
        )

    # Use OfflineExperiment for offline mode, Experiment for online mode
    if offline:
        experiment_cls = OfflineExperiment
    else:
        experiment_cls = Experiment

    exp_kwargs: dict[str, Any] = {
        "project_name": project_name,
        "workspace": workspace,
        "auto_param_logging": False,
        "auto_metric_logging": False,
        "auto_output_logging": False,
    }

    # Pass the resolved experiment name directly to Comet so that the
    # primary display name in the web console matches comet.experiment_name.
    if experiment_name_resolved:
        exp_kwargs["experiment_name"] = experiment_name_resolved

    # Add API key only for online mode
    if not offline and api_key:
        exp_kwargs["api_key"] = api_key

    # Add offline_directory for offline mode
    if offline and offline_directory:
        exp_kwargs["offline_directory"] = offline_directory

    try:
        print(
            f"[DEBUG] Creating CometML experiment with kwargs: {list(exp_kwargs.keys())}",
            flush=True,
        )
        experiment = experiment_cls(
            **{k: v for k, v in exp_kwargs.items() if v is not None}
        )
        print(
            f"[DEBUG] CometML experiment object created: {type(experiment)}", flush=True
        )

        # For offline mode, OfflineExperiment is expected
        if offline:
            if not isinstance(experiment, OfflineExperiment):
                print(
                    f"[WARNING] Expected OfflineExperiment but got {type(experiment)}",
                    flush=True,
                )
        else:
            # For online mode, check if CometML silently created an OfflineExperiment
            if isinstance(experiment, OfflineExperiment):
                raise RuntimeError(
                    "CometML created an OfflineExperiment instead of Experiment. "
                    "This usually means the API key is invalid or authentication failed. "
                    "Please verify your API key and ensure you have internet connectivity."
                )

            # Check if experiment is actually active (CometML might disable it silently)
            if hasattr(experiment, "disabled") and experiment.disabled:
                raise RuntimeError(
                    "CometML experiment was created but is disabled. "
                    "This usually means the API key was not found or is invalid. "
                    "Please verify your API key."
                )

        logger = CometExperimentLogger(experiment)
        print("[DEBUG] Configuring CometML experiment...", flush=True)
        _configure_experiment(
            logger,
            cfg,
            comet_cfg,
            offline=offline,
            experiment_name_override=experiment_name_resolved,
        )

        if offline:
            # For offline mode, just verify the experiment was created
            if hasattr(experiment, "experiment_key"):
                print(
                    f"[INFO] CometML offline experiment initialized with key: {experiment.experiment_key}",
                    flush=True,
                )
            elif hasattr(experiment, "id"):
                print(
                    f"[INFO] CometML offline experiment initialized with ID: {experiment.id}",
                    flush=True,
                )
            else:
                print(
                    "[INFO] CometML offline experiment initialized (logs will be saved locally)",
                    flush=True,
                )

            if offline_directory:
                print(f"[INFO] Offline logs directory: {offline_directory}", flush=True)
        else:
            # For online mode, verify experiment is online by checking URL or experiment key
            experiment_key = None
            url = None

            # Try to get experiment key first (more reliable indicator)
            if hasattr(experiment, "get_key"):
                try:
                    experiment_key = experiment.get_key()
                    print(
                        f"[DEBUG] CometML experiment key: {experiment_key}", flush=True
                    )
                except Exception as e:
                    print(f"[DEBUG] Could not get experiment key: {e}", flush=True)

            # Try to get URL
            if hasattr(experiment, "get_url"):
                try:
                    url = experiment.get_url()
                    if url:
                        print(
                            f"[INFO] CometML experiment initialized: {url}", flush=True
                        )
                    else:
                        print(
                            "[DEBUG] CometML experiment.get_url() returned None",
                            flush=True,
                        )
                except Exception as e:
                    print(f"[DEBUG] Could not get CometML URL: {e}", flush=True)

            # If we have either key or URL, experiment is initialized
            if not experiment_key and not url:
                # Check if experiment has other indicators of being active
                if hasattr(experiment, "id") and experiment.id:
                    print(
                        f"[INFO] CometML experiment initialized with ID: {experiment.id}",
                        flush=True,
                    )
                elif (
                    hasattr(experiment, "experiment_key") and experiment.experiment_key
                ):
                    print(
                        f"[INFO] CometML experiment initialized with key: {experiment.experiment_key}",
                        flush=True,
                    )
                else:
                    # Last resort: check if experiment is actually disabled
                    if hasattr(experiment, "disabled") and not experiment.disabled:
                        print(
                            "[INFO] CometML experiment initialized (no URL/key available, but not disabled)",
                            flush=True,
                        )
                    else:
                        raise RuntimeError(
                            "CometML experiment was created but appears to be invalid or disabled. "
                            "This usually means the API key is invalid or authentication failed. "
                            "Please verify your API key in the CometML dashboard."
                        )

            print(
                "[INFO] CometML experiment initialized successfully (online mode)",
                flush=True,
            )

        return logger
    except Exception as exc:
        import traceback

        print(f"[ERROR] Failed to initialize CometML experiment: {exc}", flush=True)
        print(f"[ERROR] Traceback: {traceback.format_exc()}", flush=True)
        # Re-raise the exception instead of falling back to NullExperimentLogger
        # This ensures we fail fast and don't silently skip logging
        raise RuntimeError(
            f"Failed to initialize CometML experiment: {exc}. "
            f"Offline mode: {offline}, API key provided: {bool(api_key)}. "
            f"Please fix the configuration and try again."
        ) from exc


def flatten_config(cfg: DictConfig) -> dict[str, Any]:
    """Flatten config to a dictionary, handling unresolved Hydra interpolations.

    First tries to resolve all interpolations. If that fails (e.g., due to
    unresolved ${hydra:runtime.*} variables), falls back to unresolved mode.
    This is expected behavior when config contains Hydra runtime variables.
    """
    try:
        # Try to resolve all interpolations (works if Hydra is running)
        data = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        # Fallback: get unresolved config (expected when Hydra runtime is not available)
        # This is safe - we just won't have resolved interpolations in the flattened dict
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
    logger: ExperimentLogger,
    cfg: DictConfig,
    comet_cfg: DictConfig,
    offline: bool = False,
    experiment_name_override: Optional[str] = None,
) -> None:
    # Get experiment_name - either use the pre-resolved override from build_experiment_logger
    # or resolve it here as a fallback.
    name: Optional[str] = experiment_name_override
    if not name:
        try:
            # Use _get_node to get the raw value without resolving interpolations
            if "experiment_name" in comet_cfg and hasattr(comet_cfg, "_get_node"):
                name_raw = comet_cfg._get_node("experiment_name")
                if name_raw is not None:
                    # Get the raw string value
                    name_str = str(
                        name_raw._value() if hasattr(name_raw, "_value") else name_raw
                    )
                    # Resolve Hydra runtime choices manually
                    name = _resolve_hydra_vars(name_str, cfg)
        except Exception as e:
            print(f"[DEBUG] Could not get experiment_name: {e}", flush=True)
            # Fallback: try direct access (may trigger resolution error, but we catch it)
            try:
                name_raw = comet_cfg.get("experiment_name", None)
                if name_raw:
                    name = _resolve_hydra_vars(str(name_raw), cfg)
            except Exception:
                pass

    if not name:
        name = _default_experiment_name(cfg)

    if name:
        logger.set_name(str(name))
        # Also log the final resolved experiment name for easy inspection in the UI
        logger.log_other("debug/experiment_name", str(name))

    # Get tags - resolve Hydra variables manually
    tags = None
    try:
        # Use _get_node to get raw values without resolving interpolations
        if "tags" in comet_cfg:
            tags_node = comet_cfg._get_node("tags")
            if tags_node is not None:
                # Get raw list value
                tags_raw = (
                    tags_node._value() if hasattr(tags_node, "_value") else tags_node
                )
                if tags_raw:
                    if isinstance(tags_raw, str):
                        tags_raw = [tags_raw]
                    # Resolve Hydra variables in each tag
                    tags = [_resolve_hydra_vars(str(tag), cfg) for tag in tags_raw]
    except Exception as e:
        print(f"[DEBUG] Could not get tags (first attempt): {e}", flush=True)
        # Fallback: try direct access
        try:
            tags_raw = comet_cfg.get("tags", None)
            if tags_raw:
                if isinstance(tags_raw, str):
                    tags_raw = [tags_raw]
                tags = [_resolve_hydra_vars(str(tag), cfg) for tag in tags_raw]
        except Exception as e2:
            print(f"[DEBUG] Could not get tags (fallback): {e2}", flush=True)

    if tags is None:
        tags = _default_tags(cfg)

    if isinstance(tags, str):
        tags = [tags]

    if tags:
        logger.add_tags([str(tag) for tag in tags])


def _resolve_hydra_vars(text: str, cfg: DictConfig) -> str:
    """Resolve Hydra runtime choice variables in a string.

    Replaces ${hydra:runtime.choices.*} with actual values from the config.
    """
    import re

    # Pattern to match ${hydra:runtime.choices.KEY}
    pattern = r"\$\{hydra:runtime\.choices\.(\w+)\}"

    def replace_var(match):
        key = match.group(1)
        try:
            # Try to get the value from hydra runtime choices
            # In Hydra, these are available via hydra.runtime.choices
            if hasattr(cfg, "hydra") and hasattr(cfg.hydra, "runtime"):
                runtime = cfg.hydra.runtime
                if hasattr(runtime, "choices") and key in runtime.choices:
                    return str(runtime.choices[key])

            # Alternative: try to get from Hydra's global state
            try:
                from hydra.core.global_hydra import GlobalHydra

                gh = GlobalHydra.instance()
                if gh.is_initialized():
                    hydra_cfg = gh.hydra
                    if hasattr(hydra_cfg, "runtime") and hasattr(
                        hydra_cfg.runtime, "choices"
                    ):
                        if key in hydra_cfg.runtime.choices:
                            return str(hydra_cfg.runtime.choices[key])
            except Exception:
                pass

            # Fallback: try to infer from config structure
            # For example, ${hydra:runtime.choices.optimizer} might correspond to the optimizer name
            # This is a heuristic and may not work for all cases
            return match.group(0)  # Return original if can't resolve
        except Exception:
            return match.group(0)  # Return original if can't resolve

    return re.sub(pattern, replace_var, text)

    logger.log_other("host", socket.gethostname())
    logger.log_other("start_time_utc", dt.datetime.utcnow().isoformat())
    # Note: save_dir is pre-resolved in run_experiment.py before build_experiment_logger() is called,
    # so this should be safe. Using _safe_get_config_value for extra safety.
    save_dir = _safe_get_config_value(cfg, "save_dir")
    if save_dir is not None:
        logger.log_other("save_dir", str(save_dir))
    seed = _safe_get_config_value(cfg, "seed")
    if seed is not None:
        logger.log_other("seed", seed)

    git_commit = _get_git_commit()
    if git_commit:
        logger.log_other("git_commit", git_commit)
    if offline:
        logger.log_other("comet.offline", True)

    if comet_cfg.get("log_code", False):
        # Use _safe_get_config_value to avoid potential interpolation errors
        # code_dir is unlikely to contain interpolations, but we handle it safely
        code_dir = _safe_get_config_value(cfg, "comet.code_dir") or os.getcwd()
        logger.log_code(folder=str(code_dir))


def _default_experiment_name(cfg: DictConfig) -> Optional[str]:
    # Note: save_dir is pre-resolved in run_experiment.py before this function is called,
    # so this should be safe. Using _safe_get_config_value for extra safety.
    save_dir = _safe_get_config_value(cfg, "save_dir")
    if save_dir:
        return Path(str(save_dir)).name
    return None


def _default_tags(cfg: DictConfig) -> list[str]:
    """Generate default tags from config, safely handling potential interpolation errors.

    These keys (model._target_, optimizer._target_, etc.) are unlikely to contain
    Hydra interpolations, but we use _safe_get_config_value for consistency and safety.
    """
    tags = []
    for key in [
        "model._target_",
        "optimizer._target_",
        "dataset._target_",
        "scheduler._target_",
    ]:
        value = _safe_get_config_value(cfg, key)
        if value:
            tags.append(str(value).split(".")[-1])
    return tags


def _safe_get_config_value(
    cfg: DictConfig, key: str, default: Optional[Any] = None
) -> Optional[Any]:
    """Safely get a config value without triggering Hydra interpolation resolution errors.

    First tries to get the raw value using _get_node() to avoid interpolation resolution.
    If that fails or the value contains unresolved interpolations, tries to resolve it.
    If resolution fails, returns the default value.

    Args:
        cfg: The config to read from
        key: The key to read (supports dot notation like "comet.code_dir")
        default: Default value to return if key is missing or resolution fails

    Returns:
        The config value or default if not found/resolution fails
    """
    try:
        # Try to get raw value without resolving interpolations
        if "." in key:
            # Handle nested keys by getting the parent node first
            parts = key.split(".")
            parent = cfg
            for part in parts[:-1]:
                if hasattr(parent, "_get_node"):
                    parent_node = parent._get_node(part)
                    if parent_node is None:
                        return default
                    parent = (
                        parent_node._value()
                        if hasattr(parent_node, "_value")
                        else parent_node
                    )
                else:
                    try:
                        parent = parent.get(part)
                        if parent is None:
                            return default
                    except Exception:
                        return default

            # Get the final node
            final_key = parts[-1]
            if hasattr(parent, "_get_node"):
                node = parent._get_node(final_key)
                if node is not None:
                    raw_value = node._value() if hasattr(node, "_value") else node
                    # Try to resolve if it's a string with interpolations
                    if isinstance(raw_value, str) and "${" in raw_value:
                        try:
                            # Try to resolve - Hydra will resolve if running
                            resolved = OmegaConf.select(cfg, key)
                            if resolved and not (
                                isinstance(resolved, str) and resolved.startswith("${")
                            ):
                                return resolved
                        except Exception:
                            pass
                    return raw_value
            else:
                # Parent is not a DictConfig, try direct access
                try:
                    value = (
                        parent.get(final_key, default)
                        if hasattr(parent, "get")
                        else default
                    )
                    return value
                except Exception:
                    return default
        else:
            # Simple key, use _get_node directly
            if hasattr(cfg, "_get_node"):
                node = cfg._get_node(key)
                if node is not None:
                    raw_value = node._value() if hasattr(node, "_value") else node
                    # Try to resolve if it's a string with interpolations
                    if isinstance(raw_value, str) and "${" in raw_value:
                        try:
                            # Try to resolve - Hydra will resolve if running
                            resolved = cfg.get(key, default)
                            if resolved and not (
                                isinstance(resolved, str) and resolved.startswith("${")
                            ):
                                return resolved
                        except Exception:
                            pass
                    return raw_value

        # Fallback: try direct access (may trigger resolution, but we catch it)
        try:
            value = OmegaConf.select(cfg, key)
            if value is not None:
                return value
        except Exception:
            pass

        # Last resort: try cfg.get() for simple keys
        if "." not in key:
            try:
                value = cfg.get(key, default)
                if value is not None:
                    return value
            except Exception:
                pass

        # Key not found or all access methods failed
        return default
    except Exception:
        return default


def _safe_select(cfg: DictConfig, key: str) -> Optional[Any]:
    """Legacy function - use _safe_get_config_value() instead.

    This function is kept for backward compatibility but may still trigger
    interpolation errors. For new code, use _safe_get_config_value().
    """
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
