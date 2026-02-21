from typing import Any

from omegaconf import ListConfig, OmegaConf


def _normalize_to_list(value: Any) -> list[Any]:
    if value is None:
        return []

    if isinstance(value, ListConfig):
        value = OmegaConf.to_container(value, resolve=True)

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    return [value]


def append_to_list(list_value: Any, extra_list: Any) -> list[Any]:
    result = _normalize_to_list(list_value)
    for item in _normalize_to_list(extra_list):
        if item != "":
            result.append(item)
    return result


def register_resolvers() -> None:
    if not OmegaConf.has_resolver("to_str"):
        OmegaConf.register_new_resolver("to_str", str)

    if not OmegaConf.has_resolver("append"):
        OmegaConf.register_new_resolver("append", append_to_list, use_cache=False)
