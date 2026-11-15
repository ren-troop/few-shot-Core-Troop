"""YAML configuration loading with dotted CLI overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


class DotDict(dict):
    """A dict whose values can be read with attribute access."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        data = dict(*args, **kwargs)
        for key, value in data.items():
            dict.__setitem__(self, key, _to_dotdict(value))

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = _to_dotdict(value)


def _to_dotdict(value: Any) -> Any:
    if isinstance(value, DotDict):
        return value
    if isinstance(value, Mapping):
        out = DotDict()
        for key, item in value.items():
            dict.__setitem__(out, key, _to_dotdict(item))
        return out
    if isinstance(value, list):
        return [_to_dotdict(v) for v in value]
    return value


def _deep_merge(base: dict, override: Mapping[str, Any]) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, Mapping)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _set_dotted(cfg: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def _parse_override(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise ValueError(f"Override must look like key=value, got: {text!r}")
    key, raw = text.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Empty key in override: {text!r}")
    value = yaml.safe_load(raw.strip())
    return key, value


def load_config(
    path: str | Path,
    overrides: Iterable[str] | None = None,
) -> DotDict:
    """Load ``path``, apply ``key=value`` overrides, and return a DotDict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Top-level YAML object must be a mapping: {path}")

    for item in overrides or []:
        key, value = _parse_override(item)
        _set_dotted(raw, key, value)

    cfg = _to_dotdict(raw)
    cfg["_config_path"] = str(path.resolve())
    return cfg


def _to_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _to_plain(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value


def save_config(cfg: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_to_plain(cfg), handle, sort_keys=False, allow_unicode=True)
