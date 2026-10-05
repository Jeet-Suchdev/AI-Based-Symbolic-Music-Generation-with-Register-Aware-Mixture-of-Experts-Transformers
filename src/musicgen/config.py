from __future__ import annotations
from pathlib import Path
from copy import deepcopy
import yaml

def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def deep_merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_config(path: str | Path) -> dict:
    cfg = load_yaml(path)
    return cfg

def set_by_dotted_key(cfg: dict, key: str, value):
    parts = key.split(".")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value

def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    if not overrides:
        return cfg
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item}")
        key, value = item.split("=", 1)
        try:
            parsed = yaml.safe_load(value)
        except yaml.YAMLError:
            parsed = value
        set_by_dotted_key(cfg, key, parsed)
    return cfg
