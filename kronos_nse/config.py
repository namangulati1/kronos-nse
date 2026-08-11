"""Config loading: YAML -> attribute-access dict, with dotted overrides."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"



class Config(dict):
    """A dict that also supports attribute access, recursively."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node: dict = self
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value


def _coerce(text: str) -> Any:
    """Parse a CLI override value using YAML rules (so 5, true, null, [a,b] work)."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load YAML config, apply environment variables, and apply ``key.subkey=value`` overrides."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    cfg = Config(copy.deepcopy(raw))

    # Environment variable overrides (Env > YAML default)
    env_mappings = {
        "KRONOS_MODEL_NAME": "model.name",
        "KRONOS_TOKENIZER_NAME": "model.tokenizer",
        "KRONOS_DEVICE": "model.device",
        "KRONOS_CACHE_DIR": "data.cache_dir",
        "KRONOS_RESULTS_DIR": "evaluate.out_dir",
    }
    for env_var, config_path in env_mappings.items():
        val = os.getenv(env_var)
        if val is not None and val.strip() != "":
            cfg.set_path(config_path, _coerce(val.strip()))

    # CLI overrides (CLI > Env > YAML default)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got: {item!r}")
        key, _, value = item.partition("=")
        cfg.set_path(key.strip(), _coerce(value.strip()))

    _validate(cfg)
    return cfg



def _validate(cfg: Config) -> None:
    lookback = cfg.get_path("forecast.lookback")
    max_context = cfg.get_path("model.max_context")
    if lookback and max_context and lookback > max_context:
        raise ValueError(
            f"forecast.lookback ({lookback}) exceeds model.max_context ({max_context}). "
            "Kronos-small/base have a 512-bar context; Kronos-mini has 2048."
        )
    if cfg.get_path("forecast.horizon", 1) < 1:
        raise ValueError("forecast.horizon must be >= 1")
    if cfg.get_path("forecast.n_samples", 1) < 2:
        raise ValueError(
            "forecast.n_samples must be >= 2. Kronos is a stochastic sampler; "
            "single-path forecasts carry no distributional information."
        )
