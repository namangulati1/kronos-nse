"""Shared CLI plumbing for the scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kronos_nse.config import load_config  # noqa: E402



def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    p.add_argument("--symbols", nargs="*", default=None,
                   help="Override the symbol list from the config.")
    p.add_argument("--set", nargs="*", default=[], metavar="key.sub=value",
                   help="Dotted config overrides, e.g. --set forecast.horizon=10")
    return p


def build_config(args):
    cfg = load_config(args.config, args.set)
    if args.symbols:
        cfg.set_path("data.symbols", list(args.symbols))
    return cfg


def data_kwargs(cfg) -> dict:
    d = cfg.data
    return {
        "interval": d.get("interval", "1d"),
        "start": d.get("start"),
        "end": d.get("end"),
        "auto_adjust": bool(d.get("auto_adjust", True)),
        "cache_dir": str(REPO_ROOT / d.get("cache_dir", "data/cache")),
    }


def out_dir(cfg) -> Path:
    p = REPO_ROOT / cfg.get_path("evaluate.out_dir", "results")
    p.mkdir(parents=True, exist_ok=True)
    return p


def tag_for(symbol: str, cfg) -> str:
    safe = symbol.replace("^", "IDX_").replace(".", "_")
    return f"{safe}__{cfg.data.get('interval', '1d')}__h{cfg.forecast.horizon}"
