"""Locate the upstream Kronos source tree and put it on ``sys.path``.

Kronos is not published to PyPI - ``from model import Kronos`` only works from
a checkout of the upstream repository. Rather than vendoring (and slowly
drifting from) a copy of that code, we clone it into ``third_party/Kronos``
and import from there.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

UPSTREAM_URL = "https://github.com/shiyu-coder/Kronos.git"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KRONOS_HOME = REPO_ROOT / "third_party" / "Kronos"


def kronos_home() -> Path:
    """Where the upstream checkout lives (override with ``KRONOS_HOME``)."""
    env = os.environ.get("KRONOS_HOME")
    return Path(env).expanduser().resolve() if env else DEFAULT_KRONOS_HOME


def is_installed(path: Path | None = None) -> bool:
    path = path or kronos_home()
    return (path / "model" / "kronos.py").exists()


def clone(path: Path | None = None, ref: str | None = None) -> Path:
    """Shallow-clone the upstream repo. Idempotent."""
    path = path or kronos_home()
    if is_installed(path):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1"]
    if ref:
        cmd += ["--branch", ref]
    cmd += [UPSTREAM_URL, str(path)]
    print(f"[vendor] cloning Kronos -> {path}")
    subprocess.run(cmd, check=True)
    if not is_installed(path):
        raise RuntimeError(f"Clone finished but {path}/model/kronos.py is missing.")
    return path


def ensure_on_path(auto_clone: bool = False) -> Path:
    """Make ``import model`` resolve to the upstream Kronos package."""
    path = kronos_home()
    if not is_installed(path):
        if auto_clone:
            clone(path)
        else:
            raise RuntimeError(
                f"Kronos source not found at {path}.\n"
                "Run `make setup` (or `python scripts/00_setup.py`) to clone it, "
                "or point KRONOS_HOME at an existing checkout."
            )
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


def import_kronos(auto_clone: bool = False):
    """Return ``(Kronos, KronosTokenizer, KronosPredictor)`` from upstream."""
    ensure_on_path(auto_clone=auto_clone)
    from model import Kronos, KronosPredictor, KronosTokenizer  # type: ignore

    return Kronos, KronosTokenizer, KronosPredictor
