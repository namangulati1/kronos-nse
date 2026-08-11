#!/usr/bin/env python3
"""Clone the upstream Kronos repo and check the environment."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kronos_nse import vendor  # noqa: E402


def main() -> int:
    path = vendor.clone()
    print(f"[setup] Kronos source ready at {path}")

    ok = True
    try:
        import torch

        print(f"[setup] torch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"[setup] CUDA available: {torch.cuda.get_device_name(0)}")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            print("[setup] Apple MPS available")
        else:
            print("[setup] No GPU detected - CPU inference works but is ~10-30x slower. "
                  "Consider Kronos-small over Kronos-base, and reduce forecast.n_samples.")
    except ImportError:
        print("[setup] torch MISSING - pip install -r requirements.txt")
        ok = False

    for mod in ["pandas", "numpy", "yfinance", "matplotlib", "scipy", "yaml", "einops",
                "huggingface_hub", "dotenv"]:
        try:
            __import__(mod)
        except ImportError:
            print(f"[setup] missing dependency: {mod}")
            ok = False

    # Check environment configuration (.env & env vars)
    import os
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        print(f"[setup] .env file found at {env_file}")
    else:
        print("[setup] NOTICE: .env file not found. Run 'cp .env.example .env' to customize your setup.")

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("[setup] HF_TOKEN is configured in environment.")
    else:
        print("[setup] HF_TOKEN not set (optional for public models, recommended if rate limited).")

    kronos_home_env = os.getenv("KRONOS_HOME")
    if kronos_home_env:
        print(f"[setup] KRONOS_HOME override active: {kronos_home_env}")

    try:
        Kronos, KronosTokenizer, _ = vendor.import_kronos()
        print("[setup] upstream Kronos classes import cleanly")
    except Exception as exc:  # noqa: BLE001
        print(f"[setup] could not import upstream Kronos: {exc}")
        ok = False


    print("[setup] OK" if ok else "[setup] issues found - see above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
