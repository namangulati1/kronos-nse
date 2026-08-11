#!/usr/bin/env python3
"""Fetch -> demo -> evaluate -> backtest, in one go."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["01_fetch_data.py", "02_demo_forecast.py", "03_run_eval.py", "04_run_backtest.py"]


def main() -> int:
    passthrough = sys.argv[1:]
    for step in STEPS:
        print(f"\n{'=' * 78}\n== {step}\n{'=' * 78}")
        extra = ["--cost-sweep"] if step == "04_run_backtest.py" else []
        rc = subprocess.run(
            [sys.executable, str(HERE / step), *passthrough, *extra], cwd=HERE
        ).returncode
        if rc != 0:
            print(f"[run_all] {step} exited {rc}; stopping.")
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
