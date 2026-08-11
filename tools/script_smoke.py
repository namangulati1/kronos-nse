#!/usr/bin/env python3
"""Exercise the 03/04 CLI scripts end to end with no network and no weights.

Patches the data loader to emit synthetic bars and the model loader to build a
tiny random Kronos, then calls each script's ``main()``. This catches wiring
bugs in the scripts themselves (tags, cache paths, save/load round-trips,
summary tables) that the unit tests do not reach.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from tools.offline_dryrun import synthetic, tiny_runner  # noqa: E402

import kronos_nse.data as dataset  # noqa: E402
import kronos_nse.predictor as predictor_mod  # noqa: E402

_FRAMES = {"^NSEI": synthetic(600, seed=1), "RELIANCE.NS": synthetic(600, seed=2)}
_RUNNER = None


def _fake_fetch(symbol, **kwargs):
    return _FRAMES[symbol]


def _fake_runner(*args, **kwargs):
    return _RUNNER


def main() -> int:
    global _RUNNER
    _RUNNER = tiny_runner()

    dataset.fetch = _fake_fetch
    predictor_mod.KronosRunner = _fake_runner

    import importlib

    for mod_name in ["03_run_eval", "04_run_backtest"]:
        mod = importlib.import_module(mod_name)
        mod.dataset.fetch = _fake_fetch
        if hasattr(mod, "KronosRunner"):
            mod.KronosRunner = _fake_runner

        argv = sys.argv
        sys.argv = [
            mod_name,
            "--symbols", "^NSEI", "RELIANCE.NS",
            "--set",
            "forecast.lookback=100", "forecast.horizon=5", "forecast.n_samples=8",
            "forecast.sample_batch=8", "evaluate.start=null", "evaluate.stride=10",
            "evaluate.max_windows=15", "evaluate.out_dir=results/script_smoke",
            "model.max_context=128", "backtest.threshold_bps=5",
            "backtest.n_bootstrap=200",
        ]
        if mod_name == "04_run_backtest":
            sys.argv.append("--cost-sweep")

        print(f"\n{'=' * 70}\n== {mod_name}\n{'=' * 70}")
        rc = mod.main()
        sys.argv = argv
        if rc != 0:
            print(f"[smoke] {mod_name} returned {rc}")
            return rc

    print("\n[smoke] scripts 03 and 04 ran clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
