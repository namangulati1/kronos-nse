#!/usr/bin/env python3
"""End-to-end dry run with NO network and NO pretrained weights.

Builds a tiny randomly-initialised Kronos and a synthetic price series, then
drives the real evaluate -> score -> backtest -> plot path. Use it to verify an
install, or after changing the harness, before spending GPU time on the real
thing. The numbers it prints are meaningless by construction: a random model on
a random walk should score ~0.5 directional accuracy and a Sharpe near zero.
That is the point - if this run shows a great Sharpe, the harness is broken.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from kronos_nse import backtest as bt  # noqa: E402
from kronos_nse import evaluate, plotting  # noqa: E402
from kronos_nse.config import Config  # noqa: E402
from kronos_nse.predictor import KronosRunner  # noqa: E402
from kronos_nse.vendor import import_kronos  # noqa: E402

OUT = REPO_ROOT / "results" / "dryrun"


def synthetic(n: int = 600, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0003, 0.011, n)
    close = 20000.0 * np.exp(np.cumsum(ret))
    wick = np.abs(rng.normal(0, 0.004, n)) * close
    open_ = close * np.exp(rng.normal(0, 0.003, n))
    df = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + wick,
            "low": np.minimum(open_, close) - wick,
            "close": close,
            "volume": rng.lognormal(14, 0.35, n),
        },
        index=pd.bdate_range("2023-01-02", periods=n, freq="B"),
    )
    df.index.name = "timestamp"
    df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)
    return df


def tiny_runner() -> KronosRunner:
    Kronos, KronosTokenizer, KronosPredictor = import_kronos(auto_clone=True)
    tok = KronosTokenizer(
        d_in=6, d_model=48, n_heads=2, ff_dim=96, n_enc_layers=2, n_dec_layers=2,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        s1_bits=5, s2_bits=5, beta=0.0, gamma0=1.0, gamma=1.0, zeta=1.0, group_size=1,
    )
    mdl = Kronos(
        s1_bits=5, s2_bits=5, n_layers=2, d_model=48, n_heads=2, ff_dim=96,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        token_dropout_p=0.0, learn_te=True,
    )
    tok.eval()
    mdl.eval()
    pred = KronosPredictor(mdl, tok, device="cpu", max_context=128, clip=5)
    return KronosRunner(device="cpu", max_context=128, _predictor=pred)


def main() -> int:
    cfg = Config({
        "forecast": {"lookback": 120, "horizon": 5, "n_samples": 16,
                     "temperature": 1.0, "top_p": 0.9, "top_k": 0, "sample_batch": 16},
        "evaluate": {"start": None, "end": None, "stride": 3, "max_windows": 40,
                     "seed": 0, "out_dir": str(OUT)},
        "backtest": {"signal": "expected_return", "sizing": "threshold",
                     "threshold_bps": 10, "threshold_prob": 0.03, "max_leverage": 1.0,
                     "allow_short": True, "overlap": True, "execution": "next_open",
                     "cost_bps_roundtrip": 25, "ann_factor": 252, "n_bootstrap": 300},
        "model": {"name": "random-init/tiny"},
    })

    print("== building tiny random model (no weights downloaded) ==")
    runner = tiny_runner()
    df = synthetic()

    print("== walk-forward evaluation ==")
    recs, samples = evaluate.run_symbol("SYNTH", df, runner, cfg)
    evaluate.save(OUT, "synth", recs, samples)

    sc = evaluate.score(recs, samples, int(cfg.forecast.horizon))
    print("\n-- forecast quality (expect noise) --")
    for k in ["n_windows", "directional_accuracy", "da_ci_lo", "da_ci_hi", "brier_skill",
              "crps", "coverage_90", "kronos_rmse", "bl_naive_rmse",
              "skill_vs_bl_naive_rmse", "vol_spearman"]:
        if k in sc:
            print(f"   {k:<28} {sc[k]:.4f}")

    print("\n== backtest ==")
    prices = df.loc[recs["decision_ts"].min():recs["target_ts"].max()]
    res = bt.run(recs, prices, cfg)
    for k in ["n_periods", "cagr", "ann_vol", "sharpe", "sharpe_ci_lo", "sharpe_ci_hi",
              "sharpe_perm_pvalue", "max_drawdown", "ann_turnover", "cost_drag_annual",
              "breakeven_cost_bps_roundtrip", "exposure"]:
        if k in res.stats:
            print(f"   {k:<28} {res.stats[k]:.4f}")
    print(f"   {'benchmark sharpe':<28} {res.benchmark_stats['sharpe']:.4f}")

    print("\n== cost sweep ==")
    print(bt.sweep_costs(recs, prices, cfg).round(3).to_string(index=False))

    print("\n== plots ==")
    fs = runner.sample(df.iloc[-125:-5], df.index[-5:], n_samples=16, batch_size=16)
    for p in [
        plotting.plot_forecast(df.iloc[-125:-5], fs, df["close"].iloc[-5:],
                               OUT / "figures" / "forecast.png", "dry run - fan chart"),
        plotting.plot_equity(res, OUT / "figures" / "equity.png", "dry run - equity"),
        plotting.plot_diagnostics(recs, samples, OUT / "figures" / "diagnostics.png",
                                  "dry run - diagnostics"),
    ]:
        print(f"   {p}")

    print("\nDry run complete. Every number above is noise by construction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
