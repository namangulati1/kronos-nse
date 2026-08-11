#!/usr/bin/env python3
"""Backtest the cached forecasts. Fast - no model inference, reads records CSVs."""

from __future__ import annotations

import pandas as pd

from _common import base_parser, build_config, data_kwargs, out_dir, tag_for

from kronos_nse import backtest as bt
from kronos_nse import data as dataset
from kronos_nse import evaluate, plotting


def main() -> int:
    parser = base_parser("Backtest cached Kronos forecasts")
    parser.add_argument("--cost-sweep", action="store_true",
                        help="Also report Sharpe across a grid of transaction costs.")
    args = parser.parse_args()
    cfg = build_config(args)
    odir = out_dir(cfg)

    rows, bench_rows = [], []
    for symbol in cfg.data.symbols:
        tag = tag_for(symbol, cfg)
        try:
            recs, _ = evaluate.load(odir, tag)
        except FileNotFoundError:
            print(f"[bt] {symbol}: no records found (run 03_run_eval.py first)")
            continue

        prices = dataset.fetch(symbol, **data_kwargs(cfg))
        # Restrict prices to the evaluated span (+ tail for the final exit).
        lo = recs["decision_ts"].min()
        hi = recs["target_ts"].max()
        prices = prices.loc[(prices.index >= lo) & (prices.index <= hi)]

        try:
            res = bt.run(recs, prices, cfg)
        except ValueError as exc:
            print(f"[bt] {symbol}: {exc}")
            continue

        res.daily.to_csv(odir / f"backtest_{tag}.csv")
        plotting.plot_equity(
            res, odir / "figures" / f"equity_{tag}.png",
            title=f"{symbol} - Kronos {cfg.backtest.signal}/{cfg.backtest.sizing}, "
                  f"{cfg.backtest.cost_bps_roundtrip}bps round trip, {cfg.backtest.execution}",
        )

        rows.append({"symbol": symbol, **res.stats})
        bench_rows.append({"symbol": symbol, **res.benchmark_stats})

        if args.cost_sweep:
            sweep = bt.sweep_costs(recs, prices, cfg)
            sweep.insert(0, "symbol", symbol)
            sweep.to_csv(odir / f"cost_sweep_{tag}.csv", index=False)
            print(f"  cost sweep {symbol}: " + "  ".join(
                f"{int(r.cost_bps_roundtrip)}bps->{r.sharpe:+.2f}" for r in sweep.itertuples()))

    if not rows:
        print("[bt] nothing to report")
        return 1

    strat = pd.DataFrame(rows)
    bench = pd.DataFrame(bench_rows)
    strat.to_csv(odir / "backtest_summary.csv", index=False)
    bench.to_csv(odir / "benchmark_summary.csv", index=False)

    show = ["symbol", "cagr", "ann_vol", "sharpe", "sharpe_ci_lo", "sharpe_ci_hi",
            "sharpe_perm_pvalue", "sharpe_gross", "max_drawdown", "hit_rate",
            "ann_turnover", "cost_drag_annual", "breakeven_cost_bps_roundtrip",
            "exposure", "time_in_market"]
    show = [c for c in show if c in strat.columns]

    pd.set_option("display.width", 250, "display.max_columns", 60)
    print("\n=== strategy (net of costs) ===")
    print(strat[show].round(4).to_string(index=False))
    print("\n=== buy & hold benchmark ===")
    print(bench[["symbol", "cagr", "ann_vol", "sharpe", "max_drawdown"]].round(4).to_string(index=False))

    print("\nHow to read this: sharpe_perm_pvalue is the share of block-shuffled versions "
          "of the same position series that matched or beat the realised Sharpe. Above ~0.10 "
          "means the timing carried no information. breakeven_cost_bps_roundtrip is the "
          "execution cost at which the gross edge is exactly consumed - if it is below what "
          "you actually pay, the strategy is not tradeable.")
    print(f"\nsaved -> {odir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
