#!/usr/bin/env python3
"""Download and cache NSE OHLCV data, then print a data-quality report."""

from __future__ import annotations

import pandas as pd

from _common import base_parser, build_config, data_kwargs, out_dir

from kronos_nse import data as dataset


def main() -> int:
    parser = base_parser("Fetch NSE data")
    parser.add_argument("--refresh", action="store_true", help="Ignore the cache and re-download.")
    args = parser.parse_args()
    cfg = build_config(args)

    frames = dataset.load_many(
        cfg.data.symbols, refresh=args.refresh, **data_kwargs(cfg)
    )

    report = pd.DataFrame([dataset.sanity_report(df, sym) for sym, df in frames.items()])
    path = out_dir(cfg) / "data_quality.csv"
    report.to_csv(path, index=False)

    pd.set_option("display.width", 200, "display.max_columns", 50)
    print("\n=== data quality ===")
    print(report.to_string(index=False))
    print(f"\nsaved -> {path}")

    suspicious = report[(report.ohlc_violations > 0) | (report.n_moves_gt_20pct > 3)]
    if len(suspicious):
        print("\n[warn] check these symbols for bad splits / bad ticks:")
        print(suspicious[["symbol", "ohlc_violations", "n_moves_gt_20pct",
                          "max_abs_bar_move_pct"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
