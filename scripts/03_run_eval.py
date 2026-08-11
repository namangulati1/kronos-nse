#!/usr/bin/env python3
"""Walk-forward forecast evaluation. This is the expensive step - it caches
per-window records so the backtest can be re-run without touching the GPU."""

from __future__ import annotations

import json
import time

import pandas as pd

from _common import base_parser, build_config, data_kwargs, out_dir, tag_for

from kronos_nse import data as dataset
from kronos_nse import evaluate, plotting
from kronos_nse.predictor import KronosRunner, set_seed


def main() -> int:
    parser = base_parser("Walk-forward evaluation")
    args = parser.parse_args()
    cfg = build_config(args)
    set_seed(int(cfg.get_path("evaluate.seed", 0)))
    odir = out_dir(cfg)

    runner = KronosRunner(
        model_name=cfg.model.name,
        tokenizer_name=cfg.model.tokenizer,
        device=cfg.model.get("device", "auto"),
        max_context=int(cfg.model.max_context),
        clip=int(cfg.model.get("clip", 5)),
    )

    all_scores = []
    for symbol in cfg.data.symbols:
        df = dataset.fetch(symbol, **data_kwargs(cfg))
        t0 = time.time()
        try:
            recs, samples = evaluate.run_symbol(symbol, df, runner, cfg)
        except ValueError as exc:
            print(f"[eval] {symbol}: skipped - {exc}")
            continue

        tag = tag_for(symbol, cfg)
        evaluate.save(odir, tag, recs, samples)

        sc = evaluate.score(recs, samples, int(cfg.forecast.horizon),
                            seed=int(cfg.get_path("evaluate.seed", 0)))
        sc["symbol"] = symbol
        sc["seconds"] = round(time.time() - t0, 1)
        all_scores.append(sc)

        print(f"  {symbol}: DA={sc['directional_accuracy']:.3f} "
              f"[{sc['da_ci_lo']:.3f}, {sc['da_ci_hi']:.3f}]  "
              f"vs naive RMSE skill={sc['skill_vs_bl_naive_rmse']:+.3f}  "
              f"CRPS={sc['crps']:.4f}  ({sc['seconds']}s)")

        plotting.plot_diagnostics(
            recs, samples,
            odir / "figures" / f"diagnostics_{tag}.png",
            title=f"{symbol} - forecast diagnostics ({sc['n_windows']} windows, h={cfg.forecast.horizon})",
        )

    if not all_scores:
        print("[eval] nothing evaluated")
        return 1

    scores = pd.DataFrame(all_scores)
    cols = ["symbol", "n_windows", "directional_accuracy", "da_ci_lo", "da_ci_hi",
            "up_base_rate", "brier_skill", "crps", "coverage_90",
            "kronos_rmse", "bl_naive_rmse", "skill_vs_bl_naive_rmse",
            "bl_momentum_da", "vol_spearman", "vol_spearman_baseline", "seconds"]
    cols = [c for c in cols if c in scores.columns]

    scores.to_csv(odir / "eval_scores.csv", index=False)
    with open(odir / "eval_config.json", "w", encoding="utf-8") as fh:
        json.dump(dict(cfg), fh, indent=2, default=str)

    pd.set_option("display.width", 250, "display.max_columns", 60)
    print("\n=== forecast quality ===")
    print(scores[cols].round(4).to_string(index=False))
    print(f"\nsaved -> {odir / 'eval_scores.csv'}")
    print("\nRead this table as: directional_accuracy near 0.50 with a CI spanning 0.50 "
          "means no directional edge. skill_vs_bl_naive_rmse <= 0 means the model is no "
          "better than assuming price does not change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
