#!/usr/bin/env python3
"""Single-window sanity demo: forecast the last N bars and plot against reality.

Run this first. If the fan chart looks like noise around the last price, that is
the honest answer for a 5-day horizon - not a bug.
"""

from __future__ import annotations

import numpy as np

from _common import base_parser, build_config, data_kwargs, out_dir

from kronos_nse import data as dataset
from kronos_nse import plotting
from kronos_nse.predictor import KronosRunner, set_seed


def main() -> int:
    parser = base_parser("Demo forecast on one window")
    parser.add_argument("--holdout", action="store_true", default=True,
                        help="Hold out the last `horizon` bars so the forecast can be scored.")
    parser.add_argument("--live", dest="holdout", action="store_false",
                        help="Forecast forward from the latest bar instead (nothing to score).")
    args = parser.parse_args()
    cfg = build_config(args)
    set_seed(int(cfg.get_path("evaluate.seed", 0)))

    runner = KronosRunner(
        model_name=cfg.model.name,
        tokenizer_name=cfg.model.tokenizer,
        device=cfg.model.get("device", "auto"),
        max_context=int(cfg.model.max_context),
        clip=int(cfg.model.get("clip", 5)),
    )

    lookback = int(cfg.forecast.lookback)
    horizon = int(cfg.forecast.horizon)
    odir = out_dir(cfg)

    for symbol in cfg.data.symbols:
        df = dataset.fetch(symbol, **data_kwargs(cfg))

        if args.holdout:
            cut = len(df) - horizon
            ctx = df.iloc[max(0, cut - lookback):cut]
            future_ts = df.index[cut:cut + horizon]
            actual = df["close"].iloc[cut:cut + horizon]
        else:
            ctx = df.iloc[-lookback:]
            freq = ctx.index.to_series().diff().median()
            future_ts = [ctx.index[-1] + freq * (k + 1) for k in range(horizon)]
            actual = None

        fs = runner.sample(
            context=ctx,
            future_timestamps=future_ts,
            n_samples=int(cfg.forecast.n_samples),
            temperature=float(cfg.forecast.temperature),
            top_p=float(cfg.forecast.top_p),
            top_k=int(cfg.forecast.top_k),
            batch_size=int(cfg.get_path("forecast.sample_batch", 30)),
        )
        s = fs.summary()
        line = (f"{symbol:<14} last={fs.last_close:10.2f}  "
                f"E[ret]={s['exp_logret']*100:+6.2f}%  "
                f"P(up)={s['p_up']:.2f}  "
                f"90% CI=[{s['q05']*100:+.2f}%, {s['q95']*100:+.2f}%]")
        if actual is not None:
            realised = float(np.log(actual.iloc[-1] / fs.last_close))
            hit = "HIT " if np.sign(s["exp_logret"]) == np.sign(realised) else "miss"
            line += f"  actual={realised*100:+6.2f}%  [{hit}]"
        print(line)

        plotting.plot_forecast(
            ctx, fs, actual,
            odir / "figures" / f"forecast_{symbol.replace('^', 'IDX_').replace('.', '_')}.png",
            title=f"{symbol} - Kronos {horizon}-bar forecast "
                  f"({fs.n_samples} sampled paths, {cfg.model.name.split('/')[-1]})",
        )

    print(f"\nfigures -> {odir / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
