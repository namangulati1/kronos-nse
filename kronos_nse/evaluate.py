"""Walk-forward evaluation of Kronos forecasts against realised NSE bars.

Protocol for every decision point ``t``:
  1. Feed the model bars ``[t-lookback+1 .. t]`` (inclusive of t's close).
  2. Draw K sampled paths for bars ``[t+1 .. t+H]``.
  3. Compare against what actually happened.

The only forward-looking input is the *calendar* of future bar timestamps
(Kronos conditions on time features, so it needs them). Trading dates are known
in advance from the exchange holiday list, so this is not an information leak -
but it does mean the harness silently assumes no unscheduled market closure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics
from .predictor import KronosRunner


def decision_indices(
    df: pd.DataFrame,
    lookback: int,
    horizon: int,
    start: str | None = None,
    end: str | None = None,
    stride: int = 1,
    max_windows: int | None = None,
) -> list[int]:
    """Positional indices of usable decision bars."""
    first = lookback - 1
    last = len(df) - horizon - 1
    if last < first:
        raise ValueError(
            f"Series has {len(df)} bars; need at least lookback+horizon "
            f"({lookback}+{horizon}) to form one window."
        )

    idx = np.arange(first, last + 1)
    ts = df.index[idx]
    if start:
        idx = idx[ts >= pd.Timestamp(start)]
        ts = df.index[idx]
    if end:
        idx = idx[ts <= pd.Timestamp(end)]

    idx = idx[::stride]
    if max_windows:
        idx = idx[:max_windows]
    return [int(i) for i in idx]


def _baselines(context_close: np.ndarray, horizon: int) -> dict:
    """Cheap competitors the model has to beat to be worth anything."""
    log_close = np.log(context_close)
    step = np.diff(log_close)
    drift = float(step.mean() * horizon) if len(step) else 0.0
    mom_lb = min(20, len(step))
    momentum = float(log_close[-1] - log_close[-1 - mom_lb]) if len(log_close) > mom_lb else 0.0
    return {
        "bl_naive": 0.0,                  # random walk: best guess is no change
        "bl_drift": drift,                # historical drift extrapolated
        "bl_momentum": momentum,          # 20-bar trend continuation
        "ctx_vol": float(step.std(ddof=1)) if len(step) > 1 else float("nan"),
    }


def run_symbol(
    symbol: str,
    df: pd.DataFrame,
    runner: KronosRunner,
    cfg,
    progress: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Evaluate one symbol. Returns (records, terminal_return_samples)."""
    lookback = int(cfg.forecast.lookback)
    horizon = int(cfg.forecast.horizon)

    idxs = decision_indices(
        df,
        lookback=lookback,
        horizon=horizon,
        start=cfg.get_path("evaluate.start"),
        end=cfg.get_path("evaluate.end"),
        stride=int(cfg.get_path("evaluate.stride", 1)),
        max_windows=cfg.get_path("evaluate.max_windows"),
    )
    if not idxs:
        raise ValueError(
            f"No windows for {symbol} in the requested date range. "
            "Either the range is too short or lookback is too long for the history available."
        )
    print(f"[eval] {symbol}: {len(idxs)} windows, lookback={lookback}, horizon={horizon}, "
          f"samples={cfg.forecast.n_samples}")

    iterator = idxs
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(idxs, desc=f"  {symbol}", unit="win")
        except ImportError:
            pass

    close = df["close"].to_numpy(dtype=float)
    records: list[dict] = []
    sample_bank: list[np.ndarray] = []

    for i in iterator:
        ctx = df.iloc[i - lookback + 1 : i + 1]
        future_ts = df.index[i + 1 : i + 1 + horizon]

        fs = runner.sample(
            context=ctx,
            future_timestamps=future_ts,
            n_samples=int(cfg.forecast.n_samples),
            temperature=float(cfg.forecast.temperature),
            top_p=float(cfg.forecast.top_p),
            top_k=int(cfg.forecast.top_k),
            batch_size=int(cfg.get_path("forecast.sample_batch", 30)),
            verbose=False,
        )

        summary = fs.summary()
        terminal_samples = fs.terminal_log_returns()

        actual_path = close[i + 1 : i + 1 + horizon]
        actual_logret = float(np.log(actual_path[-1] / close[i]))
        actual_steps = np.diff(np.log(np.concatenate([[close[i]], actual_path])))
        actual_vol = float(actual_steps.std(ddof=1)) if horizon > 1 else float("nan")

        rec = {
            "symbol": symbol,
            "decision_ts": df.index[i],
            "target_ts": df.index[i + horizon],
            "idx": i,
            "last_close": float(close[i]),
            "actual_logret": actual_logret,
            "actual_vol": actual_vol,
            "actual_max_up": float(np.log(actual_path.max() / close[i])),
            "actual_max_dn": float(np.log(actual_path.min() / close[i])),
            **summary,
            **_baselines(ctx["close"].to_numpy(dtype=float), horizon),
        }
        records.append(rec)
        sample_bank.append(terminal_samples)

    recs = pd.DataFrame(records)
    samples = np.stack(sample_bank, axis=0)  # (n_windows, K)
    return recs, samples


def score(records: pd.DataFrame, samples: np.ndarray, horizon: int, seed: int = 0) -> dict:
    """Aggregate forecast-quality metrics with block-bootstrap CIs."""
    actual = records["actual_logret"].to_numpy(dtype=float)
    pred = records["exp_logret"].to_numpy(dtype=float)
    p_up = records["p_up"].to_numpy(dtype=float)
    n = len(actual)

    da = metrics.directional_accuracy(pred, actual)
    hits = np.sign(pred[actual != 0]) == np.sign(actual[actual != 0])
    _, da_lo, da_hi = metrics.block_bootstrap_ci(
        hits.astype(float), np.mean, block=horizon, n_boot=1500, seed=seed
    )

    out: dict = {
        "n_windows": n,
        "directional_accuracy": da,
        "da_ci_lo": da_lo,
        "da_ci_hi": da_hi,
        # p-value ignores overlap, so treat it as an upper bound on significance
        "da_binomial_p_naive": metrics.binomial_p_value(int(hits.sum()), int(len(hits))),
        "up_base_rate": float((actual > 0).mean()),
        "brier": metrics.brier_score(p_up, actual),
        "brier_skill": metrics.brier_skill(p_up, actual),
        "crps": metrics.crps_empirical(samples, actual),
        "coverage_90": metrics.coverage(samples, actual, 0.05, 0.95),
        "coverage_50": metrics.coverage(samples, actual, 0.25, 0.75),
    }

    kr = metrics.error_stats(pred, actual)
    out.update({f"kronos_{k}": v for k, v in kr.items()})

    for name in ["bl_naive", "bl_drift", "bl_momentum"]:
        b = records[name].to_numpy(dtype=float)
        bs = metrics.error_stats(b, actual)
        out.update({f"{name}_{k}": v for k, v in bs.items()})
        out[f"{name}_da"] = metrics.directional_accuracy(b, actual)
        out[f"skill_vs_{name}_rmse"] = 1.0 - kr["rmse"] / bs["rmse"] if bs["rmse"] > 0 else float("nan")

    # Volatility forecasting is the claim Kronos-style models usually deliver on,
    # even when direction is a coin flip - so score it separately.
    if records["actual_vol"].notna().any():
        pv = records["pred_realised_vol"].to_numpy(dtype=float)
        av = records["actual_vol"].to_numpy(dtype=float)
        ok = np.isfinite(pv) & np.isfinite(av)
        if ok.sum() > 5:
            from scipy import stats as sps

            out["vol_spearman"] = float(sps.spearmanr(pv[ok], av[ok]).statistic)
            out["vol_pearson"] = float(np.corrcoef(pv[ok], av[ok])[0, 1])
            cv = records["ctx_vol"].to_numpy(dtype=float)
            ok2 = ok & np.isfinite(cv)
            out["vol_spearman_baseline"] = float(sps.spearmanr(cv[ok2], av[ok2]).statistic)
            out["vol_bias"] = float(np.mean(pv[ok] - av[ok]))

    return out


def save(out_dir: str | Path, tag: str, records: pd.DataFrame, samples: np.ndarray) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records.to_csv(out / f"records_{tag}.csv", index=False)
    np.savez_compressed(out / f"samples_{tag}.npz", terminal=samples)


def load(out_dir: str | Path, tag: str) -> tuple[pd.DataFrame, np.ndarray]:
    out = Path(out_dir)
    recs = pd.read_csv(out / f"records_{tag}.csv", parse_dates=["decision_ts", "target_ts"])
    samples = np.load(out / f"samples_{tag}.npz")["terminal"]
    return recs, samples
