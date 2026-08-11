"""Matplotlib figures for the report. Deliberately plain and print-friendly."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INK = "#1f2933"
ACCENT = "#2f6f9f"
WARM = "#c1663a"
MUTED = "#9aa5b1"
GRID = "#e4e7eb"


def _style(ax, title: str = "", ylabel: str = "", xlabel: str = "") -> None:
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=10)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_forecast(
    history: pd.DataFrame,
    samples,
    actual: pd.Series | None,
    out_path: str | Path,
    title: str = "",
    context_bars: int = 60,
) -> Path:
    """Fan chart: sampled paths + median, against what actually happened."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist = history.iloc[-context_bars:]
    close_paths = samples.close_paths
    anchor_ts, anchor_px = hist.index[-1], float(hist["close"].iloc[-1])

    # Stitch the forecast onto the last observed bar so the fan reads as a
    # continuation rather than a floating blob.
    ts = pd.DatetimeIndex([anchor_ts]).append(pd.DatetimeIndex(samples.timestamps))
    anchor_col = np.full((close_paths.shape[0], 1), anchor_px)
    paths = np.concatenate([anchor_col, close_paths], axis=1)

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=140)
    ax.plot(hist.index, hist["close"], color=INK, linewidth=1.4, label="history")

    for lo, hi, a in [(5, 95, 0.12), (25, 75, 0.20)]:
        ax.fill_between(
            ts,
            np.percentile(paths, lo, axis=0),
            np.percentile(paths, hi, axis=0),
            color=ACCENT, alpha=a, linewidth=0,
            label=f"Kronos {lo}-{hi}%",
        )
    ax.plot(ts, np.median(paths, axis=0), color=ACCENT, linewidth=1.8, label="Kronos median")

    if actual is not None and len(actual):
        act_ts = pd.DatetimeIndex([anchor_ts]).append(pd.DatetimeIndex(actual.index))
        act_v = np.concatenate([[anchor_px], np.asarray(actual.values, dtype=float)])
        ax.plot(act_ts, act_v, color=WARM, linewidth=1.8, label="actual")

    ax.axvline(hist.index[-1], color=MUTED, linewidth=0.9, linestyle="--")
    _style(ax, title, "price")
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper left")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_equity(result, out_path: str | Path, title: str = "") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), dpi=140, sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.2, 1.2]})
    eq, bench = result.equity, result.benchmark_equity

    axes[0].plot(eq.index, eq.values, color=ACCENT, linewidth=1.8, label="Kronos strategy (net)")
    axes[0].plot(bench.index, bench.values, color=MUTED, linewidth=1.4, label="buy & hold")
    axes[0].axhline(1.0, color=GRID, linewidth=0.8)
    _style(axes[0], title, "growth of 1")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")

    dd = eq / eq.cummax() - 1.0
    axes[1].fill_between(dd.index, dd.values, 0, color=WARM, alpha=0.35, linewidth=0)
    _style(axes[1], "Drawdown", "")

    axes[2].plot(result.daily.index, result.daily["position"], color=INK, linewidth=1.0)
    axes[2].axhline(0, color=GRID, linewidth=0.8)
    _style(axes[2], "Net exposure", "position")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_diagnostics(records: pd.DataFrame, samples: np.ndarray, out_path: str | Path,
                     title: str = "") -> Path:
    """Calibration (PIT), predicted-vs-actual scatter, and vol tracking."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .metrics import pit_values

    actual = records["actual_logret"].to_numpy(dtype=float)
    pred = records["exp_logret"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=140)

    pit = pit_values(samples, actual)
    axes[0].hist(pit, bins=10, range=(0, 1), color=ACCENT, alpha=0.8, edgecolor="white")
    axes[0].axhline(len(pit) / 10, color=WARM, linestyle="--", linewidth=1.2)
    _style(axes[0], "Calibration (PIT)  flat = well calibrated", "windows", "quantile of outcome")

    axes[1].scatter(pred * 100, actual * 100, s=12, color=ACCENT, alpha=0.55, linewidth=0)
    lim = max(np.abs(np.concatenate([pred, actual])).max() * 100 * 1.1, 0.1)
    axes[1].plot([-lim, lim], [-lim, lim], color=MUTED, linewidth=0.9, linestyle="--")
    axes[1].axhline(0, color=GRID, linewidth=0.8)
    axes[1].axvline(0, color=GRID, linewidth=0.8)
    _style(axes[1], "Forecast vs realised", "actual %", "predicted %")

    if records["actual_vol"].notna().any():
        axes[2].scatter(records["pred_realised_vol"] * 100, records["actual_vol"] * 100,
                        s=12, color=WARM, alpha=0.55, linewidth=0)
        _style(axes[2], "Volatility forecast", "actual bar vol %", "predicted bar vol %")
    else:
        axes[2].axis("off")

    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
