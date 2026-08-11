"""Forecast-quality and portfolio metrics, plus block-bootstrap inference.

Everything here assumes *overlapping* walk-forward windows, which means the
per-window errors are autocorrelated. Naive standard errors would be far too
optimistic, so confidence intervals use a moving-block bootstrap with block
length tied to the forecast horizon.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# forecast quality
# ---------------------------------------------------------------------------


def directional_accuracy(pred: np.ndarray, actual: np.ndarray) -> float:
    """Fraction of windows where the sign of the forecast matched reality.

    Zero-forecasts count as misses rather than half-credit: a model that
    refuses to take a side earns nothing.
    """
    mask = actual != 0
    if mask.sum() == 0:
        return float("nan")
    return float((np.sign(pred[mask]) == np.sign(actual[mask])).mean())


def brier_score(prob_up: np.ndarray, actual: np.ndarray) -> float:
    """Mean squared error of the P(up) probability forecast. Lower is better."""
    outcome = (actual > 0).astype(float)
    return float(np.mean((prob_up - outcome) ** 2))


def brier_skill(prob_up: np.ndarray, actual: np.ndarray) -> float:
    """Brier score vs the climatological base rate. >0 means genuine skill."""
    outcome = (actual > 0).astype(float)
    base = outcome.mean()
    ref = float(np.mean((base - outcome) ** 2))
    if ref == 0:
        return float("nan")
    return float(1.0 - brier_score(prob_up, actual) / ref)


def crps_empirical(samples: np.ndarray, actual: np.ndarray) -> float:
    """CRPS from Monte-Carlo draws, via the energy form.

        CRPS = E|X - y| - 0.5 * E|X - X'|

    ``samples`` is (N, K) - K draws for each of N windows.
    """
    n, k = samples.shape
    term1 = np.abs(samples - actual[:, None]).mean(axis=1)
    srt = np.sort(samples, axis=1)
    # E|X - X'| computed in O(K log K) via the sorted-sample identity.
    weights = (2 * np.arange(1, k + 1) - k - 1)
    term2 = 2.0 * (srt * weights).sum(axis=1) / (k * k)
    return float(np.mean(term1 - 0.5 * term2))


def pit_values(samples: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Probability-integral transform: where the outcome fell in the forecast
    distribution. A calibrated model gives a uniform histogram."""
    return (samples < actual[:, None]).mean(axis=1)


def coverage(samples: np.ndarray, actual: np.ndarray, lo: float = 0.05, hi: float = 0.95) -> float:
    """Empirical hit rate of the nominal (hi-lo) predictive interval."""
    low = np.quantile(samples, lo, axis=1)
    high = np.quantile(samples, hi, axis=1)
    return float(np.mean((actual >= low) & (actual <= high)))


def error_stats(pred: np.ndarray, actual: np.ndarray) -> dict:
    err = pred - actual
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "corr": float(np.corrcoef(pred, actual)[0, 1]) if len(pred) > 2 and np.std(pred) > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# portfolio
# ---------------------------------------------------------------------------


def sharpe(returns: np.ndarray, ann_factor: int = 252, rf_annual: float = 0.0) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return float("nan")
    excess = r - rf_annual / ann_factor
    sd = excess.std(ddof=1)
    # Guard against a degenerate (constant) return series: floating-point noise
    # can leave sd at ~1e-19 and produce an absurd Sharpe.
    if not np.isfinite(sd) or sd < 1e-12:
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(ann_factor))


def sortino(returns: np.ndarray, ann_factor: int = 252) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    downside = r[r < 0]
    if len(downside) < 2:
        return float("nan")
    dd = np.sqrt(np.mean(downside**2))
    if dd == 0:
        return float("nan")
    return float(r.mean() / dd * np.sqrt(ann_factor))


def max_drawdown(returns: np.ndarray) -> float:
    r = np.nan_to_num(np.asarray(returns, dtype=float))
    if len(r) == 0:
        return 0.0
    # Prepend the starting capital so a loss on the very first bar counts as
    # drawdown rather than establishing the high-water mark below par.
    equity = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def cagr(returns: np.ndarray, ann_factor: int = 252) -> float:
    r = np.nan_to_num(np.asarray(returns, dtype=float))
    if len(r) == 0:
        return float("nan")
    total = float(np.prod(1.0 + r))
    years = len(r) / ann_factor
    if years <= 0 or total <= 0:
        return float("nan")
    return float(total ** (1.0 / years) - 1.0)


def summarise_returns(returns: np.ndarray, ann_factor: int = 252) -> dict:
    r = np.nan_to_num(np.asarray(returns, dtype=float))
    return {
        "n_periods": int(len(r)),
        "total_return": float(np.prod(1.0 + r) - 1.0),
        "cagr": cagr(r, ann_factor),
        "ann_vol": float(r.std(ddof=1) * np.sqrt(ann_factor)) if len(r) > 1 else float("nan"),
        "sharpe": sharpe(r, ann_factor),
        "sortino": sortino(r, ann_factor),
        "max_drawdown": max_drawdown(r),
        "calmar": (cagr(r, ann_factor) / abs(max_drawdown(r))) if max_drawdown(r) < 0 else float("nan"),
        "hit_rate": float((r[r != 0] > 0).mean()) if (r != 0).any() else float("nan"),
        "best_period": float(r.max()) if len(r) else float("nan"),
        "worst_period": float(r.min()) if len(r) else float("nan"),
    }


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------


def block_bootstrap_ci(
    values: np.ndarray,
    stat_fn,
    block: int = 5,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Moving-block bootstrap CI for a statistic of a serially-correlated series.

    Returns ``(point_estimate, lo, hi)``.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    point = float(stat_fn(values))
    if n < 2 * block:
        return point, float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]).reshape(n_boot, -1)[:, :n]
    draws = values[idx]

    stats = np.array([stat_fn(row) for row in draws], dtype=float)
    stats = stats[np.isfinite(stats)]
    if len(stats) == 0:
        return point, float("nan"), float("nan")
    return point, float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def binomial_p_value(successes: int, trials: int, p0: float = 0.5) -> float:
    """Two-sided exact test that a hit rate differs from ``p0``."""
    from scipy import stats as sps

    if trials == 0:
        return float("nan")
    return float(sps.binomtest(successes, trials, p0, alternative="two-sided").pvalue)


def permutation_sharpe_pvalue(
    strategy_returns: np.ndarray,
    market_returns: np.ndarray,
    positions: np.ndarray,
    block: int = 5,
    n_perm: int = 1000,
    ann_factor: int = 252,
    seed: int = 0,
) -> float:
    """How often does a block-shuffled version of the *same* position series
    beat the realised Sharpe? This kills most 'my backtest works' illusions:
    it preserves the position distribution and the market path, and only
    destroys the timing alignment between them.
    """
    rng = np.random.default_rng(seed)
    observed = sharpe(strategy_returns, ann_factor)
    n = len(positions)
    if n < 2 * block or not np.isfinite(observed):
        return float("nan")

    n_blocks = int(np.ceil(n / block))
    beats = 0
    for _ in range(n_perm):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).reshape(-1)[:n]
        shuffled = positions[idx]
        s = sharpe(shuffled * market_returns, ann_factor)
        if np.isfinite(s) and s >= observed:
            beats += 1
    return float((beats + 1) / (n_perm + 1))


def as_frame(d: dict) -> pd.DataFrame:
    return pd.DataFrame({"metric": list(d.keys()), "value": list(d.values())})
