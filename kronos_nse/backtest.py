"""Turn Kronos forecasts into positions and score the resulting P&L.

Execution model (default ``next_open``)
---------------------------------------
A decision made on the close of bar ``i`` cannot be traded until the next
session opens. So the position goes on at ``open[i+1]`` and every bar is marked
open-to-open::

    r_exec[d] = open[d+1] / open[d] - 1

That removes the single most common backtest lie - earning the overnight gap
that your own signal was computed from. ``execution: close`` is provided for
comparison; expect it to look materially better and to be unachievable.

Overlapping horizons
--------------------
With ``horizon = H`` and a decision every bar, H forecasts are live at once.
Capital is split evenly across them, so the book on bar ``d`` is the mean of the
last H target positions. This is the standard overlapping-portfolio construction
and it also damps turnover.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import metrics

EPS = 1e-12


@dataclass
class BacktestResult:
    daily: pd.DataFrame                 # per-bar positions, returns, costs
    stats: dict = field(default_factory=dict)
    benchmark_stats: dict = field(default_factory=dict)

    @property
    def equity(self) -> pd.Series:
        return (1.0 + self.daily["net_return"]).cumprod()

    @property
    def benchmark_equity(self) -> pd.Series:
        return (1.0 + self.daily["market_return"]).cumprod()


# ---------------------------------------------------------------------------
# signal construction
# ---------------------------------------------------------------------------


def build_signal(records: pd.DataFrame, kind: str = "expected_return") -> pd.Series:
    """Raw edge per decision bar. Units are log-return except for ``prob_up``."""
    if kind == "expected_return":
        sig = records["exp_logret"]
    elif kind == "vol_scaled":
        # Shrink the forecast when the model itself is unsure. Rescaled back to
        # return units so a single threshold still means something.
        std = records["std_logret"].replace(0, np.nan)
        sig = records["exp_logret"] / (std + EPS) * std.mean()
    elif kind == "prob_up":
        sig = records["p_up"] - 0.5
    else:
        raise ValueError(f"Unknown signal kind: {kind!r}")
    return pd.Series(sig.to_numpy(dtype=float), index=records["decision_ts"].to_numpy())


def size_positions(
    signal: pd.Series,
    sizing: str = "threshold",
    threshold: float = 0.0025,
    max_leverage: float = 1.0,
    allow_short: bool = True,
) -> pd.Series:
    """Map edge -> target exposure in [-max_leverage, max_leverage]."""
    s = signal.to_numpy(dtype=float)

    if sizing == "sign":
        pos = np.sign(s)
    elif sizing == "threshold":
        pos = np.where(np.abs(s) > threshold, np.sign(s), 0.0)
    elif sizing == "continuous":
        # Saturate at 4x the threshold so a huge forecast doesn't dominate.
        scale = max(threshold * 4.0, EPS)
        pos = np.clip(s / scale, -1.0, 1.0)
    else:
        raise ValueError(f"Unknown sizing: {sizing!r}")

    pos = pos * max_leverage
    if not allow_short:
        pos = np.maximum(pos, 0.0)
    return pd.Series(pos, index=signal.index)


# ---------------------------------------------------------------------------
# position -> P&L
# ---------------------------------------------------------------------------


def _execution_returns(prices: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "next_open":
        r = prices["open"].shift(-1) / prices["open"] - 1.0
    elif mode == "close":
        r = prices["close"].shift(-1) / prices["close"] - 1.0
    else:
        raise ValueError(f"Unknown execution mode: {mode!r}")
    return r


def _book_from_decisions(
    prices: pd.DataFrame,
    target: pd.Series,
    horizon: int,
    execution: str,
    overlap: bool,
) -> pd.Series:
    """Spread each decision's target position across the bars it is live for."""
    n = len(prices)
    pos_index = {ts: k for k, ts in enumerate(prices.index)}
    book = np.zeros(n, dtype=float)
    weight = np.zeros(n, dtype=float)

    # next_open: live over bars [i+1, i+H].  close: live over bars [i, i+H-1].
    offset = 1 if execution == "next_open" else 0
    span = horizon if overlap else 1

    for ts, p in target.items():
        i = pos_index.get(ts)
        if i is None:
            continue
        lo = i + offset
        hi = min(lo + span, n)
        if lo >= n:
            continue
        book[lo:hi] += p
        weight[lo:hi] += 1.0

    live = weight > 0
    book[live] /= weight[live]
    return pd.Series(book, index=prices.index)


def run(
    records: pd.DataFrame,
    prices: pd.DataFrame,
    cfg,
) -> BacktestResult:
    """Backtest one symbol's forecast records against its price series."""
    bt = cfg.backtest
    horizon = int(cfg.forecast.horizon)
    ann = int(bt.get("ann_factor", 252))

    signal = build_signal(records, bt.get("signal", "expected_return"))
    kind = bt.get("signal", "expected_return")
    threshold = (
        float(bt.get("threshold_prob", 0.03))
        if kind == "prob_up"
        else float(bt.get("threshold_bps", 25)) / 10_000.0
    )
    target = size_positions(
        signal,
        sizing=bt.get("sizing", "threshold"),
        threshold=threshold,
        max_leverage=float(bt.get("max_leverage", 1.0)),
        allow_short=bool(bt.get("allow_short", True)),
    )

    book = _book_from_decisions(
        prices, target, horizon, bt.get("execution", "next_open"), bool(bt.get("overlap", True))
    )
    r_exec = _execution_returns(prices, bt.get("execution", "next_open"))

    # Only score the span the strategy was actually live over.
    live = book.abs() > 0
    if not live.any():
        raise ValueError(
            "The strategy never took a position. Lower backtest.threshold_bps "
            "or switch sizing to 'sign'."
        )
    first, last = np.argmax(live.to_numpy()), n_last(live)
    sl = slice(first, last + 1)

    book_s = book.iloc[sl]
    r_s = r_exec.iloc[sl]

    turnover = book_s.diff().abs()
    turnover.iloc[0] = abs(book_s.iloc[0])
    # A round trip is |Δ|=1 in and |Δ|=1 out, so half the round-trip cost per unit.
    cost_per_unit = float(bt.get("cost_bps_roundtrip", 25)) / 2.0 / 10_000.0
    cost = turnover * cost_per_unit

    gross = book_s * r_s
    net = gross - cost

    daily = pd.DataFrame(
        {
            "position": book_s,
            "market_return": r_s,
            "gross_return": gross,
            "cost": cost,
            "turnover": turnover,
            "net_return": net,
        }
    ).dropna(subset=["market_return"])

    stats = metrics.summarise_returns(daily["net_return"].to_numpy(), ann)
    bench = metrics.summarise_returns(daily["market_return"].to_numpy(), ann)

    gross_stats = metrics.summarise_returns(daily["gross_return"].to_numpy(), ann)
    stats["sharpe_gross"] = gross_stats["sharpe"]
    stats["cost_drag_annual"] = float(daily["cost"].mean() * ann)
    stats["ann_turnover"] = float(daily["turnover"].mean() * ann)
    stats["exposure"] = float(daily["position"].abs().mean())
    stats["time_in_market"] = float((daily["position"].abs() > 0).mean())
    stats["long_frac"] = float((daily["position"] > 0).mean())
    stats["short_frac"] = float((daily["position"] < 0).mean())

    _, lo, hi = metrics.block_bootstrap_ci(
        daily["net_return"].to_numpy(),
        lambda x: metrics.sharpe(x, ann),
        block=max(horizon, 5),
        n_boot=int(bt.get("n_bootstrap", 2000)),
        seed=int(cfg.get_path("evaluate.seed", 0)),
    )
    stats["sharpe_ci_lo"], stats["sharpe_ci_hi"] = lo, hi
    stats["sharpe_perm_pvalue"] = metrics.permutation_sharpe_pvalue(
        daily["net_return"].to_numpy(),
        daily["market_return"].to_numpy(),
        daily["position"].to_numpy(),
        block=max(horizon, 5),
        n_perm=min(int(bt.get("n_bootstrap", 2000)), 1000),
        ann_factor=ann,
        seed=int(cfg.get_path("evaluate.seed", 0)),
    )
    # Breakeven cost: how expensive can execution get before the edge dies?
    if daily["turnover"].mean() > 0:
        stats["breakeven_cost_bps_roundtrip"] = float(
            daily["gross_return"].mean() / daily["turnover"].mean() * 2 * 10_000
        )

    return BacktestResult(daily=daily, stats=stats, benchmark_stats=bench)


def n_last(mask: pd.Series) -> int:
    arr = mask.to_numpy()
    return int(len(arr) - 1 - np.argmax(arr[::-1]))


def sweep_costs(
    records: pd.DataFrame,
    prices: pd.DataFrame,
    cfg,
    cost_grid: list[float] | None = None,
) -> pd.DataFrame:
    """Sharpe as a function of assumed transaction cost. If the edge only exists
    at 0 bps, it does not exist."""
    cost_grid = cost_grid or [0, 5, 10, 15, 20, 25, 35, 50, 75, 100]
    rows = []
    import copy as _copy

    for c in cost_grid:
        cfg2 = _copy.deepcopy(cfg)
        cfg2.set_path("backtest.cost_bps_roundtrip", c)
        cfg2.set_path("backtest.n_bootstrap", 200)
        try:
            res = run(records, prices, cfg2)
            rows.append({
                "cost_bps_roundtrip": c,
                "sharpe": res.stats["sharpe"],
                "cagr": res.stats["cagr"],
                "total_return": res.stats["total_return"],
            })
        except Exception as exc:  # noqa: BLE001
            rows.append({"cost_bps_roundtrip": c, "sharpe": np.nan, "error": str(exc)})
    return pd.DataFrame(rows)
