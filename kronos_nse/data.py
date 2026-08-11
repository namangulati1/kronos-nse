"""NSE OHLCV loading via Yahoo Finance, with on-disk caching and validation.

Yahoo notation for Indian markets:
    ^NSEI      NIFTY 50 index
    ^NSEBANK   NIFTY BANK index
    RELIANCE.NS   NSE cash equity
    RELIANCE.BO   BSE cash equity

Caveats that matter for backtests:
  * Index series (^NSEI, ^NSEBANK) carry zero/NaN volume. Kronos accepts that
    (it zero-fills), but the volume channel then contributes nothing.
  * ``auto_adjust=True`` is strongly recommended. Indian corporate actions
    (splits, bonus issues) are frequent and an unadjusted series shows them as
    genuine -50% gaps, which both the model and the backtest will believe.
  * Intraday history is capped by Yahoo at ~60 days for sub-hourly intervals
    and ~730 days for 60m.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

IST = "Asia/Kolkata"
OHLCV = ["open", "high", "low", "close", "volume"]
INTRADAY_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}


def _cache_path(cache_dir: Path, symbol: str, interval: str, auto_adjust: bool) -> Path:
    safe = symbol.replace("^", "IDX_").replace(".", "_").replace("/", "_")
    tag = "adj" if auto_adjust else "raw"
    return Path(cache_dir) / f"{safe}__{interval}__{tag}.parquet"


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase columns, IST-naive index, derived turnover, basic hygiene."""
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    missing = [c for c in ["open", "high", "low", "close"] if c not in df.columns]
    if missing:
        raise ValueError(f"Downloaded frame is missing columns: {missing}")

    if "volume" not in df.columns:
        df["volume"] = 0.0

    idx = pd.DatetimeIndex(df.index)
    # Work in IST wall-clock, tz-naive: Kronos derives hour/minute/weekday
    # features from the timestamps, so they must reflect local session time.
    if idx.tz is not None:
        idx = idx.tz_convert(IST).tz_localize(None)
    df.index = idx
    df.index.name = "timestamp"

    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[OHLCV].astype("float64")

    df["volume"] = df["volume"].fillna(0.0).clip(lower=0.0)
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    # Yahoo occasionally emits placeholder bars where OHLC are all identical
    # and volume is zero (holidays leaking in). Drop them.
    flat = (df["high"] == df["low"]) & (df["volume"] <= 0)
    df = df[~flat]

    # Turnover proxy. Upstream Kronos expects an `amount` column when volume is
    # present; typical-price * volume is the standard stand-in.
    typical = df[["open", "high", "low", "close"]].mean(axis=1)
    df["amount"] = df["volume"] * typical
    return df


def fetch(
    symbol: str,
    interval: str = "1d",
    start: str | None = "2010-01-01",
    end: str | None = None,
    auto_adjust: bool = True,
    cache_dir: str | Path = "data/cache",
    refresh: bool = False,
    retries: int = 3,
) -> pd.DataFrame:
    """Download (or load from cache) one symbol's OHLCV series."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cpath = _cache_path(cache_dir, symbol, interval, auto_adjust)

    if cpath.exists() and not refresh:
        return pd.read_parquet(cpath)

    import yfinance as yf  # imported lazily so the package works offline

    kwargs: dict = {"interval": interval, "auto_adjust": auto_adjust}
    if interval in INTRADAY_INTERVALS:
        # Yahoo rejects long ranges for intraday; ask for the max allowed.
        kwargs["period"] = "60d" if interval in {"1m", "2m", "5m", "15m", "30m"} else "730d"
    else:
        kwargs["start"] = start
        if end:
            kwargs["end"] = end

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            raw = yf.Ticker(symbol).history(**kwargs)
            if raw is None or raw.empty:
                raise ValueError(f"Yahoo returned no rows for {symbol} ({interval}).")
            df = _normalise(raw)
            if len(df) < 50:
                raise ValueError(f"Only {len(df)} usable bars for {symbol}; refusing to cache.")
            df.to_parquet(cpath)
            return df
        except Exception as exc:  # noqa: BLE001 - surfaced after retries
            last_err = exc
            if attempt < retries - 1:
                time.sleep(2.0 * (attempt + 1))

    raise RuntimeError(f"Failed to fetch {symbol} after {retries} attempts: {last_err}")


def load_many(
    symbols: list[str],
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """Fetch several symbols, skipping (loudly) any that fail."""
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            out[sym] = fetch(sym, **kwargs)
            print(f"[data] {sym:<14} {len(out[sym]):>6} bars  "
                  f"{out[sym].index[0].date()} -> {out[sym].index[-1].date()}")
        except Exception as exc:  # noqa: BLE001
            print(f"[data] {sym:<14} FAILED: {exc}")
    if not out:
        raise RuntimeError("No symbols could be loaded.")
    return out


def sanity_report(df: pd.DataFrame, symbol: str = "") -> dict:
    """Cheap data-quality summary. Worth eyeballing before trusting a backtest."""
    close = df["close"]
    ret = np.log(close).diff().dropna()
    gaps = df.index.to_series().diff().dt.days
    return {
        "symbol": symbol,
        "bars": int(len(df)),
        "start": str(df.index[0]),
        "end": str(df.index[-1]),
        "zero_volume_frac": float((df["volume"] <= 0).mean()),
        "ann_vol_pct": float(ret.std() * np.sqrt(252) * 100),
        "max_abs_bar_move_pct": float(ret.abs().max() * 100),
        "n_moves_gt_20pct": int((ret.abs() > 0.20).sum()),
        "max_calendar_gap_days": float(gaps.max()) if len(gaps.dropna()) else float("nan"),
        "ohlc_violations": int(
            ((df["high"] < df["low"])
             | (df["high"] < df["close"])
             | (df["low"] > df["close"])).sum()
        ),
    }
