"""Offline smoke tests.

These do NOT download pretrained weights. They build a tiny randomly-initialised
Kronos from the upstream classes and push synthetic OHLCV through the whole
pipeline. The point is to prove the plumbing is correct - shapes, alignment,
no look-ahead, cost accounting - not to prove anything about forecast quality.

The look-ahead test is the important one: it asserts that shifting the price
series forward by one bar changes the backtest P&L, which it must if positions
are genuinely lagged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from kronos_nse import backtest as bt  # noqa: E402
from kronos_nse import evaluate, metrics  # noqa: E402
from kronos_nse.config import Config  # noqa: E402
from kronos_nse.predictor import KronosRunner  # noqa: E402
from kronos_nse.vendor import import_kronos  # noqa: E402

LOOKBACK = 48
HORIZON = 3
N_SAMPLES = 4


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def synthetic_ohlcv(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Geometric random walk shaped like a daily NSE equity series."""
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0004, 0.012, size=n)
    close = 1000.0 * np.exp(np.cumsum(ret))
    noise = np.abs(rng.normal(0, 0.004, size=n)) * close
    open_ = close * np.exp(rng.normal(0, 0.003, size=n))
    high = np.maximum(open_, close) + noise
    low = np.minimum(open_, close) - noise
    volume = rng.lognormal(13, 0.4, size=n)
    idx = pd.bdate_range("2024-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    df.index.name = "timestamp"
    df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)
    return df


@pytest.fixture(scope="module")
def tiny_runner():
    """A randomly-initialised Kronos small enough to run on CPU in seconds."""
    Kronos, KronosTokenizer, KronosPredictor = import_kronos(auto_clone=True)

    tokenizer = KronosTokenizer(
        d_in=6, d_model=32, n_heads=2, ff_dim=64,
        n_enc_layers=2, n_dec_layers=2,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        s1_bits=4, s2_bits=4,
        beta=0.0, gamma0=1.0, gamma=1.0, zeta=1.0, group_size=1,
    )
    model = Kronos(
        s1_bits=4, s2_bits=4, n_layers=2, d_model=32, n_heads=2, ff_dim=64,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        token_dropout_p=0.0, learn_te=True,
    )
    tokenizer.eval()
    model.eval()
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=64, clip=5)
    return KronosRunner(device="cpu", max_context=64, _predictor=predictor)


@pytest.fixture(scope="module")
def cfg() -> Config:
    return Config({
        "forecast": {
            "lookback": LOOKBACK, "horizon": HORIZON, "n_samples": N_SAMPLES,
            "temperature": 1.0, "top_p": 0.9, "top_k": 0, "sample_batch": N_SAMPLES,
        },
        "evaluate": {"start": None, "end": None, "stride": 25, "max_windows": 6, "seed": 0},
        "backtest": {
            "signal": "expected_return", "sizing": "sign", "threshold_bps": 0,
            "max_leverage": 1.0, "allow_short": True, "overlap": True,
            "execution": "next_open", "cost_bps_roundtrip": 20,
            "ann_factor": 252, "n_bootstrap": 100,
        },
    })


@pytest.fixture(scope="module")
def evaluated(tiny_runner, cfg):
    df = synthetic_ohlcv()
    recs, samples = evaluate.run_symbol("SYNTH", df, tiny_runner, cfg, progress=False)
    return df, recs, samples


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------


def test_sample_shapes_and_independence(tiny_runner):
    df = synthetic_ohlcv(120)
    ctx = df.iloc[:100]
    future = df.index[100:100 + HORIZON]
    fs = tiny_runner.sample(ctx, future, n_samples=6, batch_size=3)

    assert fs.paths.shape == (6, HORIZON, 6)
    assert fs.close_paths.shape == (6, HORIZON)
    assert fs.last_close == pytest.approx(float(ctx["close"].iloc[-1]))
    assert len(fs.timestamps) == HORIZON
    # Independent stochastic draws must not collapse to one identical path.
    assert np.std(fs.terminal_log_returns()) > 0
    s = fs.summary()
    assert 0.0 <= s["p_up"] <= 1.0
    assert np.isfinite(s["exp_logret"])


def test_context_is_truncated_to_max_context(tiny_runner):
    df = synthetic_ohlcv(300)
    ctx = df.iloc[:250]                      # longer than max_context=64
    fs = tiny_runner.sample(ctx, df.index[250:250 + 2], n_samples=2, batch_size=2)
    assert fs.paths.shape == (2, 2, 6)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def test_records_are_causal(evaluated, cfg):
    df, recs, samples = evaluated
    assert len(recs) == 6
    assert samples.shape == (6, N_SAMPLES)

    for _, r in recs.iterrows():
        i = int(r["idx"])
        # last_close must be the decision bar's close, never a future one
        assert r["last_close"] == pytest.approx(float(df["close"].iloc[i]))
        assert r["decision_ts"] == df.index[i]
        assert r["target_ts"] == df.index[i + HORIZON]
        expected = float(np.log(df["close"].iloc[i + HORIZON] / df["close"].iloc[i]))
        assert r["actual_logret"] == pytest.approx(expected, rel=1e-9)


def test_decision_indices_respect_bounds():
    df = synthetic_ohlcv(200)
    idxs = evaluate.decision_indices(df, lookback=50, horizon=5)
    assert min(idxs) == 49
    assert max(idxs) == len(df) - 6          # room for the full horizon
    assert idxs == sorted(idxs)

    with pytest.raises(ValueError):
        evaluate.decision_indices(df, lookback=500, horizon=5)


def test_score_returns_finite_metrics(evaluated, cfg):
    _, recs, samples = evaluated
    sc = evaluate.score(recs, samples, HORIZON, seed=0)
    assert sc["n_windows"] == 6
    assert 0.0 <= sc["directional_accuracy"] <= 1.0
    assert 0.0 <= sc["coverage_90"] <= 1.0
    assert np.isfinite(sc["crps"]) and sc["crps"] >= 0
    assert np.isfinite(sc["kronos_rmse"])
    # naive baseline RMSE is just the RMS of realised returns
    assert sc["bl_naive_rmse"] == pytest.approx(
        float(np.sqrt(np.mean(recs["actual_logret"] ** 2))), rel=1e-9
    )


def test_save_load_roundtrip(evaluated, tmp_path):
    _, recs, samples = evaluated
    evaluate.save(tmp_path, "unit", recs, samples)
    r2, s2 = evaluate.load(tmp_path, "unit")
    assert len(r2) == len(recs)
    np.testing.assert_allclose(s2, samples)
    assert pd.api.types.is_datetime64_any_dtype(r2["decision_ts"])


# ---------------------------------------------------------------------------
# backtest mechanics
# ---------------------------------------------------------------------------


def _fake_records(df: pd.DataFrame, signals: list[float], start: int = 10) -> pd.DataFrame:
    return pd.DataFrame({
        "decision_ts": [df.index[start + k] for k in range(len(signals))],
        "target_ts": [df.index[start + k + HORIZON] for k in range(len(signals))],
        "exp_logret": signals,
        "std_logret": [0.01] * len(signals),
        "p_up": [0.5 + np.sign(s) * 0.1 for s in signals],
    })


def test_positions_are_lagged_not_contemporaneous(cfg):
    """The core anti-look-ahead check.

    A decision on bar i must not earn bar i's own move. We give the strategy a
    perfect oracle signal for one bar and verify it captures the *next* open-to-
    open move, not the one already observed.
    """
    df = synthetic_ohlcv(80, seed=3)
    c = Config({k: dict(v) for k, v in cfg.items()})
    c.set_path("forecast.horizon", 1)
    c.set_path("backtest.overlap", False)
    c.set_path("backtest.cost_bps_roundtrip", 0)

    i = 30
    recs = _fake_records(df, [0.05], start=i)          # single long call on bar i
    res = bt.run(recs, df, c)

    assert len(res.daily) == 1
    row = res.daily.iloc[0]
    assert res.daily.index[0] == df.index[i + 1]       # live from the NEXT bar
    expected = float(df["open"].iloc[i + 2] / df["open"].iloc[i + 1] - 1)
    assert row["market_return"] == pytest.approx(expected, rel=1e-9)
    assert row["position"] == pytest.approx(1.0)


def test_overlap_splits_capital_across_live_forecasts(cfg):
    df = synthetic_ohlcv(80, seed=4)
    c = Config({k: dict(v) for k, v in cfg.items()})
    c.set_path("forecast.horizon", 3)
    c.set_path("backtest.overlap", True)
    c.set_path("backtest.cost_bps_roundtrip", 0)

    # Three consecutive decisions: long, long, short.
    recs = _fake_records(df, [0.05, 0.05, -0.05], start=20)
    res = bt.run(recs, df, c)
    pos = res.daily["position"]

    assert pos.iloc[0] == pytest.approx(1.0)      # only decision 1 live
    assert pos.iloc[1] == pytest.approx(1.0)      # decisions 1+2, both long
    assert pos.iloc[2] == pytest.approx(1 / 3)    # +1, +1, -1 -> 1/3
    assert pos.abs().max() <= 1.0 + 1e-12


def test_costs_reduce_returns_monotonically(cfg):
    df = synthetic_ohlcv(200, seed=5)
    rng = np.random.default_rng(1)
    recs = _fake_records(df, list(rng.normal(0, 0.02, 60)), start=20)

    prev = None
    for cost in [0, 10, 50, 200]:
        c = Config({k: dict(v) for k, v in cfg.items()})
        c.set_path("backtest.cost_bps_roundtrip", cost)
        res = bt.run(recs, df, c)
        total = res.stats["total_return"]
        if prev is not None:
            assert total <= prev + 1e-12
        prev = total
        assert res.stats["cost_drag_annual"] >= 0


def test_long_only_never_shorts(cfg):
    df = synthetic_ohlcv(120, seed=6)
    recs = _fake_records(df, [-0.05, -0.04, 0.03, -0.02], start=20)
    c = Config({k: dict(v) for k, v in cfg.items()})
    c.set_path("backtest.allow_short", False)
    res = bt.run(recs, df, c)
    assert (res.daily["position"] >= -1e-12).all()


def test_threshold_creates_flat_periods(cfg):
    df = synthetic_ohlcv(150, seed=7)
    recs = _fake_records(df, [0.0001, 0.05, 0.0001, -0.05], start=20)
    c = Config({k: dict(v) for k, v in cfg.items()})
    c.set_path("backtest.sizing", "threshold")
    c.set_path("backtest.threshold_bps", 100)      # 1% dead zone
    res = bt.run(recs, df, c)
    assert (res.daily["position"].abs() < 1e-12).any()


def test_shifting_prices_changes_pnl(cfg):
    """If P&L were computed contemporaneously, a one-bar shift of the price
    series would leave results unchanged. It must not."""
    df = synthetic_ohlcv(160, seed=8)
    rng = np.random.default_rng(2)
    recs = _fake_records(df, list(rng.normal(0, 0.02, 40)), start=20)

    c = Config({k: dict(v) for k, v in cfg.items()})
    base = bt.run(recs, df, c).stats["total_return"]

    shifted = df.copy()
    shifted[["open", "high", "low", "close"]] = shifted[["open", "high", "low", "close"]].shift(1)
    shifted = shifted.dropna()
    shifted_ret = bt.run(recs, shifted, c).stats["total_return"]
    assert base != pytest.approx(shifted_ret, rel=1e-6)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def test_directional_accuracy_edges():
    a = np.array([1.0, -1.0, 1.0, -1.0])
    assert metrics.directional_accuracy(a, a) == 1.0
    assert metrics.directional_accuracy(-a, a) == 0.0
    assert metrics.directional_accuracy(np.zeros(4), a) == 0.0


def test_sharpe_matches_closed_form():
    # Alternating series with mean 0.001 and a known sample std.
    r = np.tile([0.011, -0.009], 126)
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert metrics.sharpe(r) == pytest.approx(expected)
    assert metrics.sharpe(-r) == pytest.approx(-expected)
    # A constant series has no risk to reward - must not return a huge number.
    assert np.isnan(metrics.sharpe(np.full(252, 0.001)))


def test_drawdown_known_values():
    assert metrics.max_drawdown(np.array([0.1, -0.5, 0.1])) == pytest.approx(-0.5)
    assert metrics.max_drawdown(np.array([0.01, 0.01])) == pytest.approx(0.0)
    assert metrics.max_drawdown(np.array([-0.2, 0.5, -0.1])) == pytest.approx(-0.2)


def test_crps_rewards_sharper_correct_forecasts():
    actual = np.array([0.0, 0.0, 0.0])
    tight = np.random.default_rng(0).normal(0, 0.001, (3, 200))
    loose = np.random.default_rng(0).normal(0, 0.05, (3, 200))
    assert metrics.crps_empirical(tight, actual) < metrics.crps_empirical(loose, actual)


def test_coverage_of_calibrated_samples():
    rng = np.random.default_rng(0)
    samples = rng.normal(0, 1, (2000, 400))
    actual = rng.normal(0, 1, 2000)
    assert 0.86 < metrics.coverage(samples, actual, 0.05, 0.95) < 0.94


def test_brier_skill_of_useless_forecast_is_about_zero():
    rng = np.random.default_rng(0)
    actual = rng.normal(0, 1, 4000)
    base = float((actual > 0).mean())
    skill = metrics.brier_skill(np.full(4000, base), actual)
    assert abs(skill) < 0.02


def test_block_bootstrap_ci_brackets_the_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 1.0, 500)
    point, lo, hi = metrics.block_bootstrap_ci(x, np.mean, block=5, n_boot=500, seed=0)
    assert lo < point < hi
    assert lo < 0.5 < hi
