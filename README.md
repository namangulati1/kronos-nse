# Kronos NSE Testbench

A walk-forward evaluation and backtesting harness for [Kronos](https://github.com/shiyu-coder/Kronos),
the open-source foundation model for financial candlesticks (AAAI 2026), applied to Indian
equity market data.

The harness is built to answer one question honestly: **does Kronos produce a tradeable edge
on NSE, after costs, out of sample?** It is deliberately biased toward finding *no* edge —
every default is the conservative one, and the reporting is designed to make a spurious
result hard to mistake for a real one. If the numbers come out good anyway, they're worth
something.

Nothing here is investment advice, and nothing here has been validated against live trading.

---

## What Kronos actually is

Kronos is a two-stage model. A tokenizer quantises OHLCV bars into hierarchical discrete
tokens, and an autoregressive transformer — pretrained on ~12B K-line records from 45 global
exchanges — generates continuations in token space. Four checkpoints exist:

| Checkpoint | Params | Context | Tokenizer |
|---|---|---|---|
| `NeoQuasar/Kronos-mini` | 4.1M | 2048 | `Kronos-Tokenizer-2k` |
| `NeoQuasar/Kronos-small` | 24.7M | 512 | `Kronos-Tokenizer-base` |
| `NeoQuasar/Kronos-base` | 102.3M | 512 | `Kronos-Tokenizer-base` |
| `Kronos-large` | 499.2M | 512 | not released |

The critical property for testing: **Kronos is a sampler, not a point forecaster.** Each call
draws a stochastic path. A single path tells you almost nothing. This harness draws 30 paths
per decision by default and treats the spread across them as the model's uncertainty, which
is where most of the usable signal lives — P(up) across paths and predicted volatility are far
more informative than a mean forecast.

Upstream's `predict(..., sample_count=N)` averages the paths and throws the distribution away.
`kronos_nse/predictor.py` recovers the individual draws by pushing N copies of the same window
through `predict_batch` with `sample_count=1`, so each batch row samples independently.

---

## Quickstart

```bash
git clone <this repo> && cd kronos-nse
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

make setup      # clones upstream Kronos into third_party/, checks your environment
make test       # unit tests — offline, no weights downloaded, ~5 seconds
make dryrun     # full pipeline on synthetic data with a random model, ~1 minute

make data       # download + cache NSE OHLCV, print a data-quality report
make demo       # one forecast per symbol + fan charts
make eval       # walk-forward evaluation  <-- the slow step
make backtest   # backtest the cached forecasts + cost sweep
```

### Environment Variables & Security Setup

Before running scripts or pushing to GitHub, copy `.env.example` to `.env` to configure your environment variables safely:

```bash
cp .env.example .env
```

Configurable environment variables in `.env`:
- `HF_TOKEN`: Optional Hugging Face API access token for authenticated downloads or rate-limit prevention.
- `KRONOS_HOME`: Path override for upstream Kronos repository checkout.
- `KRONOS_MODEL_NAME`: Default Hugging Face model repository ID.
- `KRONOS_TOKENIZER_NAME`: Default Hugging Face tokenizer repository ID.
- `KRONOS_DEVICE`: Compute device override (`auto`, `cuda:0`, `mps`, `cpu`).
- `KRONOS_CACHE_DIR`: Directory path for cached market data.
- `KRONOS_RESULTS_DIR`: Directory path for output results.

> [!IMPORTANT]
> The `.env` file is excluded from Git tracking via `.gitignore`. Never commit actual API keys or credentials to GitHub.


`make dryrun` is worth running before anything real. It drives the entire pipeline with a
randomly-initialised model on a synthetic random walk. Every number it prints should be
noise — directional accuracy near 0.50, Sharpe near zero, a permutation p-value that isn't
significant. If a dry run shows a great Sharpe, the harness is broken, not the model.

Everything is configurable from `configs/default.yaml` or inline:

```bash
python scripts/03_run_eval.py --symbols ^NSEI RELIANCE.NS \
    --set forecast.horizon=10 forecast.n_samples=50 model.name=NeoQuasar/Kronos-base
```

---

## Read this before you trust any result

### 1. Pretraining contamination is the big one

Kronos was published in August 2025 and pretrained on 45 global exchanges — which includes
Indian venues. Any backtest on data from before the checkpoint release is potentially
**in-sample for the model**, and will flatter it, possibly enormously. This is not a
hypothetical: it's the single most common way foundation-model backtests produce fake alpha.

`evaluate.start` therefore defaults to `2025-08-01`. Moving it earlier will make your results
look better and mean less. If you want more out-of-sample data, the honest options are to wait,
or to fine-tune on a pre-cutoff period and test after it.

### 2. Short-horizon direction is close to unpredictable

The base rate for a 5-day directional call on a liquid NSE name is roughly 52–54% up (equity
drift), and the achievable edge over that is small. A directional accuracy of 0.53 on 250
overlapping windows is not evidence of anything — the confidence interval is roughly ±6
percentage points. The harness reports block-bootstrap CIs precisely so this is visible
rather than hidden.

Where models like Kronos tend to genuinely add value is **volatility and distribution shape**,
not direction. `vol_spearman` (predicted vs realised bar volatility, against a
trailing-volatility baseline) is often the most interesting row in the eval table.

### 3. Overlapping windows are correlated

With `stride=1` and `horizon=5`, consecutive windows share four of five days. Naive standard
errors would be roughly √5 too small. All CIs use a moving-block bootstrap with block length
tied to the horizon. The `da_binomial_p_naive` column ignores this and is reported only as an
optimistic bound — treat any p-value near the threshold as not significant.

### 4. Costs, and the Indian specifics

Round-trip all-in cost for NSE **delivery** equity is roughly 25–30 bps: STT 0.1% on each side,
exchange transaction charges ~0.00325%, SEBI fees, 18% GST on brokerage+charges, stamp duty
0.015% on the buy, plus slippage. Discount brokers charge ₹0–20 per order, which is noise on a
large ticket and material on a small one.

- **Intraday**: ~10 bps round trip (STT only 0.025%, sell side).
- **Index futures / options**: ~2–5 bps round trip. This is where a short-horizon NIFTY signal
  would actually be implemented.

Default is `cost_bps_roundtrip: 25`. Always run `make backtest` (which includes `--cost-sweep`)
and look at where the Sharpe crosses zero. **If the edge only exists at 0 bps, it does not exist.**

`allow_short: true` is the default because it makes the test cleaner, but note that **you cannot
short cash equity in India beyond intraday**. A short-enabled equity backtest is a research
result, not an implementable strategy — implement via futures, options, or intraday.

### 5. Execution assumptions

Default `execution: next_open`. A decision taken on the close of bar *i* goes on at the open of
bar *i+1*, and every bar is marked open-to-open. This removes the most common backtest lie:
earning the overnight gap that your own signal was computed from. `execution: close` is provided
for comparison and will look materially better. It is not achievable.

The one genuine look-ahead the harness retains: it uses the *actual* future bar timestamps as
the forecast calendar, because Kronos conditions on time features. Exchange holidays are
published in advance, so this is fine — but it does assume no unscheduled closure.

---

## What gets measured

**Forecast quality** (`results/eval_scores.csv`)

Directional accuracy with a block-bootstrap CI and the up-move base rate for comparison; MAE,
RMSE, bias and correlation against three baselines — random walk (`bl_naive`, i.e. "price
doesn't change"), extrapolated historical drift (`bl_drift`), and 20-bar momentum
(`bl_momentum`); `skill_vs_bl_naive_rmse`, where ≤ 0 means the model is no better than assuming
nothing happens; Brier score and Brier skill on the P(up) probability; CRPS over the full
predictive distribution; 50% and 90% interval coverage, plus a PIT histogram that should be flat
if the model is calibrated; and Spearman/Pearson correlation of predicted vs realised volatility
against a trailing-vol baseline.

**Strategy** (`results/backtest_summary.csv`)

CAGR, annualised vol, Sharpe with a block-bootstrap CI, Sortino, max drawdown, Calmar, hit rate,
annualised turnover, cost drag, exposure and time in market — all against a buy-and-hold
benchmark on the same span. Two columns deserve particular attention:

- **`sharpe_perm_pvalue`** — the fraction of block-shuffled versions of the *same* position
  series that matched or beat the realised Sharpe. It preserves the position distribution and
  the market path and destroys only the timing alignment. Above ~0.10 means the timing carried
  no information, whatever the headline Sharpe says.
- **`breakeven_cost_bps_roundtrip`** — the execution cost at which the gross edge is exactly
  consumed. Below what you actually pay ⇒ not tradeable.

**Figures** (`results/figures/`): fan charts of sampled paths vs realised, equity curve with
drawdown and exposure panels, and a diagnostics panel (PIT calibration, forecast vs realised
scatter, volatility scatter).

---

## Configuration

Full reference in `configs/default.yaml`. The knobs that matter:

| Key | Default | Notes |
|---|---|---|
| `model.name` | `Kronos-small` | `Kronos-base` is ~4x slower for a modest quality gain; `Kronos-mini` is the only one with a 2048 context |
| `forecast.lookback` | 400 | Must be ≤ `model.max_context` |
| `forecast.horizon` | 5 | Bars ahead. 5 daily bars ≈ one week |
| `forecast.n_samples` | 30 | Below ~20 the P(up) estimate is too noisy to trade |
| `forecast.temperature` / `top_p` | 1.0 / 0.9 | Lower T ⇒ tighter, more confident, worse calibrated |
| `evaluate.start` | `2025-08-01` | **Do not move earlier** — see contamination above |
| `evaluate.stride` | 1 | Raise to 5 for a faster, non-overlapping run |
| `backtest.signal` | `expected_return` | or `prob_up`, `vol_scaled` |
| `backtest.sizing` | `threshold` | or `sign`, `continuous` |
| `backtest.cost_bps_roundtrip` | 25 | See the cost section |

### Runtime

Rough per-window cost with `lookback=400`, `horizon=5`, `n_samples=30`, Kronos-small:
~0.2 s on a modern GPU, ~2–5 s on CPU. The default config (8 symbols × ~250 windows) is
therefore ~10 minutes on GPU and a few hours on CPU. To shrink a first pass:

```bash
python scripts/03_run_eval.py --symbols ^NSEI \
    --set evaluate.stride=5 evaluate.max_windows=60 forecast.n_samples=20
```

Evaluation caches per-window records to `results/records_*.csv`, so `04_run_backtest.py` can be
re-run over different signals, sizings and cost assumptions in seconds without touching the model.

---

## Layout

```
configs/default.yaml        all tunable parameters
kronos_nse/
  vendor.py                 clones/locates upstream Kronos, puts it on sys.path
  data.py                   yfinance NSE loader, caching, quality report
  predictor.py              model loading + Monte-Carlo path sampling
  evaluate.py               walk-forward windows, records, scoring
  backtest.py               signal -> positions -> P&L with costs
  metrics.py                forecast + portfolio metrics, block bootstrap
  plotting.py               fan charts, equity curves, diagnostics
scripts/                    00_setup -> 01_fetch -> 02_demo -> 03_eval -> 04_backtest
tools/offline_dryrun.py     full pipeline, no network, no weights
tests/test_pipeline.py      19 tests incl. explicit anti-look-ahead checks
```

The test suite is worth a look before trusting the harness. `test_positions_are_lagged_not_contemporaneous`
gives the strategy a perfect oracle call on one bar and asserts it captures the *next*
open-to-open move; `test_shifting_prices_changes_pnl` asserts that shifting the price series by
one bar changes the P&L, which it could not if returns were being computed contemporaneously.

---

## Where to go if the zero-shot result is flat

That is the likely outcome, and it isn't the end of the test. In rough order of expected value:

**Fine-tune on NSE.** Upstream ships `finetune/train_tokenizer.py` and `finetune/train_predictor.py`
(two-stage, `torchrun`, multi-GPU). Zero-shot on Indian intraday is the hardest ask you can make
of the model; a tokenizer fine-tuned on NSE bar distributions is a different proposition. Train on
pre-2025-08 data, test after — that also fixes the contamination problem.

**Trade the volatility forecast instead of direction.** If `vol_spearman` beats the trailing-vol
baseline, that's a real signal with real uses: option positioning, position sizing overlays,
regime filters. It's also a much lower bar than predicting direction.

**Test intraday.** Switch `data.interval` to `15m` or `5m`. Yahoo caps intraday history at ~60
days, so for anything serious you'll want a broker feed (Zerodha Kite, Upstox, Dhan) — the loader
in `data.py` is a single function to swap. Intraday is where a 512-bar context covers a meaningful
horizon and where costs are 10 bps rather than 25.

**Cross-sectional rather than time-series.** Rank the whole NIFTY 50 by predicted return each day
and go long the top decile, short the bottom. Cross-sectional ranking is far more forgiving of a
badly-calibrated absolute forecast than a directional timing bet, and it neutralises market beta.
This is the version most likely to work, and the harness's per-symbol records are already in the
right shape to build it.

---

## Sources

- [shiyu-coder/Kronos — upstream repository](https://github.com/shiyu-coder/Kronos) (MIT)
- [Kronos: A Foundation Model for the Language of Financial Markets — arXiv:2508.02739](https://arxiv.org/abs/2508.02739)
- [NeoQuasar/Kronos-small](https://huggingface.co/NeoQuasar/Kronos-small) · [Kronos-base](https://huggingface.co/NeoQuasar/Kronos-base) · [Kronos-Tokenizer-base](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base)

This harness is MIT-licensed, as is upstream Kronos.
