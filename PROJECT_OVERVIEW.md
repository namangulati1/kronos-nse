# Kronos NSE Testbench - Project Overview & Documentation

The **Kronos NSE Testbench** is an evaluation, Monte-Carlo forecasting, and quantitative backtesting harness designed for applying the **Kronos Financial Foundation Model** ([shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos)) to Indian equity market data (NSE - National Stock Exchange).

---

## 1. Executive Summary

Financial forecasting traditionally relies on point-forecasting models (e.g. ARIMA, GARCH, linear regressions, standard neural networks) that output a single predicted price. **Kronos** represents a paradigm shift: it is a generative, autoregressive foundation model pretrained on **~12 billion K-line OHLCV records** across 45 global exchanges.

Rather than predicting a single deterministic number, Kronos is a **stochastic sampler**. It generates multiple probable future price trajectories in discrete token space. The spread and distribution across these trajectories encode model uncertainty, directional probability ($P(\text{up})$), and predicted volatility.

This testbench provides the infrastructure to download Indian market data, draw Monte-Carlo forecasts, conduct strict walk-forward out-of-sample backtests, and account for realistic Indian market micro-structure (STT, slippage, execution timing).

---

## 2. Core Concepts & Technical Architecture

```mermaid
flowchart TD
    A[NSE Market Data\nYahoo Finance / Custom API] --> B[Data Engine & Cache\nkronos_nse/data.py]
    B --> C[Monte-Carlo Sampler\nkronos_nse/predictor.py]
    
    subgraph Kronos Foundation Model
        D[Kronos Tokenizer\nOHLCV -> Discrete Tokens]
        E[Autoregressive Transformer\nK-Line Continuation]
        D --> E
    end
    
    C <--> Kronos Foundation Model
    
    C --> F[Forecast Distribution\n30+ Sampled Paths]
    F --> G[Walk-Forward Evaluator\nkronos_nse/evaluate.py]
    G --> H[Backtest & Signal Engine\nkronos_nse/backtest.py]
    
    H --> I[Performance Reports & Charts\nCAGR, Sharpe, Drawdown, Cost Sweeps]
```

### A. Two-Stage Generative Foundation Model
1. **Hierarchical Tokenizer (`Kronos-Tokenizer-base`)**: Quantizes continuous OHLCV (Open, High, Low, Close, Volume) market bars into discrete hierarchical tokens.
2. **Autoregressive Transformer (`Kronos-small` / `Kronos-base`)**: Operates on token sequences, predicting future continuations token by token.

### B. Monte-Carlo Path Recovery (`kronos_nse/predictor.py`)
Upstream Kronos outputs the mean across paths by default, throwing away distributional insights. `kronos_nse/predictor.py` passes $N$ identical copies of a context window through `predict_batch` with `sample_count=1`. This recovers $N$ independent stochastic paths to compute:
- **Expected Return**: $\mathbb{E}[\ln(P_{\text{horizon}} / P_{\text{last}})]$
- **Directional Probability ($P(\text{up})$)**: Fraction of sampled paths ending above context close.
- **Predicted Volatility**: Implied bar-to-bar volatility across sampled paths.
- **Uncertainty Bounds**: 5th, 25th, 75th, and 95th price quantiles.

### C. Pretraining Contamination Guard
Kronos was pretrained on global market data up to mid-2025. Any backtest conducted prior to August 2025 suffers from **in-sample data contamination**. This harness defaults to evaluating performance exclusively on out-of-sample data starting after **August 1, 2025** (`evaluate.start: "2025-08-01"`).

---

## 3. Key Components & File Breakdown

| File / Folder | Purpose |
| :--- | :--- |
| [configs/default.yaml](file:///Users/naman.gulati/Library/CloudStorage/OneDrive-TV18BroadcastLimited/Desktop/coding/kronos-nse/configs/default.yaml) | Primary YAML configuration for symbols, context windows, model checkpoints, backtest thresholds, and execution costs. |
| [kronos_nse/data.py](file:///Users/naman.gulati/Library/CloudStorage/OneDrive-TV18BroadcastLimited/Desktop/coding/kronos-nse/kronos_nse/data.py) | Downloads NSE OHLCV data via `yfinance`, performs corporate action adjustments, validates data quality, and manages local caching. |
| [kronos_nse/predictor.py](file:///Users/naman.gulati/Library/CloudStorage/OneDrive-TV18BroadcastLimited/Desktop/coding/kronos-nse/kronos_nse/predictor.py) | Encapsulates the Kronos model runner, device selection (`auto`, `cuda`, `mps`, `cpu`), and Monte-Carlo sampling. |
| [kronos_nse/evaluate.py](file:///Users/naman.gulati/Library/CloudStorage/OneDrive-TV18BroadcastLimited/Desktop/coding/kronos-nse/kronos_nse/evaluate.py) | Executes rolling walk-forward evaluation windows across historical data. |
| [kronos_nse/backtest.py](file:///Users/naman.gulati/Library/CloudStorage/OneDrive-TV18BroadcastLimited/Desktop/coding/kronos-nse/kronos_nse/backtest.py) | Simulates trading strategies (`next_open` execution, dead-zone thresholds, cost sweeps, position overlap, bootstrap confidence intervals). |
| [kronos_nse/metrics.py](file:///Users/naman.gulati/Library/CloudStorage/OneDrive-TV18BroadcastLimited/Desktop/coding/kronos-nse/kronos_nse/metrics.py) | Implements statistical scoring (CRPS, Brier Skill Score, Directional Accuracy, Sharpe Ratio, Max Drawdown). |
| [kronos_nse/vendor.py](file:///Users/naman.gulati/Library/CloudStorage/OneDrive-TV18BroadcastLimited/Desktop/coding/kronos-nse/kronos_nse/vendor.py) | Clones and links the upstream `shiyu-coder/Kronos` repository into `third_party/Kronos`. |
| [.env.example](file:///Users/naman.gulati/Library/CloudStorage/OneDrive-TV18BroadcastLimited/Desktop/coding/kronos-nse/.env.example) | Environment variable template for API tokens (`HF_TOKEN`) and path/device overrides. |

---

## 4. Market Microstructure & Backtest Rules for NSE

Trading Indian equities involves specific structural constraints that this testbench models:

1. **Realistic Execution Timing (`next_open`)**:
   Predictions generated at the market Close ($T$) are executed at the **Open of $T+1$**, avoiding unrealizable "on-close" fill assumptions.
2. **Indian Transaction Costs (`cost_bps_roundtrip: 25`)**:
   Accounts for Securities Transaction Tax (STT ~0.1%), exchange turnover charges, SEBI fees, stamp duty, GST, and bid-ask slippage.
3. **Threshold Dead-Zone (`threshold_bps: 25`)**:
   Positions are only taken when expected return exceeds transaction cost noise, avoiding over-trading.

---

## 5. How to Run the Project

### Step 1: Environment Setup
```bash
# Clone and enter directory
git clone https://github.com/namangulati1/kronos-nse.git
cd kronos-nse

# Setup virtual environment and dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
```

### Step 2: Initialize upstream dependencies & test installation
```bash
make setup      # Clones upstream Kronos into third_party/ Kronos
make test       # Runs offline unit test suite (~5s)
make dryrun     # Runs full pipeline simulation on synthetic random walk (~30s)
```

### Step 3: Fetch Data & Run Live Model
```bash
make data       # Downloads latest daily OHLCV for NIFTY 50, Bank Nifty, Reliance, HDFC, Infosys, etc.
make demo       # Runs live Monte-Carlo forecast demo and outputs fan charts in results/
make eval       # Runs historical walk-forward evaluation
make backtest   # Runs strategy backtest and transaction cost sweep
```

---

## 6. Output Artifacts & Metrics

After running evaluations and backtests, outputs are generated in the `results/` directory:
- **`data_quality.csv`**: Data completeness and zero-volume audit.
- **`fan_chart.png`**: Visual fan chart showing historic prices alongside 30 forecast paths and quantile bands ($q_{05}$ to $q_{95}$).
- **`equity.png`**: Strategy cumulative return vs benchmark NIFTY buy-and-hold.
- **`cost_sweep`**: Transaction cost sensitivity table showing breakeven transaction costs in basis points (bps).
