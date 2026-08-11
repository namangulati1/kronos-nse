PY ?= python3

.PHONY: help setup install data demo eval backtest all test clean ui

help:
	@echo "make install    - pip install requirements"
	@echo "make setup      - clone upstream Kronos + check environment"
	@echo "make ui         - launch interactive Web Dashboard on browser"
	@echo "make data       - download and cache NSE OHLCV"
	@echo "make demo       - one-window forecast + fan chart per symbol"
	@echo "make eval       - walk-forward forecast evaluation (slow)"
	@echo "make backtest   - backtest cached forecasts (fast)"
	@echo "make all        - everything above in order"
	@echo "make test       - unit tests (offline, no weights needed)"
	@echo "make dryrun     - full pipeline on synthetic data with a random model"

install:
	$(PY) -m pip install -r requirements.txt

setup:
	$(PY) scripts/00_setup.py

ui:
	$(PY) -m streamlit run app.py

data:
	$(PY) scripts/01_fetch_data.py


demo:
	$(PY) scripts/02_demo_forecast.py

eval:
	$(PY) scripts/03_run_eval.py

backtest:
	$(PY) scripts/04_run_backtest.py --cost-sweep

all:
	$(PY) scripts/run_all.py

test:
	$(PY) -m pytest tests -v

dryrun:
	$(PY) tools/offline_dryrun.py

clean:
	rm -rf results/* data/cache/*
