"""Kronos NSE testbench.

A walk-forward evaluation and backtesting harness for the Kronos financial
foundation model (https://github.com/shiyu-coder/Kronos) on Indian equity
market data.
"""

__version__ = "0.1.0"

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .config import Config, load_config  # noqa: F401

