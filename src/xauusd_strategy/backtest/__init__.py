"""Backtesting module."""

from xauusd_strategy.backtest.engine import BacktestEngine, BacktestResult
from xauusd_strategy.backtest.costs import CostModel
from xauusd_strategy.backtest.metrics import PerformanceMetrics

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "CostModel",
    "PerformanceMetrics",
]
