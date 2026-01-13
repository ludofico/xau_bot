"""
XAUUSD Aggressive Trading Strategy

A Python-based algorithmic trading system for XAUUSD (Gold) designed for
aggressive account growth from small capital (€250 → €500-1000/month).

Features:
- London Breakout Strategy with Asian session box
- XGBoost/LightGBM probability filter
- Fractional Kelly position sizing with compounding
- Multi-session trading (Asian, London, NY)
- vectorbt backtesting with realistic costs
- MT5 live execution adapter
"""

__version__ = "0.1.0"
__author__ = "Trading Bot"

from xauusd_strategy.config.settings import Settings
from xauusd_strategy.strategy.london_breakout import LondonBreakoutStrategy
from xauusd_strategy.risk.compound_manager import AggressiveCompoundManager

__all__ = [
    "Settings",
    "LondonBreakoutStrategy", 
    "AggressiveCompoundManager",
]
