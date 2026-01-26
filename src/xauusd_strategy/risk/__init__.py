"""Risk management module."""

from xauusd_strategy.risk.kelly import KellyCalculator, calculate_kelly_fraction
from xauusd_strategy.risk.position_sizing import PositionSizer
from xauusd_strategy.risk.compound_manager import AggressiveCompoundManager, CompoundState
from xauusd_strategy.risk.circuit_breaker import CircuitBreaker
from xauusd_strategy.risk.risk_manager import RiskManager, RiskInfo, DrawdownStats

__all__ = [
    "KellyCalculator",
    "calculate_kelly_fraction",
    "PositionSizer",
    "AggressiveCompoundManager",
    "CompoundState",
    "CircuitBreaker",
    "RiskManager",
    "RiskInfo",
    "DrawdownStats",
]
