"""Strategy module for XAUUSD trading."""

from xauusd_strategy.strategy.london_breakout import LondonBreakoutStrategy, TradeSignal, SignalType
from xauusd_strategy.strategy.signals import SignalGenerator
from xauusd_strategy.strategy.filters import EntryFilter
from xauusd_strategy.strategy.regime_detector import RegimeDetector, MarketRegime, RegimeAnalysis
from xauusd_strategy.strategy.multi_tf_aggregator import (
    MultiTFAggregator,
    Timeframe,
    TrendDirection,
    MultiTFAnalysis,
    TimeframeAnalysis
)
from xauusd_strategy.strategy.signal_aggregator import SignalAggregator, AggregatedSignal

__all__ = [
    "LondonBreakoutStrategy",
    "TradeSignal",
    "SignalType",
    "SignalGenerator",
    "EntryFilter",
    "RegimeDetector",
    "MarketRegime",
    "RegimeAnalysis",
    "MultiTFAggregator",
    "Timeframe",
    "TrendDirection",
    "MultiTFAnalysis",
    "TimeframeAnalysis",
    "SignalAggregator",
    "AggregatedSignal",
]
