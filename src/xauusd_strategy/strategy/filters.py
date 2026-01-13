"""
Entry filters for signal validation.
"""

from dataclasses import dataclass
from typing import Optional, List
import pandas as pd
import numpy as np

from xauusd_strategy.strategy.london_breakout import TradeSignal, SignalType
from xauusd_strategy.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FilterResult:
    """Result of filter evaluation."""
    passed: bool
    filter_name: str
    reason: Optional[str] = None


class EntryFilter:
    """
    Composite filter for entry validation.
    
    Applies multiple filters to determine if a signal should be taken.
    """
    
    def __init__(
        self,
        # Volatility filters
        min_atr: float = 0.5,
        max_atr: float = 5.0,
        min_atr_ratio: float = 0.8,
        max_atr_ratio: float = 2.0,
        # Momentum filters
        min_roc: float = 0.1,
        max_roc: float = 2.0,
        # Spread filter
        max_spread_pct: float = 0.02,
        # Time filters
        avoid_news_minutes: int = 30,
        # Market structure
        require_trend_alignment: bool = False,
    ):
        """
        Initialize entry filter.
        
        Args:
            min_atr: Minimum ATR value
            max_atr: Maximum ATR (avoid extreme volatility)
            min_atr_ratio: Minimum ATR relative to average
            max_atr_ratio: Maximum ATR relative to average
            min_roc: Minimum ROC for momentum
            max_roc: Maximum ROC (avoid overextension)
            max_spread_pct: Maximum spread as % of entry
            avoid_news_minutes: Minutes to avoid before/after news
            require_trend_alignment: Require trend alignment with signal
        """
        self.min_atr = min_atr
        self.max_atr = max_atr
        self.min_atr_ratio = min_atr_ratio
        self.max_atr_ratio = max_atr_ratio
        self.min_roc = min_roc
        self.max_roc = max_roc
        self.max_spread_pct = max_spread_pct
        self.avoid_news_minutes = avoid_news_minutes
        self.require_trend_alignment = require_trend_alignment
        
        # News times to avoid (major releases in CET)
        self.news_times: List[str] = [
            "14:30",  # US economic data
            "16:00",  # ISM data
            "20:00",  # FOMC
        ]
    
    def evaluate(
        self,
        signal: TradeSignal,
        df: Optional[pd.DataFrame] = None,
        current_spread: float = 0,
    ) -> List[FilterResult]:
        """
        Evaluate all filters for a signal.
        
        Args:
            signal: Trade signal to evaluate
            df: Optional DataFrame for additional context
            current_spread: Current market spread
        
        Returns:
            List of FilterResult objects
        """
        results = []
        
        # ATR filters
        results.append(self._check_atr_bounds(signal))
        
        # ROC filters
        results.append(self._check_roc_bounds(signal))
        
        # Spread filter
        if current_spread > 0:
            results.append(self._check_spread(signal, current_spread))
        
        # Time filter
        results.append(self._check_time_filter(signal))
        
        # Risk/reward filter
        results.append(self._check_risk_reward(signal))
        
        # Trend alignment (if df provided)
        if df is not None and self.require_trend_alignment:
            results.append(self._check_trend_alignment(signal, df))
        
        return results
    
    def should_take_trade(
        self,
        signal: TradeSignal,
        df: Optional[pd.DataFrame] = None,
        current_spread: float = 0,
    ) -> bool:
        """
        Check if trade should be taken (all filters pass).
        
        Args:
            signal: Trade signal to evaluate
            df: Optional DataFrame for context
            current_spread: Current spread
        
        Returns:
            True if all filters pass
        """
        results = self.evaluate(signal, df, current_spread)
        
        for result in results:
            if not result.passed:
                logger.debug(f"Filter failed: {result.filter_name} - {result.reason}")
                return False
        
        return True
    
    def _check_atr_bounds(self, signal: TradeSignal) -> FilterResult:
        """Check ATR is within bounds."""
        if signal.atr_value < self.min_atr:
            return FilterResult(
                passed=False,
                filter_name="atr_min",
                reason=f"ATR {signal.atr_value:.2f} < min {self.min_atr}"
            )
        
        if signal.atr_value > self.max_atr:
            return FilterResult(
                passed=False,
                filter_name="atr_max",
                reason=f"ATR {signal.atr_value:.2f} > max {self.max_atr}"
            )
        
        return FilterResult(passed=True, filter_name="atr_bounds")
    
    def _check_roc_bounds(self, signal: TradeSignal) -> FilterResult:
        """Check ROC is within bounds."""
        abs_roc = abs(signal.roc_value)
        
        if abs_roc < self.min_roc:
            return FilterResult(
                passed=False,
                filter_name="roc_min",
                reason=f"ROC {abs_roc:.2f}% < min {self.min_roc}%"
            )
        
        if abs_roc > self.max_roc:
            return FilterResult(
                passed=False,
                filter_name="roc_max",
                reason=f"ROC {abs_roc:.2f}% > max {self.max_roc}% (overextended)"
            )
        
        return FilterResult(passed=True, filter_name="roc_bounds")
    
    def _check_spread(self, signal: TradeSignal, spread: float) -> FilterResult:
        """Check spread is acceptable."""
        spread_pct = spread / signal.entry_price * 100
        
        if spread_pct > self.max_spread_pct:
            return FilterResult(
                passed=False,
                filter_name="spread",
                reason=f"Spread {spread_pct:.3f}% > max {self.max_spread_pct}%"
            )
        
        return FilterResult(passed=True, filter_name="spread")
    
    def _check_time_filter(self, signal: TradeSignal) -> FilterResult:
        """Check signal is not near news events."""
        if signal.timestamp is None:
            return FilterResult(passed=True, filter_name="time")
        
        current_time = signal.timestamp.time()
        
        for news_time_str in self.news_times:
            h, m = map(int, news_time_str.split(":"))
            news_minutes = h * 60 + m
            current_minutes = current_time.hour * 60 + current_time.minute
            
            diff = abs(current_minutes - news_minutes)
            
            if diff <= self.avoid_news_minutes:
                return FilterResult(
                    passed=False,
                    filter_name="news_filter",
                    reason=f"Within {self.avoid_news_minutes}min of {news_time_str} news"
                )
        
        return FilterResult(passed=True, filter_name="time")
    
    def _check_risk_reward(
        self,
        signal: TradeSignal,
        min_rr: float = 1.5
    ) -> FilterResult:
        """Check minimum risk/reward ratio."""
        if signal.risk_reward < min_rr:
            return FilterResult(
                passed=False,
                filter_name="risk_reward",
                reason=f"R:R {signal.risk_reward:.2f} < min {min_rr}"
            )
        
        return FilterResult(passed=True, filter_name="risk_reward")
    
    def _check_trend_alignment(
        self,
        signal: TradeSignal,
        df: pd.DataFrame
    ) -> FilterResult:
        """Check signal aligns with higher timeframe trend."""
        if signal.timestamp not in df.index:
            return FilterResult(passed=True, filter_name="trend_alignment")
        
        # Check EMA alignment
        if 'ema_50' not in df.columns or 'ema_200' not in df.columns:
            return FilterResult(passed=True, filter_name="trend_alignment")
        
        idx = df.index.get_loc(signal.timestamp)
        ema_50 = df['ema_50'].iloc[idx]
        ema_200 = df['ema_200'].iloc[idx]
        
        if pd.isna(ema_50) or pd.isna(ema_200):
            return FilterResult(passed=True, filter_name="trend_alignment")
        
        # Bullish trend: EMA50 > EMA200
        is_bullish = ema_50 > ema_200
        
        if signal.signal_type == SignalType.LONG and not is_bullish:
            return FilterResult(
                passed=False,
                filter_name="trend_alignment",
                reason="Long signal in bearish trend"
            )
        
        if signal.signal_type == SignalType.SHORT and is_bullish:
            return FilterResult(
                passed=False,
                filter_name="trend_alignment",
                reason="Short signal in bullish trend"
            )
        
        return FilterResult(passed=True, filter_name="trend_alignment")


class MLProbabilityFilter:
    """Filter based on ML model probability."""
    
    def __init__(self, threshold: float = 0.55):
        """
        Initialize ML filter.
        
        Args:
            threshold: Minimum probability to pass
        """
        self.threshold = threshold
    
    def evaluate(self, signal: TradeSignal) -> FilterResult:
        """Evaluate ML probability filter."""
        if signal.probability is None:
            return FilterResult(
                passed=True,
                filter_name="ml_probability",
                reason="No ML probability available"
            )
        
        if signal.probability < self.threshold:
            return FilterResult(
                passed=False,
                filter_name="ml_probability",
                reason=f"Probability {signal.probability:.2f} < threshold {self.threshold}"
            )
        
        return FilterResult(passed=True, filter_name="ml_probability")
