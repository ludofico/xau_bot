"""
Signal generation and management utilities.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict
import pandas as pd
import numpy as np

from xauusd_strategy.strategy.london_breakout import SignalType, TradeSignal
from xauusd_strategy.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SignalStats:
    """Statistics for signal analysis."""
    total_signals: int
    long_signals: int
    short_signals: int
    avg_rr_ratio: float
    avg_atr: float
    signals_by_hour: Dict[int, int]
    signals_by_day: Dict[int, int]


class SignalGenerator:
    """
    Generate and manage trading signals.
    
    Aggregates signals from multiple strategies and applies filters.
    """
    
    def __init__(self, max_signals_per_day: int = 3):
        """
        Initialize signal generator.
        
        Args:
            max_signals_per_day: Maximum signals allowed per day
        """
        self.max_signals_per_day = max_signals_per_day
        self.signals: List[TradeSignal] = []
        self._daily_counts: Dict[str, int] = {}
    
    def add_signal(self, signal: TradeSignal) -> bool:
        """
        Add a signal if within daily limits.
        
        Args:
            signal: Trade signal to add
        
        Returns:
            True if signal was added, False if rejected
        """
        if signal.timestamp is None:
            logger.warning("Signal has no timestamp, adding anyway")
            self.signals.append(signal)
            return True
        
        date_key = signal.timestamp.strftime("%Y-%m-%d")
        current_count = self._daily_counts.get(date_key, 0)
        
        if current_count >= self.max_signals_per_day:
            logger.debug(f"Daily signal limit reached for {date_key}")
            return False
        
        self.signals.append(signal)
        self._daily_counts[date_key] = current_count + 1
        
        return True
    
    def get_signals(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        signal_type: Optional[SignalType] = None
    ) -> List[TradeSignal]:
        """
        Get filtered signals.
        
        Args:
            start: Filter start datetime
            end: Filter end datetime
            signal_type: Filter by signal type
        
        Returns:
            Filtered list of signals
        """
        filtered = self.signals.copy()
        
        if start:
            filtered = [s for s in filtered if s.timestamp and s.timestamp >= start]
        
        if end:
            filtered = [s for s in filtered if s.timestamp and s.timestamp <= end]
        
        if signal_type:
            filtered = [s for s in filtered if s.signal_type == signal_type]
        
        return filtered
    
    def get_statistics(self) -> SignalStats:
        """Calculate signal statistics."""
        if not self.signals:
            return SignalStats(
                total_signals=0,
                long_signals=0,
                short_signals=0,
                avg_rr_ratio=0,
                avg_atr=0,
                signals_by_hour={},
                signals_by_day={}
            )
        
        long_count = sum(1 for s in self.signals if s.signal_type == SignalType.LONG)
        short_count = sum(1 for s in self.signals if s.signal_type == SignalType.SHORT)
        
        rr_ratios = [s.risk_reward for s in self.signals if s.risk_reward > 0]
        avg_rr = np.mean(rr_ratios) if rr_ratios else 0
        
        atrs = [s.atr_value for s in self.signals if s.atr_value > 0]
        avg_atr = np.mean(atrs) if atrs else 0
        
        # Count by hour
        by_hour: Dict[int, int] = {}
        for s in self.signals:
            if s.timestamp:
                hour = s.timestamp.hour
                by_hour[hour] = by_hour.get(hour, 0) + 1
        
        # Count by day of week
        by_day: Dict[int, int] = {}
        for s in self.signals:
            if s.timestamp:
                day = s.timestamp.dayofweek
                by_day[day] = by_day.get(day, 0) + 1
        
        return SignalStats(
            total_signals=len(self.signals),
            long_signals=long_count,
            short_signals=short_count,
            avg_rr_ratio=avg_rr,
            avg_atr=avg_atr,
            signals_by_hour=by_hour,
            signals_by_day=by_day
        )
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert signals to DataFrame."""
        if not self.signals:
            return pd.DataFrame()
        
        data = []
        for signal in self.signals:
            data.append({
                'timestamp': signal.timestamp,
                'type': signal.signal_type.name,
                'entry': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'atr': signal.atr_value,
                'roc': signal.roc_value,
                'asian_high': signal.asian_high,
                'asian_low': signal.asian_low,
                'probability': signal.probability,
                'risk_reward': signal.risk_reward,
            })
        
        df = pd.DataFrame(data)
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
        
        return df
    
    def clear(self):
        """Clear all signals."""
        self.signals = []
        self._daily_counts = {}
    
    def __len__(self) -> int:
        return len(self.signals)
    
    def __iter__(self):
        return iter(self.signals)


def signals_to_vectorbt_format(
    df: pd.DataFrame,
    signals: List[TradeSignal]
) -> Dict[str, pd.Series]:
    """
    Convert signals to vectorbt-compatible format.
    
    Args:
        df: OHLC DataFrame with same index as signals
        signals: List of TradeSignal objects
    
    Returns:
        Dictionary with entries, exits, sl_stop, tp_stop, size Series
    """
    # Initialize series
    entries = pd.Series(False, index=df.index)
    exits = pd.Series(False, index=df.index)
    direction = pd.Series(0, index=df.index)  # 1 for long, -1 for short
    sl_stops = pd.Series(np.nan, index=df.index)
    tp_stops = pd.Series(np.nan, index=df.index)
    
    for signal in signals:
        if signal.timestamp not in df.index:
            continue
        
        idx = signal.timestamp
        entries.loc[idx] = True
        direction.loc[idx] = 1 if signal.signal_type == SignalType.LONG else -1
        sl_stops.loc[idx] = signal.stop_loss
        tp_stops.loc[idx] = signal.take_profit
    
    return {
        'entries': entries,
        'exits': exits,
        'direction': direction,
        'sl_stop': sl_stops,
        'tp_stop': tp_stops,
    }
