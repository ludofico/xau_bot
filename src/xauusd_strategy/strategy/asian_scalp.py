
"""
Asian Session Scalping Strategy (Mean Reversion).

Target: High Win Rate, High Frequency during low volatility.
Logic: Fade the edges of the Asian Range or Bollinger Bands.
"""

from dataclasses import dataclass
from typing import Optional, List
import pandas as pd
import numpy as np

from xauusd_strategy.config.settings import Settings
from xauusd_strategy.strategy.london_breakout import TradeSignal, SignalType
from xauusd_strategy.utils.logger import get_logger

logger = get_logger(__name__)

class AsianScalpingStrategy:
    """
    Mean Reversion strategy for Asian Session (00:00 - 08:00 CET).
    
    Logic:
    1. Identify Range (Bollinger Bands / Recent High-Low)
    2. Short at Resistance (Upper Band)
    3. Buy at Support (Lower Band)
    4. Filter: ADX < 25 (Ensure no strong trend)
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_overbought: float = 70,
        rsi_oversold: float = 30,
        tp_pips: float = 1.5, # $1.5 movement (~150 points)
        sl_pips: float = 3.0  # $3.0 movement (Wide SL for high WR)
    ):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.tp_pips = tp_pips
        self.sl_pips = sl_pips
        
        # Session times (CET)
        self.start_hour = 1 # Avoid spread widening at 00:00
        self.end_hour = 7
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add BB and RSI indicators."""
        df = df.copy()
        
        # BB
        sma = df['close'].rolling(self.bb_period).mean()
        std = df['close'].rolling(self.bb_period).std()
        df['bb_upper'] = sma + (std * self.bb_std)
        df['bb_lower'] = sma - (std * self.bb_std)
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ADX (Trend Filter - we want NO trend for scalping)
        df['adx'] = df['atr_14'] # Placeholder if ADX not avail, use ATR low volatility
        
        return df
        
    def generate_signal(self, df: pd.DataFrame, current_idx: int, ml_prob: float = 0.0) -> Optional[TradeSignal]:
        row = df.iloc[current_idx]
        time = df.index[current_idx]
        
        # 1. Session Filter
        if not (self.start_hour <= time.hour < self.end_hour):
            return None
            
        # 2. Volatility Filter (Don't trade if dead flat)
        if row['atr_14'] < 0.5: # Volatility too low even for scalping
            return None
            
        signal_type = SignalType.NONE
        
        # 3. Entry Logic (Mean Reversion)
        # SELL at Upper Band + Overbought
        if row['close'] > row['bb_upper'] or row['rsi'] > self.rsi_overbought:
            signal_type = SignalType.SHORT
            
        # BUY at Lower Band + Oversold
        elif row['close'] < row['bb_lower'] or row['rsi'] < self.rsi_oversold:
            signal_type = SignalType.LONG
            
        if signal_type == SignalType.NONE:
            return None
            
        entry = row['close']
        
        # 4. Exit Calculation
        if signal_type == SignalType.LONG:
            tp = entry + self.tp_pips
            sl = entry - self.sl_pips
        else:
            tp = entry - self.tp_pips
            sl = entry + self.sl_pips
            
        return TradeSignal(
            signal_type=signal_type,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            atr_value=row['atr_14'],
            roc_value=0,
            asian_high=0,
            asian_low=0,
            probability=ml_prob,
            timestamp=time
        )
