"""
Momentum Stacker Strategy for XAUUSD.

High-frequency scalping strategy that uses momentum impulses to enter and stack positions.
Optimized for 1-minute to 5-minute timeframes.

Logic:
1. Trend: EMA 50
2. Momentum: RSI (14)
3. Volatility: ATR (14)
4. Entry: Impulse move in trend direction
5. Exit: Trailing Stop (Chandelier)
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
from enum import Enum

from xauusd_strategy.utils.logger import get_logger
from xauusd_strategy.strategy.london_breakout import TradeSignal, SignalType

logger = get_logger(__name__)

class MomentumStackerStrategy:
    """
    Momentum Stacker Strategy.
    
    Aims for high-frequency entries during strong momentum bursts.
    Inherently "stacks" positions if momentum persists across multiple bars.
    """
    
    def __init__(
        self,
        ema_period: int = 50,
        rsi_period: int = 14,
        atr_period: int = 14,
        rsi_buy_threshold: float = 55.0,
        rsi_sell_threshold: float = 45.0,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        sl_atr_mult: float = 1.5,
        tp_atr_mult: float = 4.0, # High R:R attempts
        trailing_sl_mult: float = 1.0,
    ):
        self.ema_period = ema_period
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.rsi_buy_threshold = rsi_buy_threshold
        self.rsi_sell_threshold = rsi_sell_threshold
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.trailing_sl_mult = trailing_sl_mult
        
        logger.info(f"MomentumStacker initialized: EMA{ema_period} RSI{rsi_period}")

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators."""
        df = df.copy()
        
        # EMA
        df['ema_trend'] = df['close'].ewm(span=self.ema_period, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=self.atr_period).mean()
        
        # ROC (Velocity)
        df['roc'] = df['close'].pct_change(periods=3) * 100
        
        return df

    def generate_signal(self, df: pd.DataFrame, idx: int, ml_prob: Optional[float] = None) -> Optional[TradeSignal]:
        if idx < self.ema_period:
            return None
            
        bar = df.iloc[idx]
        close = bar['close']
        ema = bar['ema_trend']
        rsi = bar['rsi']
        atr = bar['atr']
        roc = bar['roc']
        time = df.index[idx]
        
        if pd.isna(rsi) or pd.isna(atr):
            return None
            
        signal_type = SignalType.NONE
        
        # LONG Logic
        # 1. Trend is UP (Close > EMA)
        # 2. RSI is Bullish (>55) but not Exhausted (<70)
        # 3. Momentum is Positive (ROC > 0)
        if close > ema and rsi > self.rsi_buy_threshold and rsi < self.rsi_overbought and roc > 0:
            signal_type = SignalType.LONG
            stop_loss = close - (atr * self.sl_atr_mult)
            take_profit = close + (atr * self.tp_atr_mult)
            
        # SHORT Logic
        elif close < ema and rsi < self.rsi_sell_threshold and rsi > self.rsi_oversold and roc < 0:
            signal_type = SignalType.SHORT
            stop_loss = close + (atr * self.sl_atr_mult)
            take_profit = close - (atr * self.tp_atr_mult)
            
        else:
            return None
            
        return TradeSignal(
            signal_type=signal_type,
            entry_price=close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr_value=atr,
            roc_value=roc,
            asian_high=0, # Not used
            asian_low=0,  # Not used
            probability=ml_prob,
            timestamp=time
        )

    def generate_signals(
        self,
        df: pd.DataFrame,
        ml_probabilities: Optional[pd.Series] = None
    ) -> List[TradeSignal]:
        if 'rsi' not in df.columns:
            df = self.prepare_data(df)
            
        signals = []
        for i in range(len(df)):
            sig = self.generate_signal(df, i)
            if sig:
                signals.append(sig)
                
        return signals
