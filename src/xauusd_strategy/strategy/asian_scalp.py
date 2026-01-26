
"""
Precision Scalping Strategy - TREND-FOLLOWING with Multiple Confirmations.

Target: 85%+ Win Rate, Surgical Precision, ZERO false positives.
Logic: ONLY trade WITH the dominant trend, multiple confirmation layers.

CRITICAL RULES:
1. NEVER buy in a downtrend
2. NEVER sell in an uptrend
3. Wait for MULTIPLE confirmations before entry
4. Exit FAST - small wins, no losses
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
    PRECISION Trend-Following Scalping Strategy.
    
    CORE PRINCIPLE: Trade ONLY with the trend, NEVER against it.
    
    Multi-Layer Confirmation System:
    1. PRIMARY TREND (EMA 21 vs EMA 55): Determines allowed direction
    2. MOMENTUM (RSI direction): Must confirm trend direction
    3. PRICE ACTION: Must show trend continuation pattern
    4. VOLATILITY FILTER: ATR in acceptable range
    5. PULLBACK ENTRY: Enter on pullbacks, not chasing
    
    Win Rate Target: 85%+
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        # Trend Detection
        ema_fast: int = 21,
        ema_slow: int = 55,
        # Momentum
        rsi_period: int = 14,
        rsi_trend_bull: float = 50,  # RSI > 50 = bullish momentum
        rsi_trend_bear: float = 50,  # RSI < 50 = bearish momentum
        # Entry fine-tuning
        bb_period: int = 20,
        bb_std: float = 2.0,
        # Risk Management
        tp_pips: float = 1.0,       # Quick TP $1.0
        sl_pips: float = 0.80       # Ultra-tight SL $0.80 - R:R 1.25:1
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.rsi_trend_bull = rsi_trend_bull
        self.rsi_trend_bear = rsi_trend_bear
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.tp_pips = tp_pips
        self.sl_pips = sl_pips
        
        # Scalping active 24/7
        self.scalp_hours = list(range(0, 24))
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive indicators for precision trading."""
        df = df.copy()
        
        # === TREND DETECTION ===
        df['ema_fast'] = df['close'].ewm(span=self.ema_fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.ema_slow, adjust=False).mean()
        
        # Trend Direction: 1=Bullish, -1=Bearish, 0=Neutral
        df['trend'] = np.where(df['ema_fast'] > df['ema_slow'], 1, 
                              np.where(df['ema_fast'] < df['ema_slow'], -1, 0))
        
        # Trend Strength (EMA separation as % of price)
        df['trend_strength'] = abs(df['ema_fast'] - df['ema_slow']) / df['close'] * 100
        
        # === MOMENTUM ===
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI Direction (is RSI rising or falling?)
        df['rsi_slope'] = df['rsi'].diff(3)  # RSI change over 3 bars
        
        # === BOLLINGER BANDS (for pullback detection) ===
        sma = df['close'].rolling(self.bb_period).mean()
        std = df['close'].rolling(self.bb_period).std()
        df['bb_upper'] = sma + (std * self.bb_std)
        df['bb_lower'] = sma - (std * self.bb_std)
        df['bb_mid'] = sma
        
        # Position within BB (0=lower, 0.5=middle, 1=upper)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # === PRICE ACTION ===
        # Higher Highs / Lower Lows detection
        df['higher_high'] = df['high'] > df['high'].shift(1)
        df['lower_low'] = df['low'] < df['low'].shift(1)
        df['higher_low'] = df['low'] > df['low'].shift(1)
        df['lower_high'] = df['high'] < df['high'].shift(1)
        
        # Consecutive pattern (last 3 bars)
        df['bullish_structure'] = (
            df['higher_low'].rolling(3).sum() >= 2  # 2+ higher lows in 3 bars
        )
        df['bearish_structure'] = (
            df['lower_high'].rolling(3).sum() >= 2  # 2+ lower highs in 3 bars
        )
        
        # === VOLATILITY ===
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()
        
        # === CANDLE ANALYSIS ===
        df['candle_body'] = abs(df['close'] - df['open'])
        df['candle_range'] = df['high'] - df['low']
        df['is_bullish_candle'] = df['close'] > df['open']
        df['is_bearish_candle'] = df['close'] < df['open']
        
        # Strong candles (body > 60% of range)
        df['strong_bull_candle'] = (df['is_bullish_candle']) & (df['candle_body'] > df['candle_range'] * 0.6)
        df['strong_bear_candle'] = (df['is_bearish_candle']) & (df['candle_body'] > df['candle_range'] * 0.6)
        
        return df
        
    def generate_signal(self, df: pd.DataFrame, current_idx: int, ml_prob: float = 0.0) -> Optional[TradeSignal]:
        """
        Generate signal with SURGICAL PRECISION.
        
        Requirements for LONG:
        - Trend = BULLISH (EMA21 > EMA55)
        - RSI > 50 (bullish momentum)
        - RSI slope positive (momentum increasing)
        - Price near BB lower or middle (pullback entry)
        - Bullish candle structure
        
        Requirements for SHORT:
        - Trend = BEARISH (EMA21 < EMA55)
        - RSI < 50 (bearish momentum)
        - RSI slope negative (momentum increasing)
        - Price near BB upper or middle (pullback entry)
        - Bearish candle structure
        """
        if current_idx < 60:  # Need enough data for indicators
            return None
            
        row = df.iloc[current_idx]
        prev_row = df.iloc[current_idx - 1]
        time = df.index[current_idx]
        
        # 1. Session Filter
        if time.hour not in self.scalp_hours:
            return None
        
        # Check required columns
        required_cols = ['atr_14', 'trend', 'rsi', 'bb_position', 'ema_fast', 'ema_slow']
        for col in required_cols:
            if col not in df.columns or pd.isna(row[col]):
                return None
            
        # 2. Volatility Filter
        if row['atr_14'] < 0.5 or row['atr_14'] > 6.0:
            return None
        
        # Extract key values
        trend = row['trend']
        trend_strength = row['trend_strength']
        rsi = row['rsi']
        rsi_slope = row['rsi_slope']
        bb_pos = row['bb_position']
        close = row['close']
        ema_fast = row['ema_fast']
        ema_slow = row['ema_slow']
        
        # === CONFIRMATION COUNTERS ===
        bull_confirmations = 0
        bear_confirmations = 0
        
        # === CHECK FOR LONG ===
        # 1. Primary Trend MUST be bullish
        if trend == 1:
            bull_confirmations += 2  # Double weight for trend
            
            # 2. Trend must have minimum strength (0.05% separation)
            if trend_strength > 0.05:
                bull_confirmations += 1
            
            # 3. RSI above 50 (bullish territory)
            if rsi > 50:
                bull_confirmations += 1
                
            # 4. RSI slope positive (momentum rising)
            if rsi_slope is not None and rsi_slope > 0:
                bull_confirmations += 1
            
            # 5. Pullback entry: price in lower half of BB or near middle
            if bb_pos < 0.6:  # Below 60% of BB range = good pullback
                bull_confirmations += 1
            
            # 6. Bullish structure (higher lows)
            if row.get('bullish_structure', False):
                bull_confirmations += 1
                
            # 7. Current candle is bullish (confirmation)
            if row.get('is_bullish_candle', False):
                bull_confirmations += 1
        
        # === CHECK FOR SHORT ===
        # 1. Primary Trend MUST be bearish
        if trend == -1:
            bear_confirmations += 2  # Double weight for trend
            
            # 2. Trend must have minimum strength
            if trend_strength > 0.05:
                bear_confirmations += 1
            
            # 3. RSI below 50 (bearish territory)
            if rsi < 50:
                bear_confirmations += 1
                
            # 4. RSI slope negative (momentum falling)
            if rsi_slope is not None and rsi_slope < 0:
                bear_confirmations += 1
            
            # 5. Pullback entry: price in upper half of BB or near middle
            if bb_pos > 0.4:  # Above 40% of BB range = good pullback
                bear_confirmations += 1
            
            # 6. Bearish structure (lower highs)
            if row.get('bearish_structure', False):
                bear_confirmations += 1
                
            # 7. Current candle is bearish (confirmation)
            if row.get('is_bearish_candle', False):
                bear_confirmations += 1
        
        # === DECISION ===
        # Need MINIMUM 6 confirmations for entry (out of 8 possible)
        MIN_CONFIRMATIONS = 6
        
        signal_type = SignalType.NONE
        
        if bull_confirmations >= MIN_CONFIRMATIONS:
            signal_type = SignalType.LONG
            logger.info(
                f"🎯 PRECISION LONG | Confirmations: {bull_confirmations}/8 | "
                f"Trend: BULL | RSI: {rsi:.1f} | BB_pos: {bb_pos:.2f} | "
                f"EMA21: {ema_fast:.2f} > EMA55: {ema_slow:.2f}"
            )
            
        elif bear_confirmations >= MIN_CONFIRMATIONS:
            signal_type = SignalType.SHORT
            logger.info(
                f"🎯 PRECISION SHORT | Confirmations: {bear_confirmations}/8 | "
                f"Trend: BEAR | RSI: {rsi:.1f} | BB_pos: {bb_pos:.2f} | "
                f"EMA21: {ema_fast:.2f} < EMA55: {ema_slow:.2f}"
            )
        else:
            # Log why we didn't trade (for debugging)
            if trend == 1 and bull_confirmations >= 3:
                logger.debug(
                    f"⏸️ LONG pending: {bull_confirmations}/6 confirmations | "
                    f"RSI: {rsi:.1f} | BB: {bb_pos:.2f}"
                )
            elif trend == -1 and bear_confirmations >= 3:
                logger.debug(
                    f"⏸️ SHORT pending: {bear_confirmations}/6 confirmations | "
                    f"RSI: {rsi:.1f} | BB: {bb_pos:.2f}"
                )
            return None
            
        entry = close
        
        # Exit Calculation
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
            roc_value=rsi_slope,  # Using RSI slope as ROC proxy for this strategy
            asian_high=0.0,
            asian_low=0.0,
            probability=ml_prob,
            timestamp=time,
            source="AsianScalp"
        )
