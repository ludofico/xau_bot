"""
London Breakout Strategy for XAUUSD.

The core trading strategy optimized for aggressive account growth.
"""

from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np

from xauusd_strategy.config.settings import Settings, EntryConfig, ExitConfig
from xauusd_strategy.utils.logger import get_logger, log_trade
from xauusd_strategy.utils.time_utils import get_asian_box, is_in_session, TradingSession

logger = get_logger(__name__)


class SignalType(Enum):
    """Trade signal direction."""
    LONG = 1
    SHORT = -1
    NONE = 0


@dataclass
class TradeSignal:
    """
    Trade signal with all entry/exit parameters.
    
    Attributes:
        signal_type: Direction (LONG, SHORT, NONE)
        entry_price: Entry price
        stop_loss: Stop loss price
        take_profit: Take profit price
        atr_value: ATR at signal time
        roc_value: ROC at signal time
        asian_high: Asian session high
        asian_low: Asian session low
        probability: ML model probability (if available)
        timestamp: Signal timestamp
        risk_reward: Risk/reward ratio
    """
    signal_type: SignalType
    entry_price: float
    stop_loss: float
    take_profit: float
    atr_value: float
    roc_value: float
    asian_high: float
    asian_low: float
    probability: Optional[float] = None
    timestamp: Optional[pd.Timestamp] = None
    
    @property
    def risk_reward(self) -> float:
        """Calculate risk/reward ratio."""
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price)
        return reward / risk if risk > 0 else 0
    
    @property
    def risk_pips(self) -> float:
        """Risk in price units."""
        return abs(self.entry_price - self.stop_loss)
    
    @property
    def reward_pips(self) -> float:
        """Reward in price units."""
        return abs(self.take_profit - self.entry_price)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'signal_type': self.signal_type.name,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'atr_value': self.atr_value,
            'roc_value': self.roc_value,
            'asian_high': self.asian_high,
            'asian_low': self.asian_low,
            'probability': self.probability,
            'timestamp': str(self.timestamp) if self.timestamp else None,
            'risk_reward': self.risk_reward,
        }


class LondonBreakoutStrategy:
    """
    London Breakout Strategy optimized for aggressive XAUUSD growth.
    
    Entry Logic:
    1. Calculate Asian session box (00:00-07:00 CET)
    2. Wait for London open (08:00 CET)
    3. Entry on box breakout with:
       - ATR(14) > threshold (volatility filter)
       - ROC(5) > threshold (velocity confirmation)
       - Optional: ML probability > threshold
    
    Exit Logic:
    - SL: 1.2 × ATR (tight for R:R)
    - TP: 2.4 × ATR (R:R = 1:2)
    - Trailing: 0.8 × ATR after 1:1
    - Breakeven: Move SL to entry at 1:1 R:R
    
    Usage:
        >>> strategy = LondonBreakoutStrategy(settings)
        >>> df = strategy.prepare_data(ohlc_df)
        >>> signals = strategy.generate_signals(df)
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        # Entry parameters
        atr_period: int = 14,
        atr_min_multiplier: float = 0.5,
        roc_period: int = 5,
        roc_threshold: float = 0.15,
        # Exit parameters
        sl_atr_mult: float = 1.2,
        tp_atr_mult: float = 2.4,
        trailing_atr_mult: float = 0.8,
        breakeven_at_rr: float = 1.0,
        # Session times (CET)
        asian_start: str = "00:00",
        asian_end: str = "07:00",
        london_start: str = "08:00",
        london_end: str = "12:00",
        ny_start: str = "13:00",
        ny_end: str = "17:00",
        # ML filter
        ml_probability_threshold: float = 0.55,
    ):
        """
        Initialize strategy.
        
        Args:
            settings: Settings object (overrides individual params if provided)
            atr_period: ATR calculation period
            atr_min_multiplier: Minimum ATR relative to Asian range
            roc_period: ROC calculation period
            roc_threshold: Minimum ROC for velocity confirmation
            sl_atr_mult: Stop loss ATR multiplier
            tp_atr_mult: Take profit ATR multiplier
            trailing_atr_mult: Trailing stop ATR multiplier
            breakeven_at_rr: R:R ratio to move to breakeven
            asian_start: Asian session start (HH:MM)
            asian_end: Asian session end (HH:MM)
            london_start: London session start (HH:MM)
            london_end: London session end (HH:MM)
            ml_probability_threshold: Min ML probability for trade
        """
        if settings:
            # Use settings object
            self.atr_period = settings.entry.atr_period
            self.atr_min_multiplier = settings.entry.atr_min_multiplier
            self.roc_period = settings.entry.roc_period
            self.roc_threshold = settings.entry.roc_threshold
            self.sl_atr_mult = settings.exit.sl_atr_multiplier
            self.tp_atr_mult = settings.exit.tp_atr_multiplier
            self.trailing_atr_mult = settings.exit.trailing_atr_multiplier
            self.breakeven_at_rr = settings.exit.breakeven_after_rr
            self.asian_start = settings.asian_session.start
            self.asian_end = settings.asian_session.end
            self.london_start = settings.london_session.start
            self.london_end = settings.london_session.end
            self.ml_probability_threshold = settings.ml.probability_threshold
            # Default NY session if not in Settings
            self.ny_start = "13:00"
            self.ny_end = "17:00"
        else:
            # Use individual parameters
            self.atr_period = atr_period
            self.atr_min_multiplier = atr_min_multiplier
            self.roc_period = roc_period
            self.roc_threshold = roc_threshold
            self.sl_atr_mult = sl_atr_mult
            self.tp_atr_mult = tp_atr_mult
            self.trailing_atr_mult = trailing_atr_mult
            self.breakeven_at_rr = breakeven_at_rr
            self.asian_start = asian_start
            self.asian_end = asian_end
            self.london_start = london_start
            self.london_end = london_end
            self.ny_start = ny_start
            self.ny_end = ny_end
            self.ml_probability_threshold = ml_probability_threshold
        
        # Parse times
        self._asian_start_time = self._parse_time(self.asian_start)
        self._asian_end_time = self._parse_time(self.asian_end)
        self._london_start_time = self._parse_time(self.london_start)
        self._london_end_time = self._parse_time(self.london_end)
        self._ny_start_time = self._parse_time(self.ny_start)
        self._ny_end_time = self._parse_time(self.ny_end)
        
        logger.info(
            f"LondonBreakoutStrategy initialized: "
            f"ATR({self.atr_period}), ROC({self.roc_period}), "
            f"SL={self.sl_atr_mult}×ATR, TP={self.tp_atr_mult}×ATR"
        )
    
    def _parse_time(self, time_str: str) -> time:
        """Parse HH:MM string to time object."""
        h, m = map(int, time_str.split(":"))
        return time(h, m)
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data with required indicators.
        
        Args:
            df: OHLC DataFrame
        
        Returns:
            DataFrame with indicators added
        """
        df = df.copy()
        
        # Calculate True Range
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        
        # ATR
        df['atr'] = df['tr'].rolling(window=self.atr_period).mean()
        
        # Rate of Change (%)
        df['roc'] = (
            (df['close'] - df['close'].shift(self.roc_period)) / 
            df['close'].shift(self.roc_period) * 100
        )
        
        # EMA 200 (Trend Filter)
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # ADX Calculation (14 period)
        # 1. Directional Movement
        up = df['high'] - df['high'].shift(1)
        down = df['low'].shift(1) - df['low']
        
        # 2. +DM and -DM
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        
        # 3. Smoothed +DM/-DM and TR
        # Using Wilder's Smoothing (alpha = 1/n) equals ewm(alpha=1/14, adjust=False)
        tr_smooth = df['tr'].ewm(alpha=1/14, adjust=False).mean()
        plus_dm_smooth = pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean()
        minus_dm_smooth = pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean()
        
        # 4. DI
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        
        # 5. ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.ewm(alpha=1/14, adjust=False).mean()
        
        # Add session column
        df['hour'] = df.index.hour
        df['is_london'] = (
            (df.index.time >= self._london_start_time) & 
            (df.index.time <= self._london_end_time)
        )
        
        logger.debug(f"Data prepared: {len(df)} rows with indicators")
        
        return df
    
    def _calculate_asian_box(
        self,
        df: pd.DataFrame,
        current_time: pd.Timestamp
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate Asian session high/low for current day."""
        today = current_time.date()
        
        # Filter for Asian session today
        asian_mask = (
            (df.index.date == today) &
            (df.index.time >= self._asian_start_time) &
            (df.index.time < self._asian_end_time)
        )
        
        asian_data = df[asian_mask]
        
        if len(asian_data) == 0:
            return None, None
        
        return asian_data['high'].max(), asian_data['low'].min()

    def _calculate_london_box(
        self,
        df: pd.DataFrame,
        current_time: pd.Timestamp
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate London session high/low (08:00-13:00) for NY breakout."""
        today = current_time.date()
        
        # Filter for London session (Session 1)
        # Using configured London start to NY start as the 'Box' for NY
        mask = (
            (df.index.date == today) &
            (df.index.time >= self._london_start_time) &
            (df.index.time < self._ny_start_time)
        )
        
        data = df[mask]
        
        if len(data) == 0:
            return None, None
        
        return data['high'].max(), data['low'].min()
    
    def _is_london_session(self, timestamp: pd.Timestamp) -> bool:
        """Check if current time is in London session."""
        t = timestamp.time()
        return self._london_start_time <= t <= self._london_end_time
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        current_idx: int,
        ml_probability: Optional[float] = None
    ) -> Optional[TradeSignal]:
        """
        Generate trade signal for current bar.
        
        Args:
            df: OHLC DataFrame with indicators
            current_idx: Current bar index (integer position)
            ml_probability: Optional ML model probability
        
        Returns:
            TradeSignal if conditions met, None otherwise
        """
        if current_idx < self.atr_period + 10:
            return None  # Not enough data
        
        current_bar = df.iloc[current_idx]
        current_time = df.index[current_idx]
        current_time_val = current_time.time()
        
        # Check sessions
        is_london_open = self._london_start_time <= current_time_val <= self._london_end_time
        is_ny_open = self._ny_start_time <= current_time_val <= self._ny_end_time
        
        if not (is_london_open or is_ny_open):
            return None
        
        # Select Reference Box
        if is_london_open:
            # Trade Asian Breakout
            ref_high, ref_low = self._calculate_asian_box(df.iloc[:current_idx + 1], current_time)
        else:
            # Trade London Breakout (NY Session)
            ref_high, ref_low = self._calculate_london_box(df.iloc[:current_idx + 1], current_time)
        
        if ref_high is None or ref_low is None:
            return None
        
        box_range = ref_high - ref_low
        
        if box_range <= 0:
            return None
        
        # Get indicators
        atr = current_bar['atr']
        roc = current_bar['roc']
        close = current_bar['close']
        adx = current_bar.get('adx', 0)
        
        # Filter: Market must be trending (ADX > 20)
        # If ADX is low, market is choppy -> Ignore breakouts
        if adx < 20: 
            return None
        
        # Trend Filter: EMA 200 (if present)
        ema_200 = current_bar.get('ema_200', None)
        trend_long = True
        trend_short = True
        
        if ema_200 is not None and not np.isnan(ema_200):
            if close > ema_200:
                trend_short = False # Only Longs allowed
            else:
                trend_long = False # Only Shorts allowed
            # Uncommenting for debug
            # logger.info(f"Idx {current_idx}: C={close:.2f} E={ema_200:.2f} -> L={trend_long} S={trend_short}")
        else:
            if current_idx > 200:
                logger.warning(f"Idx {current_idx}: EMA 200 is NaN/Missing!")
        
        if pd.isna(atr) or pd.isna(roc):
            return None
        
        # Volatility filter: ATR should be above average (indicates active market)
        # Changed from comparing to Asian range (which was too strict)
        atr_ma = df['atr'].rolling(50).mean().iloc[current_idx] if current_idx >= 50 else df['atr'].iloc[:current_idx+1].mean()
        if pd.isna(atr_ma) or atr < self.atr_min_multiplier * atr_ma:
            return None
        
        # ML filter (if provided)
        if ml_probability is not None and ml_probability < self.ml_probability_threshold:
            return None
        
        # Check breakout conditions
        signal_type = SignalType.NONE
        entry_price = close
        stop_loss = 0.0
        take_profit = 0.0
        
        # LONG: Price breaks above Box high with bullish momentum AND Trend
        if close > ref_high and roc > self.roc_threshold and trend_long:
            signal_type = SignalType.SHORT # INVERTED (Mean Reversion)
            entry_price = close
            stop_loss = entry_price + (atr * self.sl_atr_mult)
            take_profit = entry_price - (atr * self.tp_atr_mult)
        
        # SHORT: Price breaks below Box low with bearish momentum AND Trend
        elif close < ref_low and roc < -self.roc_threshold and trend_short:
            signal_type = SignalType.LONG # INVERTED (Mean Reversion)
            entry_price = close
            stop_loss = entry_price - (atr * self.sl_atr_mult)
            take_profit = entry_price + (atr * self.tp_atr_mult)
        
        else:
            return None
        
        signal = TradeSignal(
            signal_type=signal_type,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr_value=atr,
            roc_value=roc,
            asian_high=ref_high, # Storing ref_high in asian_high field
            asian_low=ref_low,   # Storing ref_low in asian_low field
            probability=ml_probability,
            timestamp=current_time
        )
        
        logger.debug(
            f"Signal generated: {signal.signal_type.name} at {entry_price:.2f}, "
            f"SL={stop_loss:.2f}, TP={take_profit:.2f}, R:R={signal.risk_reward:.2f}"
        )
        
        return signal
    
    def generate_signals(
        self,
        df: pd.DataFrame,
        ml_probabilities: Optional[pd.Series] = None,
        max_signals_per_day: int = 1
    ) -> List[TradeSignal]:
        """
        Generate all signals for the dataset.
        
        Args:
            df: OHLC DataFrame (will be prepared if needed)
            ml_probabilities: Optional series of ML probabilities
            max_signals_per_day: Maximum signals per day (0 = unlimited)
        
        Returns:
            List of TradeSignal objects
        """
        # Ensure data is prepared
        if 'atr' not in df.columns:
            df = self.prepare_data(df)
        
        signals = []
        signals_today = 0
        last_signal_date = None
        
        for i in range(len(df)):
            current_time = df.index[i]
            current_date = current_time.date()
            
            # Reset counter on new day
            if last_signal_date != current_date:
                signals_today = 0
            
            # Check daily limit (0 = unlimited)
            if max_signals_per_day > 0 and signals_today >= max_signals_per_day:
                continue
            
            # Get ML probability if available
            ml_prob = None
            if ml_probabilities is not None and current_time in ml_probabilities.index:
                ml_prob = ml_probabilities.loc[current_time]
            
            signal = self.generate_signal(df, i, ml_prob)
            
            if signal:
                signals.append(signal)
                signals_today += 1
                last_signal_date = current_date
        
        logger.info(f"Generated {len(signals)} signals from {len(df)} bars")
        
        return signals
    
    def signals_to_dataframe(self, signals: List[TradeSignal]) -> pd.DataFrame:
        """Convert list of signals to DataFrame."""
        if not signals:
            return pd.DataFrame()
        
        data = [s.to_dict() for s in signals]
        df = pd.DataFrame(data)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        return df
    
    def update_trailing_stop(
        self,
        signal: TradeSignal,
        current_price: float,
    ) -> float:
        """
        Update trailing stop based on price movement.
        
        Args:
            signal: Original trade signal
            current_price: Current market price
        
        Returns:
            Updated stop loss price
        """
        atr = signal.atr_value
        
        # Calculate current P&L in R multiples
        risk = abs(signal.entry_price - signal.stop_loss)
        
        if signal.signal_type == SignalType.LONG:
            current_pnl = current_price - signal.entry_price
            current_rr = current_pnl / risk if risk > 0 else 0
            
            # Move to breakeven at configured R:R
            if current_rr >= self.breakeven_at_rr:
                new_sl = max(signal.entry_price, signal.stop_loss)
            else:
                new_sl = signal.stop_loss
            
            # Trail after breakeven
            if current_rr >= self.breakeven_at_rr:
                trail_sl = current_price - (atr * self.trailing_atr_mult)
                new_sl = max(new_sl, trail_sl)
            
            return new_sl
        
        else:  # SHORT
            current_pnl = signal.entry_price - current_price
            current_rr = current_pnl / risk if risk > 0 else 0
            
            if current_rr >= self.breakeven_at_rr:
                new_sl = min(signal.entry_price, signal.stop_loss)
            else:
                new_sl = signal.stop_loss
            
            if current_rr >= self.breakeven_at_rr:
                trail_sl = current_price + (atr * self.trailing_atr_mult)
                new_sl = min(new_sl, trail_sl)
            
            return new_sl
    
    def get_strategy_params(self) -> dict:
        """Get strategy parameters as dictionary."""
        return {
            'atr_period': self.atr_period,
            'atr_min_multiplier': self.atr_min_multiplier,
            'roc_period': self.roc_period,
            'roc_threshold': self.roc_threshold,
            'sl_atr_mult': self.sl_atr_mult,
            'tp_atr_mult': self.tp_atr_mult,
            'trailing_atr_mult': self.trailing_atr_mult,
            'breakeven_at_rr': self.breakeven_at_rr,
            'asian_session': f"{self.asian_start}-{self.asian_end}",
            'london_session': f"{self.london_start}-{self.london_end}",
            'ml_threshold': self.ml_probability_threshold,
        }
