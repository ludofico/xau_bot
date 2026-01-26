"""
Multi-Timeframe Analysis Aggregator for XAUUSD Trading.

Provides:
- Unified analysis across M1, M5, H1, D1 timeframes
- Trend alignment scoring
- Signal confluence detection
- Strategy selection based on timeframe context

Design:
- D1 provides macro direction (weight: 40%)
- H1 provides intermediate trend (weight: 30%)
- M5 provides entry timing (weight: 20%)
- M1 provides execution precision (weight: 10%)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Tuple
import pandas as pd
import numpy as np

from xauusd_strategy.strategy.regime_detector import RegimeDetector, MarketRegime, RegimeAnalysis
from xauusd_strategy.utils.logger import get_logger

logger = get_logger(__name__)


class Timeframe(Enum):
    """Trading timeframes."""
    M1 = "M1"
    M5 = "M5"
    H1 = "H1"
    D1 = "D1"
    
    @property
    def weight(self) -> float:
        """Default weight for confluence scoring."""
        weights = {
            Timeframe.D1: 0.40,
            Timeframe.H1: 0.30,
            Timeframe.M5: 0.20,
            Timeframe.M1: 0.10
        }
        return weights.get(self, 0.0)


class TrendDirection(Enum):
    """Trend direction across timeframes."""
    STRONG_UP = "strong_up"
    UP = "up"
    NEUTRAL = "neutral"
    DOWN = "down"
    STRONG_DOWN = "strong_down"


@dataclass
class TimeframeAnalysis:
    """Analysis for a single timeframe."""
    timeframe: Timeframe
    regime: MarketRegime
    trend_direction: TrendDirection
    strength: float  # 0.0 to 1.0
    ema_trend: int  # 1=bullish, -1=bearish, 0=neutral
    rsi: float
    atr_pct: float  # ATR as % of price
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "timeframe": self.timeframe.value,
            "regime": self.regime.value,
            "trend_direction": self.trend_direction.value,
            "strength": round(self.strength, 3),
            "ema_trend": self.ema_trend,
            "rsi": round(self.rsi, 1),
            "atr_pct": round(self.atr_pct, 4)
        }


@dataclass
class MultiTFAnalysis:
    """Complete multi-timeframe analysis."""
    timestamp: pd.Timestamp
    analyses: Dict[Timeframe, TimeframeAnalysis]
    
    # Aggregate metrics
    confluence_score: float  # -1.0 (bearish) to 1.0 (bullish)
    alignment_score: float  # 0.0 (conflicting) to 1.0 (aligned)
    dominant_trend: TrendDirection
    recommended_bias: str  # "long", "short", "neutral"
    
    # Context
    volatility_regime: str  # "low", "normal", "high"
    notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON output."""
        return {
            "timestamp": str(self.timestamp),
            "confluence_score": round(self.confluence_score, 3),
            "alignment_score": round(self.alignment_score, 3),
            "dominant_trend": self.dominant_trend.value,
            "recommended_bias": self.recommended_bias,
            "volatility_regime": self.volatility_regime,
            "notes": self.notes,
            "timeframes": {tf.value: a.to_dict() for tf, a in self.analyses.items()}
        }


class MultiTFAggregator:
    """
    Multi-Timeframe Analysis Aggregator.
    
    Combines analysis from multiple timeframes to determine
    trading bias, confluence, and optimal strategy selection.
    """
    
    def __init__(
        self,
        weights: Optional[Dict[Timeframe, float]] = None,
        ema_fast: int = 21,
        ema_slow: int = 55,
        rsi_period: int = 14,
        atr_period: int = 14
    ):
        """
        Initialize aggregator.
        
        Args:
            weights: Custom weights for each timeframe (default: D1=0.4, H1=0.3, M5=0.2, M1=0.1)
            ema_fast: Fast EMA period
            ema_slow: Slow EMA period
            rsi_period: RSI period
            atr_period: ATR period
        """
        self.weights = weights or {
            Timeframe.D1: 0.40,
            Timeframe.H1: 0.30,
            Timeframe.M5: 0.20,
            Timeframe.M1: 0.10
        }
        
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        
        self.regime_detector = RegimeDetector(
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            atr_period=atr_period
        )
    
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add required indicators to dataframe."""
        df = df.copy()
        
        # EMA
        df['ema_fast'] = df['close'].ewm(span=self.ema_fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.ema_slow, adjust=False).mean()
        df['ema_trend'] = np.where(
            df['ema_fast'] > df['ema_slow'], 1,
            np.where(df['ema_fast'] < df['ema_slow'], -1, 0)
        )
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).ewm(span=self.rsi_period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=self.rsi_period, adjust=False).mean()
        rs = gain / (loss + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR as percentage
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].ewm(span=self.atr_period, adjust=False).mean()
        df['atr_pct'] = df['atr'] / df['close']
        
        return df
    
    def _determine_trend_direction(
        self,
        ema_trend: int,
        rsi: float,
        regime: MarketRegime
    ) -> Tuple[TrendDirection, float]:
        """
        Determine trend direction and strength.
        
        Returns:
            Tuple of (direction, strength)
        """
        # Base direction from EMA
        if ema_trend == 1:
            base = TrendDirection.UP
        elif ema_trend == -1:
            base = TrendDirection.DOWN
        else:
            base = TrendDirection.NEUTRAL
        
        # Strength factors
        strength = 0.5  # Base strength
        
        # RSI confirmation
        if base == TrendDirection.UP and rsi > 50:
            strength += 0.2
            if rsi > 60:
                strength += 0.1
        elif base == TrendDirection.DOWN and rsi < 50:
            strength += 0.2
            if rsi < 40:
                strength += 0.1
        
        # Regime confirmation
        if regime == MarketRegime.TRENDING_UP and base == TrendDirection.UP:
            base = TrendDirection.STRONG_UP
            strength += 0.2
        elif regime == MarketRegime.TRENDING_DOWN and base == TrendDirection.DOWN:
            base = TrendDirection.STRONG_DOWN
            strength += 0.2
        
        # Cap strength
        strength = min(1.0, strength)
        
        return base, strength
    
    def analyze_timeframe(
        self,
        df: pd.DataFrame,
        timeframe: Timeframe
    ) -> TimeframeAnalysis:
        """
        Analyze a single timeframe.
        
        Args:
            df: OHLC data for the timeframe
            timeframe: Which timeframe this is
            
        Returns:
            TimeframeAnalysis with trend and regime info
        """
        if len(df) < 60:
            logger.warning(f"Insufficient data for {timeframe.value}")
            return TimeframeAnalysis(
                timeframe=timeframe,
                regime=MarketRegime.UNKNOWN,
                trend_direction=TrendDirection.NEUTRAL,
                strength=0.0,
                ema_trend=0,
                rsi=50.0,
                atr_pct=0.0
            )
        
        # Add indicators
        df_prep = self._add_indicators(df)
        
        # Get regime
        regime_analysis = self.regime_detector.detect(df_prep)
        
        # Get current values
        current = df_prep.iloc[-1]
        ema_trend = int(current['ema_trend'])
        rsi = current['rsi']
        atr_pct = current['atr_pct']
        
        # Determine direction and strength
        direction, strength = self._determine_trend_direction(
            ema_trend, rsi, regime_analysis.regime
        )
        
        return TimeframeAnalysis(
            timeframe=timeframe,
            regime=regime_analysis.regime,
            trend_direction=direction,
            strength=strength,
            ema_trend=ema_trend,
            rsi=rsi,
            atr_pct=atr_pct
        )
    
    def aggregate(
        self,
        data: Dict[Timeframe, pd.DataFrame]
    ) -> MultiTFAnalysis:
        """
        Aggregate analysis from multiple timeframes.
        
        Args:
            data: Dictionary of timeframe -> OHLC DataFrame
            
        Returns:
            MultiTFAnalysis with confluence and recommendations
        """
        analyses: Dict[Timeframe, TimeframeAnalysis] = {}
        
        # Analyze each timeframe
        for tf, df in data.items():
            if df is not None and not df.empty:
                analyses[tf] = self.analyze_timeframe(df, tf)
        
        if not analyses:
            return self._empty_analysis()
        
        # Calculate confluence score (-1 to 1)
        confluence = 0.0
        total_weight = 0.0
        
        for tf, analysis in analyses.items():
            weight = self.weights.get(tf, 0.1)
            total_weight += weight
            
            # Direction contribution
            direction_value = {
                TrendDirection.STRONG_UP: 1.0,
                TrendDirection.UP: 0.5,
                TrendDirection.NEUTRAL: 0.0,
                TrendDirection.DOWN: -0.5,
                TrendDirection.STRONG_DOWN: -1.0
            }.get(analysis.trend_direction, 0.0)
            
            confluence += weight * direction_value * analysis.strength
        
        if total_weight > 0:
            confluence /= total_weight
        
        # Calculate alignment score (0 to 1)
        directions = [a.ema_trend for a in analyses.values()]
        if len(directions) > 1:
            # All same direction = 1.0, mixed = lower
            if all(d == 1 for d in directions):
                alignment = 1.0
            elif all(d == -1 for d in directions):
                alignment = 1.0
            elif all(d == 0 for d in directions):
                alignment = 0.5  # All neutral
            else:
                # Count agreement
                bullish = sum(1 for d in directions if d == 1)
                bearish = sum(1 for d in directions if d == -1)
                alignment = max(bullish, bearish) / len(directions)
        else:
            alignment = 0.5
        
        # Dominant trend from higher timeframes
        if Timeframe.D1 in analyses:
            dominant = analyses[Timeframe.D1].trend_direction
        elif Timeframe.H1 in analyses:
            dominant = analyses[Timeframe.H1].trend_direction
        else:
            # Use confluence to determine
            if confluence > 0.3:
                dominant = TrendDirection.UP
            elif confluence < -0.3:
                dominant = TrendDirection.DOWN
            else:
                dominant = TrendDirection.NEUTRAL
        
        # Recommended bias
        if confluence > 0.3 and alignment > 0.6:
            bias = "long"
        elif confluence < -0.3 and alignment > 0.6:
            bias = "short"
        else:
            bias = "neutral"
        
        # Volatility regime (from M5 or M1)
        if Timeframe.M5 in analyses:
            atr_pct = analyses[Timeframe.M5].atr_pct
        elif Timeframe.M1 in analyses:
            atr_pct = analyses[Timeframe.M1].atr_pct
        else:
            atr_pct = 0.002
        
        if atr_pct < 0.0015:
            vol_regime = "low"
        elif atr_pct < 0.003:
            vol_regime = "normal"
        else:
            vol_regime = "high"
        
        # Generate notes
        notes = []
        if alignment > 0.8:
            notes.append("Strong multi-TF alignment")
        elif alignment < 0.4:
            notes.append("⚠️ Conflicting signals across timeframes")
        
        if abs(confluence) > 0.6:
            notes.append(f"High confluence: {bias.upper()} bias")
        
        # Timestamp from most granular available
        timestamp = None
        for tf in [Timeframe.M1, Timeframe.M5, Timeframe.H1, Timeframe.D1]:
            if tf in data and data[tf] is not None and not data[tf].empty:
                timestamp = data[tf].index[-1]
                break
        
        return MultiTFAnalysis(
            timestamp=timestamp or pd.Timestamp.now(),
            analyses=analyses,
            confluence_score=confluence,
            alignment_score=alignment,
            dominant_trend=dominant,
            recommended_bias=bias,
            volatility_regime=vol_regime,
            notes=notes
        )
    
    def _empty_analysis(self) -> MultiTFAnalysis:
        """Return empty analysis when no data available."""
        return MultiTFAnalysis(
            timestamp=pd.Timestamp.now(),
            analyses={},
            confluence_score=0.0,
            alignment_score=0.0,
            dominant_trend=TrendDirection.NEUTRAL,
            recommended_bias="neutral",
            volatility_regime="unknown",
            notes=["No data available"]
        )
    
    def get_entry_quality(self, analysis: MultiTFAnalysis, signal_direction: int) -> float:
        """
        Score entry quality based on multi-TF alignment.
        
        Args:
            analysis: Multi-TF analysis
            signal_direction: 1 for long, -1 for short
            
        Returns:
            Quality score from 0.0 (poor) to 1.0 (excellent)
        """
        quality = 0.5  # Base quality
        
        # Confluence alignment with signal
        if signal_direction == 1:
            if analysis.confluence_score > 0:
                quality += analysis.confluence_score * 0.3
            else:
                quality -= abs(analysis.confluence_score) * 0.2
        else:
            if analysis.confluence_score < 0:
                quality += abs(analysis.confluence_score) * 0.3
            else:
                quality -= analysis.confluence_score * 0.2
        
        # Overall alignment bonus
        quality += analysis.alignment_score * 0.2
        
        # Volatility adjustment
        if analysis.volatility_regime == "high":
            quality -= 0.1
        elif analysis.volatility_regime == "low":
            quality -= 0.05
        
        return max(0.0, min(1.0, quality))
    
    def should_filter_signal(
        self,
        analysis: MultiTFAnalysis,
        signal_direction: int,
        min_quality: float = 0.4
    ) -> Tuple[bool, str]:
        """
        Check if a signal should be filtered based on multi-TF context.
        
        Args:
            analysis: Multi-TF analysis
            signal_direction: 1 for long, -1 for short
            min_quality: Minimum quality threshold
            
        Returns:
            Tuple of (should_filter: bool, reason: str)
        """
        quality = self.get_entry_quality(analysis, signal_direction)
        
        if quality < min_quality:
            return True, f"Low entry quality ({quality:.2f} < {min_quality})"
        
        # Check for conflicting D1 trend
        if Timeframe.D1 in analysis.analyses:
            d1_trend = analysis.analyses[Timeframe.D1].ema_trend
            if (signal_direction == 1 and d1_trend == -1) or \
               (signal_direction == -1 and d1_trend == 1):
                if analysis.alignment_score < 0.5:
                    return True, "Counter-trend to D1 with poor alignment"
        
        return False, ""
