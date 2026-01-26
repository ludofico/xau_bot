"""
Market Regime Detector for XAUUSD Trading.

Classifies market conditions into regimes to enable strategy selection:
- TRENDING_UP: Strong bullish trend (use breakout/momentum strategies)
- TRENDING_DOWN: Strong bearish trend (use breakout/momentum strategies)  
- RANGING: Low volatility consolidation (use mean-reversion/scalping)
- HIGH_VOLATILITY: Unstable conditions (reduce size or avoid trading)

Features:
- ADX for trend strength
- ATR percentile for volatility regime
- Price structure analysis (higher highs/lows)
- Multi-timeframe context (M5 micro, H1 macro)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Dict
import pandas as pd
import numpy as np

from xauusd_strategy.utils.logger import get_logger

logger = get_logger(__name__)


class MarketRegime(Enum):
    """Market regime classification."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    UNKNOWN = "unknown"


@dataclass
class RegimeAnalysis:
    """Complete regime analysis result."""
    regime: MarketRegime
    confidence: float  # 0.0 to 1.0
    adx: float
    atr_percentile: float
    trend_strength: float  # -1.0 (bearish) to 1.0 (bullish)
    volatility_state: str  # "low", "normal", "high", "extreme"
    
    # Multi-timeframe context
    micro_regime: Optional[MarketRegime] = None
    macro_regime: Optional[MarketRegime] = None
    alignment_score: float = 0.0  # How aligned micro/macro are
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "regime": self.regime.value,
            "confidence": round(self.confidence, 3),
            "adx": round(self.adx, 2),
            "atr_percentile": round(self.atr_percentile, 2),
            "trend_strength": round(self.trend_strength, 3),
            "volatility_state": self.volatility_state,
            "micro_regime": self.micro_regime.value if self.micro_regime else None,
            "macro_regime": self.macro_regime.value if self.macro_regime else None,
            "alignment_score": round(self.alignment_score, 3)
        }


class RegimeDetector:
    """
    Market Regime Detector using technical indicators.
    
    Classification Logic:
    1. ADX > threshold → Trending (direction from EMA/price structure)
    2. ADX < threshold + ATR low → Ranging
    3. ATR > 90th percentile → High Volatility
    
    Parameters are tuned for XAUUSD M5/H1 timeframes.
    """
    
    def __init__(
        self,
        adx_period: int = 14,
        adx_trend_threshold: float = 25.0,
        adx_strong_threshold: float = 40.0,
        atr_period: int = 14,
        atr_lookback: int = 100,
        ema_fast: int = 21,
        ema_slow: int = 55,
        structure_lookback: int = 20
    ):
        """
        Initialize Regime Detector.
        
        Args:
            adx_period: Period for ADX calculation
            adx_trend_threshold: ADX above this = trending market
            adx_strong_threshold: ADX above this = strong trend
            atr_period: Period for ATR calculation
            atr_lookback: Bars to use for ATR percentile ranking
            ema_fast: Fast EMA for trend direction
            ema_slow: Slow EMA for trend confirmation
            structure_lookback: Bars to analyze for price structure
        """
        self.adx_period = adx_period
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_strong_threshold = adx_strong_threshold
        self.atr_period = atr_period
        self.atr_lookback = atr_lookback
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.structure_lookback = structure_lookback
    
    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add required indicators to dataframe.
        
        Adds: ADX, +DI, -DI, ATR, EMA_fast, EMA_slow, trend direction.
        """
        df = df.copy()
        
        # True Range
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        
        # ATR
        df['atr'] = df['tr'].ewm(span=self.atr_period, adjust=False).mean()
        
        # Directional Movement
        df['up_move'] = df['high'] - df['high'].shift(1)
        df['down_move'] = df['low'].shift(1) - df['low']
        
        df['plus_dm'] = np.where(
            (df['up_move'] > df['down_move']) & (df['up_move'] > 0),
            df['up_move'], 0
        )
        df['minus_dm'] = np.where(
            (df['down_move'] > df['up_move']) & (df['down_move'] > 0),
            df['down_move'], 0
        )
        
        # Smoothed DI
        df['plus_di'] = 100 * (
            df['plus_dm'].ewm(span=self.adx_period, adjust=False).mean() /
            df['atr']
        )
        df['minus_di'] = 100 * (
            df['minus_dm'].ewm(span=self.adx_period, adjust=False).mean() /
            df['atr']
        )
        
        # DX and ADX
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'] + 1e-8)
        df['adx'] = df['dx'].ewm(span=self.adx_period, adjust=False).mean()
        
        # EMAs for trend direction
        df['ema_fast'] = df['close'].ewm(span=self.ema_fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.ema_slow, adjust=False).mean()
        
        # Trend direction: +1 bullish, -1 bearish, 0 neutral
        df['ema_trend'] = np.where(
            df['ema_fast'] > df['ema_slow'], 1,
            np.where(df['ema_fast'] < df['ema_slow'], -1, 0)
        )
        
        # ATR percentile ranking
        df['atr_percentile'] = df['atr'].rolling(self.atr_lookback).apply(
            lambda x: (x.iloc[-1] > x).sum() / len(x) * 100 if len(x) > 0 else 50,
            raw=False
        )
        
        return df
    
    def _analyze_price_structure(self, df: pd.DataFrame) -> float:
        """
        Analyze price structure for trend confirmation.
        
        Returns:
            Float from -1.0 (bearish structure) to 1.0 (bullish structure)
        """
        if len(df) < self.structure_lookback:
            return 0.0
        
        window = df.tail(self.structure_lookback)
        
        # Find swing highs and lows
        highs = window['high'].values
        lows = window['low'].values
        
        # Simple structure: compare first half vs second half
        mid = len(window) // 2
        
        first_half_high = highs[:mid].max()
        second_half_high = highs[mid:].max()
        first_half_low = lows[:mid].min()
        second_half_low = lows[mid:].min()
        
        # Higher highs and higher lows = bullish
        # Lower highs and lower lows = bearish
        hh = 1 if second_half_high > first_half_high else -1
        hl = 1 if second_half_low > first_half_low else -1
        
        return (hh + hl) / 2.0
    
    def _classify_volatility(self, atr_percentile: float) -> str:
        """Classify volatility state from ATR percentile."""
        if atr_percentile < 25:
            return "low"
        elif atr_percentile < 60:
            return "normal"
        elif atr_percentile < 90:
            return "high"
        else:
            return "extreme"
    
    def detect(self, df: pd.DataFrame) -> RegimeAnalysis:
        """
        Detect current market regime.
        
        Args:
            df: OHLC DataFrame with at least 100 bars
            
        Returns:
            RegimeAnalysis with regime classification and metrics
        """
        if len(df) < max(self.atr_lookback, 60):
            logger.warning("Insufficient data for regime detection")
            return RegimeAnalysis(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                adx=0.0,
                atr_percentile=50.0,
                trend_strength=0.0,
                volatility_state="unknown"
            )
        
        # Add indicators
        df_prep = self.prepare_indicators(df)
        
        # Get latest values
        current = df_prep.iloc[-1]
        adx = current['adx']
        atr_percentile = current['atr_percentile']
        ema_trend = current['ema_trend']
        plus_di = current['plus_di']
        minus_di = current['minus_di']
        
        # Analyze price structure
        structure = self._analyze_price_structure(df_prep)
        
        # Volatility classification
        vol_state = self._classify_volatility(atr_percentile)
        
        # Regime classification logic
        regime = MarketRegime.UNKNOWN
        confidence = 0.0
        
        # Check for extreme volatility first
        if vol_state == "extreme":
            regime = MarketRegime.HIGH_VOLATILITY
            confidence = min(1.0, atr_percentile / 100)
        
        # Trending market (ADX > threshold)
        elif adx > self.adx_trend_threshold:
            # Determine direction
            if plus_di > minus_di and ema_trend >= 0:
                regime = MarketRegime.TRENDING_UP
            elif minus_di > plus_di and ema_trend <= 0:
                regime = MarketRegime.TRENDING_DOWN
            else:
                # Mixed signals - use structure
                regime = MarketRegime.TRENDING_UP if structure > 0 else MarketRegime.TRENDING_DOWN
            
            # Confidence based on ADX strength and alignment
            adx_conf = min(1.0, adx / self.adx_strong_threshold)
            alignment_conf = abs(structure)
            confidence = (adx_conf * 0.7) + (alignment_conf * 0.3)
        
        # Ranging market (low ADX, low volatility)
        else:
            regime = MarketRegime.RANGING
            # Confidence: lower ADX = more confident it's ranging
            confidence = 1.0 - (adx / self.adx_trend_threshold)
        
        # High volatility override (not extreme, but elevated)
        if vol_state == "high" and regime == MarketRegime.RANGING:
            regime = MarketRegime.HIGH_VOLATILITY
            confidence = atr_percentile / 100
        
        # Trend strength: combine DI difference with structure
        trend_strength = 0.0
        if regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            di_diff = (plus_di - minus_di) / (plus_di + minus_di + 1e-8)
            trend_strength = (di_diff + structure) / 2
        
        return RegimeAnalysis(
            regime=regime,
            confidence=round(confidence, 3),
            adx=round(adx, 2),
            atr_percentile=round(atr_percentile, 2),
            trend_strength=round(trend_strength, 3),
            volatility_state=vol_state
        )
    
    def detect_multi_timeframe(
        self,
        df_micro: pd.DataFrame,
        df_macro: pd.DataFrame
    ) -> RegimeAnalysis:
        """
        Detect regime with multi-timeframe context.
        
        Args:
            df_micro: Lower timeframe data (e.g., M5)
            df_macro: Higher timeframe data (e.g., H1)
            
        Returns:
            RegimeAnalysis with micro/macro context
        """
        micro_analysis = self.detect(df_micro)
        macro_analysis = self.detect(df_macro)
        
        # Calculate alignment score
        alignment = 0.0
        if micro_analysis.regime == macro_analysis.regime:
            alignment = 1.0
        elif (micro_analysis.regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN] and
              macro_analysis.regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]):
            # Both trending but different directions = negative alignment
            if micro_analysis.regime != macro_analysis.regime:
                alignment = -0.5
        elif MarketRegime.RANGING in [micro_analysis.regime, macro_analysis.regime]:
            # One ranging, one trending = partial alignment
            alignment = 0.3
        
        # Final regime: prefer macro for direction, micro for timing
        final_regime = macro_analysis.regime
        if macro_analysis.regime == MarketRegime.RANGING and micro_analysis.regime != MarketRegime.RANGING:
            # Macro ranging but micro showing trend = early trend detection
            final_regime = micro_analysis.regime
            alignment = 0.4  # Lower confidence
        
        # Adjusted confidence based on alignment
        final_confidence = (
            (macro_analysis.confidence * 0.6) +
            (micro_analysis.confidence * 0.4)
        ) * (0.5 + alignment * 0.5)
        
        return RegimeAnalysis(
            regime=final_regime,
            confidence=round(final_confidence, 3),
            adx=macro_analysis.adx,  # Use macro ADX
            atr_percentile=micro_analysis.atr_percentile,  # Use micro ATR for sizing
            trend_strength=macro_analysis.trend_strength,
            volatility_state=micro_analysis.volatility_state,
            micro_regime=micro_analysis.regime,
            macro_regime=macro_analysis.regime,
            alignment_score=round(alignment, 3)
        )
    
    def get_strategy_recommendation(self, analysis: RegimeAnalysis) -> Dict:
        """
        Get strategy recommendations based on regime.
        
        Returns:
            Dict with recommended strategies and position sizing multiplier
        """
        recommendations = {
            "preferred_strategies": [],
            "avoid_strategies": [],
            "size_multiplier": 1.0,
            "notes": []
        }
        
        if analysis.regime == MarketRegime.TRENDING_UP:
            recommendations["preferred_strategies"] = ["london_breakout", "momentum"]
            recommendations["avoid_strategies"] = ["mean_reversion"]
            recommendations["size_multiplier"] = 1.0 + (analysis.confidence * 0.2)
            recommendations["notes"].append("Trend following favored")
            
        elif analysis.regime == MarketRegime.TRENDING_DOWN:
            recommendations["preferred_strategies"] = ["london_breakout", "momentum"]
            recommendations["avoid_strategies"] = ["mean_reversion"]
            recommendations["size_multiplier"] = 1.0 + (analysis.confidence * 0.2)
            recommendations["notes"].append("Trend following favored (short bias)")
            
        elif analysis.regime == MarketRegime.RANGING:
            recommendations["preferred_strategies"] = ["asian_scalp", "mean_reversion"]
            recommendations["avoid_strategies"] = ["breakout", "momentum"]
            recommendations["size_multiplier"] = 0.8
            recommendations["notes"].append("Mean reversion strategies preferred")
            
        elif analysis.regime == MarketRegime.HIGH_VOLATILITY:
            recommendations["preferred_strategies"] = []
            recommendations["avoid_strategies"] = ["all"]
            recommendations["size_multiplier"] = 0.5 if analysis.atr_percentile < 95 else 0.0
            recommendations["notes"].append("High volatility - reduce exposure or avoid")
        
        # Adjust for alignment
        if analysis.alignment_score < 0:
            recommendations["size_multiplier"] *= 0.7
            recommendations["notes"].append("Multi-TF misalignment - reduce size")
        elif analysis.alignment_score > 0.8:
            recommendations["size_multiplier"] *= 1.1
            recommendations["notes"].append("Strong multi-TF alignment")
        
        return recommendations
