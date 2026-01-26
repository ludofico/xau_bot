"""
Signal Aggregator & Execution Engine.

Combines signals from multiple strategies, applies filters,
integrates risk management, and generates final execution commands.

Features:
- Consolidates signals from London Breakout, Asian Scalp, etc.
- Weighs signals by market regime and multi-timeframe alignment
- Integrates with RiskManager for position sizing and safety checks
- Generates structured JSON output for execution audit
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
import numpy as np
import pandas as pd

from xauusd_strategy.strategy.london_breakout import TradeSignal, SignalType
from xauusd_strategy.strategy.regime_detector import RegimeAnalysis, MarketRegime
from xauusd_strategy.strategy.multi_tf_aggregator import MultiTFAnalysis, Timeframe
from xauusd_strategy.risk.risk_manager import RiskManager, RiskInfo
from xauusd_strategy.ai.news_calendar import NewsImpact
from xauusd_strategy.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AggregatedSignal:
    """Final aggregated signal ready for execution."""
    timestamp: str  # ISO8601
    signal_id: str
    strategy_source: str
    direction: str  # "BUY" or "SELL"
    
    # Trade params
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float  # In lots
    
    # Metrics
    confidence_score: float  # 0-100
    risk_reward: float
    expected_value: float
    
    # Context
    market_regime: str
    multi_tf_alignment: str
    news_status: str
    
    # Risk info
    risk_info: Dict
    
    # Rationale
    rationale: List[str]
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        data = asdict(self)
        return json.dumps(data, indent=2)


class SignalAggregator:
    """
    Central signal aggregation and processing engine.
    """
    
    def __init__(
        self,
        risk_manager: RiskManager,
        min_confidence: float = 60.0,
        regime_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize Aggregator.
        
        Args:
            risk_manager: Initialized RiskManager instance
            min_confidence: Minimum confidence score to execute
            regime_weights: Weight of strategies in different regimes
        """
        self.risk_manager = risk_manager
        self.min_confidence = min_confidence
        self.regime_weights = regime_weights or {
            "trending_up": 1.2,
            "trending_down": 1.2,
            "ranging": 0.8,
            "high_volatility": 0.5
        }
    
    def process_signal(
        self,
        signal: TradeSignal,
        regime_analysis: RegimeAnalysis,
        multi_tf_analysis: MultiTFAnalysis,
        news_impact: NewsImpact,
        current_equity: float,
        current_positions: int = 0
    ) -> Optional[AggregatedSignal]:
        """
        Process a raw strategy signal into an actionable command.
        
        Args:
            signal: Raw trade signal from strategy
            regime_analysis: Current market regime
            multi_tf_analysis: Multi-timeframe context
            news_impact: Current news impact
            current_equity: Current account equity
            current_positions: Open positions count
            
        Returns:
            AggregatedSignal if approved, None if filtered
        """
        # 1. Update risk manager equity
        self.risk_manager.update_equity(current_equity)
        
        # 2. Check basic validity
        if not signal or signal.signal_type == SignalType.NONE:
            return None
        
        rationale = [f"Source: {signal.source or 'Unknown Strategy'}"]
        confidence = 50.0  # Base confidence
        
        # 3. Regime Alignment Check
        regime_score = self._check_regime_alignment(signal, regime_analysis)
        confidence += regime_score
        rationale.append(f"Regime: {regime_analysis.regime.value} (Score: {regime_score:+})")
        
        # 4. Multi-TF Alignment Check
        mtf_score = self._check_multi_tf(signal, multi_tf_analysis)
        confidence += mtf_score
        rationale.append(f"Multi-TF: {multi_tf_analysis.alignment_score:.2f} (Score: {mtf_score:+})")
        
        # 5. News Check
        if news_impact.halt_trading:
            logger.warning(f"Signal rejected due to news halt: {news_impact.notes}")
            return None
        
        if news_impact.level in ["High", "Critical"]:
            confidence -= 20
            rationale.append(f"News Warning: {news_impact.level}")
        
        # 6. ML Confirmation (if present in signal)
        if hasattr(signal, 'probability') and signal.probability:
            ml_boost = (signal.probability - 0.5) * 100
            confidence += ml_boost
            rationale.append(f"ML Model: {signal.probability:.2f} (Score: {ml_boost:+.1f})")
        
        # Cap confidence
        confidence = max(0.0, min(100.0, confidence))
        
        # 7. Filter low confidence
        if confidence < self.min_confidence:
            logger.info(f"Signal rejected: Low confidence {confidence:.1f} < {self.min_confidence}")
            return None
        
        # 8. Risk Management
        sl_dist = abs(signal.entry_price - signal.stop_loss)
        
        # Get multipliers
        regime_mult = 1.0
        if regime_analysis.regime == MarketRegime.HIGH_VOLATILITY:
            regime_mult = 0.5
        elif regime_analysis.regime == MarketRegime.RANGING:
            regime_mult = 0.8
            
        vol_mult = 1.0
        if multi_tf_analysis.volatility_regime == "high":
            vol_mult = 0.7
            
        risk_info = self.risk_manager.assess_risk(
            sl_distance=sl_dist,
            current_positions=current_positions,
            news_multiplier=news_impact.size_multiplier,
            regime_multiplier=regime_mult,
            volatility_multiplier=vol_mult
        )
        
        if not risk_info.can_trade:
            logger.warning(f"Signal rejected by Risk Manager: {risk_info.notes}")
            return None
            
        rationale.extend(risk_info.notes)
        
        # 9. Construct Final Signal
        direction = "BUY" if signal.signal_type == SignalType.LONG else "SELL"
        
        # Generate ID
        ts_str = datetime.now().isoformat()
        sig_id = f"{direction}_{int(datetime.now().timestamp())}"
        
        return AggregatedSignal(
            timestamp=ts_str,
            signal_id=sig_id,
            strategy_source=getattr(signal, 'source', 'unknown'),
            direction=direction,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            position_size=risk_info.position_size,
            confidence_score=round(confidence, 1),
            risk_reward=round(signal.risk_reward, 2) if hasattr(signal, 'risk_reward') else 0.0,
            expected_value=0.0,  # Could calculate based on winrate * reward
            market_regime=regime_analysis.regime.value,
            multi_tf_alignment=f"Score: {multi_tf_analysis.alignment_score:.2f}",
            news_status=news_impact.level,
            risk_info=risk_info.to_dict(),
            rationale=rationale
        )
    
    def _check_regime_alignment(self, signal: TradeSignal, regime: RegimeAnalysis) -> float:
        """Calculate confidence adjustment based on regime."""
        score = 0.0
        is_long = signal.signal_type == SignalType.LONG
        
        if regime.regime == MarketRegime.TRENDING_UP:
            if is_long: score += 20
            else: score -= 30  # Counter-trend penalty
            
        elif regime.regime == MarketRegime.TRENDING_DOWN:
            if not is_long: score += 20
            else: score -= 30
            
        elif regime.regime == MarketRegime.RANGING:
            # Strategies should handle ranging, but generally lower confidence
            score -= 10
            
        elif regime.regime == MarketRegime.HIGH_VOLATILITY:
            score -= 40
            
        return score
    
    def _check_multi_tf(self, signal: TradeSignal, mtf: MultiTFAnalysis) -> float:
        """Calculate confidence adjustment based on Multi-TF."""
        score = 0.0
        is_long = signal.signal_type == SignalType.LONG
        
        # Alignment score (0-1) -> map to -20 to +20
        # If alignment is high, valid; if low, chop
        score += (mtf.alignment_score - 0.5) * 40
        
        # Confluence direction check
        if is_long and mtf.confluence_score > 0.2:
            score += 15
        elif not is_long and mtf.confluence_score < -0.2:
            score += 15
        elif (is_long and mtf.confluence_score < -0.2) or (not is_long and mtf.confluence_score > 0.2):
            score -= 25  # Fighting the aggregate trend
            
        return score
