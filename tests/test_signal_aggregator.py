"""
Tests for Signal Aggregator.

Verifies:
- Signal validation and confidence scoring
- Regime and Multi-TF adjustments
- Risk Manager integration
- JSON output generation
"""

import pytest
import json
from datetime import datetime
from unittest.mock import MagicMock

from xauusd_strategy.strategy.signal_aggregator import (
    SignalAggregator,
    AggregatedSignal
)
from xauusd_strategy.strategy.london_breakout import TradeSignal, SignalType
from xauusd_strategy.strategy.regime_detector import RegimeAnalysis, MarketRegime
from xauusd_strategy.strategy.multi_tf_aggregator import MultiTFAnalysis, TrendDirection, TimeframeAnalysis, Timeframe
from xauusd_strategy.risk.risk_manager import RiskManager, RiskInfo
from xauusd_strategy.ai.news_calendar import NewsImpact

@pytest.fixture
def mock_risk_manager():
    """Create mock risk manager."""
    rm = MagicMock(spec=RiskManager)
    
    # Setup default response
    info = RiskInfo(
        can_trade=True,
        position_size=1.0,
        max_loss=100.0,
        risk_percent=1.0,
        size_multiplier=1.0,
        circuit_status="active",
        kelly_size=1.0,
        news_multiplier=1.0,
        regime_multiplier=1.0,
        volatility_multiplier=1.0,
        notes=["Risk Check OK"]
    )
    
    rm.assess_risk.return_value = info
    return rm

@pytest.fixture
def aggregator(mock_risk_manager):
    """Create SignalAggregator instance."""
    return SignalAggregator(
        risk_manager=mock_risk_manager,
        min_confidence=50.0
    )

@pytest.fixture
def sample_signal():
    """Create sample trade signal."""
    return TradeSignal(
        signal_type=SignalType.LONG,
        entry_price=2650.0,
        stop_loss=2645.0,
        take_profit=2660.0,
        timestamp=datetime.now(),
        source="TestStrategy",
        probability=0.7,
        atr_value=5.0,
        roc_value=0.2,
        asian_high=2655.0,
        asian_low=2640.0
    )

@pytest.fixture
def bullish_regime():
    """Create bullish regime analysis."""
    return RegimeAnalysis(
        regime=MarketRegime.TRENDING_UP,
        confidence=0.8,
        adx=30.0,
        atr_percentile=50.0,
        trend_strength=0.5,
        volatility_state="normal"
    )

@pytest.fixture
def bullish_mtf():
    """Create bullish multi-tf analysis."""
    return MultiTFAnalysis(
        timestamp=datetime.now(),
        analyses={},
        confluence_score=0.8,
        alignment_score=0.9,
        dominant_trend=TrendDirection.UP,
        recommended_bias="long",
        volatility_regime="normal",
        notes=[]
    )

@pytest.fixture
def neutral_news():
    """Create neutral news impact."""
    return NewsImpact(
        level="Low",
        sentiment="Neutral",
        halt_trading=False,
        size_multiplier=1.0
    )

class TestSignalAggregator:
    """Tests for SignalAggregator class."""
    
    def test_process_valid_signal(
        self,
        aggregator,
        sample_signal,
        bullish_regime,
        bullish_mtf,
        neutral_news
    ):
        """Test processing a valid, high-confidence signal."""
        
        result = aggregator.process_signal(
            signal=sample_signal,
            regime_analysis=bullish_regime,
            multi_tf_analysis=bullish_mtf,
            news_impact=neutral_news,
            current_equity=10000.0
        )
        
        assert result is not None
        assert isinstance(result, AggregatedSignal)
        assert result.direction == "BUY"
        assert result.confidence_score > 60.0 # Base 50 + Regime 20 + MTF 15 + ML 20 = ~100
        
        # Verify RiskManager was called
        aggregator.risk_manager.assess_risk.assert_called_once()
        
    def test_filter_low_confidence(
        self,
        aggregator,
        sample_signal,
        bullish_regime,
        bullish_mtf,
        neutral_news
    ):
        """Test that low confidence signals are filtered."""
        # Make everything bad
        bad_regime = RegimeAnalysis(
            regime=MarketRegime.TRENDING_DOWN, # Against signal
            confidence=0.8,
            adx=30,
            atr_percentile=50,
            trend_strength=-0.5,
            volatility_state="normal"
        )
        
        bad_mtf = MultiTFAnalysis(
            timestamp=datetime.now(),
            analyses={},
            confluence_score=-0.8, # Against signal
            alignment_score=0.9,
            dominant_trend=TrendDirection.DOWN,
            recommended_bias="short",
            volatility_regime="normal",
            notes=[]
        )
        
        # Signal has ML prob 0.7 which adds +20, but regime -30 and MTF -25
        # 50 + 20 - 30 - 25 = 15 confidence
        
        result = aggregator.process_signal(
            signal=sample_signal,
            regime_analysis=bad_regime,
            multi_tf_analysis=bad_mtf,
            news_impact=neutral_news,
            current_equity=10000.0
        )
        
        assert result is None
        
    def test_news_halt(
        self,
        aggregator,
        sample_signal,
        bullish_regime,
        bullish_mtf
    ):
        """Test signal rejection during news halt."""
        halt_news = NewsImpact(
            level="Critical",
            sentiment="Neutral",
            halt_trading=True,
            size_multiplier=0.0,
            notes=["NFP HALT"]
        )
        
        result = aggregator.process_signal(
            signal=sample_signal,
            regime_analysis=bullish_regime,
            multi_tf_analysis=bullish_mtf,
            news_impact=halt_news,
            current_equity=10000.0
        )
        
        assert result is None
        
    def test_risk_manager_rejection(
        self,
        aggregator,
        sample_signal,
        bullish_regime,
        bullish_mtf,
        neutral_news
    ):
        """Test rejection when risk manager says no."""
        # Setup risk manager to reject
        info = RiskInfo(
            can_trade=False, # REJECTED
            position_size=0,
            max_loss=0,
            risk_percent=0,
            size_multiplier=0,
            circuit_status="triggered",
            kelly_size=0,
            news_multiplier=1.0,
            regime_multiplier=1.0,
            volatility_multiplier=1.0,
            notes=["Circuit Breaker"]
        )
        aggregator.risk_manager.assess_risk.return_value = info
        
        result = aggregator.process_signal(
            signal=sample_signal,
            regime_analysis=bullish_regime,
            multi_tf_analysis=bullish_mtf,
            news_impact=neutral_news,
            current_equity=10000.0
        )
        
        assert result is None
        
    def test_json_output(
        self,
        aggregator,
        sample_signal,
        bullish_regime,
        bullish_mtf,
        neutral_news
    ):
        """Test valid JSON generation."""
        result = aggregator.process_signal(
            signal=sample_signal,
            regime_analysis=bullish_regime,
            multi_tf_analysis=bullish_mtf,
            news_impact=neutral_news,
            current_equity=10000.0
        )
        
        json_str = result.to_json()
        data = json.loads(json_str)
        
        assert data['direction'] == "BUY"
        assert data['entry_price'] == 2650.0
        assert "risk_info" in data
        assert "rationale" in data
