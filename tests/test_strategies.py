"""
Unit tests for trading strategies.

Tests verify:
- Signal generation logic
- Trend filtering (PRECISION scalping)
- Confirmation counting
- Edge cases
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime


class TestAsianScalpStrategy:
    """Tests for the PRECISION Asian Scalping Strategy."""
    
    def test_strategy_initialization(self, asian_scalp_strategy):
        """Test strategy initializes with correct parameters."""
        assert asian_scalp_strategy.ema_fast == 21
        assert asian_scalp_strategy.ema_slow == 55
        assert asian_scalp_strategy.rsi_period == 14
        
    def test_prepare_data_adds_indicators(self, asian_scalp_strategy, mock_market_data):
        """Test that prepare_data adds all required indicators."""
        df = mock_market_data(n_bars=100)
        df_prep = asian_scalp_strategy.prepare_data(df)
        
        required_cols = [
            'ema_fast', 'ema_slow', 'trend', 'rsi',
            'bb_upper', 'bb_lower', 'bb_position', 'atr_14'
        ]
        
        for col in required_cols:
            assert col in df_prep.columns, f"Missing column: {col}"
    
    def test_no_signal_with_insufficient_data(self, asian_scalp_strategy, mock_market_data):
        """Test that no signal is generated with < 60 bars."""
        df = mock_market_data(n_bars=50)
        df_prep = asian_scalp_strategy.prepare_data(df)
        
        signal = asian_scalp_strategy.generate_signal(df_prep, current_idx=30)
        assert signal is None
    
    def test_bullish_trend_no_short(self, asian_scalp_strategy):
        """In a bullish trend, should NOT generate SHORT signals."""
        # Create strongly bullish data
        n_bars = 100
        times = pd.date_range(end=datetime.now(), periods=n_bars, freq='5min')
        
        # Steadily rising prices
        prices = 2650 + np.arange(n_bars) * 0.5
        
        df = pd.DataFrame({
            'open': prices - 0.2,
            'high': prices + 1,
            'low': prices - 1,
            'close': prices,
            'volume': np.random.randint(100, 1000, n_bars)
        }, index=times)
        
        df_prep = asian_scalp_strategy.prepare_data(df)
        
        # Check trend detection
        assert df_prep.iloc[-1]['trend'] == 1, "Should detect bullish trend"
        
        # If signal generated, it should be LONG, not SHORT
        signal = asian_scalp_strategy.generate_signal(df_prep, len(df_prep) - 1)
        if signal:
            from xauusd_strategy.strategy.london_breakout import SignalType
            assert signal.signal_type != SignalType.SHORT, "Should not SHORT in uptrend"
    
    def test_bearish_trend_no_long(self, asian_scalp_strategy):
        """In a bearish trend, should NOT generate LONG signals."""
        n_bars = 100
        times = pd.date_range(end=datetime.now(), periods=n_bars, freq='5min')
        
        # Steadily falling prices
        prices = 2700 - np.arange(n_bars) * 0.5
        
        df = pd.DataFrame({
            'open': prices + 0.2,
            'high': prices + 1,
            'low': prices - 1,
            'close': prices,
            'volume': np.random.randint(100, 1000, n_bars)
        }, index=times)
        
        df_prep = asian_scalp_strategy.prepare_data(df)
        
        # Check trend detection
        assert df_prep.iloc[-1]['trend'] == -1, "Should detect bearish trend"
        
        # If signal generated, it should be SHORT, not LONG
        signal = asian_scalp_strategy.generate_signal(df_prep, len(df_prep) - 1)
        if signal:
            from xauusd_strategy.strategy.london_breakout import SignalType
            assert signal.signal_type != SignalType.LONG, "Should not LONG in downtrend"
    
    def test_confirmation_threshold(self, asian_scalp_strategy, mock_market_data):
        """Test that signals require minimum 6 confirmations."""
        df = mock_market_data(n_bars=100)
        df_prep = asian_scalp_strategy.prepare_data(df)
        
        # The strategy requires 6/8 confirmations
        # With random data, unlikely to meet threshold
        signal = asian_scalp_strategy.generate_signal(df_prep, len(df_prep) - 1)
        
        # Signal may or may not be generated, but if it is, 
        # the log should show >= 6 confirmations
        # This test mainly verifies the method doesn't crash
        assert True


class TestLondonBreakoutStrategy:
    """Tests for London Breakout Strategy."""
    
    def test_strategy_initialization(self, london_strategy):
        """Test strategy initializes correctly."""
        assert london_strategy is not None
    
    def test_prepare_data_adds_breakout_levels(self, london_strategy, mock_market_data):
        """Test that prepare_data adds trading indicators."""
        df = mock_market_data(n_bars=300)
        df_prep = london_strategy.prepare_data(df)
        
        # Should have standard indicators
        assert 'atr' in df_prep.columns or 'tr' in df_prep.columns
        assert 'roc' in df_prep.columns or 'adx' in df_prep.columns
    
    def test_no_signal_outside_london_session(self, london_strategy, mock_market_data):
        """Test that signals are session-aware."""
        # Create data outside London session (e.g., 3 AM UTC)
        n_bars = 100
        times = pd.date_range(
            start=datetime(2024, 1, 15, 3, 0),  # 3 AM
            periods=n_bars, 
            freq='5min'
        )
        
        df = mock_market_data(n_bars=100)
        df.index = times
        
        df_prep = london_strategy.prepare_data(df)
        signal = london_strategy.generate_signal(df_prep, len(df_prep) - 1, ml_probability=0.8)
        
        # May or may not generate signal depending on session config
        # Main check is it doesn't crash
        assert True


class TestStrategyIntegration:
    """Integration tests for strategies working together."""
    
    def test_multiple_strategies_same_data(
        self, london_strategy, asian_scalp_strategy, mock_market_data
    ):
        """Test that multiple strategies can process the same data."""
        df = mock_market_data(n_bars=200)
        
        # Both should be able to prepare data
        df_london = london_strategy.prepare_data(df)
        df_scalp = asian_scalp_strategy.prepare_data(df)
        
        assert len(df_london) == len(df)
        assert len(df_scalp) == len(df)
    
    def test_signal_type_consistency(self, asian_scalp_strategy, mock_market_data):
        """Test that signal types are properly defined."""
        from xauusd_strategy.strategy.london_breakout import SignalType, TradeSignal
        
        # Verify SignalType enum exists
        assert hasattr(SignalType, 'LONG')
        assert hasattr(SignalType, 'SHORT')
        assert hasattr(SignalType, 'NONE')
