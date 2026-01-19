"""
Pytest fixtures for XAUUSD Trading Strategy tests.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def mock_market_data():
    """Generate realistic mock OHLCV data for testing."""
    def _generate(n_bars: int = 200, trend: str = "neutral", volatility: float = 1.0):
        """
        Generate mock market data with controllable characteristics.
        
        Args:
            n_bars: Number of bars to generate
            trend: 'bullish', 'bearish', or 'neutral'
            volatility: Multiplier for price movement (1.0 = normal)
        """
        np.random.seed(42)  # Reproducible for tests
        
        times = pd.date_range(
            end=datetime.now(), 
            periods=n_bars, 
            freq='5min'
        )
        
        # Base price with trend component
        base_price = 2650.0
        trend_factor = {
            'bullish': 0.02,
            'bearish': -0.02,
            'neutral': 0.0
        }.get(trend, 0.0)
        
        prices = [base_price]
        for i in range(1, n_bars):
            # Random walk with trend bias
            change = np.random.normal(trend_factor, 0.5 * volatility)
            prices.append(prices[-1] + change)
        
        prices = np.array(prices)
        
        # Generate OHLC from prices
        data = {
            'open': prices + np.random.uniform(-1, 1, n_bars) * volatility,
            'close': prices,
            'high': prices + np.abs(np.random.normal(1, 0.5, n_bars)) * volatility,
            'low': prices - np.abs(np.random.normal(1, 0.5, n_bars)) * volatility,
            'volume': np.random.randint(100, 10000, n_bars)
        }
        
        df = pd.DataFrame(data, index=times)
        
        # Ensure OHLC consistency
        df['high'] = df[['open', 'close', 'high']].max(axis=1)
        df['low'] = df[['open', 'close', 'low']].min(axis=1)
        
        return df
    
    return _generate


@pytest.fixture
def bullish_data(mock_market_data):
    """Generate bullish trend data."""
    return mock_market_data(n_bars=200, trend='bullish')


@pytest.fixture
def bearish_data(mock_market_data):
    """Generate bearish trend data."""
    return mock_market_data(n_bars=200, trend='bearish')


@pytest.fixture
def test_settings():
    """Load test-safe settings."""
    from xauusd_strategy.config.settings import Settings
    
    config_path = Path(__file__).parent.parent / "config" / "aggressive.yaml"
    if config_path.exists():
        return Settings.from_yaml(str(config_path))
    
    # Fallback to defaults if no config
    return Settings()


@pytest.fixture
def london_strategy(test_settings):
    """Create London Breakout strategy instance."""
    from xauusd_strategy.strategy.london_breakout import LondonBreakoutStrategy
    return LondonBreakoutStrategy(settings=test_settings)


@pytest.fixture
def asian_scalp_strategy(test_settings):
    """Create Asian Scalping strategy instance."""
    from xauusd_strategy.strategy.asian_scalp import AsianScalpingStrategy
    return AsianScalpingStrategy(settings=test_settings)


@pytest.fixture
def sample_trades():
    """Sample trade results for Monte Carlo testing."""
    return [
        10.50, -5.20, 15.00, -3.50, 8.20,
        -7.00, 22.30, -4.10, 12.00, -8.50,
        18.00, -6.20, 9.50, -5.00, 14.20,
        -9.00, 11.00, -4.50, 16.80, -7.20
    ]
