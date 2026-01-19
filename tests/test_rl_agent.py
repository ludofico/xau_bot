"""
Tests for RL Agent feature alignment.

Verifies the 154-dimensional observation space is correctly implemented.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime


class TestDeepScalperAgent:
    """Tests for DeepScalper RL Agent."""
    
    def test_agent_initialization(self):
        """Test agent initializes without errors."""
        from xauusd_strategy.rl.agent import DeepScalperAgent
        
        agent = DeepScalperAgent()
        
        assert agent.window_size == 30
        assert agent.initial_balance == 250.0
        assert agent.max_lot == 0.10
        assert len(agent.NORM_COLS) == 5
    
    def test_norm_cols_definition(self):
        """Test normalized columns are correctly defined."""
        from xauusd_strategy.rl.agent import DeepScalperAgent
        
        expected_cols = ['returns', 'atr_pct', 'vol_norm', 'rsi_norm', 'momentum']
        
        assert DeepScalperAgent.NORM_COLS == expected_cols
    
    def test_observation_shape(self, mock_market_data):
        """Test observation has correct shape (154 dimensions)."""
        from xauusd_strategy.rl.agent import DeepScalperAgent
        
        agent = DeepScalperAgent()
        df = mock_market_data(n_bars=100)
        
        # Prepare features
        df_prep = agent._prepare_features(df)
        
        # Build observation
        obs = agent._build_observation(df_prep, balance=250.0, position_size=0.0)
        
        # Expected: 5 features × 30 window + 4 state = 154
        expected_shape = 5 * 30 + 4
        assert obs.shape[0] == expected_shape, \
            f"Observation shape {obs.shape[0]} != expected {expected_shape}"
    
    def test_feature_preparation(self, mock_market_data):
        """Test feature preparation adds all required columns."""
        from xauusd_strategy.rl.agent import DeepScalperAgent
        
        agent = DeepScalperAgent()
        df = mock_market_data(n_bars=100)
        
        df_prep = agent._prepare_features(df)
        
        for col in agent.NORM_COLS:
            assert col in df_prep.columns, f"Missing column: {col}"
    
    def test_feature_values_clipped(self, mock_market_data):
        """Test that feature values are clipped to [-10, 10]."""
        from xauusd_strategy.rl.agent import DeepScalperAgent
        
        agent = DeepScalperAgent()
        df = mock_market_data(n_bars=100)
        
        df_prep = agent._prepare_features(df)
        
        for col in agent.NORM_COLS:
            assert df_prep[col].max() <= 10, f"{col} exceeds 10"
            assert df_prep[col].min() >= -10, f"{col} below -10"
    
    def test_predict_returns_valid_action(self, mock_market_data):
        """Test predict returns action in valid range [0-3]."""
        from xauusd_strategy.rl.agent import DeepScalperAgent
        
        agent = DeepScalperAgent()
        df = mock_market_data(n_bars=100)
        
        # Predict action
        action = agent.predict(df, balance=250.0, position_size=0.0)
        
        # Action should be 0-3 (even without loaded model, should return 0)
        assert 0 <= action <= 3
    
    def test_predict_with_insufficient_data(self, mock_market_data):
        """Test predict returns Hold (0) with insufficient data."""
        from xauusd_strategy.rl.agent import DeepScalperAgent
        
        agent = DeepScalperAgent()
        df = mock_market_data(n_bars=20)  # Less than window_size
        
        action = agent.predict(df, balance=250.0, position_size=0.0)
        
        assert action == 0  # Should hold
    
    def test_state_normalization(self):
        """Test account state is properly normalized."""
        from xauusd_strategy.rl.agent import DeepScalperAgent
        
        agent = DeepScalperAgent()
        
        # Create minimal DataFrame with required columns
        df = pd.DataFrame({
            'returns': [0.0] * 30,
            'atr_pct': [0.0] * 30,
            'vol_norm': [0.0] * 30,
            'rsi_norm': [0.0] * 30,
            'momentum': [0.0] * 30
        })
        
        # Test with different balance scenarios
        obs1 = agent._build_observation(df, balance=250.0, position_size=0.0)
        obs2 = agent._build_observation(df, balance=500.0, position_size=0.05)
        
        # State values are last 4 elements
        state1 = obs1[-4:]
        state2 = obs2[-4:]
        
        # At initial balance, first state should be 0
        assert abs(state1[0]) < 0.01  # (250-250)/250 = 0
        
        # At 2x balance, first state should be 1.0
        assert abs(state2[0] - 1.0) < 0.01  # (500-250)/250 = 1.0


class TestRLEnvironmentAlignment:
    """Tests to verify agent and environment are aligned."""
    
    def test_observation_space_matches_env(self):
        """Verify agent observation matches environment expectation."""
        from xauusd_strategy.rl.env import XauUsdEnv
        from xauusd_strategy.rl.agent import DeepScalperAgent
        
        # Create minimal data for env
        df = pd.DataFrame({
            'open': [2650.0] * 100,
            'high': [2652.0] * 100,
            'low': [2648.0] * 100,
            'close': [2650.0] * 100,
            'volume': [1000] * 100
        }, index=pd.date_range('2024-01-01', periods=100, freq='5min'))
        
        env = XauUsdEnv(df, window_size=30, mode="discrete")
        agent = DeepScalperAgent()
        
        # Both should expect 154 dimensions
        env_obs_shape = env.observation_space.shape[0]
        agent_obs_shape = 5 * agent.window_size + 4
        
        assert env_obs_shape == agent_obs_shape, \
            f"Env expects {env_obs_shape} but agent produces {agent_obs_shape}"
