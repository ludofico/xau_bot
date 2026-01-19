
"""
DeepScalper Agent Wrapper.

Loads trained PPO model and provides interface for LiveTrader.

CRITICAL: Observation space MUST match env.py exactly:
- 5 normalized features × 30 window = 150
- 4 account state values = 4
- TOTAL = 154 dimensions

Features: returns, atr_pct, vol_norm, rsi_norm, momentum
"""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional
import pickle

from xauusd_strategy.utils.logger import get_logger

logger = get_logger(__name__)


class DeepScalperAgent:
    """
    DeepScalper Agent with ALIGNED observation space.
    
    Uses EXACTLY the same 5 normalized features as env.py:
    - returns: price change %
    - atr_pct: ATR as % of price
    - vol_norm: normalized volume
    - rsi_norm: RSI normalized [-1, 1]
    - momentum: 5-bar momentum
    """
    
    # Feature columns MUST match env.py
    NORM_COLS = ['returns', 'atr_pct', 'vol_norm', 'rsi_norm', 'momentum']
    
    def __init__(self, model_path: str = "models/rl_deepscalper/final_model.zip"):
        self.model_path = Path(model_path)
        self.vec_normalize_path = self.model_path.parent / "vec_normalize.pkl"
        self.model = None
        self.vec_normalize_stats = None
        self.window_size = 30  # Must match env training
        self.initial_balance = 250.0
        self.max_lot = 0.10
        self._load()
        
    def _load(self):
        """Load model and VecNormalize stats if available."""
        if self.model_path.exists():
            try:
                self.model = PPO.load(self.model_path)
                logger.info(f"✅ DeepScalper Agent loaded from {self.model_path}")
                
                # Load VecNormalize stats for proper observation normalization
                if self.vec_normalize_path.exists():
                    with open(self.vec_normalize_path, 'rb') as f:
                        self.vec_normalize_stats = pickle.load(f)
                    logger.info("✅ VecNormalize stats loaded")
            except Exception as e:
                logger.error(f"Failed to load RL Agent: {e}")
        else:
            logger.warning(f"RL Model not found at {self.model_path}")
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare normalized features EXACTLY matching env.py.
        
        This ensures the agent sees the same observation space as training.
        """
        df = df.copy()
        
        # 1. Returns (always normalized around 0)
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change().fillna(0)
        
        # 2. ATR percentage
        if 'atr_pct' not in df.columns:
            if 'atr_14' in df.columns:
                df['atr_pct'] = df['atr_14'] / df['close']
            else:
                df['atr_pct'] = (df['high'] - df['low']) / df['close']
        
        # 3. Volume normalized
        if 'vol_norm' not in df.columns:
            if 'volume' in df.columns and df['volume'].std() > 0:
                vol_mean = df['volume'].rolling(100, min_periods=1).mean()
                vol_std = df['volume'].rolling(100, min_periods=1).std().replace(0, 1)
                df['vol_norm'] = ((df['volume'] - vol_mean) / vol_std).clip(-3, 3)
            else:
                df['vol_norm'] = 0.0
        
        # 4. RSI normalized [-1, 1]
        if 'rsi_norm' not in df.columns:
            if 'rsi_14' in df.columns:
                df['rsi_norm'] = (df['rsi_14'] - 50) / 50
            elif 'rsi' in df.columns:
                df['rsi_norm'] = (df['rsi'] - 50) / 50
            else:
                df['rsi_norm'] = 0.0
        
        # 5. Momentum (5-bar)
        if 'momentum' not in df.columns:
            df['momentum'] = df['close'].pct_change(5).fillna(0).clip(-0.05, 0.05) * 10
        
        # Clip all features for stability
        for col in self.NORM_COLS:
            df[col] = df[col].fillna(0).clip(-10, 10)
        
        return df
    
    def _build_observation(self, df: pd.DataFrame, balance: float, position_size: float) -> np.ndarray:
        """
        Build observation vector EXACTLY matching env.py.
        
        Shape: (154,) = 5 features × 30 window + 4 state values
        """
        # Extract only the normalized columns
        window = df[self.NORM_COLS].iloc[-self.window_size:].values
        obs = np.clip(window.flatten(), -10, 10).astype(np.float32)
        
        # Account state (matching env.py exactly)
        equity = balance  # Simplified - no unrealized in inference
        unrealized = 0.0
        
        state = np.array([
            (balance - self.initial_balance) / self.initial_balance,
            position_size / self.max_lot,
            (equity - self.initial_balance) / self.initial_balance,
            np.clip(unrealized / 50.0, -1, 1)
        ], dtype=np.float32)
        
        return np.concatenate([obs, state])
            
    def predict(self, df_window: pd.DataFrame, balance: float, position_size: float) -> int:
        """
        Predict action for current market state.
        
        Actions:
        0: Hold
        1: Buy
        2: Sell
        3: Close
        
        Returns:
            int: Action to take (0-3)
        """
        if self.model is None:
            return 0  # Hold (Safe default)
            
        if len(df_window) < self.window_size:
            logger.warning("Insufficient data for RL Agent")
            return 0
        
        try:
            # Prepare features (aligns with env.py)
            df_prepared = self._prepare_features(df_window)
            
            # Build observation vector
            obs = self._build_observation(df_prepared, balance, position_size)
            
            # Verify shape
            expected_shape = (self.window_size * len(self.NORM_COLS)) + 4
            if obs.shape[0] != expected_shape:
                logger.error(f"Observation shape mismatch: {obs.shape[0]} != {expected_shape}")
                return 0
            
            # Predict action
            action, _states = self.model.predict(obs, deterministic=True)
            
            return int(action)
            
        except Exception as e:
            logger.error(f"RL Prediction error: {e}")
            return 0  # Safe fallback
    
    def get_confidence(self, df_window: pd.DataFrame, balance: float, position_size: float) -> float:
        """
        Get confidence score for the predicted action.
        
        Returns probability of the chosen action from the policy.
        """
        if self.model is None:
            return 0.0
            
        try:
            df_prepared = self._prepare_features(df_window)
            obs = self._build_observation(df_prepared, balance, position_size)
            
            # Get action probabilities
            import torch
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                dist = self.model.policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs.numpy()[0]
                return float(probs.max())
        except:
            return 0.5  # Neutral confidence
