
"""
Custom Gym Environment for XAUUSD Scalping (DeepScalper).

Features:
- Realistic Spread & Commission simulation
- Continuous Action Space (Position Sizing) or Discrete (Buy/Sell/Hold)
- Reward Function: Sharpe Ratio driven (Profit - Volatility Penalty)
- NORMALIZED OBSERVATIONS for stable training
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict

class XauUsdEnv(gym.Env):
    """
    XAUUSD Trading Environment with NORMALIZED observations.
    
    Observation Space (Normalized):
    - Returns instead of raw prices
    - Normalized indicators
    - Scaled account state
    
    Action Space (Discrete):
    0: Hold
    1: Buy (Long)
    2: Sell (Short)
    3: Close All
    
    Reward:
    - Realized PnL (scaled)
    - Unrealized PnL change (scaled)
    - Small penalty for holding (funding cost/risk)
    """
    
    def __init__(self, df: pd.DataFrame, window_size: int = 30, initial_balance: float = 250.0, mode: str = "discrete"):
        super(XauUsdEnv, self).__init__()
        
        self.df = df.copy()
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.mode = mode
        
        # Pre-compute normalized features for efficiency
        self._precompute_normalized_features()
        
        if mode == "continuous":
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(4)
        
        # Observation Space (REDUCED for faster learning)
        self.n_features = 5  # returns, atr_pct, vol_pct, rsi_norm, momentum
        self.obs_shape = (window_size * self.n_features) + 4
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.obs_shape,), dtype=np.float32
        )
        
        # State
        self.balance = initial_balance
        self.position = 0.0 
        self.entry_price = 0.0
        self.current_step = 0
        self.max_steps = len(df) - window_size - 1
        
        # Production Constraints
        self.max_lot = 0.10
        self.min_lot = 0.01
        self.spread = 0.25 
        self.commission = 7.0
        
        self.reward_history = []
        
    def _precompute_normalized_features(self):
        """Pre-compute normalized features once for efficiency."""
        df = self.df
        
        # Check if robust features are already computed (from train_rl.py)
        robust_cols = ['returns', 'trend', 'vol_regime', 'deviation', 'momentum']
        has_robust = all(col in df.columns for col in robust_cols)
        
        if has_robust:
            # Use pre-computed robust features
            self.norm_cols = robust_cols
            logger.info("Using robust features from training pipeline") if hasattr(self, 'logger') else None
        else:
            # Compute basic features as fallback
            df['returns'] = df['close'].pct_change().fillna(0)
            
            if 'atr_14' in df.columns:
                df['atr_pct'] = df['atr_14'] / df['close']
            else:
                df['atr_pct'] = (df['high'] - df['low']) / df['close']
            
            if 'volume' in df.columns and df['volume'].std() > 0:
                df['vol_norm'] = (df['volume'] - df['volume'].rolling(100).mean()) / (df['volume'].rolling(100).std() + 1e-8)
                df['vol_norm'] = df['vol_norm'].fillna(0).clip(-3, 3)
            else:
                df['vol_norm'] = 0.0
            
            if 'rsi_14' in df.columns:
                df['rsi_norm'] = (df['rsi_14'] - 50) / 50
            else:
                df['rsi_norm'] = 0.0
            
            df['momentum'] = df['close'].pct_change(5).fillna(0).clip(-0.05, 0.05) * 10
            
            self.norm_cols = ['returns', 'atr_pct', 'vol_norm', 'rsi_norm', 'momentum']
        
        # Fill NaNs and clip for stability
        for col in self.norm_cols:
            df[col] = df[col].fillna(0).clip(-10, 10)
        
        self.df = df
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.current_step = 0
        self.reward_history = []
        return self._next_observation(), {}
        
    def _next_observation(self):
        start = self.current_step
        end = self.current_step + self.window_size
        
        window = self.df[self.norm_cols].iloc[start:end].values
        
        idx = end - 1
        current_price = self.df.iloc[idx]['close']
        unrealized = 0.0
        if self.position != 0:
            unrealized = (current_price - self.entry_price) * self.position * 100
        
        obs = np.clip(window.flatten(), -10, 10).astype(np.float32)
        
        equity = self.balance + unrealized
        state = np.array([
            (self.balance - self.initial_balance) / self.initial_balance,
            self.position / self.max_lot,
            (equity - self.initial_balance) / self.initial_balance,
            np.clip(unrealized / 50.0, -1, 1)
        ], dtype=np.float32)
        
        return np.concatenate([obs, state])
        
    def step(self, action):
        self.current_step += 1
        idx = self.current_step + self.window_size - 1
        current_price = self.df.iloc[idx]['close']
        
        reward = 0.0
        done = False
        
        if self.mode == "continuous":
            side_score = action[0]
            size_score = (action[1] + 1) / 2
            
            if side_score > 0.5:
                target_pos = max(self.min_lot, size_score * self.max_lot)
            elif side_score < -0.5:
                target_pos = -max(self.min_lot, size_score * self.max_lot)
            else:
                target_pos = 0.0
        else:
            targets = [0.0, 0.03, -0.03, 0.0]
            target_pos = targets[action]

        if target_pos != self.position:
            if self.position != 0:
                side = 1 if self.position > 0 else -1
                exit_price = current_price - (side * (self.spread / 2))
                pnl = (exit_price - self.entry_price) * self.position * 100
                comm = abs(self.position) * self.commission
                realized = pnl - comm
                self.balance += realized
                reward += realized / 10.0
                self.position = 0.0
            
            if target_pos != 0:
                side = 1 if target_pos > 0 else -1
                entry_price = current_price + (side * (self.spread / 2))
                self.position = target_pos
                self.entry_price = entry_price
                self.balance -= (abs(self.position) * (self.commission / 2))

        if self.position != 0:
            unrealized = (current_price - self.entry_price) * self.position * 100
            reward += np.clip(unrealized / 100.0, -0.2, 0.2)
            
        if self.position == 0:
            reward -= 0.001

        if self.current_step >= self.max_steps: 
            done = True
        if self.balance < self.initial_balance * 0.5:
            done = True
            reward -= 5.0
            
        reward = np.clip(reward, -5.0, 5.0)
        self.reward_history.append(reward)
            
        return self._next_observation(), reward, done, False, {}

    def render(self):
        print(f"Step: {self.current_step}, Balance: ${self.balance:.2f}, Pos: {self.position:.2f} lots")

