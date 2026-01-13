
"""
Custom Gym Environment for XAUUSD Scalping (DeepScalper).

Features:
- Realistic Spread & Commission simulation
- Continuous Action Space (Position Sizing) or Discrete (Buy/Sell/Hold)
- Reward Function: Sharpe Ratio driven (Profit - Volatility Penalty)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict

class XauUsdEnv(gym.Env):
    """
    XAUUSD Trading Environment.
    
    Observation Space:
    - Window of N candles (OHLCV) + Technical Indicators
    - Current Account PnL, Open Position status
    
    Action Space (Discrete):
    0: Hold
    1: Buy (Long)
    2: Sell (Short)
    3: Close All
    
    Reward:
    - Realized PnL
    - Unrealized PnL change
    - Small penalty for holding (funding cost/risk)
    """
    
    def __init__(self, df: pd.DataFrame, window_size: int = 60, initial_balance: float = 250.0, mode: str = "continuous"):
        super(XauUsdEnv, self).__init__()
        
        self.df = df
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.mode = mode
        
        if mode == "continuous":
            # Action[0]: Side (-1 to 1). < -0.5 Short, > 0.5 Long, else Close/Hold
            # Action[1]: Sizing (0 to 1). Multiplier for max risk.
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(4)
        
        # Observation Space: 
        n_features = len(df.columns)
        self.obs_shape = (window_size * n_features) + 4 # +4 for balance, pos, equity, unrealized
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32
        )
        
        # State
        self.balance = initial_balance
        self.position = 0.0 
        self.entry_price = 0.0
        self.current_step = 0
        self.max_steps = len(df) - window_size - 1
        
        # Production Constraints
        self.max_lot = 0.50 # Hard limit for safety
        self.min_lot = 0.01
        self.spread = 0.25 
        self.commission = 7.0 
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.current_step = 0
        return self._next_observation(), {}
        
    def _next_observation(self):
        start = self.current_step
        end = self.current_step + self.window_size
        window = self.df.iloc[start:end].values
        
        idx = end - 1
        current_price = self.df.iloc[idx]['close']
        unrealized = 0.0
        if self.position != 0:
            unrealized = (current_price - self.entry_price) * self.position * 100
            
        obs = window.flatten().astype(np.float32)
        state = np.array([
            self.balance / 1000.0, # Normalization
            self.position * 10.0,
            (self.balance + unrealized) / 1000.0,
            unrealized / 100.0
        ], dtype=np.float32)
        
        return np.concatenate([obs, state])
        
    def step(self, action):
        self.current_step += 1
        idx = self.current_step + self.window_size - 1
        current_price = self.df.iloc[idx]['close']
        
        reward = 0.0
        done = False
        
        # Parse Action
        if self.mode == "continuous":
            side_score = action[0]
            size_score = (action[1] + 1) / 2 # Scale -1,1 to 0,1
            
            # Action Logic
            if side_score > 0.5: # Want to be LONG
                target_pos = max(self.min_lot, size_score * self.max_lot)
            elif side_score < -0.5: # Want to be SHORT
                target_pos = -max(self.min_lot, size_score * self.max_lot)
            else: # Want to be NEUTRAL
                target_pos = 0.0
        else:
            # Discrete mapper
            targets = [0.0, 0.01, -0.01, 0.0]
            target_pos = targets[action]

        # Execution Logic (Close/Flip)
        if target_pos != self.position:
            # 1. Close current position
            if self.position != 0:
                side = 1 if self.position > 0 else -1
                exit_price = current_price - (side * (self.spread / 2))
                pnl = (exit_price - self.entry_price) * self.position * 100
                comm = abs(self.position) * self.commission
                self.balance += (pnl - comm)
                reward += (pnl - comm)
                self.position = 0.0
            
            # 2. Open new position
            if target_pos != 0:
                side = 1 if target_pos > 0 else -1
                entry_price = current_price + (side * (self.spread / 2))
                self.position = target_pos
                self.entry_price = entry_price
                # Entry commission
                self.balance -= (abs(self.position) * (self.commission/2))

        # Reward Shaping: Volatility Penalty & Time decay
        if self.position != 0:
            unrealized = (current_price - self.entry_price) * self.position * 100
            # Small penalty for high risk (Position Sizing)
            risk_penalty = (abs(self.position) / self.max_lot) * 0.01
            reward += (unrealized * 0.05) - risk_penalty

        if self.current_step >= self.max_steps: done = True
        if self.balance < self.initial_balance * 0.3: # Stop at 70% drawdown
             done = True
             reward -= 50
            
        return self._next_observation(), reward, done, False, {}

    def render(self):
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Pos: {self.position}")
