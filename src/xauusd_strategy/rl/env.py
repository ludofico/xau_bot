
"""
Custom Gym Environment for XAUUSD Scalping (DeepScalper).

Features:
- Realistic Spread & Commission simulation
- Continuous Action Space (Position Sizing) or Discrete (Buy/Sell/Hold)
- Reward Function V3: BALANCED for train/test alignment
- NORMALIZED OBSERVATIONS for stable training

REWARD FUNCTION V3 (Balanced):
- Minimal holding penalty
- Realistic trading costs
- Small profit bonus (not inflated)
- Reduced end-of-episode bonus (only if profitable)
- NO equity high bonus (was causing reward inflation)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict

class XauUsdEnv(gym.Env):
    """
    XAUUSD Trading Environment with BALANCED reward function.
    
    Observation Space (Normalized):
    - Returns instead of raw prices
    - Normalized indicators
    - Scaled account state
    
    Action Space (Discrete):
    0: Hold
    1: Buy (Long)
    2: Sell (Short)
    3: Close All
    
    Reward V3 (Balanced):
    - Realized PnL scaled appropriately
    - Small bonus for winning trades
    - Minimal holding penalty
    - Moderate end-of-episode bonus
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
        
        # Observation Space
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
        
        # Realistic Trading Costs
        self.max_lot = 0.10
        self.min_lot = 0.01
        self.spread = 0.15       # ECN spread
        self.commission = 5.0    # Competitive broker
        
        self.reward_history = []
        self.max_balance = initial_balance
        self.trade_count = 0
        self.winning_trades = 0
        
    def _precompute_normalized_features(self):
        """Pre-compute normalized features once for efficiency."""
        df = self.df
        
        robust_cols = ['returns', 'trend', 'vol_regime', 'deviation', 'momentum']
        has_robust = all(col in df.columns for col in robust_cols)
        
        if has_robust:
            self.norm_cols = robust_cols
        else:
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
        
        for col in self.norm_cols:
            df[col] = df[col].fillna(0).clip(-10, 10)
        
        self.df = df
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.max_balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.current_step = 0
        self.reward_history = []
        self.trade_count = 0
        self.winning_trades = 0
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

        # === POSITION CHANGE ===
        if target_pos != self.position:
            # Close existing
            if self.position != 0:
                side = 1 if self.position > 0 else -1
                exit_price = current_price - (side * (self.spread / 2))
                pnl = (exit_price - self.entry_price) * self.position * 100
                comm = abs(self.position) * self.commission
                realized = pnl - comm
                self.balance += realized
                self.trade_count += 1
                
                # BALANCED REWARD: Scale by P&L
                if realized > 0:
                    self.winning_trades += 1
                    reward += (realized / 5.0) + 0.1  # Small bonus (was 0.5)
                else:
                    reward += realized / 8.0  # Slightly softer loss penalty
                
                self.position = 0.0
            
            # Open new
            if target_pos != 0:
                side = 1 if target_pos > 0 else -1
                entry_price = current_price + (side * (self.spread / 2))
                self.position = target_pos
                self.entry_price = entry_price
                self.balance -= (abs(self.position) * (self.commission / 2))

        # === UNREALIZED FEEDBACK ===
        if self.position != 0:
            unrealized = (current_price - self.entry_price) * self.position * 100
            reward += np.clip(unrealized / 80.0, -0.3, 0.3)
        
        # === MINIMAL HOLDING PENALTY ===
        if self.position == 0:
            reward -= 0.0001
        
        # === TRACK MAX BALANCE (no bonus, just track) ===
        if self.balance > self.max_balance:
            self.max_balance = self.balance
        
        # === DRAWDOWN PENALTY ===
        drawdown = (self.max_balance - self.balance) / self.max_balance
        if drawdown > 0.1:
            reward -= drawdown * 0.5

        # === TERMINAL CONDITIONS ===
        if self.current_step >= self.max_steps: 
            done = True
            # MODERATE end-of-episode bonus (only if profitable)
            final_return = (self.balance - self.initial_balance) / self.initial_balance
            if final_return > 0:
                reward += final_return * 2  # Reduced from 10 to 2
            
        if self.balance < self.initial_balance * 0.5:
            done = True
            reward -= 3.0
            
        reward = np.clip(reward, -5.0, 5.0)
        self.reward_history.append(reward)
            
        return self._next_observation(), reward, done, False, {}

    def render(self):
        wr = self.winning_trades / max(1, self.trade_count) * 100
        print(f"Step: {self.current_step}, Balance: ${self.balance:.2f}, Pos: {self.position:.2f}, Trades: {self.trade_count}, WR: {wr:.0f}%")



