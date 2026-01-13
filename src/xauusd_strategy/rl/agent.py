
"""
DeepScalper Agent Wrapper.

Loads trained PPO model and provides interface for LiveTrader.
"""

from stable_baselines3 import PPO
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple, Dict

from xauusd_strategy.utils.logger import get_logger

logger = get_logger(__name__)

class DeepScalperAgent:
    def __init__(self, model_path: str = "models/rl_deepscalper/final_model.zip"):
        self.model_path = Path(model_path)
        self.model = None
        self.window_size = 60
        self._load()
        
    def _load(self):
        if self.model_path.exists():
            try:
                self.model = PPO.load(self.model_path)
                logger.info(f"DeepScalper Agent loaded from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load RL Agent: {e}")
        else:
            logger.warning(f"RL Model not found at {self.model_path}")
            
    def predict(self, df_window: pd.DataFrame, balance: float, position_size: float) -> int:
        """
        Predict action for current market state.
        
        Actions:
        0: Hold
        1: Buy
        2: Sell
        3: Close
        """
        if self.model is None:
            return 0 # Hold (Safe default)
            
        # Construct Observation Vector matching Env
        # 1. Flatten Data Window
        # Ensure we have exactly window_size rows
        if len(df_window) < self.window_size:
            logger.warning("Insufficient data for RL Agent")
            return 0
            
        window = df_window.iloc[-self.window_size:].values.flatten().astype(np.float32)
        
        # 2. Account State
        state = np.array([balance, position_size], dtype=np.float32)
        
        # 3. Concatenate
        obs = np.concatenate([window, state])
        
        # Predict
        action, _states = self.model.predict(obs, deterministic=True)
        
        return int(action)
