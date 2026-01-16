
"""
RL Trainer (DeepScalper).

Trains a PPO Agent on the XauUsdEnv.
Uses VecNormalize for observation/reward normalization.
"""

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from pathlib import Path
import numpy as np

from xauusd_strategy.rl.env import XauUsdEnv
from xauusd_strategy.utils.logger import get_logger
from xauusd_strategy.data.fetcher import DataFetcher
from xauusd_strategy.data.processor import DataProcessor
from xauusd_strategy.data.features import FeatureEngineer

logger = get_logger(__name__)

def train_rl_agent(total_timesteps=200_000):
    model_path = Path("models/rl_deepscalper")
    model_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Fetch & Prepare Data
    logger.info("Fetching data for RL training...")
    fetcher = DataFetcher(source='yfinance')
    from datetime import datetime, timedelta
    end = datetime.now()
    start = end - timedelta(days=59)
    df_raw = fetcher.fetch(start, end, "5m")
    
    processor = DataProcessor(timezone="Europe/Berlin")
    df = processor.process(df_raw)
    
    # Add BASIC Features only (no transformers)
    eng = FeatureEngineer()
    df = eng.compute_all(df, include_ml_features=True)
    df = df.dropna()
    
    logger.info(f"Data ready: {len(df)} bars for training")
    
    # 2. Create Environment with Normalization
    def make_env():
        env = XauUsdEnv(df.copy(), window_size=30, mode="discrete")
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_env])
    
    env = VecNormalize(
        env, 
        norm_obs=True, 
        norm_reward=True, 
        clip_obs=10.0, 
        clip_reward=10.0,
        gamma=0.99
    )
    
    # 3. Create PPO Agent
    model = PPO(
        "MlpPolicy", 
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./logs/rl_tensorboard/",
        device="auto"
    )
    
    # 4. Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=20000, 
        save_path=str(model_path),
        name_prefix='deepscalper_ppo'
    )
    
    # 5. Train
    logger.info(f"Starting Training for {total_timesteps} steps...")
    logger.info("Watch for explained_variance > 0.1 as sign of learning!")
    
    model.learn(
        total_timesteps=total_timesteps, 
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # 6. Save Final Model AND VecNormalize stats
    model.save(model_path / "final_model")
    env.save(str(model_path / "vec_normalize.pkl"))
    logger.info("Training Complete. Model and normalizer saved.")

if __name__ == "__main__":
    train_rl_agent()

