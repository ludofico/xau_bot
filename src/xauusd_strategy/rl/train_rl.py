
"""
RL Trainer (DeepScalper).

Trains a PPO Agent on the XauUsdEnv.
"""

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from pathlib import Path

from xauusd_strategy.rl.env import XauUsdEnv
from xauusd_strategy.utils.logger import get_logger
from xauusd_strategy.data.fetcher import DataFetcher
from xauusd_strategy.data.processor import DataProcessor
from xauusd_strategy.ml.features import MLFeatureEngineer

logger = get_logger(__name__)

def train_rl_agent(total_timesteps=100_000):
    model_path = Path("models/rl_deepscalper")
    model_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Fetch & Prepare Data (Need a LOT of data for RL)
    logger.info("Fetching data for RL training...")
    # Ideally 1-2 years. For now using what we have (60 days 5m) or fetching more
    fetcher = DataFetcher(source='yfinance')
    from datetime import datetime, timedelta
    end = datetime.now()
    start = end - timedelta(days=59) 
    df_raw = fetcher.fetch(start, end, "5m")
    
    processor = DataProcessor(timezone="Europe/Berlin")
    df = processor.process(df_raw)
    
    # Add Features (Technical Indicators + 'The Linguist' Embeddings)
    # Note: Transformers take time, use a smaller window or skip if data too large
    eng = MLFeatureEngineer(use_transformers=True)
    df = eng.prepare_ml_features(df)
    df = df.dropna()
    
    logger.info(f"Data ready: {len(df)} bars")
    
    # 2. Create Environment
    # Vectorized environment for faster training
    env = DummyVecEnv([lambda: XauUsdEnv(df, window_size=60)])
    
    # 3. Create Agent (PPO)
    # MLP Policy (FEED FORWARD) - Good for numerical features
    # MultiInputPolicy allows the agent to process both raw embeddings and indicators
    model = PPO(
        "MlpPolicy", 
        env,
        verbose=1,
        learning_rate=0.0002, # Slightly lower for continuous control
        n_steps=4096,         # More steps for complex policy
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01, # Encourage exploration
        tensorboard_log="./logs/rl_tensorboard/"
    )
    
    # 4. Train
    logger.info(f"Starting Training for {total_timesteps} steps...")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path=str(model_path),
        name_prefix='deepscalper_ppo'
    )
    
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    
    # 5. Save Final
    model.save(model_path / "final_model")
    logger.info("Training Complete. Model Saved.")
    
if __name__ == "__main__":
    train_rl_agent()
