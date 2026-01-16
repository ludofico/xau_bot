
"""
RL Trainer (DeepScalper) - Anti-Overfitting Version.

Features:
- Train/Test split (80/20)
- Early stopping when explained_variance > 0.6
- Evaluation on held-out test data
- Higher entropy for exploration
- VecNormalize for stability
"""

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from pathlib import Path

from xauusd_strategy.rl.env import XauUsdEnv
from xauusd_strategy.utils.logger import get_logger
from xauusd_strategy.data.fetcher import DataFetcher
from xauusd_strategy.data.processor import DataProcessor
from xauusd_strategy.data.features import FeatureEngineer

logger = get_logger(__name__)


class EarlyStoppingCallback(BaseCallback):
    """Stop training when explained_variance exceeds threshold (overfitting sign)."""
    
    def __init__(self, threshold: float = 0.6, patience: int = 5, verbose: int = 1):
        super().__init__(verbose)
        self.threshold = threshold
        self.patience = patience
        self.counter = 0
        self.best_reward = -np.inf
        
    def _on_step(self) -> bool:
        # Check every 2048 steps (after each rollout)
        if self.n_calls % 2048 == 0 and len(self.model.ep_info_buffer) > 0:
            # Get explained variance from logger
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                logs = self.logger.name_to_value
                exp_var = logs.get('train/explained_variance', 0)
                ep_rew = logs.get('rollout/ep_rew_mean', -np.inf)
                
                if exp_var > self.threshold:
                    self.counter += 1
                    if self.verbose > 0:
                        logger.warning(f"⚠️ Overfitting detected: explained_variance={exp_var:.3f} > {self.threshold}")
                    
                    if self.counter >= self.patience:
                        logger.critical(f"🛑 Early stopping triggered after {self.patience} consecutive overfitting signals")
                        return False
                else:
                    self.counter = 0  # Reset counter
                    
                # Track best reward for model selection
                if ep_rew > self.best_reward:
                    self.best_reward = ep_rew
                    
        return True


def train_rl_agent(total_timesteps=100_000):
    """Train RL agent with anti-overfitting measures."""
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
    
    # Add features
    eng = FeatureEngineer()
    df = eng.compute_all(df, include_ml_features=True)
    df = df.dropna()
    
    logger.info(f"Total data: {len(df)} bars")
    
    # 2. TRAIN/TEST SPLIT (80/20) - Prevent overfitting!
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    
    logger.info(f"Train: {len(df_train)} bars | Test: {len(df_test)} bars")
    
    # 3. Create Training Environment
    def make_train_env():
        env = XauUsdEnv(df_train.copy(), window_size=30, mode="discrete")
        env = Monitor(env)
        return env
    
    train_env = DummyVecEnv([make_train_env])
    train_env = VecNormalize(
        train_env, 
        norm_obs=True, 
        norm_reward=True, 
        clip_obs=10.0, 
        clip_reward=10.0,
        gamma=0.99
    )
    
    # 4. Create Test Environment for Evaluation
    def make_test_env():
        env = XauUsdEnv(df_test.copy(), window_size=30, mode="discrete")
        env = Monitor(env)
        return env
    
    test_env = DummyVecEnv([make_test_env])
    # Use same normalization stats from training
    test_env = VecNormalize(
        test_env,
        norm_obs=True,
        norm_reward=False,  # Don't normalize reward for evaluation
        clip_obs=10.0,
        training=False  # Don't update stats during eval
    )
    
    # 5. Create PPO Agent with ANTI-OVERFITTING hyperparameters
    model = PPO(
        "MlpPolicy", 
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=5,          # Reduced from 10 (less overfitting)
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,       # INCREASED from 0.01 (more exploration)
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./logs/rl_tensorboard/",
        device="auto"
    )
    
    # 6. Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path=str(model_path),
        name_prefix='deepscalper_ppo'
    )
    
    early_stopping = EarlyStoppingCallback(
        threshold=0.6,  # Stop if explained_variance > 0.6
        patience=5,     # Allow 5 consecutive warnings before stopping
        verbose=1
    )
    
    eval_callback = EvalCallback(
        test_env,
        best_model_save_path=str(model_path / "best"),
        log_path=str(model_path / "eval_logs"),
        eval_freq=10000,
        n_eval_episodes=3,
        deterministic=True,
        verbose=1
    )
    
    # 7. Train with all callbacks
    logger.info(f"Starting Training for {total_timesteps} steps...")
    logger.info("Anti-overfitting: Early stopping at explained_variance > 0.6")
    logger.info("Target: ep_rew_mean > 40 on TEST data for doubling potential")
    
    try:
        model.learn(
            total_timesteps=total_timesteps, 
            callback=[checkpoint_callback, early_stopping, eval_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    
    # 8. Save Final Model
    model.save(model_path / "final_model")
    train_env.save(str(model_path / "vec_normalize.pkl"))
    
    # 9. Final Evaluation on TEST data
    logger.info("\n" + "="*50)
    logger.info("FINAL EVALUATION ON UNSEEN TEST DATA")
    logger.info("="*50)
    
    # Sync normalization stats
    test_env.obs_rms = train_env.obs_rms
    test_env.ret_rms = train_env.ret_rms
    
    mean_reward, std_reward = evaluate_policy(
        model, test_env, n_eval_episodes=5, deterministic=True
    )
    
    logger.info(f"Test Performance: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    if mean_reward > 40:
        logger.info("✅ TARGET MET! Model ready for live trading (doubling potential)")
    elif mean_reward > 20:
        logger.info("⚠️ Moderate performance. May need more training or data.")
    else:
        logger.warning("❌ Below target. Consider more data or hyperparameter tuning.")
    
    logger.info(f"\nBest model saved to: {model_path / 'best'}")
    logger.info(f"Final model saved to: {model_path / 'final_model.zip'}")

if __name__ == "__main__":
    train_rl_agent()


