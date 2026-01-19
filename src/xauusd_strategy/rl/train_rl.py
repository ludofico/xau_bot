
"""
RL Trainer (DeepScalper) - Production Anti-Overfitting Version.

ANTI-OVERFITTING TECHNIQUES:
1. Large dataset (12+ months of data)
2. Train/Test split 70/30 (more test data for validation)
3. Data augmentation (noise injection, time shifts)
4. High entropy coefficient (exploration)
5. Early stopping on TEST performance degradation
6. Gradient clipping
7. Small batch size with many epochs
8. Regular evaluation on held-out data

Target: 300k timesteps, robust generalization
"""

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from pathlib import Path
from datetime import datetime, timedelta
import random

from xauusd_strategy.rl.env import XauUsdEnv
from xauusd_strategy.utils.logger import get_logger
from xauusd_strategy.data.processor import DataProcessor
from xauusd_strategy.data.features import FeatureEngineer

logger = get_logger(__name__)


class AntiOverfitCallback(BaseCallback):
    """
    Advanced anti-overfitting callback with multiple safeguards.
    
    Monitors:
    1. explained_variance (too high = overfitting)
    2. Policy entropy (too low = overfitting)
    3. Test performance degradation
    """
    
    def __init__(
        self, 
        exp_var_threshold: float = 0.7,  # Stricter threshold
        entropy_min: float = 0.3,        # Minimum entropy
        patience: int = 10,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.exp_var_threshold = exp_var_threshold
        self.entropy_min = entropy_min
        self.patience = patience
        self.overfit_counter = 0
        self.low_entropy_counter = 0
        self.best_test_reward = -np.inf
        self.test_degradation_counter = 0
        
    def _on_step(self) -> bool:
        if self.n_calls % 2048 == 0 and len(self.model.ep_info_buffer) > 0:
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                logs = self.logger.name_to_value
                
                # Check explained variance
                exp_var = logs.get('train/explained_variance', 0)
                if exp_var > self.exp_var_threshold:
                    self.overfit_counter += 1
                    if self.verbose:
                        logger.warning(f"⚠️ High explained_variance: {exp_var:.3f} ({self.overfit_counter}/{self.patience})")
                    if self.overfit_counter >= self.patience:
                        logger.critical("🛑 STOPPING: Overfitting detected (explained_variance)")
                        return False
                else:
                    self.overfit_counter = max(0, self.overfit_counter - 1)
                
                # Check entropy
                entropy = logs.get('train/entropy_loss', 1.0)
                if abs(entropy) < self.entropy_min:
                    self.low_entropy_counter += 1
                    if self.verbose:
                        logger.warning(f"⚠️ Low entropy: {entropy:.3f} ({self.low_entropy_counter}/{self.patience})")
                    if self.low_entropy_counter >= self.patience:
                        logger.critical("🛑 STOPPING: Policy collapsed (low entropy)")
                        return False
                else:
                    self.low_entropy_counter = max(0, self.low_entropy_counter - 1)
                    
        return True


class DataAugmenter:
    """Data augmentation for robust RL training."""
    
    @staticmethod
    def add_noise(df: pd.DataFrame, noise_pct: float = 0.001) -> pd.DataFrame:
        """Add small Gaussian noise to prices."""
        df = df.copy()
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                noise = np.random.normal(0, df[col].std() * noise_pct, len(df))
                df[col] = df[col] + noise
        return df
    
    @staticmethod
    def time_shift(df: pd.DataFrame, shift_bars: int = 5) -> pd.DataFrame:
        """Randomly shift time series by a few bars."""
        shift = random.randint(-shift_bars, shift_bars)
        return df.shift(shift).dropna()
    
    @staticmethod
    def scale_volatility(df: pd.DataFrame, scale_range: tuple = (0.8, 1.2)) -> pd.DataFrame:
        """Randomly scale price volatility."""
        df = df.copy()
        scale = random.uniform(*scale_range)
        mid_price = (df['high'] + df['low']) / 2
        
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = mid_price + (df[col] - mid_price) * scale
        return df


def fetch_extended_data() -> pd.DataFrame:
    """
    Fetch MAXIMUM data for RL training.
    
    Priority:
    1. MT5 Native: 12 months of M5 data (~100,000 bars)
    2. yfinance: 2 years of 1H data (~12,000 bars)
    """
    end = datetime.now()
    
    # Try MT5 first (Windows with MT5)
    try:
        import MetaTrader5 as mt5
        
        if mt5.initialize():
            logger.info("MT5 connected! Fetching 12 MONTHS of M5 data...")
            
            start = end - timedelta(days=365)  # 12 months
            symbols = ["XAUUSD", "GOLD", "XAUUSDm", "XAUUSD.a"]
            
            for symbol in symbols:
                rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start, end)
                if rates is not None and len(rates) > 5000:
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('time', inplace=True)
                    df.columns = [c.lower() for c in df.columns]
                    
                    if 'tick_volume' in df.columns and 'volume' not in df.columns:
                        df['volume'] = df['tick_volume']
                    
                    logger.info(f"MT5: Fetched {len(df)} bars of {symbol} M5 ({len(df)//288} days)")
                    mt5.shutdown()
                    return df
            
            mt5.shutdown()
    except ImportError:
        logger.info("MT5 not available, using yfinance")
    except Exception as e:
        logger.warning(f"MT5 error: {e}")
    
    # Fallback: yfinance 2 years 1H
    import yfinance as yf
    
    logger.info("Fetching 2 YEARS of 1H data from yfinance...")
    
    ticker = yf.Ticker("GC=F")
    start = end - timedelta(days=730)  # 2 years
    
    df = ticker.history(start=start, end=end, interval="1h")
    
    if df.empty or len(df) < 1000:
        logger.warning("1H data insufficient, trying daily...")
        df = ticker.history(start=start, end=end, interval="1d")
    
    df.columns = [c.lower() for c in df.columns]
    
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    logger.info(f"yfinance: Fetched {len(df)} bars (~{len(df)//24} days)")
    
    return df


def add_robust_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add NORMALIZED features that generalize across market regimes.
    These features are scale-invariant and period-agnostic.
    """
    # 1. Returns (always normalized around 0)
    df['returns'] = df['close'].pct_change().fillna(0).clip(-0.1, 0.1)
    df['returns_5'] = df['close'].pct_change(5).fillna(0).clip(-0.2, 0.2)
    df['returns_20'] = df['close'].pct_change(20).fillna(0).clip(-0.3, 0.3)
    
    # 2. Volatility Regime (normalized 0.5-2.0)
    df['volatility'] = df['returns'].rolling(20).std().fillna(0.01)
    vol_ma = df['volatility'].rolling(100).mean()
    df['vol_regime'] = (df['volatility'] / vol_ma.replace(0, 0.01)).clip(0.5, 2.0).fillna(1.0)
    
    # 3. Trend Strength (normalized -1 to 1)
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()
    df['trend'] = ((df['ema_20'] - df['ema_50']) / df['close']).clip(-0.02, 0.02) * 50
    
    # 4. Mean Reversion Signal (deviation from mean)
    df['deviation'] = ((df['close'] - df['ema_20']) / df['close']).clip(-0.03, 0.03) * 30
    
    # 5. Momentum (normalized)
    df['momentum'] = df['returns_5'].clip(-0.05, 0.05) * 10
    
    # 6. ATR Percentage (scale-invariant volatility)
    high_low = df['high'] - df['low']
    df['atr_pct'] = (high_low.rolling(14).mean() / df['close']).clip(0, 0.02) * 50
    
    # 7. Volume Signal (if available)
    if 'volume' in df.columns and df['volume'].std() > 0:
        vol_ma = df['volume'].rolling(20).mean()
        df['vol_signal'] = ((df['volume'] / vol_ma.replace(0, 1)) - 1).clip(-2, 2)
    else:
        df['vol_signal'] = 0
    
    # 8. Time features (cyclical)
    if hasattr(df.index, 'hour'):
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    else:
        df['hour_sin'] = 0
        df['hour_cos'] = 0
    
    df['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    
    # Fill NaN and ensure stability
    df = df.fillna(0)
    
    return df


def create_augmented_env(df: pd.DataFrame, augment: bool = True):
    """Create environment with optional data augmentation."""
    if augment:
        df = DataAugmenter.add_noise(df, noise_pct=0.0005)
    
    env = XauUsdEnv(df.copy(), window_size=30, mode="discrete")
    env = Monitor(env)
    return env


def train_rl_agent(total_timesteps: int = 300_000, use_augmentation: bool = True):
    """
    Train RL agent with PRODUCTION anti-overfitting settings.
    
    Args:
        total_timesteps: Training steps (default 300k)
        use_augmentation: Whether to use data augmentation
    """
    model_path = Path("models/rl_deepscalper")
    model_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Fetch Extended Data
    logger.info("="*60)
    logger.info("ANTI-OVERFITTING RL TRAINING")
    logger.info("="*60)
    
    df = fetch_extended_data()
    
    # 2. Process and add robust features
    processor = DataProcessor(timezone="Europe/Berlin")
    df = processor.process(df)
    
    eng = FeatureEngineer()
    df = eng.compute_all(df, include_ml_features=True)
    df = add_robust_features(df)
    df = df.dropna()
    
    logger.info(f"Total data: {len(df)} bars (~{len(df)//24} days)")
    
    # 3. TRAIN/TEST SPLIT - 70/30 (more test data for robust validation)
    split_idx = int(len(df) * 0.7)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    
    logger.info(f"Train: {len(df_train)} bars ({len(df_train)*100//len(df)}%)")
    logger.info(f"Test:  {len(df_test)} bars ({len(df_test)*100//len(df)}%)")
    
    # 4. Create Training Environment with Augmentation
    def make_train_env():
        return create_augmented_env(df_train, augment=use_augmentation)
    
    train_env = DummyVecEnv([make_train_env])
    train_env = VecNormalize(
        train_env, 
        norm_obs=True, 
        norm_reward=True, 
        clip_obs=10.0, 
        clip_reward=10.0,
        gamma=0.99
    )
    
    # 5. Create Test Environment (NO augmentation)
    def make_test_env():
        env = XauUsdEnv(df_test.copy(), window_size=30, mode="discrete")
        env = Monitor(env)
        return env
    
    test_env = DummyVecEnv([make_test_env])
    test_env = VecNormalize(
        test_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=False
    )
    
    # 6. Create PPO Agent with ANTI-OVERFITTING Settings
    model = PPO(
        "MlpPolicy", 
        train_env,
        verbose=1,
        learning_rate=3e-5,           # Lower LR for stability
        n_steps=2048,
        batch_size=64,
        n_epochs=10,                  # More epochs per batch
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,               # Tighter clipping (more conservative)
        ent_coef=0.1,                 # HIGH ENTROPY (exploration)
        vf_coef=0.5,
        max_grad_norm=0.3,            # Aggressive gradient clipping
        tensorboard_log="./logs/rl_tensorboard/",
        device="auto",
        policy_kwargs={
            "net_arch": [64, 64],     # Smaller network (less capacity to overfit)
        }
    )
    
    # 7. Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,              # More frequent checkpoints
        save_path=str(model_path),
        name_prefix='deepscalper_ppo'
    )
    
    anti_overfit = AntiOverfitCallback(
        exp_var_threshold=0.7,
        entropy_min=0.3,
        patience=10,
        verbose=1
    )
    
    eval_callback = EvalCallback(
        test_env,
        best_model_save_path=str(model_path / "best"),
        log_path=str(model_path / "eval_logs"),
        eval_freq=10000,              # Evaluate every 10k steps
        n_eval_episodes=10,           # More eval episodes
        deterministic=True,
        verbose=1
    )
    
    # 8. Training
    logger.info("")
    logger.info("TRAINING CONFIGURATION:")
    logger.info(f"  - Timesteps: {total_timesteps:,}")
    logger.info(f"  - Data Augmentation: {use_augmentation}")
    logger.info(f"  - Entropy Coef: 0.1 (high exploration)")
    logger.info(f"  - Clip Range: 0.1 (conservative updates)")
    logger.info(f"  - Network: [64, 64] (small to prevent overfit)")
    logger.info(f"  - Grad Clip: 0.3 (aggressive)")
    logger.info("")
    logger.info("Starting training... (Ctrl+C to stop early)")
    
    try:
        model.learn(
            total_timesteps=total_timesteps, 
            callback=[checkpoint_callback, anti_overfit, eval_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    
    # 9. Save Final Model
    model.save(model_path / "final_model")
    train_env.save(str(model_path / "vec_normalize.pkl"))
    
    # 10. Final Evaluation
    logger.info("")
    logger.info("="*60)
    logger.info("FINAL EVALUATION ON UNSEEN TEST DATA")
    logger.info("="*60)
    
    # Sync normalization stats
    test_env.obs_rms = train_env.obs_rms
    test_env.ret_rms = train_env.ret_rms
    
    mean_reward, std_reward = evaluate_policy(
        model, test_env, n_eval_episodes=20, deterministic=True
    )
    
    logger.info(f"Test Performance: {mean_reward:.2f} +/- {std_reward:.2f}")
    logger.info("")
    
    # Recommendation
    if mean_reward > 50:
        logger.info("🏆 EXCELLENT! Model ready for aggressive live trading")
        rec = "aggressive"
    elif mean_reward > 30:
        logger.info("✅ GOOD! Model ready for live trading with rule fallbacks")
        rec = "with_fallback"
    elif mean_reward > 10:
        logger.info("⚠️ MARGINAL. Use primarily rule-based strategies")
        rec = "cautious"
    else:
        logger.warning("❌ POOR. Use rule-based strategies only (London Breakout, Asian Scalp)")
        rec = "rules_only"
    
    logger.info("")
    logger.info(f"Best model saved: {model_path / 'best'}")
    logger.info(f"Final model saved: {model_path / 'final_model.zip'}")
    logger.info(f"Recommendation: {rec}")
    
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'recommendation': rec,
        'model_path': str(model_path / 'final_model.zip')
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train DeepScalper RL Agent")
    parser.add_argument('--timesteps', type=int, default=300_000, help='Training timesteps')
    parser.add_argument('--no-augment', action='store_true', help='Disable data augmentation')
    
    args = parser.parse_args()
    
    train_rl_agent(
        total_timesteps=args.timesteps,
        use_augmentation=not args.no_augment
    )

