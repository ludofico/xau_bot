
"""
RL Trainer (DeepScalper) - Production Version.

Features:
- 12 MONTHS of data (1H timeframe from yfinance)
- 300k timesteps for robust learning
- Train/Test split (80/20)
- Early stopping when explained_variance > 0.6
- Robust features (market regime, momentum, volatility)
- Rule-based fallback ready
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

from xauusd_strategy.rl.env import XauUsdEnv
from xauusd_strategy.utils.logger import get_logger
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
        if self.n_calls % 2048 == 0 and len(self.model.ep_info_buffer) > 0:
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
                    self.counter = 0
                    
                if ep_rew > self.best_reward:
                    self.best_reward = ep_rew
                    
        return True


def fetch_extended_data() -> pd.DataFrame:
    """
    Fetch extended data for RL training.
    
    Priority:
    1. MT5 Native: 6 months of M5 data (~35,000 bars) - Windows only
    2. yfinance: 12 months of 1H data (~6,000 bars) - Cross-platform fallback
    """
    end = datetime.now()
    
    # Try MT5 first (Windows server with MT5 installed)
    try:
        import MetaTrader5 as mt5
        
        if mt5.initialize():
            logger.info("MT5 connected! Fetching 6 MONTHS of M5 data...")
            
            start = end - timedelta(days=180)  # 6 months
            
            # XAUUSD or Gold symbol (depends on broker)
            symbols_to_try = ["XAUUSD", "GOLD", "XAUUSDm", "XAUUSD.a"]
            
            for symbol in symbols_to_try:
                rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start, end)
                if rates is not None and len(rates) > 1000:
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('time', inplace=True)
                    df.columns = [c.lower() for c in df.columns]
                    
                    # Rename tick_volume to volume if needed
                    if 'tick_volume' in df.columns and 'volume' not in df.columns:
                        df['volume'] = df['tick_volume']
                    
                    logger.info(f"MT5: Fetched {len(df)} bars of {symbol} M5 from {df.index[0]} to {df.index[-1]}")
                    mt5.shutdown()
                    return df
            
            logger.warning("MT5: No valid XAUUSD data found, trying yfinance...")
            mt5.shutdown()
        else:
            logger.warning(f"MT5 init failed: {mt5.last_error()}")
            
    except ImportError:
        logger.info("MT5 not available (not Windows), using yfinance fallback")
    except Exception as e:
        logger.warning(f"MT5 error: {e}, using yfinance fallback")
    
    # Fallback: yfinance 1H data (12 months)
    import yfinance as yf
    
    logger.info("Fetching 12 MONTHS of 1H data from yfinance (fallback)...")
    
    ticker = yf.Ticker("GC=F")  # Gold Futures
    start = end - timedelta(days=365)
    
    df = ticker.history(start=start, end=end, interval="1h")
    
    if df.empty:
        logger.warning("1H data failed, trying daily...")
        df = ticker.history(start=start, end=end, interval="1d")
    
    df.columns = [c.lower() for c in df.columns]
    
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    logger.info(f"yfinance: Fetched {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    
    return df


def add_robust_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add robust features that generalize across market regimes.
    These are less period-specific than raw indicators.
    """
    # 1. Returns (always normalized around 0)
    df['returns'] = df['close'].pct_change().fillna(0)
    df['returns_5'] = df['close'].pct_change(5).fillna(0)
    df['returns_20'] = df['close'].pct_change(20).fillna(0)
    
    # 2. Volatility Regime (normalized)
    df['volatility'] = df['returns'].rolling(20).std().fillna(0)
    df['vol_regime'] = (df['volatility'] / df['volatility'].rolling(100).mean()).fillna(1)
    df['vol_regime'] = df['vol_regime'].clip(0.5, 2.0)
    
    # 3. Trend Strength (normalized -1 to 1)
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()
    df['trend'] = ((df['ema_20'] - df['ema_50']) / df['close']).clip(-0.02, 0.02) * 50
    
    # 4. Mean Reversion Signal
    df['deviation'] = ((df['close'] - df['ema_20']) / df['close']).clip(-0.03, 0.03) * 30
    
    # 5. Momentum (normalized)
    df['momentum'] = df['returns_5'].clip(-0.05, 0.05) * 10
    
    # 6. ATR Percentage
    high_low = df['high'] - df['low']
    df['atr_pct'] = high_low.rolling(14).mean() / df['close']
    df['atr_pct'] = df['atr_pct'].fillna(0).clip(0, 0.02) * 50
    
    # 7. Volume Signal
    if 'volume' in df.columns and df['volume'].std() > 0:
        df['vol_signal'] = ((df['volume'] / df['volume'].rolling(20).mean()) - 1).clip(-2, 2)
    else:
        df['vol_signal'] = 0
    
    # 8. Hour of day (cyclical)
    if hasattr(df.index, 'hour'):
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    else:
        df['hour_sin'] = 0
        df['hour_cos'] = 0
    
    # 9. Day of week (cyclical)
    df['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    
    df = df.fillna(0)
    return df


def train_rl_agent(total_timesteps=300_000):
    """Train RL agent with production-grade settings."""
    model_path = Path("models/rl_deepscalper")
    model_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Fetch Extended Data (12 months)
    df = fetch_extended_data()
    
    # 2. Process and add robust features
    processor = DataProcessor(timezone="Europe/Berlin")
    df = processor.process(df)
    
    eng = FeatureEngineer()
    df = eng.compute_all(df, include_ml_features=True)
    df = add_robust_features(df)
    df = df.dropna()
    
    logger.info(f"Total data: {len(df)} bars (~{len(df)//24} days)")
    
    # 3. TRAIN/TEST SPLIT (80/20)
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    
    logger.info(f"Train: {len(df_train)} bars | Test: {len(df_test)} bars")
    
    # 4. Create Training Environment
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
    
    # 5. Create Test Environment
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
    
    # 6. Create PPO Agent (Production Settings)
    model = PPO(
        "MlpPolicy", 
        train_env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./logs/rl_tensorboard/",
        device="auto"
    )
    
    # 7. Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=25000, 
        save_path=str(model_path),
        name_prefix='deepscalper_ppo'
    )
    
    early_stopping = EarlyStoppingCallback(
        threshold=0.6,
        patience=10,
        verbose=1
    )
    
    eval_callback = EvalCallback(
        test_env,
        best_model_save_path=str(model_path / "best"),
        log_path=str(model_path / "eval_logs"),
        eval_freq=25000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1
    )
    
    # 8. Train
    logger.info(f"Starting PRODUCTION Training: {total_timesteps} steps")
    logger.info(f"Data: ~{len(df_train)//24} days training, ~{len(df_test)//24} days testing")
    logger.info("Target: ep_rew_mean > 40 on TEST data for doubling potential")
    logger.info("Fallback: London Breakout + Asian Scalp always available")
    
    try:
        model.learn(
            total_timesteps=total_timesteps, 
            callback=[checkpoint_callback, early_stopping, eval_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    
    # 9. Save Final Model
    model.save(model_path / "final_model")
    train_env.save(str(model_path / "vec_normalize.pkl"))
    
    # 10. Final Evaluation
    logger.info("\n" + "="*50)
    logger.info("FINAL EVALUATION ON UNSEEN TEST DATA")
    logger.info("="*50)
    
    test_env.obs_rms = train_env.obs_rms
    test_env.ret_rms = train_env.ret_rms
    
    mean_reward, std_reward = evaluate_policy(
        model, test_env, n_eval_episodes=10, deterministic=True
    )
    
    logger.info(f"Test Performance: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    if mean_reward > 40:
        logger.info("✅ TARGET MET! Model ready for live trading")
    elif mean_reward > 20:
        logger.info("⚠️ Moderate. Use with rule-based fallbacks.")
    elif mean_reward > 0:
        logger.info("⚠️ Marginal. Enable rule-based as primary.")
    else:
        logger.warning("❌ Use rule-based strategies only (London Breakout, Asian Scalp).")
    
    logger.info(f"\nBest model: {model_path / 'best'}")
    logger.info(f"Final model: {model_path / 'final_model.zip'}")


if __name__ == "__main__":
    train_rl_agent()
