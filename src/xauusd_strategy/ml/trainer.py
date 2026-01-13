
"""
ML Trainer Script.

Trains XGBoost model on generated dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from xauusd_strategy.ml.model import MLProbabilityFilter
from xauusd_strategy.utils.logger import get_logger

logger = get_logger(__name__)

def train_model():
    data_path = Path("data/training_data.csv")
    model_path = Path("models/ml_filter_doubler.pkl")
    
    if not data_path.exists():
        logger.error("Dataset not found! Run generator.py first.")
        return
        
    logger.info("Loading dataset...")
    df = pd.read_csv(data_path)
    
    # Check balance
    win_rate = df['target'].mean()
    logger.info(f"Dataset Size: {len(df)} samples")
    logger.info(f"Class Balance (Win Rate): {win_rate:.2%}")
    
    feature_cols = [c for c in df.columns if c not in [
        'target', 'timestamp', 'signal_type', 'pnl_r', 
        'entry_price', 'stop_loss', 'take_profit'
    ]]
    
    logger.info(f"Training on {len(feature_cols)} features...")
    
    # Train
    ml = MLProbabilityFilter(probability_threshold=0.60) # High conviction
    metrics = ml.train(df, df['target'])
    
    logger.info("--- Training metrics ---")
    logger.info(metrics)
    
    # Save
    ml.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
if __name__ == "__main__":
    train_model()
