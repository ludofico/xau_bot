
"""
Data generator for ML training.

Generates dataset by:
1. Fetching historical data
2. Generating strategy signals (all potential entries)
3. Labeling them based on future outcome (Win/Loss)
4. extracting features
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from xauusd_strategy.data.fetcher import DataFetcher
from xauusd_strategy.data.processor import DataProcessor
from xauusd_strategy.ml.features import MLFeatureEngineer
from xauusd_strategy.strategy.london_breakout import LondonBreakoutStrategy
from xauusd_strategy.utils.logger import get_logger

logger = get_logger(__name__)

class DatasetGenerator:
    def __init__(self, data_path: str = "data/raw"):
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
    def generate_dataset(
        self,
        days: int = 365,
        output_file: str = "data/training_data.csv"
    ):
        """Generate labeled dataset for training."""
        logger.info(f"Generating dataset for last {days} days...")
        
        # 1. Fetch Data
        # ideally use MT5 for tick precision, but yfinance for length
        fetcher = DataFetcher(source='yfinance') 
        end = datetime.now()
        start = end - timedelta(days=days)
        
        # We need 1m data for precise labeling, or 5m for features
        # yfinance limit: 60 days for 5m. For backtesting longer, we need hourly or external data
        # For this prototype, we'll use what we can get (60 days 5m)
        df_raw = fetcher.fetch(start, end, "5m")
        
        # 2. Process
        processor = DataProcessor(timezone="Europe/Berlin")
        df = processor.process(df_raw)
        
        # 3. Add Features
        eng = MLFeatureEngineer()
        df = eng.prepare_ml_features(df)
        
        # 4. Generate Signals (Loose filter to get MANY examples)
        # We want to train the model to distinguish good/bad
        # So we temporarily RELAX the strict filters in the strategy
        strategy = LondonBreakoutStrategy()
        
        # Override params to get more candidates
        strategy.atr_min_multiplier = 0.1 # Very loose
        strategy.roc_threshold = 0.05    # Very loose
        
        # We need to manually iterate to label signals
        signals_data = []
        
        df_prepared = strategy.prepare_data(df)
        
        logger.info("Labeling signals...")
        
        # Vectorized signal scanning would be faster, but let's loop for precision labeling
        for i in range(100, len(df_prepared) - 100): # Buffer for rolling & future lookahead
            
            # Check for potential setup
            signal = strategy.generate_signal(df_prepared, i, ml_probability=1.0) # Force pass ml check
            
            if signal:
                # 5. Label Outcome
                # We need to see if it hit TP before SL
                label, outcome_data = self._label_signal(
                    signal, 
                    df_prepared.iloc[i+1:] # Future data
                )
                
                # Check for "Partial Win" or "Breakeven" logic?
                # For classification 1=Win (TP or >1R), 0=Loss
                
                # Extract Features at time i
                row = df_prepared.iloc[i].to_dict()
                row['target'] = label
                row['signal_type'] = signal.signal_type.name
                row['pnl_r'] = outcome_data['pnl_r']
                
                signals_data.append(row)
        
        if not signals_data:
            logger.warning("No signals generated! Check strategy logic.")
            return
            
        # 6. Save
        df_train = pd.DataFrame(signals_data)
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_train.to_csv(out_path, index=False)
        
        logger.info(f"Saved {len(df_train)} labeled samples to {out_path}")
        logger.info(f"Win Rate in dataset: {df_train['target'].mean():.2%}")
        
    def _label_signal(self, signal, future_df: pd.DataFrame) -> tuple:
        """
        Determine if signal was a winner.
        Returns (1/0, details_dict)
        """
        entry = signal.entry_price
        sl = signal.stop_loss
        tp = signal.take_profit
        
        is_long = (signal.signal_type.name == 'LONG')
        
        for i in range(len(future_df)):
            high = future_df.iloc[i]['high']
            low = future_df.iloc[i]['low']
            
            # Check Stop Loss first (conservative)
            if is_long:
                if low <= sl:
                    return 0, {'reason': 'sl', 'pnl_r': -1.0}
                if high >= tp:
                    return 1, {'reason': 'tp', 'pnl_r': 2.0} # Assuming 1:2
            else:
                if high >= sl:
                    return 0, {'reason': 'sl', 'pnl_r': -1.0}
                if low <= tp:
                    return 1, {'reason': 'tp', 'pnl_r': 2.0}
            
            # Timeout (End of day?)
            # If trade lasts > 8 hours, close it?
            if i > 96: # 8 hours * 12 (5m bars)
                current_close = future_df.iloc[i]['close']
                # Calculate PnL R
                risk = abs(entry - sl)
                if risk == 0: return 0, {}
                
                if is_long:
                    pnl = current_close - entry
                else: 
                    pnl = entry - current_close
                    
                r_multiple = pnl / risk
                
                # Treat > 0.5R as a "Win" for training purposes? 
                # Or adhere strictly to TP? 
                # Let's say > 0 is a win for now (Direction was right)
                return (1 if r_multiple > 0 else 0), {'reason': 'time', 'pnl_r': r_multiple}
                
        return 0, {'reason': 'end_of_data', 'pnl_r': 0}

if __name__ == "__main__":
    gen = DatasetGenerator()
    gen.generate_dataset(days=59) # Max for yfinance 5m
