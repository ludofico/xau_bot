
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from xauusd_strategy.config.settings import Settings
from xauusd_strategy.data.fetcher import DataFetcher
from xauusd_strategy.data.processor import DataProcessor
from xauusd_strategy.data.features import FeatureEngineer
from xauusd_strategy.strategy.london_breakout import LondonBreakoutStrategy
from xauusd_strategy.ml.model import MLProbabilityFilter

@dataclass
class Position:
    entry_price: float
    size: float
    open_time: pd.Timestamp

@dataclass
class PyramidState:
    positions: List[Position] = field(default_factory=list)
    sl_price: float = 0.0
    direction: int = 0  # 1 for Buy, -1 for Sell
    peak_price: float = 0.0

class PyramidSimulator:
    def __init__(self, step_dollars: float = 2.5, max_layers: int = 4, lot_size: float = 0.03):
        self.step = step_dollars
        self.max_layers = max_layers
        self.lot_size = lot_size
        self.sl_buffer = 0.5  # $0.50 protection buffer when moving SL
        self.initial_sl_dist = 2.5 # Initial SL distance ($2.50)
        
        self.balance = 250.0
        self.equity_curve = []
        self.trades = []
        
    def run(self, df: pd.DataFrame, signals: pd.Series):
        state = PyramidState()
        
        print(f"Starting balance: ${self.balance:.2f}")
        
        for i in range(len(df)):
            if i == 0: continue
            
            bar = df.iloc[i]
            prev_bar = df.iloc[i-1]
            timestamp = df.index[i]
            current_price = bar['close'] # Simplification using close
            
            # 1. Manage Active Games
            if len(state.positions) > 0:
                # Check for SL hit
                if (state.direction == 1 and bar['low'] <= state.sl_price) or \
                   (state.direction == -1 and bar['high'] >= state.sl_price):
                    
                    # Calculate PnL
                    exit_price = state.sl_price
                    total_pnl = 0
                    for pos in state.positions:
                        if state.direction == 1:
                            pnl = (exit_price - pos.entry_price) * pos.size * 100 # $100 per lot per $1
                        else:
                            pnl = (pos.entry_price - exit_price) * pos.size * 100
                        total_pnl += pnl
                        # Commission ($7/lot)
                        total_pnl -= (7.0 * pos.size)
                    
                    self.balance += total_pnl
                    self.trades.append({
                        'time': timestamp,
                        'type': 'SL' if total_pnl < 0 else 'TP',
                        'pnl': total_pnl,
                        'layers': len(state.positions),
                        'balance': self.balance
                    })
                    
                    # Reset state
                    state = PyramidState()
                
                # Check for Scaling In (Pyramiding)
                elif len(state.positions) < self.max_layers:
                    last_pos = state.positions[-1]
                    
                    if state.direction == 1:
                        # Buy Logic
                        if current_price >= last_pos.entry_price + self.step:
                            # Add Layer
                            new_pos = Position(current_price, self.lot_size, timestamp)
                            state.positions.append(new_pos)
                            
                            # Move SL to breakeven/protection of PREVIOUS positions
                            # Logic: Lock profit. Set collective SL to LastPos Entry (+ buffer?)
                            # User Logic: new_sl = last_pos.entry_open - SL_PROTECTION (Wait, user code logic was tricky)
                            # Let's use: SL = NewEntry - Step + Buffer (Essentially Last Entry)
                            new_sl = current_price - self.step + self.sl_buffer
                            state.sl_price = max(state.sl_price, new_sl)
                            
                    else:
                        # Sell Logic
                        if current_price <= last_pos.entry_price - self.step:
                            # Add Layer
                            new_pos = Position(current_price, self.lot_size, timestamp)
                            state.positions.append(new_pos)
                            
                            new_sl = current_price + self.step - self.sl_buffer
                            state.sl_price = min(state.sl_price, new_sl)

            # 2. Check for New Entries (if flat)
            if len(state.positions) == 0:
                signal = signals.get(timestamp, 0)
                
                if signal != 0:
                    state.direction = signal
                    state.positions.append(Position(current_price, self.lot_size, timestamp))
                    
                    if signal == 1:
                        state.sl_price = current_price - self.initial_sl_dist
                    else:
                        state.sl_price = current_price + self.initial_sl_dist
            
            self.equity_curve.append(self.balance)

        print(f"Final balance: ${self.balance:.2f}")
        return pd.DataFrame(self.trades)

def main():
    # Setup
    settings = Settings.aggressive()
    fetcher = DataFetcher(source="yfinance")
    data = fetcher.fetch_recent(days=60, timeframe="5m")
    
    # Process
    processor = DataProcessor()
    data = processor.process(data)
    
    # ML Filter Load
    try:
        ml_filter = MLProbabilityFilter()
        ml_filter.load(Path("models/ml_filter_doubler.pkl"))
        probs = ml_filter.predict(MLProbabilityFilter().prepare_features(data))
        probs_series = pd.Series(probs, index=data.index)
        print("ML Model loaded successfully")
    except Exception as e:
        print(f"Model not found or error ({e}), skipping ML filter")
        probs_series = pd.Series(0.6, index=data.index) # Default to some probability

    # Generate Basic Signals
    strategy = LondonBreakoutStrategy()
    data = strategy.prepare_data(data)
    
    # generate simplistic signals series
    signals = pd.Series(0, index=data.index)
    
    for i in range(50, len(data)):
        idx = data.index[i]
        bar = data.iloc[i]
        
        # Simple Breakout Logic + ML
        prob = probs_series.iloc[i] if i < len(probs_series) else 0
        
        if prob > 0.55: # Lowered threshold to see pyramid mechanics
            # Check for breakout visual (simplified)
            # Long
            if bar['close'] > data['high'].iloc[i-10:i].max():
                signals.iloc[i] = 1
            # Short
            elif bar['close'] < data['low'].iloc[i-10:i].min():
                signals.iloc[i] = -1

    # Run Simulator
    sim = PyramidSimulator(step_dollars=2.5, max_layers=4, lot_size=0.03)
    trades = sim.run(data, signals)
    
    if not trades.empty:
        print("\n--- Trade Analysis ---")
        print(trades['pnl'].describe())
        print(f"Win Rate: {len(trades[trades['pnl']>0]) / len(trades) * 100:.1f}%")
        print(f"Max Layer Reached: {trades['layers'].max()}")

if __name__ == "__main__":
    main()
