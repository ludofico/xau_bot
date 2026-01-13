
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import sys
from pathlib import Path
import itertools

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from xauusd_strategy.config.settings import Settings
from xauusd_strategy.data.fetcher import DataFetcher
from xauusd_strategy.data.processor import DataProcessor
from xauusd_strategy.strategy.london_breakout import LondonBreakoutStrategy
from xauusd_strategy.ml.model import MLProbabilityFilter
from xauusd_strategy.ml.features import MLFeatureEngineer

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

class PyramidSimulator:
    def __init__(self, step_dollars: float, max_layers: int, lot_size: float = 0.03, initial_sl: float = 2.5):
        self.step = step_dollars
        self.max_layers = max_layers
        self.lot_size = lot_size
        self.sl_buffer = 0.5
        self.initial_sl_dist = initial_sl
        
        self.balance = 250.0
        self.max_balance = 250.0
        self.max_exclude_balance = 250.0 # Excursion

    def run(self, df: pd.DataFrame, signals: pd.Series) -> Dict:
        state = PyramidState()
        trades = []
        equity_curve = []
        
        for i in range(len(df)):
            if i == 0: continue
            
            bar = df.iloc[i]
            timestamp = df.index[i]
            current_price = bar['close']
            
            # --- MANAGING POSITIONS ---
            if len(state.positions) > 0:
                # Check SL
                hit_sl = False
                if state.direction == 1:
                    if bar['low'] <= state.sl_price: hit_sl = True
                else:
                    if bar['high'] >= state.sl_price: hit_sl = True
                
                if hit_sl:
                    pnl_total = 0
                    for pos in state.positions:
                        diff = (state.sl_price - pos.entry_price) if state.direction == 1 else (pos.entry_price - state.sl_price)
                        pnl = (diff * pos.size * 100) - (7.0 * pos.size) # Commission
                        pnl_total += pnl
                    
                    self.balance += pnl_total
                    trades.append(pnl_total)
                    state = PyramidState() # Reset
                
                # Check Scaling
                elif len(state.positions) < self.max_layers:
                    last_pos = state.positions[-1]
                    added = False
                    
                    if state.direction == 1:
                        if current_price >= last_pos.entry_price + self.step:
                            new_pos = Position(current_price, self.lot_size, timestamp)
                            state.positions.append(new_pos)
                            # Collective SL: Previous Entry (Lock Profit)
                            state.sl_price = max(state.sl_price, current_price - self.step + self.sl_buffer)
                            added = True
                    elif state.direction == -1:
                        if current_price <= last_pos.entry_price - self.step:
                            new_pos = Position(current_price, self.lot_size, timestamp)
                            state.positions.append(new_pos)
                            state.sl_price = min(state.sl_price, current_price + self.step - self.sl_buffer)
                            added = True
            
            # --- CHECK ENTRIES ---
            if len(state.positions) == 0:
                signal = signals.get(timestamp, 0)
                if signal != 0:
                    state.direction = signal
                    state.positions.append(Position(current_price, self.lot_size, timestamp))
                    if signal == 1:
                        state.sl_price = current_price - self.initial_sl_dist
                    else:
                        state.sl_price = current_price + self.initial_sl_dist
            
            # Track Stats
            if self.balance > self.max_balance: self.max_balance = self.balance
            
            # Ruin check
            if self.balance < 50: # Bust
                break
        
        total_ret = (self.balance - 250) / 250 * 100
        dd = (self.max_balance - self.balance) / self.max_balance * 100 if self.max_balance > 0 else 0
        
        return {
            "Final Balance": self.balance,
            "Return %": total_ret,
            "Drawdown %": dd,
            "Trades": len(trades),
            "Trade List": trades
        }

def run_scenarios():
    print("Loading Data & Model...")
    fetcher = DataFetcher(source="yfinance")
    data = fetcher.fetch_recent(days=60, timeframe="5m")
    processor = DataProcessor()
    data = processor.process(data)
    
    # Feature Engineering
    eng = MLFeatureEngineer()
    f_df = eng.prepare_ml_features(data)
    
    # ML Model
    ml_filter = MLProbabilityFilter()
    try:
        ml_filter.load(Path("models/ml_filter_doubler.pkl"))
        probs = ml_filter.predict(f_df)
        probs_series = pd.Series(probs, index=data.index)
        print("Model Loaded.")
    except:
        print("Model FAIL. Using Random.")
        probs_series = pd.Series(0.5, index=data.index)

    strategy = LondonBreakoutStrategy()
    df_ind = strategy.prepare_data(data)
    
    # Scenarios
    thresholds = [0.55, 0.60, 0.65, 0.70]
    steps = [2.0, 2.5, 3.0]
    layers = [3, 4, 5]
    
    results = []
    
    for th in thresholds:
        # Pre-calculate signals for this threshold
        signals = pd.Series(0, index=data.index)
        
        for i in range(50, len(data)):
             # Re-use logical checks from strategy
             # Simplified for speed: If ML > TH -> Check basic trend
             prob = probs_series.iloc[i]
             if prob > th:
                 # Generate signal
                 # Hacky reuse of strategy object logic requires loops which is slow, just doing simple logic here
                 # London Session?
                 t = data.index[i].time()
                 if t >= pd.Timestamp("08:00").time() and t <= pd.Timestamp("16:00").time():
                     bar = data.iloc[i]
                     # Long
                     if bar['close'] > data['high'].iloc[i-10:i].max():
                         signals.iloc[i] = 1
                     elif bar['close'] < data['low'].iloc[i-10:i].min():
                         signals.iloc[i] = -1

        for step, max_l in itertools.product(steps, layers):
            sim = PyramidSimulator(step_dollars=step, max_layers=max_l)
            res = sim.run(data, signals)
            
            res.update({
                "Threshold": th,
                "Step": step,
                "Layers": max_l
            })
            results.append(res)
            print(f"Scen: TH={th} Step={step} L={max_l} -> Bal=${res['Final Balance']:.2f}")

    # Summary
    res_df = pd.DataFrame(results)
    print("\n--- TOP 5 PERFORMANCES ---")
    print(res_df.sort_values("Final Balance", ascending=False).head(5)[["Threshold", "Step", "Layers", "Final Balance", "Return %", "Drawdown %"]])

if __name__ == "__main__":
    run_scenarios()
