import sys
import os
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from xauusd_strategy.data.fetcher import DataFetcher
from xauusd_strategy.backtest.engine import BacktestEngine, BacktestResult
from xauusd_strategy.strategy.london_breakout import LondonBreakoutStrategy
from xauusd_strategy.utils.logger import get_logger

logger = get_logger("AntiMartingale")

class AntiMartingaleEngine(BacktestEngine):
    def _simulate_trades(self, data, signals):
        trades = []
        balance = self.initial_balance
        base_risk_pct = 2.0
        current_risk_pct = base_risk_pct
        streak = 0
        
        for signal in signals:
            if signal.timestamp is None: continue
            entry_idx = data.index.get_loc(signal.timestamp)
            if entry_idx >= len(data) - 1: continue
            
            # Anti-Martingale Sizing
            # Streak 0: 2%
            # Streak 1: 4%
            # Streak 2: 8%
            # Streak 3+: 10% (Capped)
            if streak == 0: current_risk_pct = 2.0
            elif streak == 1: current_risk_pct = 4.0
            elif streak == 2: current_risk_pct = 8.0
            else: current_risk_pct = 10.0
            
            risk_amount = balance * (current_risk_pct / 100)
            stop_distance = abs(signal.entry_price - signal.stop_loss)
            if stop_distance == 0: continue
            
            lots = risk_amount * 1.08 / (stop_distance * 100)
            lots = max(0.01, min(lots, 10.0))
            
            # Entry Cost
            entry_price = self.cost_model.apply_entry_cost(signal.entry_price, signal.signal_type.name == 'LONG')
            
            # Exit matches engine logic...
            # (Simplified for brevity, assuming standard engine exit logic is reusable if we could override just sizing)
            # Since we can't easily inject sizing into parent method, we copy-paste the core loop logic here or just modify risk_pct dynamically if possible.
            # Actually, `run` calls `_simulate_trades`, so overriding this IS the way.
            
            # ... Copying Exit Logic ...
            exit_price, exit_reason, exit_time = self._simulate_trade_exit(data, entry_idx, signal, entry_price)
            
            if exit_price is None: continue
            
            # PnL Calc
            pnl_points = (exit_price - entry_price) if signal.signal_type.name == 'LONG' else (entry_price - exit_price)
            pnl_usd = pnl_points * lots * 100
            commission = self.cost_model.commission_per_lot * lots
            pnl_usd -= commission
            pnl_eur = pnl_usd / 1.08
            
            balance += pnl_eur
            
            # Streak Logic
            if pnl_eur > 0:
                streak += 1
            else:
                streak = 0
                
            trades.append({
                'entry_time': signal.timestamp,
                'exit_time': exit_time, # Added missing field
                'direction': signal.signal_type.name,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'lots': lots,
                'pnl': pnl_eur,
                'pnl_pct': pnl_eur / balance * 100,
                'balance': balance,
                'streak': streak,
                'risk_used': current_risk_pct
            })
            
        return pd.DataFrame(trades)

def main():
    fetcher = DataFetcher()
    data = fetcher.fetch_recent(days=60, timeframe="5m")
    
    logger.info("🚀 STARTING ANTI-MARTINGALE BACKTEST 🚀")
    
    engine = AntiMartingaleEngine(initial_balance=250)
    strategy = LondonBreakoutStrategy(
        sl_atr_mult=1.2, # Verified
        tp_atr_mult=3.0  # Verified
    )
    
    result = engine.run(data, strategy)
    engine.print_summary(result)

if __name__ == "__main__":
    main()
