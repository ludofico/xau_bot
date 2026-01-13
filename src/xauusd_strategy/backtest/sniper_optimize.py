import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from xauusd_strategy.data.fetcher import DataFetcher
from xauusd_strategy.backtest.engine import BacktestEngine
from xauusd_strategy.strategy.london_breakout import LondonBreakoutStrategy
from xauusd_strategy.utils.logger import get_logger

logger = get_logger("SniperOpt")

def main():
    # 1. Fetch Data
    fetcher = DataFetcher()
    data = fetcher.fetch_recent(days=60, timeframe="5m")
    
    if data is None or data.empty:
        logger.error("No data fetched")
        return

    logger.info("🎯 STARTING SNIPER BACKTEST (High Risk Mean Reversion) 🎯")
    
    # 2. Configure Sniper Engine (High Leverage, High Risk)
    engine = BacktestEngine(
        initial_balance=250,
        leverage=500,
        risk_pct=8.0,      # <--- AGGRESSIVE RISK (8% per trade)
        max_risk_pct=15.0,
        use_compounding=True
    )
    
    # 3. Strategy: The verified Mean Reversion logic
    # But letting winners run longer (TP 4.0)
    strategy = LondonBreakoutStrategy(
        atr_period=14,
        roc_period=5,
        sl_atr_mult=1.0,  # Tighter Stop
        tp_atr_mult=4.0,  # Home Run Target
        ml_probability_threshold=0.55 # Only high prob
    )
    
    result = engine.run(data, strategy)
    
    # 4. Print
    engine.print_summary(result)

if __name__ == "__main__":
    main()
