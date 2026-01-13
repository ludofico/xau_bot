import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from xauusd_strategy.data.fetcher import DataFetcher
from xauusd_strategy.backtest.engine import BacktestEngine
from xauusd_strategy.strategy.momentum_stacker import MomentumStackerStrategy
from xauusd_strategy.utils.logger import get_logger

logger = get_logger("StackerOpt")

def main():
    # 1. Fetch High Res Data (Recent 60 days)
    # Using '5m' to simulate high frequency relative to 1H
    fetcher = DataFetcher()
    data = fetcher.fetch_recent(days=60, timeframe="5m")
    
    if data is None or data.empty:
        logger.error("No data fetched")
        return

    logger.info("⚠️ STARTING MOMENTUM STACKER BACKTEST ⚠️")
    
    # 2. Configure aggressive engine
    engine = BacktestEngine(
        initial_balance=250,
        leverage=500,
        risk_pct=1.0, # Lower risk per trade because we STACK many trades
        use_compounding=True
    )
    
    # 3. Run Strategy
    strategy = MomentumStackerStrategy(
        sl_atr_mult=1.5,
        tp_atr_mult=5.0 # Let winners run hard
    )
    
    result = engine.run(data, strategy)
    
    # 4. Print
    engine.print_summary(result)

if __name__ == "__main__":
    main()
