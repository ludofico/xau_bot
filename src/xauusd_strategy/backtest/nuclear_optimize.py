
import logging
from xauusd_strategy.data.fetcher import fetch_xauusd_data
from xauusd_strategy.strategy.london_breakout import LondonBreakoutStrategy
from xauusd_strategy.backtest.engine import BacktestEngine

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    
    import datetime
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=60)
    
    # 1. Fetch Data
    df = fetch_xauusd_data(start=start_date, end=end_date)
    
    # 2. Strategy: Verified Profitable "Mean Reversion"
    # Includes new NY Session Logic automatically
    strategy = LondonBreakoutStrategy(
        sl_atr_mult=1.2,
        tp_atr_mult=3.0,
        roc_threshold=0.15,
        atr_min_multiplier=0.5
    )
    
    # 3. Nuclear Engine: 5% Risk, 500x Leverage
    print("\n⚠️ IGNITION: STARTING NUCLEAR BACKTEST ⚠️")
    print("Target: €1000/month (400% ROI)")
    print("Risk: 5.0% per trade | Strategy: Double Session Fade (London + NY)")
    print("-" * 50)
    
    engine = BacktestEngine(
        initial_balance=250,
        risk_pct=5.0, # NUCLEAR RISK
        leverage=500,
        use_compounding=True
    )
    
    # 4. Run
    result = engine.run(df, strategy)
    engine.print_summary(result)

if __name__ == "__main__":
    main()
