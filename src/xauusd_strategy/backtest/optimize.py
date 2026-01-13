
import itertools
import pandas as pd
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from xauusd_strategy.data.fetcher import fetch_xauusd_data
from xauusd_strategy.backtest.engine import BacktestEngine
from xauusd_strategy.strategy.london_breakout import LondonBreakoutStrategy
from xauusd_strategy.utils.logger import get_logger

logger = get_logger("Optimizer")

def run_single_backtest(args):
    """Run a single backtest with specific parameters."""
    params, df = args
    
    # Initialize strategy with these params
    strategy = LondonBreakoutStrategy(
        atr_period=14,
        roc_period=5,
        roc_threshold=params['roc'],
        sl_atr_mult=params['sl'],
        tp_atr_mult=params['tp'],
        atr_min_multiplier=params['atr_mult'] # Volatility filter
    )
    
    engine = BacktestEngine(initial_balance=250.0)
    result = engine.run(df, strategy)
    
    return {
        'params': params,
        'return_pct': result.total_return_pct,
        'max_dd': result.max_drawdown_pct,
        'sharpe': result.sharpe_ratio,
        'trades': result.total_trades,
        'win_rate': result.win_rate_pct,
        'final_balance': result.final_balance
    }

def main():
    logger.info("Starting Grid Search Optimization")
    
    # 1. Fetch Data
    from datetime import datetime, timedelta
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    
    logger.info("Fetching Data...")
    df = fetch_xauusd_data(start_date, end_date, timeframe="5m", source="yfinance")
    
    if df.empty:
        logger.error("No data fetched.")
        return

    # 2. Define Parameter Grid
    # Focused on finding a profitable baseline
    param_grid = {
        'sl': [1.0, 1.2, 1.5],          # Stop Loss (ATR multiplier)
        'tp': [2.0, 2.5, 3.0],          # Take Profit (ATR multiplier)
        'roc': [0.1, 0.15],             # Momentum threshold
        'atr_mult': [0.5]               # Volatility filter (Fixed)
    }
    
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    logger.info(f"Testing {len(combinations)} combinations...")
    
    # 3. Run Optimization (Parallel)
    # Prepare args for map
    # Note: DataFrame copying might be heavy for multiprocessing, but with 60 days on M1/M2 it's fine.
    # For larger datasets, use a shared memory approach or process pool with data loaded in workers.
    # Here we'll just run plain map or simple loop if simpler. 
    # Let's try simple loop first to avoid pickling issues with DataFrames in some envs, 
    # or just pass the data if small enough.
    
    results = []
    total = len(combinations)
    
    for i, params in enumerate(combinations):
        if i % 10 == 0:
            print(f"Processing {i}/{total}...", end='\r')
        
        # Determine R:R requirement (Trade needs to justify risk)
        if params['tp'] < params['sl']: 
            continue # Skip bad R:R
            
        res = run_single_backtest((params, df))
        results.append(res)
        
    print(f"Processing {total}/{total} - Done.")
    
    # 4. Analyze Results
    results_df = pd.DataFrame(results)
    
    # Filter for minimum number of trades to be statistically relevant (e.g., > 10)
    # And positive return
    valid_results = results_df[results_df['trades'] > 5].copy()
    
    if valid_results.empty:
        logger.warning("No profitable configurations found with > 5 trades!")
        return

    # Sort by Net Profit (Final Balance)
    top_results = valid_results.sort_values(by='final_balance', ascending=False).head(10)
    
    print("\n" + "="*80)
    print("TOP 10 CONFIGURATIONS (Base Strategy)")
    print("="*80)
    print(f"{'SL':<6} {'TP':<6} {'ROC':<6} {'ATR_M':<6} | {'Return':<8} {'DD%':<6} {'Trades':<6} {'Win%':<6} {'Balance':<8}")
    print("-" * 80)
    
    for _, row in top_results.iterrows():
        p = row['params']
        print(f"{p['sl']:<6.1f} {p['tp']:<6.1f} {p['roc']:<6.2f} {p['atr_mult']:<6.1f} | "
              f"{row['return_pct']:<7.1f}% {row['max_dd']:<6.1f} {row['trades']:<6} {row['win_rate']:<6.1f} €{row['final_balance']:.0f}")
    print("="*80)

if __name__ == "__main__":
    main()
