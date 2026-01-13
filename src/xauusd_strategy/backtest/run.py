
import argparse
from datetime import datetime, timedelta
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from xauusd_strategy.config.settings import Settings
from xauusd_strategy.data.fetcher import fetch_xauusd_data
from xauusd_strategy.backtest.engine import BacktestEngine
from xauusd_strategy.strategy.london_breakout import LondonBreakoutStrategy
from xauusd_strategy.utils.logger import get_logger

logger = get_logger("BacktestRunner")

def main():
    parser = argparse.ArgumentParser(description="Run XAUUSD Strategy Backtest")
    parser.add_argument("--days", type=int, default=30, help="Days of history to backtest (default: 30)")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--ml", action="store_true", help="Enable ML Probability Filter")
    parser.add_argument("--balance", type=float, default=250.0, help="Initial Balance (EUR)")
    parser.add_argument("--config", type=str, default="aggressive", help="Config mode (aggressive/conservative)")
    
    args = parser.parse_args()
    
    # 1. Setup Timeframe
    if args.start and args.end:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
    else:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
    
    logger.info(f"Backtesting {args.config.upper()} Strategy")
    logger.info(f"Period: {start_date.date()} to {end_date.date()}")
    
    # 2. Fetch Data
    logger.info("Fetching Market Data...")
    try:
        # Try yfinance first for backtesting as it is free and available
        df = fetch_xauusd_data(start_date, end_date, timeframe="5m", source="yfinance")
        if df.empty:
            logger.error("No data fetched from yfinance. Trying CSV if available or Fail.")
            return
    except Exception as e:
        logger.error(f"Data Fetch Error: {e}")
        return

    # 3. Setup ML (Optional)
    ml_probs = None
    if args.ml:
        logger.info("Loading ML Model...")
        try:
            from xauusd_strategy.ml.model import MLProbabilityFilter
            from xauusd_strategy.ml.features import MLFeatureEngineer
            
            model_path = Path("models/ml_filter_doubler.pkl")
            if model_path.exists():
                model = MLProbabilityFilter(model_path=model_path)
            else:
                logger.warning(f"Model not found at {model_path}, initializing new model")
                model = MLProbabilityFilter()
            
            eng = MLFeatureEngineer()
            
            # df_features = eng.compute_all(df, include_ml_features=True) # Redundant
            df_features = eng.prepare_ml_features(df) # Ensure full ML set
            
            logger.info("Predicting probabilities...")
            ml_probs = pd.Series(model.predict(df_features), index=df.index)
            logger.info("ML Predictions generated.")
            
        except ImportError:
            logger.error("ML modules not found. Running without ML.")
        except Exception as e:
            logger.error(f"ML Error: {e}. Running without ML.")
    
    # 4. Initialize Strategy
    # Using default aggressive settings hardcoded for now or loaded from Settings if needed
    # For now, relying on Strategy class defaults which mirror aggressive.yaml
    strategy = LondonBreakoutStrategy()
    
    # 5. Run Backtest
    engine = BacktestEngine(initial_balance=args.balance)
    result = engine.run(df, strategy, ml_probabilities=ml_probs)
    
    # 6. Print Summary
    engine.print_summary(result)

if __name__ == "__main__":
    main()
