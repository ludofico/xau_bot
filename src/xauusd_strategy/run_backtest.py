#!/usr/bin/env python3
"""
Main entry point for running XAUUSD backtests.

Usage:
    python -m xauusd_strategy.run_backtest --config config/aggressive.yaml
    python -m xauusd_strategy.run_backtest --start 2024-01-01 --end 2024-12-31
    python -m xauusd_strategy.run_backtest --ml-model models/ml_filter.pkl
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys
import pandas as pd
import numpy as np

from xauusd_strategy.config.settings import Settings
from xauusd_strategy.data.fetcher import DataFetcher
from xauusd_strategy.data.processor import DataProcessor
from xauusd_strategy.data.features import FeatureEngineer
from xauusd_strategy.strategy.london_breakout import LondonBreakoutStrategy
from xauusd_strategy.backtest.engine import BacktestEngine
from xauusd_strategy.backtest.costs import CostModel
from xauusd_strategy.risk.compound_manager import AggressiveCompoundManager
from xauusd_strategy.ml.model import MLProbabilityFilter
from xauusd_strategy.ml.features import MLFeatureEngineer
from xauusd_strategy.utils.logger import setup_logger, get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run XAUUSD London Breakout backtest"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file"
    )
    
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--source",
        type=str,
        choices=["yfinance", "mt5", "csv"],
        default="yfinance",
        help="Data source"
    )
    
    parser.add_argument(
        "--timeframe",
        type=str,
        default="5m",
        help="Candle timeframe (5m, 15m, 1h)"
    )
    
    parser.add_argument(
        "--balance",
        type=float,
        default=250,
        help="Initial balance in EUR"
    )
    
    parser.add_argument(
        "--risk",
        type=float,
        default=2.5,
        help="Risk per trade (%)"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["conservative", "aggressive", "ultra"],
        default="aggressive",
        help="Risk mode preset"
    )
    
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run walk-forward validation"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)"
    )
    
    parser.add_argument(
        "--ml-model",
        type=str,
        default=None,
        help="Path to trained ML filter model (.pkl)"
    )
    
    parser.add_argument(
        "--ml-threshold",
        type=float,
        default=0.55,
        help="ML probability threshold (default: 0.55)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logger(log_level=log_level)
    logger = get_logger(__name__)
    
    logger.info("=" * 60)
    logger.info("XAUUSD London Breakout Strategy - Backtest")
    logger.info("=" * 60)
    
    # Load settings
    if args.config:
        settings = Settings.from_yaml(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        if args.mode == "conservative":
            settings = Settings.conservative()
        elif args.mode == "ultra":
            settings = Settings.ultra_aggressive()
        else:
            settings = Settings.aggressive()
        logger.info(f"Using {args.mode} preset")
    
    # Override with command line args
    if args.balance:
        settings.account.initial_balance = args.balance
    if args.risk:
        settings.risk.risk_per_trade_pct = args.risk
    
    # Determine date range
    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
    else:
        end_date = datetime.now()
    
    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
    else:
        # Default to last 60 days (yfinance limit for intraday)
        start_date = end_date - timedelta(days=60)
    
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Initial balance: €{settings.account.initial_balance}")
    logger.info(f"Risk per trade: {settings.risk.risk_per_trade_pct}%")
    
    # Fetch data
    logger.info(f"Fetching data from {args.source}...")
    fetcher = DataFetcher(source=args.source)
    
    try:
        raw_data = fetcher.fetch(start_date, end_date, args.timeframe)
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        sys.exit(1)
    
    if raw_data.empty:
        logger.error("No data fetched")
        sys.exit(1)
    
    logger.info(f"Fetched {len(raw_data)} candles")
    
    # Process data
    processor = DataProcessor()
    data = processor.process(raw_data)
    logger.info(f"Processed data: {len(data)} candles")
    
    # Add features (use ML feature engineer if ML model is used)
    if args.ml_model:
        feature_eng = MLFeatureEngineer()
        data = feature_eng.prepare_ml_features(data)
    else:
        feature_eng = FeatureEngineer()
        data = feature_eng.compute_all(data)
    
    # Load ML filter if specified
    ml_probabilities = None
    if args.ml_model:
        ml_model_path = Path(args.ml_model)
        if ml_model_path.exists():
            logger.info(f"Loading ML filter from {ml_model_path}")
            ml_filter = MLProbabilityFilter(probability_threshold=args.ml_threshold)
            ml_filter.load(ml_model_path)
            
            # Generate probabilities for all bars
            feature_cols = [c for c in data.columns if c in MLProbabilityFilter.FEATURE_COLUMNS or 
                           c.startswith('atr') or c.startswith('roc') or c.startswith('ema') or
                           c.startswith('volatility') or c in ['hour', 'day_of_week', 'is_london', 'is_overlap',
                                                                'rsi_14', 'bb_position', 'asian_range']]
            X_all = data[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
            probabilities = ml_filter.predict(X_all)
            ml_probabilities = pd.Series(probabilities, index=data.index)
            logger.info(f"ML filter loaded: threshold={args.ml_threshold}")
        else:
            logger.warning(f"ML model file not found: {ml_model_path}")
    
    # Create strategy
    strategy = LondonBreakoutStrategy(settings=settings)
    logger.info(f"Strategy params: {strategy.get_strategy_params()}")
    
    # Create cost model
    cost_model = CostModel.from_broker_type(settings.account.broker_type)
    logger.info(f"Cost model: {cost_model}")
    
    # Create backtest engine
    engine = BacktestEngine(
        initial_balance=settings.account.initial_balance,
        leverage=settings.account.leverage,
        cost_model=cost_model,
        use_compounding=True,
        risk_pct=settings.risk.risk_per_trade_pct,
        max_risk_pct=settings.risk.max_risk_per_trade_pct
    )
    
    # Run backtest
    if args.walk_forward:
        logger.info("Running walk-forward validation...")
        results = engine.run_walk_forward(data, strategy)
        
        # Aggregate results
        total_return = sum(r.total_return_pct for r in results) / len(results)
        avg_sharpe = sum(r.sharpe_ratio for r in results) / len(results)
        avg_win_rate = sum(r.win_rate_pct for r in results) / len(results)
        
        print("\n" + "=" * 60)
        print("WALK-FORWARD VALIDATION RESULTS")
        print("=" * 60)
        print(f"Folds: {len(results)}")
        print(f"Avg Return per Fold: {total_return:.1f}%")
        print(f"Avg Sharpe Ratio: {avg_sharpe:.2f}")
        print(f"Avg Win Rate: {avg_win_rate:.1f}%")
        
        for i, r in enumerate(results, 1):
            print(f"\nFold {i}: Return={r.total_return_pct:.1f}%, "
                  f"Trades={r.total_trades}, DD={r.max_drawdown_pct:.1f}%")
        print("=" * 60)
        
    else:
        logger.info("Running backtest..." + (" (with ML filter)" if ml_probabilities is not None else ""))
        result = engine.run(data, strategy, ml_probabilities=ml_probabilities)
        
        # Print results
        engine.print_summary(result)
        
        # Save results if output specified
        if args.output:
            import json
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            
            logger.info(f"Results saved to {output_path}")
        
        # Return result for programmatic use
        return result


if __name__ == "__main__":
    main()
