#!/usr/bin/env python3
"""
ML Filter Training and Backtesting Pipeline.

This script:
1. Fetches XAUUSD data
2. Generates signals from London Breakout strategy
3. Simulates trades to create labels (1=hit TP, 0=hit SL)
4. Trains XGBoost/LightGBM model to predict winning trades
5. Runs backtest with ML filter enabled

Usage:
    python -m xauusd_strategy.train_ml_filter --days 60 --output models/ml_filter.pkl
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

from xauusd_strategy.config.settings import Settings
from xauusd_strategy.data.fetcher import DataFetcher
from xauusd_strategy.data.processor import DataProcessor
from xauusd_strategy.data.features import FeatureEngineer
from xauusd_strategy.strategy.london_breakout import LondonBreakoutStrategy, SignalType
from xauusd_strategy.ml.model import MLProbabilityFilter
from xauusd_strategy.ml.features import MLFeatureEngineer
from xauusd_strategy.backtest.engine import BacktestEngine
from xauusd_strategy.backtest.costs import CostModel
from xauusd_strategy.utils.logger import setup_logger, get_logger

logger = get_logger(__name__)


def create_training_labels(
    data: pd.DataFrame,
    signals: list,
    cost_model: CostModel
) -> pd.DataFrame:
    """
    Create training labels from signal outcomes.
    
    For each signal, simulate the trade and label:
    - 1 = Trade hit Take Profit first
    - 0 = Trade hit Stop Loss first
    
    Args:
        data: OHLC DataFrame
        signals: List of TradeSignal objects
        cost_model: Cost model for realistic prices
    
    Returns:
        DataFrame with signal features and labels
    """
    records = []
    
    for signal in signals:
        if signal.timestamp is None or signal.timestamp not in data.index:
            continue
        
        entry_idx = data.index.get_loc(signal.timestamp)
        
        if entry_idx >= len(data) - 10:
            continue  # Need room for trade to play out
        
        # Apply entry cost
        is_long = signal.signal_type == SignalType.LONG
        entry_price = cost_model.apply_entry_cost(signal.entry_price, is_long)
        
        # Simulate trade outcome
        label = 0  # Default: loss
        exit_reason = "unknown"
        bars_held = 0
        max_favorable = 0
        max_adverse = 0
        
        for i in range(entry_idx + 1, min(entry_idx + 200, len(data))):
            bar = data.iloc[i]
            bars_held = i - entry_idx
            
            if is_long:
                # Track max favorable/adverse excursion
                max_favorable = max(max_favorable, bar['high'] - entry_price)
                max_adverse = max(max_adverse, entry_price - bar['low'])
                
                # Check SL hit
                if bar['low'] <= signal.stop_loss:
                    label = 0
                    exit_reason = "stop_loss"
                    break
                
                # Check TP hit
                if bar['high'] >= signal.take_profit:
                    label = 1
                    exit_reason = "take_profit"
                    break
            else:
                max_favorable = max(max_favorable, entry_price - bar['low'])
                max_adverse = max(max_adverse, bar['high'] - entry_price)
                
                if bar['high'] >= signal.stop_loss:
                    label = 0
                    exit_reason = "stop_loss"
                    break
                
                if bar['low'] <= signal.take_profit:
                    label = 1
                    exit_reason = "take_profit"
                    break
        
        # Extract features at signal time
        signal_row = data.loc[signal.timestamp]
        
        record = {
            'timestamp': signal.timestamp,
            'direction': 1 if is_long else -1,
            'label': label,
            'exit_reason': exit_reason,
            'bars_held': bars_held,
            'max_favorable': max_favorable,
            'max_adverse': max_adverse,
            'atr_value': signal.atr_value,
            'roc_value': signal.roc_value,
            'asian_range': signal.asian_high - signal.asian_low,
            'entry_price': entry_price,
        }
        
        # Add all available features
        for col in data.columns:
            if col not in ['open', 'high', 'low', 'close', 'volume']:
                value = signal_row.get(col)
                if value is not None and not (isinstance(value, float) and pd.isna(value)):
                    record[col] = value
        
        records.append(record)
    
    return pd.DataFrame(records)


def train_ml_filter(
    training_data: pd.DataFrame,
    model_type: str = "xgboost",
    probability_threshold: float = 0.55
) -> tuple:
    """
    Train ML filter on labeled data.
    
    Args:
        training_data: DataFrame with features and 'label' column
        model_type: "xgboost" or "lightgbm"
        probability_threshold: Threshold for predictions
    
    Returns:
        Tuple of (trained filter, training metrics)
    """
    logger.info(f"Training ML filter on {len(training_data)} samples")
    
    # Prepare features (exclude non-feature columns)
    exclude_cols = ['timestamp', 'label', 'exit_reason', 'bars_held', 
                    'max_favorable', 'max_adverse', 'entry_price', 'direction']
    
    feature_cols = [c for c in training_data.columns if c not in exclude_cols]
    
    X = training_data[feature_cols].copy()
    y = training_data['label']
    
    # Handle missing values
    X = X.fillna(0)
    
    # Replace infinities
    X = X.replace([np.inf, -np.inf], 0)
    
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Label distribution: {y.value_counts().to_dict()}")
    
    # Create and train filter
    ml_filter = MLProbabilityFilter(
        model_type=model_type,
        probability_threshold=probability_threshold
    )
    
    metrics = ml_filter.train(X, y, validation_split=0.2)
    
    return ml_filter, metrics


def run_backtest_with_ml(
    data: pd.DataFrame,
    strategy: LondonBreakoutStrategy,
    ml_filter: MLProbabilityFilter,
    settings: Settings
) -> dict:
    """
    Run backtest with ML filter enabled.
    
    Args:
        data: OHLC DataFrame with features
        strategy: Strategy instance
        ml_filter: Trained ML filter
        settings: Strategy settings
    
    Returns:
        Dictionary with backtest results (with and without ML)
    """
    # Create engine
    cost_model = CostModel.from_broker_type(settings.account.broker_type)
    engine = BacktestEngine(
        initial_balance=settings.account.initial_balance,
        leverage=settings.account.leverage,
        cost_model=cost_model,
        risk_pct=settings.risk.risk_per_trade_pct
    )
    
    # Run without ML filter
    logger.info("Running backtest WITHOUT ML filter...")
    result_no_ml = engine.run(data, strategy, ml_probabilities=None)
    
    # Generate ML probabilities for all bars
    logger.info("Generating ML probabilities...")
    
    # Get feature columns that the model was trained on
    feature_cols = [c for c in data.columns if c in MLProbabilityFilter.FEATURE_COLUMNS or 
                   c.startswith('atr') or c.startswith('roc') or c.startswith('ema') or
                   c.startswith('volatility') or c in ['hour', 'day_of_week', 'is_london', 'is_overlap',
                                                        'rsi_14', 'bb_position', 'asian_range']]
    
    X_all = data[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    probabilities = ml_filter.predict(X_all)
    ml_proba_series = pd.Series(probabilities, index=data.index)
    
    # Run with ML filter
    logger.info("Running backtest WITH ML filter...")
    result_with_ml = engine.run(data, strategy, ml_probabilities=ml_proba_series)
    
    return {
        'no_ml': result_no_ml,
        'with_ml': result_with_ml,
        'improvement': {
            'win_rate_diff': result_with_ml.win_rate_pct - result_no_ml.win_rate_pct,
            'profit_factor_diff': result_with_ml.profit_factor - result_no_ml.profit_factor,
            'return_diff': result_with_ml.total_return_pct - result_no_ml.total_return_pct,
            'trades_filtered': result_no_ml.total_trades - result_with_ml.total_trades,
        }
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train ML filter for XAUUSD strategy")
    parser.add_argument("--days", type=int, default=60, help="Days of data to use")
    parser.add_argument("--output", type=str, default="models/ml_filter.pkl", help="Output model path")
    parser.add_argument("--model", type=str, choices=["xgboost", "lightgbm"], default="xgboost")
    parser.add_argument("--threshold", type=float, default=0.55, help="Probability threshold")
    parser.add_argument("--mode", type=str, choices=["conservative", "aggressive", "ultra"], default="aggressive")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    # Setup
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logger(log_level=log_level)
    
    logger.info("=" * 60)
    logger.info("XAUUSD ML Filter Training Pipeline")
    logger.info("=" * 60)
    
    # Load settings
    if args.mode == "conservative":
        settings = Settings.conservative()
    elif args.mode == "ultra":
        settings = Settings.ultra_aggressive()
    else:
        settings = Settings.aggressive()
    
    # Fetch data
    logger.info(f"Fetching {args.days} days of data...")
    fetcher = DataFetcher(source="yfinance")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    raw_data = fetcher.fetch(start_date, end_date, "5m")
    
    if raw_data.empty:
        logger.error("No data fetched!")
        return
    
    logger.info(f"Fetched {len(raw_data)} candles")
    
    # Process data
    processor = DataProcessor()
    data = processor.process(raw_data)
    
    # Add features
    feature_eng = MLFeatureEngineer()
    data = feature_eng.prepare_ml_features(data)
    
    logger.info(f"Processed data with {len(data.columns)} features")
    
    # Split train/test
    split_idx = int(len(data) * 0.7)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    logger.info(f"Train: {len(train_data)}, Test: {len(test_data)} candles")
    
    # Create strategy with PERMISSIVE parameters for training
    # This generates more signals for ML training, then we use ML to filter
    logger.info("Creating permissive strategy for training data collection...")
    training_strategy = LondonBreakoutStrategy(
        atr_period=14,
        atr_min_multiplier=0.1,  # Almost no filter
        roc_period=5,
        roc_threshold=0.0,       # No momentum required
        sl_atr_mult=1.2,         # Match ultra-aggressive SL
        tp_atr_mult=2.4,         # Match ultra-aggressive TP (1:2)
        ml_probability_threshold=0.0  # No ML filter during training
    )
    
    # Generate signals on training data using permissive strategy
    # Use unlimited signals per day for ML training data collection
    logger.info("Generating training signals (unlimited per day for ML training)...")
    train_signals = training_strategy.generate_signals(train_data, max_signals_per_day=0)
    logger.info(f"Generated {len(train_signals)} signals on training data")
    
    if len(train_signals) < 20:
        logger.error("Not enough signals for training. Need at least 20.")
        return
    
    # Create training labels
    cost_model = CostModel.from_broker_type(settings.account.broker_type)
    training_df = create_training_labels(train_data, train_signals, cost_model)
    
    logger.info(f"Created {len(training_df)} labeled samples")
    logger.info(f"Win rate in train: {training_df['label'].mean():.1%}")
    
    # Train ML filter
    ml_filter, metrics = train_ml_filter(
        training_df,
        model_type=args.model,
        probability_threshold=args.threshold
    )
    
    logger.info(f"Training metrics: {metrics}")
    
    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ml_filter.save(output_path)
    logger.info(f"Model saved to {output_path}")
    
    # Create production strategy with normal parameters for backtest
    production_strategy = LondonBreakoutStrategy(settings=settings)
    
    # Run comparison backtest on test data
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON BACKTEST ON TEST DATA")
    logger.info("=" * 60)
    
    results = run_backtest_with_ml(test_data, production_strategy, ml_filter, settings)
    
    # Print comparison
    no_ml = results['no_ml']
    with_ml = results['with_ml']
    improvement = results['improvement']
    
    print("\n" + "=" * 70)
    print("                    ML FILTER COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Metric':<25} {'Without ML':>15} {'With ML':>15} {'Change':>12}")
    print("-" * 70)
    print(f"{'Total Trades':<25} {no_ml.total_trades:>15} {with_ml.total_trades:>15} {improvement['trades_filtered']:>+12}")
    print(f"{'Win Rate':<25} {no_ml.win_rate_pct:>14.1f}% {with_ml.win_rate_pct:>14.1f}% {improvement['win_rate_diff']:>+11.1f}%")
    print(f"{'Profit Factor':<25} {no_ml.profit_factor:>15.2f} {with_ml.profit_factor:>15.2f} {improvement['profit_factor_diff']:>+12.2f}")
    print(f"{'Total Return':<25} {no_ml.total_return_pct:>14.1f}% {with_ml.total_return_pct:>14.1f}% {improvement['return_diff']:>+11.1f}%")
    print(f"{'Max Drawdown':<25} {no_ml.max_drawdown_pct:>14.1f}% {with_ml.max_drawdown_pct:>14.1f}%")
    print(f"{'Sharpe Ratio':<25} {no_ml.sharpe_ratio:>15.2f} {with_ml.sharpe_ratio:>15.2f}")
    print("-" * 70)
    print(f"{'Final Balance':<25} €{no_ml.final_balance:>13.2f} €{with_ml.final_balance:>13.2f}")
    print("=" * 70)
    
    # Feature importance
    print("\n📊 Top 10 Features:")
    for i, (feat, imp) in enumerate(ml_filter.get_feature_importance().head(10).items(), 1):
        print(f"  {i:2}. {feat:<30} {imp:.4f}")
    
    print("\n✅ ML Filter training complete!")
    print(f"   Model saved to: {output_path}")
    print(f"   Threshold: {args.threshold}")
    print(f"   Use with: --ml-model {output_path}")


if __name__ == "__main__":
    main()
