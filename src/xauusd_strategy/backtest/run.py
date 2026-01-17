
"""
XAUUSD Backtest Runner - Full Stack Version.

Features:
- MT5 data when available, yfinance fallback
- XGBoost ML Filter integration
- RL DeepScalper agent signals
- Multi-strategy backtesting (London Breakout + Asian Scalp + RL)
"""

import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from xauusd_strategy.config.settings import Settings
from xauusd_strategy.backtest.engine import BacktestEngine
from xauusd_strategy.strategy.london_breakout import LondonBreakoutStrategy, TradeSignal, SignalType
from xauusd_strategy.utils.logger import get_logger

logger = get_logger("BacktestRunner")


def fetch_data_mt5_or_yfinance(start_date, end_date, timeframe="5m"):
    """
    Fetch data from MT5 if available, otherwise yfinance.
    MT5 has more historical data for M5.
    """
    # Try MT5 first
    try:
        import MetaTrader5 as mt5
        
        if mt5.initialize():
            logger.info("MT5 connected! Fetching data from MT5...")
            
            tf_map = {
                "1m": mt5.TIMEFRAME_M1,
                "5m": mt5.TIMEFRAME_M5,
                "15m": mt5.TIMEFRAME_M15,
                "1h": mt5.TIMEFRAME_H1,
            }
            mt5_tf = tf_map.get(timeframe, mt5.TIMEFRAME_M5)
            
            symbols_to_try = ["XAUUSD", "GOLD", "XAUUSDm", "XAUUSD.a"]
            
            for symbol in symbols_to_try:
                rates = mt5.copy_rates_range(symbol, mt5_tf, start_date, end_date)
                if rates is not None and len(rates) > 100:
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('time', inplace=True)
                    df.columns = [c.lower() for c in df.columns]
                    
                    if 'tick_volume' in df.columns and 'volume' not in df.columns:
                        df['volume'] = df['tick_volume']
                    
                    logger.info(f"MT5: Fetched {len(df)} bars of {symbol}")
                    mt5.shutdown()
                    return df
            
            mt5.shutdown()
            logger.warning("MT5: No valid data found, using yfinance fallback")
            
    except ImportError:
        logger.info("MT5 not available, using yfinance")
    except Exception as e:
        logger.warning(f"MT5 error: {e}, using yfinance")
    
    # Fallback: yfinance
    from xauusd_strategy.data.fetcher import fetch_xauusd_data
    return fetch_xauusd_data(start_date, end_date, timeframe=timeframe, source="yfinance")


def main():
    parser = argparse.ArgumentParser(description="Run XAUUSD Full Stack Backtest")
    parser.add_argument("--days", type=int, default=30, help="Days of history")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--balance", type=float, default=250.0, help="Initial Balance (EUR)")
    parser.add_argument("--no-ml", action="store_true", help="Disable ML Filter")
    parser.add_argument("--no-rl", action="store_true", help="Disable RL Agent")
    
    args = parser.parse_args()
    
    # 1. Setup Timeframe
    if args.start and args.end:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
    else:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
    
    logger.info(f"FULL STACK Backtest: {start_date.date()} to {end_date.date()}")
    
    # 2. Fetch Data (MT5 or yfinance)
    logger.info("Fetching Market Data...")
    df = fetch_data_mt5_or_yfinance(start_date, end_date, "5m")
    
    if df is None or df.empty:
        logger.error("No data fetched!")
        return
    
    logger.info(f"Data: {len(df)} bars")
    
    # 3. Initialize Strategy
    strategy = LondonBreakoutStrategy()
    df_prep = strategy.prepare_data(df)
    
    # 4. Load ML Model (XGBoost)
    ml_probs = None
    if not args.no_ml:
        try:
            from xauusd_strategy.ml.model import MLProbabilityFilter
            from xauusd_strategy.ml.features import MLFeatureEngineer
            
            model_path = Path("models/ml_filter_doubler.pkl")
            if model_path.exists():
                model = MLProbabilityFilter(model_path=model_path)
                eng = MLFeatureEngineer()
                df_features = eng.prepare_ml_features(df_prep)
                ml_probs = pd.Series(model.predict(df_features), index=df_prep.index[:len(df_features)])
                logger.info(f"ML Filter loaded: {len(ml_probs)} predictions")
            else:
                logger.warning(f"ML model not found at {model_path}")
        except Exception as e:
            logger.error(f"ML Error: {e}")
    
    # 5. Load RL Agent
    rl_agent = None
    if not args.no_rl:
        try:
            from xauusd_strategy.rl.agent import DeepScalperAgent
            rl_agent = DeepScalperAgent()
            if rl_agent.model is None:
                logger.warning("RL Model not loaded - no model file found")
                rl_agent = None
            else:
                logger.info("RL DeepScalper Agent loaded")
        except Exception as e:
            logger.error(f"RL Error: {e}")
    
    # 6. Generate Signals from ALL strategies
    logger.info("Generating signals from all strategies...")
    all_signals = []
    
    # 6a. London Breakout Signals (with ML filter)
    breakout_signals = strategy.generate_signals(df_prep, ml_probs)
    for s in breakout_signals:
        setattr(s, 'source', "LondonBreakout")
    all_signals.extend(breakout_signals)
    logger.info(f"London Breakout: {len(breakout_signals)} signals")
    
    # 6b. Asian Scalp Signals
    try:
        from xauusd_strategy.strategy.asian_scalp import AsianScalpStrategy
        scalp_strategy = AsianScalpStrategy()
        scalp_signals = []
        for i in range(50, len(df_prep)):
            sig = scalp_strategy.generate_signal(df_prep, i, ml_probs.iloc[i] if ml_probs is not None and i < len(ml_probs) else 0.5)
            if sig:
                setattr(sig, 'source', "AsianScalp")
                sig.timestamp = df_prep.index[i]
                scalp_signals.append(sig)
        all_signals.extend(scalp_signals)
        logger.info(f"Asian Scalp: {len(scalp_signals)} signals")
    except Exception as e:
        logger.warning(f"Asian Scalp Error: {e}")
    
    # 6c. RL Agent Signals
    if rl_agent:
        rl_signals = []
        window = 30
        for i in range(window, len(df_prep)):
            try:
                sub_df = df_prep.iloc[i-window:i+1].copy()
                action = rl_agent.predict(sub_df, balance=args.balance, position=0.0)
                
                if action in [1, 2]:  # Buy or Sell
                    row = df_prep.iloc[i]
                    atr = row.get('atr_14', 2.0)
                    close = row['close']
                    
                    if action == 1:  # Long
                        sig = TradeSignal(
                            signal_type=SignalType.LONG,
                            entry_price=close,
                            stop_loss=close - (atr * 1.5),
                            take_profit=close + (atr * 2.0),
                            atr_value=atr,
                            roc_value=0, asian_high=0, asian_low=0,
                            probability=0.7,
                            timestamp=df_prep.index[i]
                        )
                    else:  # Short
                        sig = TradeSignal(
                            signal_type=SignalType.SHORT,
                            entry_price=close,
                            stop_loss=close + (atr * 1.5),
                            take_profit=close - (atr * 2.0),
                            atr_value=atr,
                            roc_value=0, asian_high=0, asian_low=0,
                            probability=0.7,
                            timestamp=df_prep.index[i]
                        )
                    setattr(sig, 'source', "DeepScalper_RL")
                    rl_signals.append(sig)
            except Exception as e:
                pass  # Skip errors silently
        
        all_signals.extend(rl_signals)
        logger.info(f"RL DeepScalper: {len(rl_signals)} signals")
    
    # Remove duplicate timestamps (keep highest probability)
    if all_signals:
        signals_by_time = {}
        for s in all_signals:
            ts = s.timestamp
            if ts not in signals_by_time or s.probability > signals_by_time[ts].probability:
                signals_by_time[ts] = s
        all_signals = sorted(signals_by_time.values(), key=lambda x: x.timestamp)
    
    logger.info(f"Total Signals (deduplicated): {len(all_signals)}")
    
    # 7. Run Backtest
    engine = BacktestEngine(initial_balance=args.balance)
    
    # Custom run with pre-generated signals
    if all_signals:
        trade_log = engine._simulate_trades(df_prep, all_signals)
        if not trade_log.empty:
            result = engine._calculate_metrics(df_prep, trade_log)
        else:
            result = engine._empty_result()
    else:
        result = engine._empty_result()
    
    # 8. Print Summary
    print("\n" + "=" * 60)
    print("   FULL STACK BACKTEST - XAUUSD (MT5 + ML + RL)")
    print("=" * 60)
    print(f"  Data Source:      {'MT5' if 'MT5' in str(type(df)) else 'yfinance'}")
    print(f"  ML Filter:        {'ON' if ml_probs is not None else 'OFF'}")
    print(f"  RL Agent:         {'ON' if rl_agent else 'OFF'}")
    print("-" * 60)
    print(f"  Initial Balance:  €{result.initial_balance:.2f}")
    print(f"  Final Balance:    €{result.final_balance:.2f}")
    print(f"  Peak Balance:     €{result.peak_balance:.2f}")
    print("-" * 60)
    print(f"  Total Return:     {result.total_return_pct:+.1f}%")
    print(f"  Monthly Return:   {result.monthly_return_pct:+.1f}%")
    print("-" * 60)
    print(f"  Sharpe Ratio:     {result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown:     {result.max_drawdown_pct:.1f}%")
    print("-" * 60)
    print(f"  Total Trades:     {result.total_trades}")
    print(f"  Win Rate:         {result.win_rate_pct:.1f}%")
    print(f"  Profit Factor:    {result.profit_factor:.2f}")
    print("-" * 60)
    
    target_500 = "✅" if result.monthly_return_pct >= 200 else "❌"
    target_1000 = "✅" if result.monthly_return_pct >= 400 else "❌"
    
    print(f"  €500/month Target (200%):  {target_500}")
    print(f"  €1000/month Target (400%): {target_1000}")
    print("=" * 60 + "\n")
    
    # Log individual trade sources
    if not result.trade_log.empty and 'source' in df_prep.columns:
        logger.info("Trade breakdown by strategy:")
        # This would require tracking source in trade_log

if __name__ == "__main__":
    main()
