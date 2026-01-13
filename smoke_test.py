
import pandas as pd
import numpy as np
from datetime import datetime
from xauusd_strategy.config.settings import Settings
from xauusd_strategy.strategy.london_breakout import LondonBreakoutStrategy
from xauusd_strategy.strategy.asian_scalp import AsianScalpingStrategy
from xauusd_strategy.ml.model import MLProbabilityFilter
from xauusd_strategy.utils.logger import get_logger

logger = get_logger("SmokeTest")

def generate_mock_data(n_bars=200):
    """Generate mock OHLCV data for testing."""
    times = pd.date_range(end=datetime.now(), periods=n_bars, freq='5min')
    data = {
        'open': np.random.uniform(2030, 2040, n_bars),
        'high': np.random.uniform(2040, 2050, n_bars),
        'low': np.random.uniform(2020, 2030, n_bars),
        'close': np.random.uniform(2030, 2040, n_bars),
        'volume': np.random.randint(100, 1000, n_bars)
    }
    df = pd.DataFrame(data, index=times)
    # Ensure High is really high and Low is really low
    df['high'] = df[['open', 'close', 'high']].max(axis=1) + 1
    df['low'] = df[['open', 'close', 'low']].min(axis=1) - 1
    return df

def run_test():
    logger.info("Starting System Smoke Test...")
    
    # 1. Load Settings
    settings = Settings.from_yaml("config/aggressive.yaml")
    logger.info("Settings loaded.")
    
    # 2. Mock Data
    df = generate_mock_data(300)
    logger.info(f"Mock data generated: {len(df)} bars.")
    
    # 3. Test London Breakout Strategy
    breakout = LondonBreakoutStrategy(settings=settings)
    df_prep = breakout.prepare_data(df)
    logger.info("Breakout: Data preparation (indicators) successful.")
    
    # Test Signal Generation
    idx = len(df_prep) - 1
    signal = breakout.generate_signal(df_prep, idx, ml_probability=0.65)
    if signal:
        logger.info(f"Breakout: Signal generated! Type: {signal.signal_type}")
    else:
        logger.info("Breakout: No signal (Normal for random data).")
        
    # 4. Test Asian Scalp Strategy
    scalper = AsianScalpingStrategy(settings=settings)
    # Using same prep for now as it uses standard indicators
    sig_scalp = scalper.generate_signal(df_prep, idx, ml_prob=0.65)
    if sig_scalp:
        logger.info(f"Scalper: Signal generated! Type: {sig_scalp.signal_type}")
    else:
        logger.info("Scalper: No signal (Normal for random data).")
        
    # 5. Test ML Model Load
    ml_filter = MLProbabilityFilter()
    try:
        # Load doesn't return anything, just call it
        ml_filter.load("models/ml_filter_doubler.pkl")
        if ml_filter.model is not None:
            logger.info("ML Filter: Model loaded successfully.")
            # Mock feature vector
            from xauusd_strategy.ml.features import MLFeatureEngineer
            eng = MLFeatureEngineer()
            f_df = eng.prepare_ml_features(df_prep)
            prob = ml_filter.predict(f_df.iloc[[-1]])[0]
            logger.info(f"ML Filter: Test prediction successful. Prob: {prob:.2f}")
        else:
            logger.warning("ML Filter: Model found but not initialized correctly.")
    except Exception as e:
        logger.error(f"ML Filter Load Error (Expected if file missing): {e}")

    # 6. Test RL Agent Load
    try:
        from xauusd_strategy.rl.agent import DeepScalperAgent
        agent = DeepScalperAgent()
        if agent.model:
            logger.info("RL Agent: Model loaded successfully.")
            action = agent.predict(df_prep, 250.0, 0.0)
            logger.info(f"RL Agent: Prediction successful. Action: {action}")
        else:
            logger.info("RL Agent: Model zip not found (Normal if not trained yet).")
    except Exception as e:
        logger.error(f"RL Agent Error: {e}")

    logger.info("Smoke Test Complete.")

if __name__ == "__main__":
    run_test()
