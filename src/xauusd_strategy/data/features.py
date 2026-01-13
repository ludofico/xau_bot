"""
Feature engineering for trading strategies and ML models.

Computes technical indicators and derived features.
"""

from typing import List, Optional
import pandas as pd
import numpy as np

from xauusd_strategy.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Compute technical indicators and features for trading strategies.
    
    All features are computed in-place on the DataFrame.
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        pass
    
    def compute_all(
        self,
        df: pd.DataFrame,
        include_ml_features: bool = False
    ) -> pd.DataFrame:
        """
        Compute all features.
        
        Args:
            df: OHLC DataFrame
            include_ml_features: Include ML-specific features
        
        Returns:
            DataFrame with all features
        """
        df = df.copy()
        
        # Core indicators
        df = self.add_atr(df)
        df = self.add_roc(df)
        df = self.add_ema(df)
        
        # Session features
        df = self.add_session_features(df)
        
        # Volatility features
        df = self.add_volatility_features(df)
        
        if include_ml_features:
            df = self.add_ml_features(df)
        
        return df
    
    def add_atr(
        self,
        df: pd.DataFrame,
        periods: List[int] = [14, 5, 20]
    ) -> pd.DataFrame:
        """
        Add Average True Range (ATR) indicators.
        
        Args:
            df: OHLC DataFrame
            periods: ATR periods to compute
        
        Returns:
            DataFrame with ATR columns
        """
        # True Range
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        
        for period in periods:
            df[f'atr_{period}'] = df['tr'].rolling(window=period).mean()
        
        # ATR as percentage of price
        df['atr_pct'] = df['atr_14'] / df['close'] * 100
        
        return df
    
    def add_roc(
        self,
        df: pd.DataFrame,
        periods: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """
        Add Rate of Change (ROC) indicators.
        
        Args:
            df: OHLC DataFrame
            periods: ROC periods to compute
        
        Returns:
            DataFrame with ROC columns
        """
        for period in periods:
            df[f'roc_{period}'] = (
                (df['close'] - df['close'].shift(period)) / 
                df['close'].shift(period) * 100
            )
        
        return df
    
    def add_ema(
        self,
        df: pd.DataFrame,
        periods: List[int] = [9, 20, 50, 200]
    ) -> pd.DataFrame:
        """
        Add Exponential Moving Averages (EMA).
        
        Args:
            df: OHLC DataFrame
            periods: EMA periods to compute
        
        Returns:
            DataFrame with EMA columns
        """
        for period in periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            
            # EMA slope (direction)
            df[f'ema_{period}_slope'] = df[f'ema_{period}'].diff(5)
        
        # Price relative to EMAs
        if 'ema_20' in df.columns:
            df['price_to_ema20'] = (df['close'] - df['ema_20']) / df['ema_20'] * 100
        
        if 'ema_50' in df.columns:
            df['price_to_ema50'] = (df['close'] - df['ema_50']) / df['ema_50'] * 100
        
        return df
    
    def add_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add session-based features.
        
        Args:
            df: OHLC DataFrame with DatetimeIndex
        
        Returns:
            DataFrame with session features
        """
        # Time features
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        df['day_of_week'] = df.index.dayofweek
        
        # Session flags (CET times)
        df['is_asian'] = ((df['hour'] >= 0) & (df['hour'] < 7)).astype(int)
        df['is_london'] = ((df['hour'] >= 8) & (df['hour'] < 17)).astype(int)
        df['is_ny'] = ((df['hour'] >= 14) & (df['hour'] < 21)).astype(int)
        df['is_overlap'] = ((df['hour'] >= 14) & (df['hour'] < 17)).astype(int)
        
        return df
    
    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility-based features.
        
        Args:
            df: OHLC DataFrame
        
        Returns:
            DataFrame with volatility features
        """
        # Candle size
        df['candle_size'] = df['high'] - df['low']
        df['candle_body'] = abs(df['close'] - df['open'])
        df['candle_body_ratio'] = df['candle_body'] / df['candle_size'].replace(0, np.nan)
        
        # Upper/lower wick
        df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
        
        # Rolling volatility
        df['volatility_5'] = df['close'].pct_change().rolling(5).std() * 100
        df['volatility_20'] = df['close'].pct_change().rolling(20).std() * 100
        
        # Volatility ratio (current vs average)
        if 'atr_14' in df.columns:
            atr_avg = df['atr_14'].rolling(50).mean()
            df['atr_ratio'] = df['atr_14'] / atr_avg.replace(0, np.nan)
        
        return df
    
    def add_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features specifically for ML models.
        
        Args:
            df: OHLC DataFrame
        
        Returns:
            DataFrame with ML features
        """
        # Momentum features
        df['momentum_5'] = df['close'].diff(5)
        df['momentum_10'] = df['close'].diff(10)
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_lower'] = sma_20 - (std_20 * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma_20 * 100
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Stochastic
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14).replace(0, np.nan)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # Volume features (if available)
        if 'volume' in df.columns and df['volume'].sum() > 0:
            df['volume_ma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma_20'].replace(0, np.nan)
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'roc_5_lag_{lag}'] = df['roc_5'].shift(lag) if 'roc_5' in df.columns else np.nan
        
        return df
    
    def get_asian_box_features(
        self,
        df: pd.DataFrame,
        asian_start: str = "00:00",
        asian_end: str = "07:00"
    ) -> pd.DataFrame:
        """
        Calculate Asian session box features for each day.
        
        Args:
            df: OHLC DataFrame
            asian_start: Session start time
            asian_end: Session end time
        
        Returns:
            DataFrame with daily Asian box features
        """
        from xauusd_strategy.utils.time_utils import get_asian_box
        
        df = df.copy()
        
        # Initialize columns
        df['asian_high'] = np.nan
        df['asian_low'] = np.nan
        df['asian_range'] = np.nan
        df['distance_from_asian_high'] = np.nan
        df['distance_from_asian_low'] = np.nan
        
        # Calculate for each unique date
        unique_dates = np.unique(df.index.date)
        for date in unique_dates:
            asian_high, asian_low = get_asian_box(df, pd.Timestamp(date), asian_start, asian_end)
            
            if asian_high is None:
                continue
            
            day_mask = df.index.date == date
            df.loc[day_mask, 'asian_high'] = asian_high
            df.loc[day_mask, 'asian_low'] = asian_low
            df.loc[day_mask, 'asian_range'] = asian_high - asian_low
        
        # Distance from box
        df['distance_from_asian_high'] = df['close'] - df['asian_high']
        df['distance_from_asian_low'] = df['close'] - df['asian_low']
        
        return df
    
    def get_feature_names(self, include_ml: bool = False) -> List[str]:
        """
        Get list of all computed feature names.
        
        Args:
            include_ml: Include ML-specific features
        
        Returns:
            List of feature column names
        """
        features = [
            # ATR
            'atr_14', 'atr_5', 'atr_20', 'atr_pct', 'tr',
            # ROC
            'roc_5', 'roc_10', 'roc_20',
            # EMA
            'ema_9', 'ema_20', 'ema_50', 'ema_200',
            'ema_9_slope', 'ema_20_slope', 'ema_50_slope', 'ema_200_slope',
            'price_to_ema20', 'price_to_ema50',
            # Session
            'hour', 'minute', 'day_of_week',
            'is_asian', 'is_london', 'is_ny', 'is_overlap',
            # Volatility
            'candle_size', 'candle_body', 'candle_body_ratio',
            'upper_wick', 'lower_wick',
            'volatility_5', 'volatility_20', 'atr_ratio',
        ]
        
        if include_ml:
            features.extend([
                'momentum_5', 'momentum_10',
                'rsi_14',
                'macd', 'macd_signal', 'macd_hist',
                'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
                'stoch_k', 'stoch_d',
                'volume_ma_20', 'volume_ratio',
            ])
            
            # Lagged features
            for lag in [1, 2, 3, 5]:
                features.extend([f'close_lag_{lag}', f'roc_5_lag_{lag}'])
        
        return features
