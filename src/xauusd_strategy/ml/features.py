"""
Feature engineering specific to ML models.
"""

import pandas as pd
import numpy as np
from typing import Optional, List

from xauusd_strategy.data.features import FeatureEngineer
from xauusd_strategy.utils.logger import get_logger
# from xauusd_strategy.ml.embeddings import TransformerEmbedder

logger = get_logger(__name__)


class MLFeatureEngineer(FeatureEngineer):
    """
    Extended feature engineering for ML models.
    
    Adds additional features optimized for ML performance:
    - Lag features
    - Rolling statistics
    - Cross-sectional features
    - Target encoding (for training)
    """
    
    def __init__(self, look_back_periods: List[int] = [5, 10, 20, 50], use_transformers: bool = False):
        """
        Initialize ML feature engineer.
        
        Args:
            look_back_periods: Periods for rolling features
            use_transformers: Whether to compute 'The Linguist' embeddings
        """
        super().__init__()
        self.look_back_periods = look_back_periods
        self.use_transformers = use_transformers
        self._embedder = None
        
        if use_transformers:
            from xauusd_strategy.ml.embeddings import TransformerEmbedder
            self._embedder = TransformerEmbedder()
    
    def prepare_ml_features(
        self,
        df: pd.DataFrame,
        target: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Prepare full ML feature set.
        
        Args:
            df: OHLC DataFrame
            target: Optional target series for target encoding
        
        Returns:
            DataFrame with ML features
        """
        if len(df) > 100: logger.debug(f"Preparing ML features for {len(df)} rows")
        
        # Start with base features
        df = self.compute_all(df, include_ml_features=True)
        
        # Add Asian box features
        df = self.get_asian_box_features(df)
        
        # Add lag features
        df = self._add_lag_features(df)
        
        # Add rolling statistics
        df = self._add_rolling_stats(df)
        
        # Add cross-sectional features
        df = self._add_cross_features(df)
        
        # Add Transformer Embeddings ('The Linguist')
        if self.use_transformers and self._embedder:
            df = self._embedder.add_transformer_features(df)
        
        # Add target encoding if target provided
        if target is not None:
            df = self._add_target_encoding(df, target)
        
        # Clean up
        df = df.replace([np.inf, -np.inf], np.nan)
        
        logger.debug(f"ML features ready: {len(df.columns)} columns")
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged versions of key features."""
        lag_cols = ['close', 'atr_14', 'roc_5', 'volatility_5']
        
        for col in lag_cols:
            if col not in df.columns:
                continue
            
            for lag in [1, 2, 3, 5, 10]:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def _add_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistics."""
        for period in self.look_back_periods:
            # Rolling returns
            returns = df['close'].pct_change()
            df[f'return_mean_{period}'] = returns.rolling(period).mean()
            df[f'return_std_{period}'] = returns.rolling(period).std()
            df[f'return_skew_{period}'] = returns.rolling(period).skew()
            
            # Rolling volatility
            if 'atr_14' in df.columns:
                df[f'atr_mean_{period}'] = df['atr_14'].rolling(period).mean()
                df[f'atr_std_{period}'] = df['atr_14'].rolling(period).std()
            
            # Rolling high/low range
            df[f'range_mean_{period}'] = (df['high'] - df['low']).rolling(period).mean()
        
        return df
    
    def _add_cross_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-sectional features (interactions)."""
        # ATR * ROC interaction
        if 'atr_14' in df.columns and 'roc_5' in df.columns:
            df['atr_roc_interaction'] = df['atr_14'] * df['roc_5']
        
        # Volatility regime
        if 'volatility_20' in df.columns:
            vol_median = df['volatility_20'].rolling(100).median()
            df['high_vol_regime'] = (df['volatility_20'] > vol_median).astype(int)
        
        # Trend strength
        if 'ema_20' in df.columns and 'ema_50' in df.columns:
            df['trend_strength'] = (df['ema_20'] - df['ema_50']) / df['atr_14'] if 'atr_14' in df.columns else 0
        
        # Distance from EMA as percentage of ATR
        if 'price_to_ema20' in df.columns and 'atr_14' in df.columns:
            df['ema_distance_atr'] = df['price_to_ema20'] / (df['atr_14'] / df['close'] * 100)
        
        return df
    
    def _add_target_encoding(
        self,
        df: pd.DataFrame,
        target: pd.Series,
        encoding_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Add target encoding for categorical features.
        
        Uses expanding mean to avoid lookahead bias.
        """
        if encoding_cols is None:
            encoding_cols = ['hour', 'day_of_week']
        
        for col in encoding_cols:
            if col not in df.columns:
                continue
            
            # Expanding mean by category
            target_aligned = target.reindex(df.index)
            
            for cat in df[col].unique():
                mask = df[col] == cat
                expanding_mean = target_aligned[mask].expanding().mean().shift(1)
                df.loc[mask, f'{col}_target_enc'] = expanding_mean
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of all ML feature names."""
        base_features = super().get_feature_names(include_ml=True)
        
        # Add lag features
        lag_features = []
        for col in ['close', 'atr_14', 'roc_5', 'volatility_5']:
            for lag in [1, 2, 3, 5, 10]:
                lag_features.append(f'{col}_lag_{lag}')
        
        # Add rolling features
        rolling_features = []
        for period in self.look_back_periods:
            rolling_features.extend([
                f'return_mean_{period}',
                f'return_std_{period}',
                f'return_skew_{period}',
                f'atr_mean_{period}',
                f'range_mean_{period}'
            ])
        
        # Add cross features
        cross_features = [
            'atr_roc_interaction',
            'high_vol_regime',
            'trend_strength',
            'ema_distance_atr'
        ]
        
        return base_features + lag_features + rolling_features + cross_features
