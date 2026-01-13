"""
Data preprocessing and cleaning utilities.

Handles resampling, gap filling, and data quality checks.
"""

from datetime import datetime, time
from typing import Optional, Tuple
import pandas as pd
import numpy as np

from xauusd_strategy.utils.logger import get_logger

logger = get_logger(__name__)


class DataProcessor:
    """
    Preprocess OHLC data for strategy use.
    
    Features:
    - Resample to different timeframes
    - Fill gaps and missing data
    - Remove outliers
    - Timezone handling
    - Data quality validation
    """
    
    def __init__(self, timezone: str = "Europe/Berlin"):
        """
        Initialize processor.
        
        Args:
            timezone: Target timezone (default CET)
        """
        self.timezone = timezone
    
    def process(
        self,
        df: pd.DataFrame,
        resample_to: Optional[str] = None,
        fill_gaps: bool = True,
        remove_weekends: bool = True,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Full preprocessing pipeline.
        
        Args:
            df: Raw OHLC DataFrame
            resample_to: Target timeframe (e.g., "5T" for 5 minutes)
            fill_gaps: Whether to forward-fill gaps
            remove_weekends: Remove weekend data
            validate: Run data quality checks
        
        Returns:
            Processed DataFrame
        """
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return df
        
        logger.info(f"Processing {len(df)} rows")
        
        # Ensure datetime index
        df = self._ensure_datetime_index(df)
        
        # Localize/convert timezone
        df = self._handle_timezone(df)
        
        # Remove weekend data
        if remove_weekends:
            df = self._remove_weekends(df)
        
        # Resample if requested
        if resample_to:
            df = self._resample(df, resample_to)
        
        # Fill gaps
        if fill_gaps:
            df = self._fill_gaps(df)
        
        # Remove outliers
        df = self._remove_outliers(df)
        
        # Validate
        if validate:
            self._validate(df)
        
        logger.info(f"Processing complete: {len(df)} rows")
        
        return df
    
    def _ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has a DatetimeIndex."""
        if not isinstance(df.index, pd.DatetimeIndex):
            # Try to find datetime column
            datetime_cols = ['datetime', 'time', 'date', 'timestamp']
            for col in datetime_cols:
                if col in df.columns:
                    df = df.set_index(pd.to_datetime(df[col]))
                    df = df.drop(columns=[col], errors='ignore')
                    break
            else:
                # Try to parse index
                df.index = pd.to_datetime(df.index)
        
        return df
    
    def _handle_timezone(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle timezone conversion."""
        if df.index.tz is None:
            # Assume UTC if no timezone
            df.index = df.index.tz_localize('UTC')
        
        # Convert to target timezone
        df.index = df.index.tz_convert(self.timezone)
        
        return df
    
    def _remove_weekends(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove weekend data (Saturday and Sunday)."""
        original_len = len(df)
        df = df[df.index.dayofweek < 5]
        removed = original_len - len(df)
        
        if removed > 0:
            logger.debug(f"Removed {removed} weekend rows")
        
        return df
    
    def _resample(self, df: pd.DataFrame, rule: str) -> pd.DataFrame:
        """
        Resample OHLC data to a different timeframe.
        
        Args:
            df: OHLC DataFrame
            rule: Pandas resample rule (e.g., "5T", "1H", "1D")
        
        Returns:
            Resampled DataFrame
        """
        logger.debug(f"Resampling to {rule}")
        
        resampled = df.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Remove rows where all values are NaN
        resampled = resampled.dropna(how='all')
        
        return resampled
    
    def _fill_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill gaps in data using forward fill.
        
        Only fills during trading hours.
        """
        # Forward fill close price
        df['close'] = df['close'].ffill()
        
        # For gaps, set OHLC to previous close
        mask = df['open'].isna()
        df.loc[mask, 'open'] = df.loc[mask, 'close']
        df.loc[mask, 'high'] = df.loc[mask, 'close']
        df.loc[mask, 'low'] = df.loc[mask, 'close']
        df.loc[mask, 'volume'] = 0
        
        return df
    
    def _remove_outliers(
        self,
        df: pd.DataFrame,
        std_threshold: float = 5.0
    ) -> pd.DataFrame:
        """
        Remove extreme price outliers.
        
        Args:
            df: OHLC DataFrame
            std_threshold: Number of standard deviations for outlier detection
        
        Returns:
            DataFrame with outliers removed
        """
        original_len = len(df)
        
        # Calculate returns
        returns = df['close'].pct_change()
        
        # Find outliers
        mean_ret = returns.mean()
        std_ret = returns.std()
        
        outlier_mask = abs(returns - mean_ret) > (std_threshold * std_ret)
        
        # Remove outlier rows
        df = df[~outlier_mask]
        
        removed = original_len - len(df)
        if removed > 0:
            logger.warning(f"Removed {removed} outlier rows")
        
        return df
    
    def _validate(self, df: pd.DataFrame) -> bool:
        """
        Validate data quality.
        
        Checks:
        - No NaN values in OHLC
        - High >= Low
        - Open and Close within High-Low range
        - Positive volume
        """
        issues = []
        
        # Check for NaN
        nan_counts = df[['open', 'high', 'low', 'close']].isna().sum()
        if nan_counts.any():
            issues.append(f"NaN values found: {nan_counts.to_dict()}")
        
        # Check High >= Low
        invalid_hl = (df['high'] < df['low']).sum()
        if invalid_hl > 0:
            issues.append(f"{invalid_hl} rows with High < Low")
        
        # Check OHLC consistency
        open_invalid = ((df['open'] > df['high']) | (df['open'] < df['low'])).sum()
        close_invalid = ((df['close'] > df['high']) | (df['close'] < df['low'])).sum()
        
        if open_invalid > 0:
            issues.append(f"{open_invalid} rows with Open outside High-Low range")
        if close_invalid > 0:
            issues.append(f"{close_invalid} rows with Close outside High-Low range")
        
        # Check volume
        negative_volume = (df['volume'] < 0).sum()
        if negative_volume > 0:
            issues.append(f"{negative_volume} rows with negative volume")
        
        if issues:
            for issue in issues:
                logger.warning(f"Data validation issue: {issue}")
            return False
        
        logger.debug("Data validation passed")
        return True
    
    def get_trading_hours_mask(
        self,
        df: pd.DataFrame,
        start_hour: int = 0,
        end_hour: int = 24
    ) -> pd.Series:
        """
        Create mask for trading hours.
        
        Args:
            df: DataFrame with DatetimeIndex
            start_hour: Start hour (0-23)
            end_hour: End hour (0-23)
        
        Returns:
            Boolean Series
        """
        hours = df.index.hour
        
        if start_hour <= end_hour:
            return (hours >= start_hour) & (hours < end_hour)
        else:
            # Overnight session
            return (hours >= start_hour) | (hours < end_hour)
    
    def split_train_test(
        self,
        df: pd.DataFrame,
        test_ratio: float = 0.2,
        split_date: Optional[datetime] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and test sets.
        
        Args:
            df: OHLC DataFrame
            test_ratio: Ratio of data for test set (if no split_date)
            split_date: Specific date to split on
        
        Returns:
            Tuple of (train_df, test_df)
        """
        if split_date:
            train = df[df.index < split_date]
            test = df[df.index >= split_date]
        else:
            split_idx = int(len(df) * (1 - test_ratio))
            train = df.iloc[:split_idx]
            test = df.iloc[split_idx:]
        
        logger.info(f"Split: {len(train)} train, {len(test)} test rows")
        
        return train, test
