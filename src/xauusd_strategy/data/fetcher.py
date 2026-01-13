"""
Data fetching module for XAUUSD historical and real-time data.

Supports multiple data sources:
- yfinance (free, delayed)
- MetaTrader5 (real-time, requires MT5 terminal)
- CSV files (offline)
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Literal
import pandas as pd

from xauusd_strategy.utils.logger import get_logger

logger = get_logger(__name__)


class BaseDataFetcher(ABC):
    """Abstract base class for data fetchers."""
    
    @abstractmethod
    def fetch(
        self,
        start: datetime,
        end: datetime,
        timeframe: str = "5m"
    ) -> pd.DataFrame:
        """Fetch OHLC data for the specified period."""
        pass


class YFinanceFetcher(BaseDataFetcher):
    """Fetch data from yfinance (free, delayed)."""
    
    SYMBOL = "GC=F"  # Gold futures
    
    def __init__(self):
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError:
            raise ImportError("yfinance not installed. Run: pip install yfinance")
    
    def fetch(
        self,
        start: datetime,
        end: datetime,
        timeframe: str = "5m"
    ) -> pd.DataFrame:
        """
        Fetch XAUUSD data from yfinance.
        
        Note: yfinance has limitations on intraday data (max 60 days for 5m).
        For longer periods, use daily data or MT5.
        
        Args:
            start: Start datetime
            end: End datetime
            timeframe: Candle timeframe (5m, 15m, 1h, 1d)
        
        Returns:
            OHLCV DataFrame
        """
        logger.info(f"Fetching {self.SYMBOL} from yfinance: {start} to {end}")
        
        # Map timeframe to yfinance interval
        interval_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d",
        }
        
        interval = interval_map.get(timeframe.lower(), "5m")
        
        ticker = self.yf.Ticker(self.SYMBOL)
        
        # yfinance limitation: max 60 days for intraday data
        if interval in ["1m", "5m", "15m", "30m"]:
            max_days = 60
            if (end - start).days > max_days:
                logger.warning(
                    f"yfinance limits intraday data to {max_days} days. "
                    "Truncating request."
                )
                start = end - timedelta(days=max_days)
        
        df = ticker.history(
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True
        )
        
        if df.empty:
            logger.warning("No data returned from yfinance")
            return pd.DataFrame()
        
        # Standardize column names
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })
        
        # Keep only OHLCV
        df = df[["open", "high", "low", "close", "volume"]]
        
        logger.info(f"Fetched {len(df)} candles from yfinance")
        
        return df


class MT5Fetcher(BaseDataFetcher):
    """Fetch data from MetaTrader5 terminal."""
    
    SYMBOL = "XAUUSD"
    
    def __init__(self, login: Optional[int] = None, password: Optional[str] = None, server: Optional[str] = None):
        self.login = login
        self.password = password
        self.server = server
        self._connected = False
        
        try:
            import MetaTrader5 as mt5
            self.mt5 = mt5
        except ImportError:
            raise ImportError("MetaTrader5 not installed. Run: pip install MetaTrader5")
    
    def connect(self) -> bool:
        """Initialize connection to MT5 terminal."""
        if self._connected:
            return True
        
        if not self.mt5.initialize():
            logger.error(f"MT5 initialize failed: {self.mt5.last_error()}")
            return False
        
        # Login if credentials provided
        if self.login and self.password and self.server:
            if not self.mt5.login(self.login, self.password, self.server):
                logger.error(f"MT5 login failed: {self.mt5.last_error()}")
                return False
        
        self._connected = True
        logger.info("MT5 connected successfully")
        return True
    
    def disconnect(self):
        """Shutdown MT5 connection."""
        if self._connected:
            self.mt5.shutdown()
            self._connected = False
    
    def fetch(
        self,
        start: datetime,
        end: datetime,
        timeframe: str = "5m"
    ) -> pd.DataFrame:
        """
        Fetch XAUUSD data from MT5.
        
        Args:
            start: Start datetime
            end: End datetime
            timeframe: Candle timeframe
        
        Returns:
            OHLCV DataFrame
        """
        if not self.connect():
            return pd.DataFrame()
        
        # Map timeframe to MT5 constant
        timeframe_map = {
            "1m": self.mt5.TIMEFRAME_M1,
            "5m": self.mt5.TIMEFRAME_M5,
            "15m": self.mt5.TIMEFRAME_M15,
            "30m": self.mt5.TIMEFRAME_M30,
            "1h": self.mt5.TIMEFRAME_H1,
            "4h": self.mt5.TIMEFRAME_H4,
            "1d": self.mt5.TIMEFRAME_D1,
        }
        
        tf = timeframe_map.get(timeframe.lower(), self.mt5.TIMEFRAME_M5)
        
        logger.info(f"Fetching {self.SYMBOL} from MT5: {start} to {end}")
        
        rates = self.mt5.copy_rates_range(
            self.SYMBOL,
            tf,
            start,
            end
        )
        
        if rates is None or len(rates) == 0:
            logger.warning(f"No data returned from MT5: {self.mt5.last_error()}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Rename columns
        df = df.rename(columns={
            'tick_volume': 'volume'
        })
        
        # Keep only OHLCV
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        logger.info(f"Fetched {len(df)} candles from MT5")
        
        return df


class CSVFetcher(BaseDataFetcher):
    """Fetch data from CSV files."""
    
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
    
    def fetch(
        self,
        start: datetime,
        end: datetime,
        timeframe: str = "5m"
    ) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Expected file format: XAUUSD_{timeframe}.csv
        Expected columns: datetime, open, high, low, close, volume
        
        Args:
            start: Start datetime
            end: End datetime
            timeframe: Candle timeframe
        
        Returns:
            OHLCV DataFrame
        """
        file_path = self.data_dir / f"XAUUSD_{timeframe}.csv"
        
        if not file_path.exists():
            logger.error(f"Data file not found: {file_path}")
            return pd.DataFrame()
        
        logger.info(f"Loading data from {file_path}")
        
        df = pd.read_csv(file_path, parse_dates=['datetime'], index_col='datetime')
        
        # Filter by date range
        df = df[(df.index >= start) & (df.index <= end)]
        
        logger.info(f"Loaded {len(df)} candles from CSV")
        
        return df


class DataFetcher:
    """
    Unified data fetcher interface.
    
    Usage:
        >>> fetcher = DataFetcher(source="yfinance")
        >>> df = fetcher.fetch(start, end, "5m")
    """
    
    def __init__(
        self,
        source: Literal["yfinance", "mt5", "csv"] = "yfinance",
        **kwargs
    ):
        """
        Initialize data fetcher.
        
        Args:
            source: Data source ("yfinance", "mt5", "csv")
            **kwargs: Source-specific arguments
        """
        self.source = source
        
        if source == "yfinance":
            self._fetcher = YFinanceFetcher()
        elif source == "mt5":
            self._fetcher = MT5Fetcher(**kwargs)
        elif source == "csv":
            self._fetcher = CSVFetcher(**kwargs)
        else:
            raise ValueError(f"Unknown data source: {source}")
    
    def fetch(
        self,
        start: datetime,
        end: datetime,
        timeframe: str = "5m"
    ) -> pd.DataFrame:
        """Fetch data from configured source."""
        return self._fetcher.fetch(start, end, timeframe)
    
    def fetch_recent(
        self,
        days: int = 30,
        timeframe: str = "5m"
    ) -> pd.DataFrame:
        """Fetch recent data (last N days)."""
        end = datetime.now()
        start = end - timedelta(days=days)
        return self.fetch(start, end, timeframe)


def fetch_xauusd_data(
    start: datetime,
    end: datetime,
    timeframe: str = "5m",
    source: str = "yfinance",
    **kwargs
) -> pd.DataFrame:
    """
    Convenience function to fetch XAUUSD data.
    
    Args:
        start: Start datetime
        end: End datetime
        timeframe: Candle timeframe
        source: Data source
        **kwargs: Source-specific arguments
    
    Returns:
        OHLCV DataFrame
    """
    fetcher = DataFetcher(source=source, **kwargs)
    return fetcher.fetch(start, end, timeframe)
