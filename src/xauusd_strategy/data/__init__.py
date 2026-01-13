"""Data module for fetching and processing XAUUSD data."""

from xauusd_strategy.data.fetcher import DataFetcher, fetch_xauusd_data
from xauusd_strategy.data.processor import DataProcessor
from xauusd_strategy.data.features import FeatureEngineer

__all__ = [
    "DataFetcher",
    "fetch_xauusd_data",
    "DataProcessor",
    "FeatureEngineer",
]
