"""
Time utilities for session detection and timezone handling.

All times are assumed to be in CET (Central European Time) unless specified.
"""

from datetime import datetime, time, timedelta
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd


class TradingSession(Enum):
    """Trading session identifiers."""
    ASIAN = "asian"
    LONDON = "london"
    NY = "ny"
    OVERLAP_LONDON_NY = "london_ny_overlap"
    OFF_HOURS = "off_hours"


@dataclass
class SessionTimes:
    """Session start and end times."""
    name: TradingSession
    start: time
    end: time
    
    def contains(self, t: time) -> bool:
        """Check if time is within session."""
        if self.start <= self.end:
            return self.start <= t <= self.end
        else:
            # Handles overnight sessions
            return t >= self.start or t <= self.end


# Default session definitions (CET timezone)
DEFAULT_SESSIONS = {
    TradingSession.ASIAN: SessionTimes(
        name=TradingSession.ASIAN,
        start=time(0, 0),
        end=time(7, 0)
    ),
    TradingSession.LONDON: SessionTimes(
        name=TradingSession.LONDON,
        start=time(8, 0),
        end=time(16, 30)
    ),
    TradingSession.NY: SessionTimes(
        name=TradingSession.NY,
        start=time(14, 30),
        end=time(21, 0)
    ),
    TradingSession.OVERLAP_LONDON_NY: SessionTimes(
        name=TradingSession.OVERLAP_LONDON_NY,
        start=time(14, 30),
        end=time(16, 30)
    ),
}


def get_session_times(
    session: TradingSession,
    custom_start: Optional[str] = None,
    custom_end: Optional[str] = None
) -> SessionTimes:
    """
    Get session times with optional custom override.
    
    Args:
        session: Session identifier
        custom_start: Optional custom start time (HH:MM format)
        custom_end: Optional custom end time (HH:MM format)
    
    Returns:
        SessionTimes object
    """
    default = DEFAULT_SESSIONS.get(session)
    
    if not default:
        raise ValueError(f"Unknown session: {session}")
    
    start = default.start
    end = default.end
    
    if custom_start:
        h, m = map(int, custom_start.split(":"))
        start = time(h, m)
    
    if custom_end:
        h, m = map(int, custom_end.split(":"))
        end = time(h, m)
    
    return SessionTimes(name=session, start=start, end=end)


def is_in_session(
    timestamp: datetime | pd.Timestamp,
    session: TradingSession,
    custom_start: Optional[str] = None,
    custom_end: Optional[str] = None
) -> bool:
    """
    Check if timestamp is within a trading session.
    
    Args:
        timestamp: Timestamp to check
        session: Session to check against
        custom_start: Optional custom start time
        custom_end: Optional custom end time
    
    Returns:
        True if timestamp is within session
    """
    session_times = get_session_times(session, custom_start, custom_end)
    t = timestamp.time() if hasattr(timestamp, 'time') else timestamp
    
    return session_times.contains(t)


def get_current_session(
    timestamp: datetime | pd.Timestamp,
) -> TradingSession:
    """
    Determine which trading session a timestamp falls into.
    
    Args:
        timestamp: Timestamp to check
    
    Returns:
        TradingSession enum value
    """
    t = timestamp.time() if hasattr(timestamp, 'time') else timestamp
    
    # Check overlap first (most volatile)
    if DEFAULT_SESSIONS[TradingSession.OVERLAP_LONDON_NY].contains(t):
        return TradingSession.OVERLAP_LONDON_NY
    
    # Check individual sessions
    for session, times in DEFAULT_SESSIONS.items():
        if session != TradingSession.OVERLAP_LONDON_NY and times.contains(t):
            return session
    
    return TradingSession.OFF_HOURS


def get_session_data(
    df: pd.DataFrame,
    session: TradingSession,
    date: Optional[datetime] = None,
    custom_start: Optional[str] = None,
    custom_end: Optional[str] = None
) -> pd.DataFrame:
    """
    Filter DataFrame for a specific session.
    
    Args:
        df: DataFrame with DatetimeIndex
        session: Session to filter
        date: Specific date (if None, returns all dates)
        custom_start: Optional custom start time
        custom_end: Optional custom end time
    
    Returns:
        Filtered DataFrame
    """
    session_times = get_session_times(session, custom_start, custom_end)
    
    # Create time mask
    times = df.index.time
    
    if session_times.start <= session_times.end:
        mask = (times >= session_times.start) & (times <= session_times.end)
    else:
        mask = (times >= session_times.start) | (times <= session_times.end)
    
    # Apply date filter if specified
    if date:
        date_mask = df.index.date == date.date()
        mask = mask & date_mask
    
    return df[mask]


def get_asian_box(
    df: pd.DataFrame,
    date: datetime | pd.Timestamp,
    start: str = "00:00",
    end: str = "07:00"
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate Asian session box (high/low) for a specific date.
    
    Args:
        df: OHLC DataFrame with DatetimeIndex
        date: Date to calculate box for
        start: Session start time (HH:MM)
        end: Session end time (HH:MM)
    
    Returns:
        Tuple of (asian_high, asian_low) or (None, None) if no data
    """
    target_date = date.date() if hasattr(date, 'date') else date
    
    start_time = time(*map(int, start.split(":")))
    end_time = time(*map(int, end.split(":")))
    
    # Filter for Asian session on target date
    mask = (
        (df.index.date == target_date) &
        (df.index.time >= start_time) &
        (df.index.time < end_time)
    )
    
    asian_data = df[mask]
    
    if len(asian_data) == 0:
        return None, None
    
    return asian_data['high'].max(), asian_data['low'].min()


def calculate_session_volatility(
    df: pd.DataFrame,
    session: TradingSession
) -> pd.Series:
    """
    Calculate average volatility (ATR) for each session.
    
    Args:
        df: OHLC DataFrame
        session: Session to analyze
    
    Returns:
        Series with date index and average ATR values
    """
    session_data = get_session_data(df, session)
    
    # Group by date and calculate range
    daily_ranges = session_data.groupby(session_data.index.date).apply(
        lambda x: x['high'].max() - x['low'].min()
    )
    
    return daily_ranges


def is_trading_day(date: datetime | pd.Timestamp) -> bool:
    """
    Check if date is a valid trading day (weekday).
    
    Note: Does not check for holidays.
    
    Args:
        date: Date to check
    
    Returns:
        True if weekday (Mon-Fri)
    """
    weekday = date.weekday() if hasattr(date, 'weekday') else date
    return weekday < 5  # Monday = 0, Friday = 4


def get_next_session_start(
    current_time: datetime,
    session: TradingSession
) -> datetime:
    """
    Get the datetime of the next session start.
    
    Args:
        current_time: Current datetime
        session: Target session
    
    Returns:
        Datetime of next session start
    """
    session_times = get_session_times(session)
    
    # Create datetime with session start time
    next_session = datetime.combine(current_time.date(), session_times.start)
    
    # If session already started today, move to tomorrow
    if current_time.time() >= session_times.start:
        next_session += timedelta(days=1)
    
    # Skip weekends
    while not is_trading_day(next_session):
        next_session += timedelta(days=1)
    
    return next_session
