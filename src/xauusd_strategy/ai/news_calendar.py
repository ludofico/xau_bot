"""
Economic Calendar and News Engine for XAUUSD Trading.

Provides:
- Economic event calendar scraping (ForexFactory)
- High-impact event detection (NFP, FOMC, CPI, GDP)
- Trading halt windows around major events
- News sentiment integration with the existing SentimentAnalyst

Features:
- Daily cache to reduce API calls
- Configurable halt windows (pre/post event)
- Risk multiplier based on event impact
"""

import os
import json
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import re

from xauusd_strategy.utils.logger import get_logger

logger = get_logger(__name__)


class EventImpact(Enum):
    """Economic event impact level."""
    CRITICAL = "critical"  # FOMC, NFP - halt trading
    HIGH = "high"          # CPI, GDP - reduce size 50%
    MEDIUM = "medium"      # Jobless claims - reduce size 25%
    LOW = "low"            # Minor data - no adjustment
    UNKNOWN = "unknown"


@dataclass
class EconomicEvent:
    """An economic calendar event."""
    timestamp: datetime
    currency: str
    event_name: str
    impact: EventImpact
    actual: Optional[str] = None
    forecast: Optional[str] = None
    previous: Optional[str] = None
    
    def affects_gold(self) -> bool:
        """Check if this event affects XAUUSD."""
        # Gold is affected by USD events and major global events
        gold_currencies = ["USD", "EUR", "CNY", "JPY", "ALL"]
        return self.currency in gold_currencies
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "currency": self.currency,
            "event_name": self.event_name,
            "impact": self.impact.value,
            "actual": self.actual,
            "forecast": self.forecast,
            "previous": self.previous
        }


@dataclass
class NewsImpact:
    """News impact assessment for trading."""
    level: str  # "Critical", "High", "Medium", "Low"
    sentiment: str  # "Bullish", "Bearish", "Neutral"
    halt_trading: bool
    halt_until: Optional[datetime] = None
    size_multiplier: float = 1.0
    upcoming_events: List[EconomicEvent] = None
    notes: List[str] = None
    
    def __post_init__(self):
        if self.upcoming_events is None:
            self.upcoming_events = []
        if self.notes is None:
            self.notes = []
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON output."""
        return {
            "level": self.level,
            "sentiment": self.sentiment,
            "halt_trading": self.halt_trading,
            "halt_until": self.halt_until.isoformat() if self.halt_until else None,
            "size_multiplier": round(self.size_multiplier, 2),
            "upcoming_events": [e.to_dict() for e in self.upcoming_events],
            "notes": self.notes
        }


class NewsCalendar:
    """
    Economic Calendar Manager with trading halt detection.
    
    Scrapes ForexFactory for economic events and determines
    if trading should be halted or position size adjusted.
    """
    
    # Known high-impact events that affect Gold
    CRITICAL_EVENTS = [
        "Non-Farm Payrolls",
        "Non-Farm Employment Change",
        "NFP",
        "FOMC",
        "Federal Funds Rate",
        "Fed Interest Rate Decision",
        "Fed Chair Powell",
        "Jackson Hole",
    ]
    
    HIGH_IMPACT_EVENTS = [
        "CPI",
        "Consumer Price Index",
        "Core CPI",
        "PPI",
        "Producer Price Index",
        "GDP",
        "Gross Domestic Product",
        "Retail Sales",
        "PCE Price Index",
        "Core PCE",
        "ISM Manufacturing",
        "ISM Services",
        "ECB Interest Rate",
        "ECB President",
        "BoE Interest Rate",
        "BoJ Interest Rate",
    ]
    
    def __init__(
        self,
        cache_dir: str = "data/news_cache",
        halt_before_critical_mins: int = 15,
        halt_after_critical_mins: int = 30,
        halt_before_high_mins: int = 5,
        halt_after_high_mins: int = 15,
    ):
        """
        Initialize News Calendar.
        
        Args:
            cache_dir: Directory for caching calendar data
            halt_before_critical_mins: Minutes before critical event to halt
            halt_after_critical_mins: Minutes after critical event to resume
            halt_before_high_mins: Minutes before high-impact event to halt
            halt_after_high_mins: Minutes after high-impact event to resume
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.halt_before_critical = timedelta(minutes=halt_before_critical_mins)
        self.halt_after_critical = timedelta(minutes=halt_after_critical_mins)
        self.halt_before_high = timedelta(minutes=halt_before_high_mins)
        self.halt_after_high = timedelta(minutes=halt_after_high_mins)
        
        self._events_cache: Dict[str, List[EconomicEvent]] = {}
    
    def _get_cache_path(self, date: datetime) -> Path:
        """Get cache file path for a date."""
        date_str = date.strftime("%Y-%m-%d")
        return self.cache_dir / f"calendar_{date_str}.json"
    
    def _load_from_cache(self, date: datetime) -> Optional[List[EconomicEvent]]:
        """Load events from cache if available and fresh."""
        cache_path = self._get_cache_path(date)
        
        if not cache_path.exists():
            return None
        
        # Cache valid for 6 hours
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if cache_age > timedelta(hours=6):
            return None
        
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            
            events = []
            for item in data:
                events.append(EconomicEvent(
                    timestamp=datetime.fromisoformat(item['timestamp']),
                    currency=item['currency'],
                    event_name=item['event_name'],
                    impact=EventImpact(item['impact']),
                    actual=item.get('actual'),
                    forecast=item.get('forecast'),
                    previous=item.get('previous')
                ))
            return events
        except Exception as e:
            logger.warning(f"Cache load error: {e}")
            return None
    
    def _save_to_cache(self, date: datetime, events: List[EconomicEvent]):
        """Save events to cache."""
        cache_path = self._get_cache_path(date)
        try:
            data = [e.to_dict() for e in events]
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Cache save error: {e}")
    
    def _classify_event(self, event_name: str) -> EventImpact:
        """Classify event impact based on name."""
        name_upper = event_name.upper()
        
        for critical in self.CRITICAL_EVENTS:
            if critical.upper() in name_upper:
                return EventImpact.CRITICAL
        
        for high in self.HIGH_IMPACT_EVENTS:
            if high.upper() in name_upper:
                return EventImpact.HIGH
        
        # Check for generic keywords
        if any(kw in name_upper for kw in ["RATE", "CHAIR", "PRESIDENT", "GOVERNOR"]):
            return EventImpact.HIGH
        
        return EventImpact.MEDIUM
    
    def fetch_events(self, date: Optional[datetime] = None) -> List[EconomicEvent]:
        """
        Fetch economic events for a date.
        
        Uses cache if available, otherwise attempts to scrape.
        Falls back to known recurring events if scraping fails.
        
        Args:
            date: Date to fetch events for (default: today)
            
        Returns:
            List of EconomicEvent objects
        """
        if date is None:
            date = datetime.now()
        
        date_key = date.strftime("%Y-%m-%d")
        
        # Check memory cache
        if date_key in self._events_cache:
            return self._events_cache[date_key]
        
        # Check file cache
        cached = self._load_from_cache(date)
        if cached:
            self._events_cache[date_key] = cached
            return cached
        
        # Attempt to scrape ForexFactory (with fallback)
        events = self._scrape_forex_factory(date)
        
        # If scraping fails, use known recurring events
        if not events:
            events = self._get_known_recurring_events(date)
        
        # Cache results
        if events:
            self._save_to_cache(date, events)
            self._events_cache[date_key] = events
        
        return events
    
    def _scrape_forex_factory(self, date: datetime) -> List[EconomicEvent]:
        """
        Attempt to scrape ForexFactory calendar.
        
        Note: This is a simplified implementation. In production,
        consider using their API or a more robust scraping solution.
        """
        events = []
        
        try:
            import requests
            from bs4 import BeautifulSoup
            
            url = f"https://www.forexfactory.com/calendar"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                logger.warning(f"ForexFactory returned status {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # ForexFactory uses specific class names for events
            # This is a simplified parser - adjust based on actual HTML structure
            event_rows = soup.find_all('tr', class_='calendar__row')
            
            for row in event_rows:
                try:
                    # Extract currency
                    currency_cell = row.find('td', class_='calendar__currency')
                    currency = currency_cell.text.strip() if currency_cell else "USD"
                    
                    # Extract event name
                    event_cell = row.find('td', class_='calendar__event')
                    event_name = event_cell.text.strip() if event_cell else ""
                    
                    if not event_name:
                        continue
                    
                    # Extract impact (ForexFactory uses color/holiday class indicators)
                    impact_cell = row.find('td', class_='calendar__impact')
                    impact_class = impact_cell.find('span').get('class', []) if impact_cell else []
                    
                    if 'high' in str(impact_class).lower() or 'red' in str(impact_class).lower():
                        base_impact = self._classify_event(event_name)
                        # Upgrade to at least HIGH if ForexFactory marks it red
                        if base_impact not in [EventImpact.CRITICAL]:
                            base_impact = EventImpact.HIGH
                    else:
                        base_impact = self._classify_event(event_name)
                    
                    # Extract time
                    time_cell = row.find('td', class_='calendar__time')
                    time_str = time_cell.text.strip() if time_cell else ""
                    
                    # Parse time (ForexFactory uses various formats)
                    event_time = self._parse_event_time(date, time_str)
                    
                    events.append(EconomicEvent(
                        timestamp=event_time,
                        currency=currency,
                        event_name=event_name,
                        impact=base_impact
                    ))
                    
                except Exception as e:
                    logger.debug(f"Error parsing event row: {e}")
                    continue
            
            logger.info(f"Scraped {len(events)} events from ForexFactory")
            return events
            
        except ImportError:
            logger.warning("requests/BeautifulSoup not available for scraping")
            return []
        except Exception as e:
            logger.warning(f"ForexFactory scrape failed: {e}")
            return []
    
    def _parse_event_time(self, date: datetime, time_str: str) -> datetime:
        """Parse ForexFactory time string to datetime."""
        if not time_str or time_str.lower() in ['all day', 'tentative', '']:
            return date.replace(hour=12, minute=0, second=0, microsecond=0)
        
        try:
            # Handle various formats: "8:30am", "2:00pm", "14:30", etc.
            time_str = time_str.lower().strip()
            
            if 'am' in time_str or 'pm' in time_str:
                # 12-hour format
                time_str = time_str.replace('am', ' AM').replace('pm', ' PM')
                parsed = datetime.strptime(time_str.strip(), "%I:%M %p")
            else:
                # 24-hour format
                parsed = datetime.strptime(time_str, "%H:%M")
            
            return date.replace(
                hour=parsed.hour,
                minute=parsed.minute,
                second=0,
                microsecond=0
            )
        except:
            return date.replace(hour=12, minute=0, second=0, microsecond=0)
    
    def _get_known_recurring_events(self, date: datetime) -> List[EconomicEvent]:
        """
        Get known recurring events for fallback.
        
        NFP: First Friday of month
        FOMC: 8 times per year (roughly every 6 weeks)
        """
        events = []
        
        # NFP: First Friday of month
        first_day = date.replace(day=1)
        days_until_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_until_friday)
        
        if date.date() == first_friday.date():
            events.append(EconomicEvent(
                timestamp=date.replace(hour=13, minute=30),  # 8:30 AM EST → 13:30 UTC
                currency="USD",
                event_name="Non-Farm Payrolls",
                impact=EventImpact.CRITICAL
            ))
        
        # FOMC dates for common months (simplified)
        # In production, use actual FOMC schedule
        fomc_dates = [
            (1, 31), (3, 20), (5, 1), (6, 12),
            (7, 31), (9, 18), (11, 6), (12, 18)
        ]
        
        for month, day in fomc_dates:
            if date.month == month and date.day == day:
                events.append(EconomicEvent(
                    timestamp=date.replace(hour=19, minute=0),  # 2:00 PM EST → 19:00 UTC
                    currency="USD",
                    event_name="FOMC Statement",
                    impact=EventImpact.CRITICAL
                ))
        
        return events
    
    def get_upcoming_events(
        self,
        hours_ahead: int = 24,
        min_impact: EventImpact = EventImpact.MEDIUM
    ) -> List[EconomicEvent]:
        """
        Get upcoming events within the specified time window.
        
        Args:
            hours_ahead: Hours to look ahead
            min_impact: Minimum impact level to include
            
        Returns:
            List of upcoming events affecting gold
        """
        now = datetime.now()
        end_time = now + timedelta(hours=hours_ahead)
        
        # Fetch today's and tomorrow's events
        today_events = self.fetch_events(now)
        tomorrow_events = self.fetch_events(now + timedelta(days=1))
        
        all_events = today_events + tomorrow_events
        
        # Filter by time window and impact
        impact_order = [EventImpact.CRITICAL, EventImpact.HIGH, EventImpact.MEDIUM, EventImpact.LOW]
        min_idx = impact_order.index(min_impact)
        
        upcoming = []
        for event in all_events:
            if now <= event.timestamp <= end_time:
                if event.affects_gold():
                    event_idx = impact_order.index(event.impact) if event.impact in impact_order else 999
                    if event_idx <= min_idx:
                        upcoming.append(event)
        
        # Sort by time
        upcoming.sort(key=lambda e: e.timestamp)
        return upcoming
    
    def assess_trading_impact(self, current_time: Optional[datetime] = None) -> NewsImpact:
        """
        Assess current news impact on trading.
        
        Returns NewsImpact with halt status and size multiplier.
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Get events within next 24 hours
        upcoming = self.get_upcoming_events(hours_ahead=24, min_impact=EventImpact.MEDIUM)
        
        halt = False
        halt_until = None
        size_mult = 1.0
        level = "Low"
        notes = []
        
        for event in upcoming:
            time_to_event = event.timestamp - current_time
            time_after_event = current_time - event.timestamp
            
            if event.impact == EventImpact.CRITICAL:
                # Check if within halt window
                if -self.halt_after_critical <= time_to_event <= self.halt_before_critical:
                    halt = True
                    halt_until = event.timestamp + self.halt_after_critical
                    level = "Critical"
                    notes.append(f"HALT: {event.event_name} at {event.timestamp.strftime('%H:%M')}")
                elif timedelta(0) < time_to_event <= timedelta(hours=2):
                    # Approaching critical event
                    size_mult = min(size_mult, 0.5)
                    level = "High" if level != "Critical" else level
                    notes.append(f"CAUTION: {event.event_name} in {int(time_to_event.total_seconds() / 60)} min")
            
            elif event.impact == EventImpact.HIGH:
                if -self.halt_after_high <= time_to_event <= self.halt_before_high:
                    halt = True
                    halt_until = event.timestamp + self.halt_after_high
                    level = "High" if level not in ["Critical"] else level
                    notes.append(f"HALT: {event.event_name} at {event.timestamp.strftime('%H:%M')}")
                elif timedelta(0) < time_to_event <= timedelta(hours=1):
                    size_mult = min(size_mult, 0.75)
                    level = "Medium" if level not in ["Critical", "High"] else level
                    notes.append(f"Upcoming: {event.event_name}")
            
            elif event.impact == EventImpact.MEDIUM:
                if timedelta(0) < time_to_event <= timedelta(minutes=30):
                    size_mult = min(size_mult, 0.9)
        
        # Determine sentiment (simplified: use existing headlines if available)
        sentiment = "Neutral"
        
        return NewsImpact(
            level=level,
            sentiment=sentiment,
            halt_trading=halt,
            halt_until=halt_until,
            size_multiplier=size_mult,
            upcoming_events=upcoming[:5],  # Top 5 upcoming
            notes=notes
        )
    
    def should_halt(self, current_time: Optional[datetime] = None) -> Tuple[bool, Optional[str]]:
        """
        Quick check if trading should be halted.
        
        Returns:
            Tuple of (should_halt: bool, reason: str or None)
        """
        impact = self.assess_trading_impact(current_time)
        
        if impact.halt_trading:
            reason = impact.notes[0] if impact.notes else "High-impact event"
            return True, reason
        
        return False, None
