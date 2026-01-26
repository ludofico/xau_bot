"""
Tests for News Calendar Engine.

Verifies:
- Event classification (CRITICAL, HIGH, MEDIUM)
- Halt window detection
- Trading impact assessment
- Known recurring events (NFP, FOMC)
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

from xauusd_strategy.ai.news_calendar import (
    NewsCalendar,
    EconomicEvent,
    NewsImpact,
    EventImpact
)


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    cache_dir = tempfile.mkdtemp()
    yield cache_dir
    shutil.rmtree(cache_dir, ignore_errors=True)


@pytest.fixture
def news_calendar(temp_cache_dir):
    """Create NewsCalendar instance with temp cache."""
    return NewsCalendar(
        cache_dir=temp_cache_dir,
        halt_before_critical_mins=15,
        halt_after_critical_mins=30,
        halt_before_high_mins=5,
        halt_after_high_mins=15
    )


class TestEconomicEvent:
    """Tests for EconomicEvent dataclass."""
    
    def test_event_creation(self):
        """Test creating an economic event."""
        event = EconomicEvent(
            timestamp=datetime(2024, 1, 5, 13, 30),
            currency="USD",
            event_name="Non-Farm Payrolls",
            impact=EventImpact.CRITICAL
        )
        
        assert event.currency == "USD"
        assert event.impact == EventImpact.CRITICAL
    
    def test_affects_gold_usd(self):
        """Test that USD events affect gold."""
        event = EconomicEvent(
            timestamp=datetime.now(),
            currency="USD",
            event_name="Fed Rate Decision",
            impact=EventImpact.CRITICAL
        )
        
        assert event.affects_gold() is True
    
    def test_affects_gold_eur(self):
        """Test that EUR events affect gold."""
        event = EconomicEvent(
            timestamp=datetime.now(),
            currency="EUR",
            event_name="ECB Rate Decision",
            impact=EventImpact.HIGH
        )
        
        assert event.affects_gold() is True
    
    def test_event_to_dict(self):
        """Test event serialization."""
        event = EconomicEvent(
            timestamp=datetime(2024, 1, 5, 13, 30),
            currency="USD",
            event_name="NFP",
            impact=EventImpact.CRITICAL,
            forecast="180K",
            previous="175K"
        )
        
        result = event.to_dict()
        assert "timestamp" in result
        assert "impact" in result
        assert result["impact"] == "critical"
        assert result["forecast"] == "180K"


class TestNewsCalendar:
    """Tests for NewsCalendar class."""
    
    def test_initialization(self, news_calendar):
        """Test calendar initializes correctly."""
        assert news_calendar.halt_before_critical == timedelta(minutes=15)
        assert news_calendar.halt_after_critical == timedelta(minutes=30)
    
    def test_classify_critical_events(self, news_calendar):
        """Test that NFP and FOMC are classified as critical."""
        assert news_calendar._classify_event("Non-Farm Payrolls") == EventImpact.CRITICAL
        assert news_calendar._classify_event("FOMC Statement") == EventImpact.CRITICAL
        assert news_calendar._classify_event("Fed Chair Powell Speaks") == EventImpact.CRITICAL
    
    def test_classify_high_events(self, news_calendar):
        """Test that CPI and GDP are classified as high."""
        assert news_calendar._classify_event("CPI m/m") == EventImpact.HIGH
        assert news_calendar._classify_event("Core CPI") == EventImpact.HIGH
        assert news_calendar._classify_event("GDP q/q") == EventImpact.HIGH
        assert news_calendar._classify_event("Retail Sales") == EventImpact.HIGH
    
    def test_classify_medium_events(self, news_calendar):
        """Test that unknown events are classified as medium."""
        assert news_calendar._classify_event("Housing Starts") == EventImpact.MEDIUM
        assert news_calendar._classify_event("Trade Balance") == EventImpact.MEDIUM
    
    def test_nfp_friday_detection(self, news_calendar):
        """Test that first Friday of month is detected for NFP."""
        # January 2024: First Friday is January 5th
        nfp_date = datetime(2024, 1, 5)
        events = news_calendar._get_known_recurring_events(nfp_date)
        
        nfp_events = [e for e in events if "Non-Farm" in e.event_name]
        assert len(nfp_events) == 1
        assert nfp_events[0].impact == EventImpact.CRITICAL
    
    def test_non_nfp_friday(self, news_calendar):
        """Test that second Friday doesn't get NFP."""
        # January 2024: Second Friday is January 12th
        non_nfp_date = datetime(2024, 1, 12)
        events = news_calendar._get_known_recurring_events(non_nfp_date)
        
        nfp_events = [e for e in events if "Non-Farm" in e.event_name]
        assert len(nfp_events) == 0
    
    def test_halt_during_critical_event(self, news_calendar):
        """Test that trading halts during critical event window."""
        # Create event 5 minutes from now
        now = datetime.now()
        event_time = now + timedelta(minutes=5)
        
        event = EconomicEvent(
            timestamp=event_time,
            currency="USD",
            event_name="Non-Farm Payrolls",
            impact=EventImpact.CRITICAL
        )
        
        # Manually inject event into cache for testing
        date_key = now.strftime("%Y-%m-%d")
        news_calendar._events_cache[date_key] = [event]
        
        # Check halt status
        impact = news_calendar.assess_trading_impact(now)
        
        assert impact.halt_trading is True
        assert impact.level == "Critical"
    
    def test_no_halt_before_window(self, news_calendar):
        """Test that trading continues when event is far away."""
        # Create event 3 hours from now
        now = datetime.now()
        event_time = now + timedelta(hours=3)
        
        event = EconomicEvent(
            timestamp=event_time,
            currency="USD",
            event_name="CPI m/m",
            impact=EventImpact.HIGH
        )
        
        date_key = now.strftime("%Y-%m-%d")
        news_calendar._events_cache[date_key] = [event]
        
        impact = news_calendar.assess_trading_impact(now)
        
        # Should not halt, but may reduce size
        assert impact.halt_trading is False
    
    def test_size_multiplier_approaching_event(self, news_calendar):
        """Test that size is reduced when approaching high-impact event."""
        now = datetime.now()
        event_time = now + timedelta(minutes=45)  # Within 1 hour
        
        event = EconomicEvent(
            timestamp=event_time,
            currency="USD",
            event_name="Fed Chair Powell Speaks",
            impact=EventImpact.CRITICAL
        )
        
        date_key = now.strftime("%Y-%m-%d")
        news_calendar._events_cache[date_key] = [event]
        
        impact = news_calendar.assess_trading_impact(now)
        
        # Should reduce size even if not halting
        assert impact.size_multiplier <= 0.5
    
    def test_impact_to_dict(self, news_calendar):
        """Test NewsImpact serialization."""
        impact = NewsImpact(
            level="High",
            sentiment="Bearish",
            halt_trading=True,
            halt_until=datetime.now() + timedelta(minutes=30),
            size_multiplier=0.5,
            notes=["NFP incoming"]
        )
        
        result = impact.to_dict()
        assert result["level"] == "High"
        assert result["halt_trading"] is True
        assert result["size_multiplier"] == 0.5
    
    def test_should_halt_quick_check(self, news_calendar):
        """Test the quick halt check method."""
        now = datetime.now()
        event_time = now + timedelta(minutes=10)
        
        event = EconomicEvent(
            timestamp=event_time,
            currency="USD",
            event_name="FOMC Statement",
            impact=EventImpact.CRITICAL
        )
        
        date_key = now.strftime("%Y-%m-%d")
        news_calendar._events_cache[date_key] = [event]
        
        should_halt, reason = news_calendar.should_halt(now)
        
        assert should_halt is True
        assert reason is not None
        assert "FOMC" in reason
    
    def test_empty_calendar_no_halt(self, news_calendar):
        """Test that empty calendar doesn't halt trading."""
        now = datetime.now()
        date_key = now.strftime("%Y-%m-%d")
        news_calendar._events_cache[date_key] = []
        
        should_halt, reason = news_calendar.should_halt(now)
        
        assert should_halt is False
        assert reason is None
