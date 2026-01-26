"""AI module for XAUUSD trading - sentiment and news analysis."""

from xauusd_strategy.ai.sentiment import SentimentAnalyst
from xauusd_strategy.ai.news_calendar import (
    NewsCalendar,
    EconomicEvent,
    NewsImpact,
    EventImpact
)

__all__ = [
    "SentimentAnalyst",
    "NewsCalendar",
    "EconomicEvent", 
    "NewsImpact",
    "EventImpact"
]
