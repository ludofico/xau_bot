"""
Circuit breaker for risk management.

Implements hard stops to protect capital during adverse conditions.
"""

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Optional, Callable
from enum import Enum

from xauusd_strategy.utils.logger import get_logger

logger = get_logger(__name__)


class BreakerState(Enum):
    """Circuit breaker state."""
    ACTIVE = "active"
    TRIGGERED = "triggered"
    COOLING_OFF = "cooling_off"
    DISABLED = "disabled"


@dataclass
class BreakerTrigger:
    """Information about a breaker trigger event."""
    reason: str
    value: float
    threshold: float
    triggered_at: datetime
    cooloff_until: Optional[datetime] = None


class CircuitBreaker:
    """
    Circuit breaker to halt trading under adverse conditions.
    
    Triggers:
    1. Daily drawdown limit
    2. Consecutive losses
    3. Maximum daily trades
    4. Equity below floor
    5. Time-based restrictions
    
    Features:
    - Automatic cooloff periods
    - Progressive triggers
    - Manual override capability
    """
    
    def __init__(
        self,
        # Drawdown limits
        daily_dd_limit_pct: float = 8.0,
        weekly_dd_limit_pct: float = 15.0,
        total_dd_limit_pct: float = 30.0,
        # Trade limits
        max_consecutive_losses: int = 5,
        max_daily_trades: int = 8,
        # Cooloff periods (minutes)
        dd_cooloff_minutes: int = 60,
        loss_streak_cooloff_minutes: int = 30,
        # Equity floor
        equity_floor_pct: float = 50.0,
        # Time restrictions
        restricted_hours: Optional[list] = None,
    ):
        """
        Initialize circuit breaker.
        
        Args:
            daily_dd_limit_pct: Daily drawdown limit (%)
            weekly_dd_limit_pct: Weekly drawdown limit (%)
            total_dd_limit_pct: Total drawdown limit (%)
            max_consecutive_losses: Max losses in a row
            max_daily_trades: Max trades per day
            dd_cooloff_minutes: Cooloff after DD trigger
            loss_streak_cooloff_minutes: Cooloff after loss streak
            equity_floor_pct: Minimum equity as % of initial
            restricted_hours: Hours when trading is disabled
        """
        self.daily_dd_limit_pct = daily_dd_limit_pct
        self.weekly_dd_limit_pct = weekly_dd_limit_pct
        self.total_dd_limit_pct = total_dd_limit_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.max_daily_trades = max_daily_trades
        self.dd_cooloff_minutes = dd_cooloff_minutes
        self.loss_streak_cooloff_minutes = loss_streak_cooloff_minutes
        self.equity_floor_pct = equity_floor_pct
        self.restricted_hours = restricted_hours or []
        
        # State
        self._state = BreakerState.ACTIVE
        self._current_trigger: Optional[BreakerTrigger] = None
        self._trigger_history: list = []
        self._manual_override = False
        
        # Callbacks
        self._on_trigger_callback: Optional[Callable] = None
        self._on_reset_callback: Optional[Callable] = None
    
    @property
    def state(self) -> BreakerState:
        """Get current breaker state."""
        self._check_cooloff()
        return self._state
    
    @property
    def is_triggered(self) -> bool:
        """Check if breaker is currently triggered."""
        return self.state in (BreakerState.TRIGGERED, BreakerState.COOLING_OFF)
    
    @property
    def can_trade(self) -> bool:
        """Check if trading is allowed."""
        if self._manual_override:
            return True
        return self.state == BreakerState.ACTIVE and not self._is_restricted_time()
    
    def check(
        self,
        daily_pnl: float,
        weekly_pnl: float,
        total_pnl: float,
        current_balance: float,
        initial_balance: float,
        consecutive_losses: int,
        daily_trades: int
    ) -> bool:
        """
        Check all breaker conditions.
        
        Args:
            daily_pnl: Today's P&L
            weekly_pnl: This week's P&L
            total_pnl: Total P&L
            current_balance: Current account balance
            initial_balance: Initial account balance
            consecutive_losses: Current loss streak
            daily_trades: Number of trades today
        
        Returns:
            True if trading is allowed, False if breaker triggered
        """
        if self._manual_override:
            return True
        
        # Check daily drawdown
        if daily_pnl < 0:
            daily_dd_pct = abs(daily_pnl) / current_balance * 100
            if daily_dd_pct >= self.daily_dd_limit_pct:
                self._trigger(
                    reason="daily_drawdown",
                    value=daily_dd_pct,
                    threshold=self.daily_dd_limit_pct,
                    cooloff_minutes=self.dd_cooloff_minutes
                )
                return False
        
        # Check weekly drawdown
        if weekly_pnl < 0:
            weekly_dd_pct = abs(weekly_pnl) / current_balance * 100
            if weekly_dd_pct >= self.weekly_dd_limit_pct:
                self._trigger(
                    reason="weekly_drawdown",
                    value=weekly_dd_pct,
                    threshold=self.weekly_dd_limit_pct,
                    cooloff_minutes=self.dd_cooloff_minutes * 4  # Longer cooloff
                )
                return False
        
        # Check total drawdown
        if total_pnl < 0:
            total_dd_pct = abs(total_pnl) / initial_balance * 100
            if total_dd_pct >= self.total_dd_limit_pct:
                self._trigger(
                    reason="total_drawdown",
                    value=total_dd_pct,
                    threshold=self.total_dd_limit_pct,
                    cooloff_minutes=None  # No auto-reset for total DD
                )
                return False
        
        # Check consecutive losses
        if consecutive_losses >= self.max_consecutive_losses:
            self._trigger(
                reason="consecutive_losses",
                value=consecutive_losses,
                threshold=self.max_consecutive_losses,
                cooloff_minutes=self.loss_streak_cooloff_minutes
            )
            return False
        
        # Check daily trade limit
        if daily_trades >= self.max_daily_trades:
            self._trigger(
                reason="max_daily_trades",
                value=daily_trades,
                threshold=self.max_daily_trades,
                cooloff_minutes=None  # Reset at end of day
            )
            return False
        
        # Check equity floor
        equity_pct = current_balance / initial_balance * 100
        if equity_pct < self.equity_floor_pct:
            self._trigger(
                reason="equity_floor",
                value=equity_pct,
                threshold=self.equity_floor_pct,
                cooloff_minutes=None  # No auto-reset
            )
            return False
        
        # Check time restrictions
        if self._is_restricted_time():
            return False
        
        return True
    
    def _trigger(
        self,
        reason: str,
        value: float,
        threshold: float,
        cooloff_minutes: Optional[int]
    ):
        """Trigger the circuit breaker."""
        now = datetime.now()
        
        cooloff_until = None
        if cooloff_minutes:
            cooloff_until = now + timedelta(minutes=cooloff_minutes)
            self._state = BreakerState.COOLING_OFF
        else:
            self._state = BreakerState.TRIGGERED
        
        self._current_trigger = BreakerTrigger(
            reason=reason,
            value=value,
            threshold=threshold,
            triggered_at=now,
            cooloff_until=cooloff_until
        )
        
        self._trigger_history.append(self._current_trigger)
        
        logger.warning(
            f"Circuit breaker TRIGGERED: {reason} "
            f"(value={value:.2f}, threshold={threshold:.2f})"
        )
        
        if cooloff_until:
            logger.info(f"Cooloff until: {cooloff_until.strftime('%H:%M:%S')}")
        
        if self._on_trigger_callback:
            self._on_trigger_callback(self._current_trigger)
    
    def _check_cooloff(self):
        """Check if cooloff period has ended."""
        if self._state != BreakerState.COOLING_OFF:
            return
        
        if self._current_trigger and self._current_trigger.cooloff_until:
            if datetime.now() >= self._current_trigger.cooloff_until:
                self.reset()
    
    def _is_restricted_time(self) -> bool:
        """Check if current time is in restricted hours."""
        if not self.restricted_hours:
            return False
        
        current_hour = datetime.now().hour
        return current_hour in self.restricted_hours
    
    def reset(self):
        """Manually reset the circuit breaker."""
        prev_state = self._state
        self._state = BreakerState.ACTIVE
        self._current_trigger = None
        
        if prev_state != BreakerState.ACTIVE:
            logger.info("Circuit breaker RESET")
            if self._on_reset_callback:
                self._on_reset_callback()
    
    def force_trigger(self, reason: str = "manual"):
        """Manually trigger the circuit breaker."""
        self._trigger(
            reason=reason,
            value=0,
            threshold=0,
            cooloff_minutes=None
        )
    
    def set_manual_override(self, enabled: bool):
        """
        Enable/disable manual override.
        
        WARNING: Use with extreme caution. Bypasses all safety checks.
        """
        self._manual_override = enabled
        if enabled:
            logger.warning("Circuit breaker OVERRIDE ENABLED - trading allowed regardless of triggers")
    
    def on_trigger(self, callback: Callable):
        """Set callback for trigger events."""
        self._on_trigger_callback = callback
    
    def on_reset(self, callback: Callable):
        """Set callback for reset events."""
        self._on_reset_callback = callback
    
    def get_status(self) -> dict:
        """Get current breaker status."""
        return {
            "state": self.state.value,
            "can_trade": self.can_trade,
            "current_trigger": self._current_trigger.reason if self._current_trigger else None,
            "trigger_count": len(self._trigger_history),
            "cooloff_remaining": self._get_cooloff_remaining(),
            "manual_override": self._manual_override,
        }
    
    def _get_cooloff_remaining(self) -> Optional[int]:
        """Get remaining cooloff time in seconds."""
        if self._state != BreakerState.COOLING_OFF:
            return None
        
        if self._current_trigger and self._current_trigger.cooloff_until:
            remaining = (self._current_trigger.cooloff_until - datetime.now()).total_seconds()
            return max(0, int(remaining))
        
        return None
    
    def get_trigger_history(self) -> list:
        """Get list of all trigger events."""
        return [
            {
                "reason": t.reason,
                "value": t.value,
                "threshold": t.threshold,
                "triggered_at": t.triggered_at.isoformat(),
            }
            for t in self._trigger_history
        ]
