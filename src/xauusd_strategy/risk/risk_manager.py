"""
Unified Risk Manager for XAUUSD Trading.

Provides:
- Dynamic position sizing (Kelly, ATR-based, news-adjusted)
- Max drawdown tracking and circuit breaker integration
- Risk-per-trade enforcement
- Trade risk scoring

Combines:
- Circuit breaker (emergency halts)
- Kelly criterion (optimal sizing)
- News impact (size reduction)
- Regime-based adjustments
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import json

from xauusd_strategy.risk.circuit_breaker import CircuitBreaker, BreakerState
from xauusd_strategy.risk.kelly import KellyCalculator
from xauusd_strategy.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RiskInfo:
    """Risk assessment for a potential trade."""
    can_trade: bool
    position_size: float
    max_loss: float
    risk_percent: float
    size_multiplier: float
    
    # Context
    circuit_status: str
    kelly_size: float
    news_multiplier: float
    regime_multiplier: float
    volatility_multiplier: float
    
    notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON output."""
        return {
            "can_trade": self.can_trade,
            "position_size": round(self.position_size, 4),
            "max_loss": round(self.max_loss, 2),
            "risk_percent": round(self.risk_percent, 2),
            "size_multiplier": round(self.size_multiplier, 3),
            "circuit_status": self.circuit_status,
            "kelly_size": round(self.kelly_size, 4),
            "news_multiplier": round(self.news_multiplier, 2),
            "regime_multiplier": round(self.regime_multiplier, 2),
            "volatility_multiplier": round(self.volatility_multiplier, 2),
            "notes": self.notes
        }


@dataclass
class DrawdownStats:
    """Drawdown tracking statistics."""
    current_equity: float
    peak_equity: float
    current_drawdown_pct: float
    max_drawdown_pct: float
    daily_pnl: float
    daily_pnl_pct: float
    weekly_pnl: float
    consecutive_losses: int
    trades_today: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "current_equity": round(self.current_equity, 2),
            "peak_equity": round(self.peak_equity, 2),
            "current_drawdown_pct": round(self.current_drawdown_pct, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_pnl_pct": round(self.daily_pnl_pct, 2),
            "weekly_pnl": round(self.weekly_pnl, 2),
            "consecutive_losses": self.consecutive_losses,
            "trades_today": self.trades_today
        }


class RiskManager:
    """
    Unified Risk Management System.
    
    Integrates:
    - Circuit breaker for emergency halts
    - Kelly criterion for optimal sizing
    - Dynamic multipliers (news, regime, volatility)
    - Drawdown tracking
    """
    
    def __init__(
        self,
        # Account settings
        initial_balance: float = 250.0,
        base_risk_pct: float = 3.0,
        max_risk_pct: float = 5.0,
        
        # Position limits
        max_open_positions: int = 5,
        max_trades_per_day: int = 10,
        min_lot_size: float = 0.01,
        max_lot_size: float = 1.0,
        
        # Drawdown limits
        max_daily_drawdown_pct: float = 10.0,
        max_weekly_drawdown_pct: float = 20.0,
        max_total_drawdown_pct: float = 40.0,
        
        # Kelly settings
        kelly_fraction: float = 0.5,
        
        # Circuit breaker settings
        consecutive_loss_limit: int = 4,
        
        # State persistence
        state_path: Optional[str] = "monitor/risk_state.json"
    ):
        """Initialize Risk Manager."""
        self.initial_balance = initial_balance
        self.base_risk_pct = base_risk_pct
        self.max_risk_pct = max_risk_pct
        
        self.max_open_positions = max_open_positions
        self.max_trades_per_day = max_trades_per_day
        self.min_lot_size = min_lot_size
        self.max_lot_size = max_lot_size
        
        self.max_daily_dd = max_daily_drawdown_pct
        self.max_weekly_dd = max_weekly_drawdown_pct
        self.max_total_dd = max_total_drawdown_pct
        
        self.kelly_fraction = kelly_fraction
        self.state_path = Path(state_path) if state_path else None
        
        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker(
            daily_dd_limit_pct=max_daily_drawdown_pct,
            weekly_dd_limit_pct=max_weekly_drawdown_pct,
            total_dd_limit_pct=max_total_drawdown_pct,
            max_consecutive_losses=consecutive_loss_limit,
            max_daily_trades=max_trades_per_day
        )
        
        # Initialize Kelly
        self.kelly = KellyCalculator(kelly_fraction=kelly_fraction)
        
        # Tracking state
        self.peak_equity = initial_balance
        self.day_start_equity = initial_balance
        self.week_start_equity = initial_balance
        self.current_equity = initial_balance
        
        self.trades_today = 0
        self.consecutive_losses = 0
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        
        self.last_day_reset = datetime.now().date()
        self.last_week_reset = datetime.now().isocalendar()[1]
        
        # Trade history for Kelly calculation
        self.trade_results: List[float] = []
        
        # Load state if exists
        self._load_state()
    
    def _load_state(self):
        """Load persisted state."""
        if self.state_path and self.state_path.exists():
            try:
                with open(self.state_path, 'r') as f:
                    state = json.load(f)
                
                self.peak_equity = state.get('peak_equity', self.initial_balance)
                self.trades_today = state.get('trades_today', 0)
                self.consecutive_losses = state.get('consecutive_losses', 0)
                self.trade_results = state.get('trade_results', [])[-50:]  # Keep last 50
                
                # Check if day/week reset needed
                saved_date = state.get('date', None)
                if saved_date and saved_date != datetime.now().date().isoformat():
                    self._reset_daily()
                
                logger.info("Risk state restored")
            except Exception as e:
                logger.warning(f"Could not load risk state: {e}")
    
    def _save_state(self):
        """Persist current state."""
        if not self.state_path:
            return
            
        try:
            self.state_path.parent.mkdir(exist_ok=True)
            state = {
                'date': datetime.now().date().isoformat(),
                'week': datetime.now().isocalendar()[1],
                'peak_equity': self.peak_equity,
                'trades_today': self.trades_today,
                'consecutive_losses': self.consecutive_losses,
                'trade_results': self.trade_results[-50:]
            }
            with open(self.state_path, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save risk state: {e}")
    
    def _reset_daily(self):
        """Reset daily counters."""
        self.trades_today = 0
        self.day_start_equity = self.current_equity
        self.daily_pnl = 0.0
        self.last_day_reset = datetime.now().date()
        logger.info("Daily risk counters reset")
    
    def _reset_weekly(self):
        """Reset weekly counters."""
        self.week_start_equity = self.current_equity
        self.weekly_pnl = 0.0
        self.last_week_reset = datetime.now().isocalendar()[1]
        logger.info("Weekly risk counters reset")
    
    def update_equity(self, equity: float):
        """
        Update current equity and check for resets.
        
        Args:
            equity: Current account equity
        """
        self.current_equity = equity
        
        # Update peak
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        # Check for daily reset
        if datetime.now().date() != self.last_day_reset:
            self._reset_daily()
        
        # Check for weekly reset
        if datetime.now().isocalendar()[1] != self.last_week_reset:
            self._reset_weekly()
        
        # Update P&L
        self.daily_pnl = equity - self.day_start_equity
        self.weekly_pnl = equity - self.week_start_equity
    
    def record_trade(self, pnl: float):
        """
        Record a completed trade.
        
        Args:
            pnl: Trade profit/loss
        """
        self.trade_results.append(pnl)
        self.trades_today += 1
        
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Update kelly with new data
        if len(self.trade_results) >= 10:
            wins = [r for r in self.trade_results if r > 0]
            losses = [r for r in self.trade_results if r < 0]
            if wins and losses:
                win_rate = len(wins) / len(self.trade_results)
                avg_win = sum(wins) / len(wins)
                avg_loss = abs(sum(losses) / len(losses))
                # Update kelly with new data (recalculate each time)
                self._last_kelly_result = self.kelly.calculate(
                    win_rate=win_rate,
                    avg_win=avg_win,
                    avg_loss=avg_loss
                )
        
        self._save_state()
    
    def get_drawdown_stats(self) -> DrawdownStats:
        """Get current drawdown statistics."""
        current_dd = (self.peak_equity - self.current_equity) / self.peak_equity * 100
        max_dd = current_dd  # Simplified - in production track historical max
        
        daily_pnl_pct = 0.0
        if self.day_start_equity > 0:
            daily_pnl_pct = self.daily_pnl / self.day_start_equity * 100
        
        return DrawdownStats(
            current_equity=self.current_equity,
            peak_equity=self.peak_equity,
            current_drawdown_pct=current_dd,
            max_drawdown_pct=max_dd,
            daily_pnl=self.daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            weekly_pnl=self.weekly_pnl,
            consecutive_losses=self.consecutive_losses,
            trades_today=self.trades_today
        )
    
    def calculate_position_size(
        self,
        sl_distance: float,
        pip_value: float = 0.86,
        news_multiplier: float = 1.0,
        regime_multiplier: float = 1.0,
        volatility_multiplier: float = 1.0
    ) -> Tuple[float, List[str]]:
        """
        Calculate position size with all adjustments.
        
        Args:
            sl_distance: Stop loss distance in price units
            pip_value: Value per pip per 0.01 lot
            news_multiplier: 0.0-1.0 based on news risk
            regime_multiplier: 0.5-1.2 based on regime
            volatility_multiplier: 0.5-1.0 based on ATR
            
        Returns:
            Tuple of (lot_size, notes)
        """
        notes = []
        
        # Base risk amount
        risk_pct = self.base_risk_pct
        risk_amount = self.current_equity * (risk_pct / 100)
        
        # Kelly adjustment (use latest result if available)
        kelly_pct = 0.0
        if hasattr(self, '_last_kelly_result') and self._last_kelly_result:
            kelly_pct = self._last_kelly_result.recommended_risk_pct
        if kelly_pct > 0 and kelly_pct < risk_pct:
            risk_pct = kelly_pct
            notes.append(f"Kelly reduced risk to {kelly_pct:.1f}%")
        
        # Cap at max risk
        if risk_pct > self.max_risk_pct:
            risk_pct = self.max_risk_pct
            notes.append(f"Capped at max risk {self.max_risk_pct}%")
        
        risk_amount = self.current_equity * (risk_pct / 100)
        
        # Apply multipliers
        combined_mult = news_multiplier * regime_multiplier * volatility_multiplier
        combined_mult = max(0.1, min(1.5, combined_mult))  # Limit range
        
        if combined_mult < 1.0:
            notes.append(f"Size reduced by {(1-combined_mult)*100:.0f}%")
        
        risk_amount *= combined_mult
        
        # Calculate lot size
        if sl_distance > 0 and pip_value > 0:
            sl_pips = sl_distance / 0.01  # Convert $ to pips
            lot_size = risk_amount / (sl_pips * pip_value * 100)
        else:
            lot_size = self.min_lot_size
        
        # Apply limits
        lot_size = round(lot_size, 2)
        lot_size = max(self.min_lot_size, min(self.max_lot_size, lot_size))
        
        return lot_size, notes
    
    def assess_risk(
        self,
        sl_distance: float,
        pip_value: float = 0.86,
        current_positions: int = 0,
        news_multiplier: float = 1.0,
        regime_multiplier: float = 1.0,
        volatility_multiplier: float = 1.0
    ) -> RiskInfo:
        """
        Assess risk for a potential new trade.
        
        Args:
            sl_distance: Stop loss distance in price units
            pip_value: Value per pip per 0.01 lot
            current_positions: Number of currently open positions
            news_multiplier: 0.0-1.0 based on news impact
            regime_multiplier: 0.5-1.2 based on market regime
            volatility_multiplier: 0.5-1.0 based on ATR level
            
        Returns:
            RiskInfo with complete risk assessment
        """
        notes = []
        can_trade = True
        
        # Check circuit breaker
        self.circuit_breaker.check(
            daily_pnl=self.daily_pnl,
            weekly_pnl=self.weekly_pnl,
            total_pnl=self.current_equity - self.initial_balance,
            current_balance=self.current_equity,
            initial_balance=self.initial_balance,
            consecutive_losses=self.consecutive_losses,
            daily_trades=self.trades_today
        )
        
        circuit_status = self.circuit_breaker.state.value
        
        if not self.circuit_breaker.can_trade:
            can_trade = False
            notes.append(f"Circuit breaker triggered: {circuit_status}")
        
        # Check position limits
        if current_positions >= self.max_open_positions:
            can_trade = False
            notes.append(f"Max positions ({self.max_open_positions}) reached")
        
        # Check daily trade limit
        if self.trades_today >= self.max_trades_per_day:
            can_trade = False
            notes.append(f"Daily trade limit ({self.max_trades_per_day}) reached")
        
        # Check news halt
        if news_multiplier == 0.0:
            can_trade = False
            notes.append("News halt in effect")
        
        # Calculate position size
        lot_size, size_notes = self.calculate_position_size(
            sl_distance=sl_distance,
            pip_value=pip_value,
            news_multiplier=news_multiplier,
            regime_multiplier=regime_multiplier,
            volatility_multiplier=volatility_multiplier
        )
        notes.extend(size_notes)
        
        # Calculate max loss for this trade
        sl_pips = sl_distance / 0.01 if sl_distance > 0 else 0
        max_loss = lot_size * sl_pips * pip_value * 100
        
        # Risk percent
        risk_pct = (max_loss / self.current_equity) * 100 if self.current_equity > 0 else 0
        
        # Combined multiplier
        combined_mult = news_multiplier * regime_multiplier * volatility_multiplier
        
        # Kelly size (before all adjustments)
        kelly_risk = 0.0
        if hasattr(self, '_last_kelly_result') and self._last_kelly_result:
            kelly_risk = self._last_kelly_result.recommended_risk_pct / 100
        kelly_amount = self.current_equity * kelly_risk
        if sl_distance > 0 and pip_value > 0:
            sl_pips = sl_distance / 0.01
            kelly_size = kelly_amount / (sl_pips * pip_value * 100) if kelly_amount > 0 else self.min_lot_size
            kelly_size = round(kelly_size, 2)
        else:
            kelly_size = self.min_lot_size
        
        return RiskInfo(
            can_trade=can_trade,
            position_size=lot_size,
            max_loss=max_loss,
            risk_percent=risk_pct,
            size_multiplier=combined_mult,
            circuit_status=circuit_status,
            kelly_size=kelly_size,
            news_multiplier=news_multiplier,
            regime_multiplier=regime_multiplier,
            volatility_multiplier=volatility_multiplier,
            notes=notes
        )
    
    def reset_circuit_breaker(self):
        """Manually reset the circuit breaker."""
        self.circuit_breaker.reset()
        logger.info("Circuit breaker manually reset")
    
    def get_status(self) -> Dict:
        """Get current risk manager status."""
        dd_stats = self.get_drawdown_stats()
        
        return {
            "equity": round(self.current_equity, 2),
            "peak_equity": round(self.peak_equity, 2),
            "drawdown_pct": round(dd_stats.current_drawdown_pct, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "trades_today": self.trades_today,
            "consecutive_losses": self.consecutive_losses,
            "circuit_status": self.circuit_breaker.state.value,
            "can_trade": self.circuit_breaker.can_trade
        }
