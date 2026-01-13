"""
Aggressive compound manager for account growth.

Implements dynamic position sizing with compounding,
anti-martingale logic, and daily tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional, Dict, List
import math

from xauusd_strategy.risk.kelly import KellyCalculator
from xauusd_strategy.risk.position_sizing import PositionSizer, PositionSize
from xauusd_strategy.utils.logger import get_logger, log_trade

logger = get_logger(__name__)


@dataclass
class CompoundState:
    """Track compounding state and performance."""
    initial_balance: float
    current_balance: float
    high_water_mark: float
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    trade_history: List[float] = field(default_factory=list)
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        return self.winning_trades / self.total_trades if self.total_trades > 0 else 0
    
    @property
    def total_return_pct(self) -> float:
        """Calculate total return percentage."""
        return (self.current_balance - self.initial_balance) / self.initial_balance * 100
    
    @property
    def drawdown_from_hwm(self) -> float:
        """Calculate current drawdown from high water mark."""
        if self.high_water_mark <= 0:
            return 0
        return (self.high_water_mark - self.current_balance) / self.high_water_mark * 100
    
    @property
    def profit_factor(self) -> float:
        """Calculate profit factor."""
        gross_profit = sum(t for t in self.trade_history if t > 0)
        gross_loss = abs(sum(t for t in self.trade_history if t < 0))
        return gross_profit / gross_loss if gross_loss > 0 else 0
    
    @property
    def avg_win(self) -> float:
        """Average winning trade."""
        wins = [t for t in self.trade_history if t > 0]
        return sum(wins) / len(wins) if wins else 0
    
    @property
    def avg_loss(self) -> float:
        """Average losing trade (positive value)."""
        losses = [t for t in self.trade_history if t < 0]
        return abs(sum(losses) / len(losses)) if losses else 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'high_water_mark': self.high_water_mark,
            'total_return_pct': self.total_return_pct,
            'drawdown_from_hwm': self.drawdown_from_hwm,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'daily_pnl': self.daily_pnl,
            'weekly_pnl': self.weekly_pnl,
        }


class AggressiveCompoundManager:
    """
    Aggressive compounding with dynamic risk scaling for small account growth.
    
    Features:
    1. Increases position size as account grows (compound)
    2. Reduces risk after losses (anti-martingale)
    3. ML probability-based sizing
    4. Daily/weekly limits and circuit breakers
    5. Streak-based adjustments
    
    Target: €250 → €500-€1000/month with controlled risk.
    
    Example:
        >>> manager = AggressiveCompoundManager(initial_balance=250)
        >>> pos = manager.calculate_position_size(
        ...     entry_price=2000,
        ...     stop_loss_price=1990,
        ...     signal_probability=0.65
        ... )
        >>> print(pos['lots'])
    """
    
    def __init__(
        self,
        initial_balance: float = 250,
        base_risk_pct: float = 2.5,
        max_risk_pct: float = 4.0,
        kelly_fraction: float = 0.5,
        daily_dd_limit: float = 8.0,
        daily_profit_limit: float = 15.0,
        leverage: int = 500,
        account_currency: str = "EUR"
    ):
        """
        Initialize compound manager.
        
        Args:
            initial_balance: Starting account balance
            base_risk_pct: Base risk per trade (%)
            max_risk_pct: Maximum risk per trade (%)
            kelly_fraction: Kelly fraction to use (0.5 = 50% Kelly)
            daily_dd_limit: Daily drawdown limit (%)
            daily_profit_limit: Daily profit target (%)
            leverage: Account leverage
            account_currency: Account currency
        """
        self.initial_balance = initial_balance
        self.base_risk_pct = base_risk_pct
        self.max_risk_pct = max_risk_pct
        self.kelly_fraction = kelly_fraction
        self.daily_dd_limit = daily_dd_limit
        self.daily_profit_limit = daily_profit_limit
        self.leverage = leverage
        
        # Initialize state
        self.state = CompoundState(
            initial_balance=initial_balance,
            current_balance=initial_balance,
            high_water_mark=initial_balance
        )
        
        # Initialize position sizer
        self.position_sizer = PositionSizer(
            balance=initial_balance,
            leverage=leverage,
            account_currency=account_currency
        )
        
        # Kelly calculator
        self.kelly_calc = KellyCalculator(kelly_fraction=kelly_fraction)
        
        # Daily tracking
        self._daily_trades: Dict[str, int] = {}
        self._current_date: Optional[date] = None
        
        logger.info(
            f"AggressiveCompoundManager initialized: "
            f"balance={initial_balance}, risk={base_risk_pct}%, "
            f"kelly={kelly_fraction}, dd_limit={daily_dd_limit}%"
        )
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        signal_probability: Optional[float] = None
    ) -> dict:
        """
        Calculate position size with aggressive compounding.
        
        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            signal_probability: ML model probability (0-1)
        
        Returns:
            Dictionary with lots, risk_amount, and metadata
        """
        # Check circuit breaker
        if self._is_circuit_breaker_triggered():
            return {
                "lots": 0,
                "reason": "circuit_breaker_active",
                "daily_pnl": self.state.daily_pnl,
                "daily_dd_pct": self._get_daily_dd_pct()
            }
        
        # Calculate dynamic risk
        dynamic_risk_pct = self._calculate_dynamic_risk(signal_probability)
        
        # Update position sizer with current balance
        self.position_sizer.update_balance(self.state.current_balance)
        
        # Calculate position
        position = self.position_sizer.calculate(
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            risk_pct=dynamic_risk_pct,
            max_risk_pct=self.max_risk_pct
        )
        
        return {
            "lots": position.lots,
            "risk_amount": position.risk_amount,
            "risk_pct": position.risk_pct,
            "effective_leverage": position.effective_leverage,
            "stop_distance": position.stop_distance,
            "position_value": position.position_value,
            "dynamic_risk_pct": dynamic_risk_pct,
            "signal_probability": signal_probability,
            "balance": self.state.current_balance,
        }
    
    def _calculate_dynamic_risk(self, signal_probability: Optional[float]) -> float:
        """
        Calculate dynamic risk based on multiple factors.
        
        Factors:
        1. Recent performance (anti-martingale)
        2. Signal quality (ML probability)
        3. Account growth tier
        4. Streak adjustments
        """
        risk_pct = self.base_risk_pct
        
        # Factor 1: Anti-martingale (reduce after losses)
        if self.state.daily_pnl < 0:
            loss_ratio = abs(self.state.daily_pnl) / self.state.current_balance
            reduction = min(0.5, loss_ratio * 2)  # Reduce up to 50%
            risk_pct *= (1 - reduction)
            logger.debug(f"Anti-martingale: risk reduced by {reduction:.0%}")
        
        # Factor 2: Increase on daily profit (but cap)
        elif self.state.daily_pnl > self.state.current_balance * 0.03:
            risk_pct *= 1.15  # 15% increase
            logger.debug("Daily profit bonus: +15% risk")
        
        # Factor 3: ML signal quality
        if signal_probability is not None:
            if signal_probability >= 0.70:
                risk_pct *= 1.25  # High confidence
                logger.debug(f"High probability signal ({signal_probability:.0%}): +25% risk")
            elif signal_probability >= 0.60:
                risk_pct *= 1.10  # Medium-high confidence
            elif signal_probability < 0.55:
                risk_pct *= 0.75  # Low confidence
                logger.debug(f"Low probability signal ({signal_probability:.0%}): -25% risk")
        
        # Factor 4: Account growth tier
        growth_pct = self.state.total_return_pct
        if growth_pct >= 100:  # Doubled account
            risk_pct *= 1.10  # Play with house money
            logger.debug("Account doubled: +10% risk bonus")
        elif growth_pct >= 50:
            risk_pct *= 1.05
        
        # Factor 5: Streak adjustments
        if self.state.consecutive_losses >= 3:
            risk_pct *= 0.7  # Reduce on losing streak
            logger.debug(f"Losing streak ({self.state.consecutive_losses}): -30% risk")
        elif self.state.consecutive_wins >= 3:
            risk_pct *= 1.1  # Slight increase on winning streak
            logger.debug(f"Winning streak ({self.state.consecutive_wins}): +10% risk")
        
        # Cap at maximum
        final_risk = min(risk_pct, self.max_risk_pct)
        
        # Minimum floor
        final_risk = max(final_risk, 0.5)
        
        return final_risk
    
    def _is_circuit_breaker_triggered(self) -> bool:
        """Check if daily drawdown limit is hit."""
        daily_dd_pct = self._get_daily_dd_pct()
        
        if daily_dd_pct >= self.daily_dd_limit:
            logger.warning(
                f"Circuit breaker triggered: daily DD {daily_dd_pct:.1f}% >= "
                f"limit {self.daily_dd_limit}%"
            )
            return True
        
        return False
    
    def _get_daily_dd_pct(self) -> float:
        """Get current daily drawdown percentage."""
        if self.state.daily_pnl >= 0:
            return 0
        return abs(self.state.daily_pnl) / self.state.current_balance * 100
    
    def update_trade_result(self, pnl: float):
        """
        Update state after a trade closes.
        
        Args:
            pnl: Trade profit/loss in account currency
        """
        is_winner = pnl > 0
        
        # Update balance
        self.state.current_balance += pnl
        self.state.daily_pnl += pnl
        self.state.weekly_pnl += pnl
        self.state.monthly_pnl += pnl
        
        # Update trade counts
        self.state.total_trades += 1
        self.state.trade_history.append(pnl)
        
        if is_winner:
            self.state.winning_trades += 1
            self.state.consecutive_wins += 1
            self.state.consecutive_losses = 0
            self.state.max_consecutive_wins = max(
                self.state.max_consecutive_wins,
                self.state.consecutive_wins
            )
            if pnl > self.state.largest_win:
                self.state.largest_win = pnl
        else:
            self.state.losing_trades += 1
            self.state.consecutive_losses += 1
            self.state.consecutive_wins = 0
            self.state.max_consecutive_losses = max(
                self.state.max_consecutive_losses,
                self.state.consecutive_losses
            )
            if pnl < self.state.largest_loss:
                self.state.largest_loss = pnl
        
        # Update high water mark
        if self.state.current_balance > self.state.high_water_mark:
            self.state.high_water_mark = self.state.current_balance
        
        # Update position sizer
        self.position_sizer.update_balance(self.state.current_balance)
        
        log_trade(
            action="RESULT",
            pnl=pnl,
            balance=self.state.current_balance,
            win_rate=f"{self.state.win_rate:.0%}",
            daily_pnl=self.state.daily_pnl
        )
    
    def reset_daily(self):
        """Reset daily counters (call at start of each trading day)."""
        self.state.daily_pnl = 0
        self._daily_trades = {}
        self._current_date = date.today()
        logger.info(f"Daily reset: balance={self.state.current_balance:.2f}")
    
    def reset_weekly(self):
        """Reset weekly counters."""
        self.state.weekly_pnl = 0
        logger.info(f"Weekly reset: weekly_pnl=0")
    
    def reset_monthly(self):
        """Reset monthly counters."""
        self.state.monthly_pnl = 0
        logger.info(f"Monthly reset: monthly_pnl=0")
    
    def get_kelly_recommendation(self) -> dict:
        """
        Get Kelly-based sizing recommendation.
        
        Returns:
            Dictionary with Kelly calculation results
        """
        if self.state.total_trades < 20:
            return {
                "error": "insufficient_trades",
                "trades_needed": 20 - self.state.total_trades,
                "using_default": self.base_risk_pct
            }
        
        try:
            result = self.kelly_calc.calculate(
                trade_results=self.state.trade_history
            )
            return {
                "full_kelly": result.full_kelly * 100,
                "fractional_kelly": result.fractional_kelly * 100,
                "recommended_risk_pct": result.recommended_risk_pct,
                "win_rate": result.win_rate,
                "edge": result.edge
            }
        except Exception as e:
            logger.warning(f"Kelly calculation failed: {e}")
            return {"error": str(e), "using_default": self.base_risk_pct}
    
    def get_monthly_projection(self) -> dict:
        """
        Project monthly returns based on current statistics.
        
        Returns:
            Dictionary with projections
        """
        if self.state.total_trades < 10:
            return {"error": "insufficient_data", "trades_needed": 10}
        
        # Average trade return
        avg_return = sum(self.state.trade_history) / len(self.state.trade_history)
        
        # Estimated trades per month (based on current frequency)
        # Assume 3 trades per day * 20 trading days
        est_trades_per_month = 60
        
        projected_monthly_pnl = avg_return * est_trades_per_month
        projected_monthly_pct = (projected_monthly_pnl / self.initial_balance) * 100
        
        # Risk-adjusted projection (account for variance)
        std_return = (
            sum((t - avg_return) ** 2 for t in self.state.trade_history) /
            len(self.state.trade_history)
        ) ** 0.5
        
        conservative_projection = (avg_return - std_return) * est_trades_per_month
        optimistic_projection = (avg_return + std_return) * est_trades_per_month
        
        return {
            "avg_trade_return": avg_return,
            "win_rate": self.state.win_rate,
            "profit_factor": self.state.profit_factor,
            "projected_monthly_pnl": projected_monthly_pnl,
            "projected_monthly_pct": projected_monthly_pct,
            "conservative_pnl": conservative_projection,
            "optimistic_pnl": optimistic_projection,
            "target_500_achievable": projected_monthly_pnl >= 500,
            "target_1000_achievable": projected_monthly_pnl >= 1000,
        }
    
    def get_status(self) -> dict:
        """Get current manager status."""
        return {
            "balance": self.state.current_balance,
            "total_return_pct": self.state.total_return_pct,
            "daily_pnl": self.state.daily_pnl,
            "weekly_pnl": self.state.weekly_pnl,
            "monthly_pnl": self.state.monthly_pnl,
            "drawdown_from_hwm": self.state.drawdown_from_hwm,
            "win_rate": self.state.win_rate,
            "total_trades": self.state.total_trades,
            "consecutive_wins": self.state.consecutive_wins,
            "consecutive_losses": self.state.consecutive_losses,
            "circuit_breaker_active": self._is_circuit_breaker_triggered(),
        }
