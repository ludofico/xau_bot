"""
Kelly Criterion calculator for optimal position sizing.

The Kelly Criterion provides the mathematically optimal bet size for
maximizing long-term growth. For trading, we use a fractional Kelly
approach to reduce variance and drawdown risk.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from xauusd_strategy.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class KellyResult:
    """Result of Kelly calculation."""
    full_kelly: float
    fractional_kelly: float
    win_rate: float
    avg_win: float
    avg_loss: float
    edge: float
    recommended_risk_pct: float


class KellyCalculator:
    """
    Kelly Criterion calculator for trading.
    
    The Kelly formula determines optimal bet sizing:
    f* = (p × b - q) / b
    
    Where:
    - p = probability of winning
    - q = probability of losing (1 - p)
    - b = ratio of average win to average loss (payoff ratio)
    
    For trading, we use fractional Kelly (typically 25-50%) because:
    1. Reduces variance/drawdowns
    2. Accounts for estimation errors
    3. More practical for real trading
    """
    
    def __init__(
        self,
        kelly_fraction: float = 0.5,
        max_risk_pct: float = 5.0,
        min_trades_required: int = 30
    ):
        """
        Initialize Kelly calculator.
        
        Args:
            kelly_fraction: Fraction of Kelly to use (0.25 = 25% Kelly)
            max_risk_pct: Maximum risk per trade in percent
            min_trades_required: Minimum trades needed for valid calculation
        """
        self.kelly_fraction = kelly_fraction
        self.max_risk_pct = max_risk_pct
        self.min_trades_required = min_trades_required
    
    def calculate(
        self,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
        trade_results: Optional[List[float]] = None
    ) -> KellyResult:
        """
        Calculate Kelly criterion.
        
        Args:
            win_rate: Win rate (0-1), or calculated from trade_results
            avg_win: Average winning trade, or calculated from trade_results
            avg_loss: Average losing trade (positive), or calculated from trade_results
            trade_results: Optional list of trade P&Ls for calculation
        
        Returns:
            KellyResult with full Kelly, fractional Kelly, and components
        
        Raises:
            ValueError: If insufficient data provided
        """
        # Calculate from trade results if provided
        if trade_results is not None:
            win_rate, avg_win, avg_loss = self._calculate_from_trades(trade_results)
        
        # Validate inputs
        if win_rate is None or avg_win is None or avg_loss is None:
            raise ValueError("Must provide either trade_results or win_rate/avg_win/avg_loss")
        
        if avg_loss <= 0:
            logger.warning("avg_loss must be positive, using absolute value")
            avg_loss = abs(avg_loss)
        
        # Calculate payoff ratio (b)
        payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        # Calculate Kelly: f* = (p × b - q) / b
        p = win_rate
        q = 1 - p
        
        if payoff_ratio <= 0:
            full_kelly = 0
        else:
            full_kelly = (p * payoff_ratio - q) / payoff_ratio
        
        # Clamp to valid range
        full_kelly = max(0, min(full_kelly, 1.0))
        
        # Apply fraction
        fractional_kelly = full_kelly * self.kelly_fraction
        
        # Cap at maximum
        recommended = min(fractional_kelly * 100, self.max_risk_pct)
        
        # Calculate edge (expected value per unit risked)
        edge = (p * payoff_ratio) - q
        
        result = KellyResult(
            full_kelly=full_kelly,
            fractional_kelly=fractional_kelly,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            edge=edge,
            recommended_risk_pct=recommended
        )
        
        logger.debug(
            f"Kelly calculated: full={full_kelly:.2%}, "
            f"fractional={fractional_kelly:.2%}, "
            f"recommended={recommended:.2f}%"
        )
        
        return result
    
    def _calculate_from_trades(
        self,
        trade_results: List[float]
    ) -> Tuple[float, float, float]:
        """Calculate win rate, avg win, avg loss from trade results."""
        if len(trade_results) < self.min_trades_required:
            logger.warning(
                f"Only {len(trade_results)} trades, need {self.min_trades_required} "
                "for reliable Kelly. Using conservative estimate."
            )
        
        wins = [t for t in trade_results if t > 0]
        losses = [t for t in trade_results if t < 0]
        
        if not wins:
            return 0.0, 0.0, 1.0
        if not losses:
            return 1.0, np.mean(wins), 0.0
        
        win_rate = len(wins) / len(trade_results)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        
        return win_rate, avg_win, avg_loss
    
    def calculate_from_strategy_stats(
        self,
        total_trades: int,
        winning_trades: int,
        total_profit: float,
        total_loss: float
    ) -> KellyResult:
        """
        Calculate Kelly from aggregate strategy statistics.
        
        Args:
            total_trades: Total number of trades
            winning_trades: Number of winning trades
            total_profit: Sum of all winning trades
            total_loss: Sum of all losing trades (positive value)
        
        Returns:
            KellyResult
        """
        if total_trades == 0:
            raise ValueError("No trades to calculate from")
        
        losing_trades = total_trades - winning_trades
        
        win_rate = winning_trades / total_trades
        avg_win = total_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
        
        return self.calculate(win_rate, avg_win, avg_loss)
    
    def optimal_growth_rate(self, kelly_result: KellyResult) -> float:
        """
        Calculate expected growth rate using Kelly sizing.
        
        G = p × log(1 + b × f) + q × log(1 - f)
        
        Args:
            kelly_result: Kelly calculation result
        
        Returns:
            Expected log growth rate per trade
        """
        p = kelly_result.win_rate
        q = 1 - p
        b = kelly_result.avg_win / kelly_result.avg_loss if kelly_result.avg_loss > 0 else 0
        f = kelly_result.fractional_kelly
        
        if f <= 0 or f >= 1:
            return 0
        
        try:
            growth = p * np.log(1 + b * f) + q * np.log(1 - f)
            return growth
        except (ValueError, ZeroDivisionError):
            return 0


def calculate_kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    kelly_fraction: float = 0.5
) -> float:
    """
    Convenience function to calculate fractional Kelly.
    
    Args:
        win_rate: Probability of winning (0-1)
        avg_win: Average winning trade amount
        avg_loss: Average losing trade amount (positive)
        kelly_fraction: Fraction of full Kelly to use
    
    Returns:
        Recommended risk percentage (0-100)
    """
    calc = KellyCalculator(kelly_fraction=kelly_fraction)
    result = calc.calculate(win_rate, avg_win, avg_loss)
    return result.recommended_risk_pct


def calculate_optimal_f(
    trade_results: List[float],
    num_simulations: int = 1000
) -> float:
    """
    Calculate optimal f using Monte Carlo simulation.
    
    More robust than Kelly for non-normal distributions.
    
    Args:
        trade_results: List of trade P&L values
        num_simulations: Number of Monte Carlo paths
    
    Returns:
        Optimal f value (fraction to risk)
    """
    if len(trade_results) < 30:
        logger.warning("Insufficient trades for reliable optimal f calculation")
    
    # Normalize results to percentage returns
    max_loss = abs(min(trade_results))
    if max_loss == 0:
        return 0.01  # Minimal risk if no losses
    
    normalized = [t / max_loss for t in trade_results]
    
    # Test different f values
    f_values = np.arange(0.01, 0.50, 0.01)
    terminal_wealth = []
    
    for f in f_values:
        wealths = []
        
        for _ in range(num_simulations):
            # Random path through trades
            path = np.random.choice(normalized, size=len(normalized), replace=True)
            
            # Calculate terminal wealth
            wealth = 1.0
            for ret in path:
                wealth *= (1 + f * ret)
                if wealth <= 0:
                    wealth = 0
                    break
            
            wealths.append(wealth)
        
        terminal_wealth.append(np.median(wealths))
    
    # Find optimal f
    optimal_idx = np.argmax(terminal_wealth)
    optimal_f = f_values[optimal_idx]
    
    logger.debug(f"Optimal f from Monte Carlo: {optimal_f:.2%}")
    
    return optimal_f
