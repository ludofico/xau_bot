"""
Performance metrics calculation utilities.
"""

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
import numpy as np


@dataclass
class PerformanceMetrics:
    """Collection of trading performance metrics."""
    
    # Returns
    total_return: float
    cagr: float
    monthly_return: float
    
    # Risk
    max_drawdown: float
    avg_drawdown: float
    volatility: float
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Win/loss
    win_rate: float
    profit_factor: float
    payoff_ratio: float
    expectancy: float
    
    # Trade stats
    total_trades: int
    avg_trade: float
    avg_winner: float
    avg_loser: float
    
    @classmethod
    def from_trades(
        cls,
        trades: pd.DataFrame,
        initial_balance: float = 250,
        trading_days_per_year: int = 252
    ) -> "PerformanceMetrics":
        """
        Calculate metrics from trade log.
        
        Args:
            trades: DataFrame with 'pnl' column
            initial_balance: Starting balance
            trading_days_per_year: Trading days per year for annualization
        
        Returns:
            PerformanceMetrics instance
        """
        if trades.empty or 'pnl' not in trades.columns:
            return cls._empty()
        
        pnls = trades['pnl'].values
        
        # Calculate cumulative returns
        final_balance = initial_balance + pnls.sum()
        total_return = (final_balance - initial_balance) / initial_balance
        
        # Time period
        if 'entry_time' in trades.columns and 'exit_time' in trades.columns:
            days = (trades['exit_time'].iloc[-1] - trades['entry_time'].iloc[0]).days
            years = max(0.1, days / 365)
        else:
            years = 1
        
        # CAGR
        cagr = (final_balance / initial_balance) ** (1 / years) - 1
        
        # Monthly return
        months = max(1, years * 12)
        monthly_return = (final_balance / initial_balance) ** (1 / months) - 1
        
        # Build equity curve
        equity = initial_balance + np.cumsum(pnls)
        
        # Drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = drawdown.max()
        avg_drawdown = drawdown.mean()
        
        # Volatility (annualized)
        returns = pnls / np.maximum(1, np.roll(equity, 1))
        volatility = np.std(returns) * np.sqrt(trading_days_per_year)
        
        # Sharpe ratio
        risk_free_rate = 0.02  # 2% annual
        excess_return = cagr - risk_free_rate
        sharpe = excess_return / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside volatility)
        negative_returns = returns[returns < 0]
        downside_vol = np.std(negative_returns) * np.sqrt(trading_days_per_year) if len(negative_returns) > 0 else 0
        sortino = excess_return / downside_vol if downside_vol > 0 else 0
        
        # Calmar ratio
        calmar = cagr / max_drawdown if max_drawdown > 0 else 0
        
        # Win/loss metrics
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        
        win_rate = len(wins) / len(pnls) if len(pnls) > 0 else 0
        
        gross_profit = wins.sum() if len(wins) > 0 else 0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        avg_winner = wins.mean() if len(wins) > 0 else 0
        avg_loser = abs(losses.mean()) if len(losses) > 0 else 0
        payoff_ratio = avg_winner / avg_loser if avg_loser > 0 else float('inf')
        
        # Expectancy (expected value per trade)
        expectancy = (win_rate * avg_winner) - ((1 - win_rate) * avg_loser)
        
        return cls(
            total_return=total_return * 100,
            cagr=cagr * 100,
            monthly_return=monthly_return * 100,
            max_drawdown=max_drawdown * 100,
            avg_drawdown=avg_drawdown * 100,
            volatility=volatility * 100,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            win_rate=win_rate * 100,
            profit_factor=profit_factor,
            payoff_ratio=payoff_ratio,
            expectancy=expectancy,
            total_trades=len(pnls),
            avg_trade=pnls.mean() if len(pnls) > 0 else 0,
            avg_winner=avg_winner,
            avg_loser=avg_loser
        )
    
    @classmethod
    def _empty(cls) -> "PerformanceMetrics":
        """Return empty metrics."""
        return cls(
            total_return=0, cagr=0, monthly_return=0,
            max_drawdown=0, avg_drawdown=0, volatility=0,
            sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
            win_rate=0, profit_factor=0, payoff_ratio=0, expectancy=0,
            total_trades=0, avg_trade=0, avg_winner=0, avg_loser=0
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'total_return_pct': self.total_return,
            'cagr_pct': self.cagr,
            'monthly_return_pct': self.monthly_return,
            'max_drawdown_pct': self.max_drawdown,
            'volatility_pct': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'win_rate_pct': self.win_rate,
            'profit_factor': self.profit_factor,
            'payoff_ratio': self.payoff_ratio,
            'expectancy': self.expectancy,
            'total_trades': self.total_trades,
            'avg_trade': self.avg_trade,
        }
    
    def __repr__(self) -> str:
        return (
            f"PerformanceMetrics(\n"
            f"  total_return={self.total_return:.1f}%, "
            f"cagr={self.cagr:.1f}%\n"
            f"  sharpe={self.sharpe_ratio:.2f}, "
            f"sortino={self.sortino_ratio:.2f}\n"
            f"  max_dd={self.max_drawdown:.1f}%, "
            f"win_rate={self.win_rate:.1f}%\n"
            f"  pf={self.profit_factor:.2f}, "
            f"expectancy={self.expectancy:.2f}\n"
            f")"
        )


def calculate_drawdown_series(equity: pd.Series) -> pd.Series:
    """Calculate drawdown series from equity curve."""
    peak = equity.cummax()
    drawdown = (peak - equity) / peak * 100
    return drawdown


def calculate_recovery_factor(
    equity: pd.Series,
    initial_balance: float
) -> float:
    """
    Calculate recovery factor (net profit / max drawdown).
    
    Higher is better. > 2 is considered good.
    """
    net_profit = equity.iloc[-1] - initial_balance
    drawdown = calculate_drawdown_series(equity)
    max_dd = drawdown.max()
    
    if max_dd == 0:
        return float('inf')
    
    return (net_profit / initial_balance * 100) / max_dd


def calculate_ulcer_index(equity: pd.Series, period: int = 14) -> float:
    """
    Calculate Ulcer Index (measures depth and duration of drawdowns).
    
    Lower is better. < 5 is considered good.
    """
    drawdown = calculate_drawdown_series(equity)
    squared_dd = drawdown ** 2
    avg_squared = squared_dd.rolling(period).mean()
    ulcer = np.sqrt(avg_squared)
    return ulcer.iloc[-1] if len(ulcer) > 0 else 0


def calculate_cpc_ratio(trades: pd.DataFrame) -> float:
    """
    Calculate CPC (Coefficient of Randomness) ratio.
    
    CPC = Profit Factor × Win Rate × Payoff Ratio
    > 1 indicates positive edge
    """
    if trades.empty or 'pnl' not in trades.columns:
        return 0
    
    pnls = trades['pnl'].values
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    
    if len(losses) == 0 or len(wins) == 0:
        return 0
    
    win_rate = len(wins) / len(pnls)
    profit_factor = wins.sum() / abs(losses.sum())
    payoff_ratio = wins.mean() / abs(losses.mean())
    
    return profit_factor * win_rate * payoff_ratio


def calculate_system_quality_number(trades: pd.DataFrame) -> float:
    """
    Calculate SQN (System Quality Number) by Van Tharp.
    
    SQN = √N × (Avg R / Std R)
    
    Interpretation:
    - 1.6 - 1.9: Below average
    - 2.0 - 2.4: Average
    - 2.5 - 2.9: Good
    - 3.0 - 5.0: Excellent
    - 5.0 - 7.0: Superb
    - > 7.0: Holy Grail (rare)
    """
    if trades.empty or 'pnl_pct' not in trades.columns:
        return 0
    
    r_values = trades['pnl_pct'].values
    n = len(r_values)
    
    if n < 30:
        return 0  # Not enough trades for reliable SQN
    
    mean_r = np.mean(r_values)
    std_r = np.std(r_values)
    
    if std_r == 0:
        return 0
    
    sqn = np.sqrt(n) * (mean_r / std_r)
    
    return sqn
