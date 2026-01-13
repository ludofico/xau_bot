"""
Backtesting engine using vectorbt.

Provides fast vectorized backtesting with realistic cost modeling.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Tuple
import pandas as pd
import numpy as np

from xauusd_strategy.strategy.london_breakout import LondonBreakoutStrategy, TradeSignal, SignalType
from xauusd_strategy.risk.compound_manager import AggressiveCompoundManager
from xauusd_strategy.backtest.costs import CostModel
from xauusd_strategy.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    # Returns
    total_return_pct: float
    monthly_return_pct: float
    annual_return_pct: float
    
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    avg_drawdown_pct: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    profit_factor: float
    
    # Trade analysis
    avg_trade_return: float
    avg_winning_trade: float
    avg_losing_trade: float
    largest_win: float
    largest_loss: float
    avg_trade_duration_hours: float
    
    # Balance
    initial_balance: float
    final_balance: float
    peak_balance: float
    
    # Equity curve
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    trade_log: pd.DataFrame
    
    def __repr__(self) -> str:
        return (
            f"BacktestResult(\n"
            f"  total_return={self.total_return_pct:.1f}%, "
            f"monthly={self.monthly_return_pct:.1f}%\n"
            f"  sharpe={self.sharpe_ratio:.2f}, "
            f"max_dd={self.max_drawdown_pct:.1f}%\n"
            f"  trades={self.total_trades}, "
            f"win_rate={self.win_rate_pct:.1f}%, "
            f"pf={self.profit_factor:.2f}\n"
            f"  balance: {self.initial_balance:.0f} → {self.final_balance:.0f}\n"
            f")"
        )
    
    def meets_target(self, monthly_target: float = 200) -> bool:
        """Check if result meets monthly target."""
        return self.monthly_return_pct >= monthly_target
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'total_return_pct': self.total_return_pct,
            'monthly_return_pct': self.monthly_return_pct,
            'annual_return_pct': self.annual_return_pct,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown_pct': self.max_drawdown_pct,
            'total_trades': self.total_trades,
            'win_rate_pct': self.win_rate_pct,
            'profit_factor': self.profit_factor,
            'avg_trade_return': self.avg_trade_return,
            'initial_balance': self.initial_balance,
            'final_balance': self.final_balance,
        }


class BacktestEngine:
    """
    Vectorized backtesting engine for XAUUSD strategies.
    
    Features:
    - Fast vectorized execution using numpy
    - Realistic cost modeling (spread, slippage, commission)
    - Compounding position sizing
    - Walk-forward validation
    - Detailed trade logging
    
    Example:
        >>> engine = BacktestEngine(initial_balance=250)
        >>> strategy = LondonBreakoutStrategy()
        >>> result = engine.run(data, strategy)
        >>> print(result)
    """
    
    def __init__(
        self,
        initial_balance: float = 250,
        leverage: int = 500,
        cost_model: Optional[CostModel] = None,
        use_compounding: bool = True,
        risk_pct: float = 2.5,
        max_risk_pct: float = 4.0
    ):
        """
        Initialize backtest engine.
        
        Args:
            initial_balance: Starting balance in EUR
            leverage: Account leverage
            cost_model: Cost model for spread/slippage
            use_compounding: Whether to compound gains
            risk_pct: Risk per trade (%)
            max_risk_pct: Maximum risk per trade (%)
        """
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.cost_model = cost_model or CostModel()
        self.use_compounding = use_compounding
        self.risk_pct = risk_pct
        self.max_risk_pct = max_risk_pct
        
        logger.info(
            f"BacktestEngine initialized: balance={initial_balance}, "
            f"leverage={leverage}, compounding={use_compounding}"
        )
    
    def run(
        self,
        data: pd.DataFrame,
        strategy: LondonBreakoutStrategy,
        ml_probabilities: Optional[pd.Series] = None
    ) -> BacktestResult:
        """
        Run backtest on historical data.
        
        Args:
            data: OHLC DataFrame
            strategy: Trading strategy instance
            ml_probabilities: Optional ML model probabilities
        
        Returns:
            BacktestResult with all metrics
        """
        logger.info(f"Starting backtest: {len(data)} bars")
        
        # Prepare data with indicators
        data = strategy.prepare_data(data)
        
        # Generate signals
        signals = strategy.generate_signals(data, ml_probabilities)
        
        if not signals:
            logger.warning("No signals generated")
            return self._empty_result()
        
        # Simulate trading
        trade_log = self._simulate_trades(data, signals)
        
        if trade_log.empty:
            logger.warning("No trades executed")
            return self._empty_result()
        
        # Calculate metrics
        result = self._calculate_metrics(data, trade_log)
        
        logger.info(f"Backtest complete: {result}")
        
        return result
    
    def _simulate_trades(
        self,
        data: pd.DataFrame,
        signals: List[TradeSignal]
    ) -> pd.DataFrame:
        """Simulate trade execution with compounding."""
        trades = []
        balance = self.initial_balance
        
        for signal in signals:
            if signal.timestamp is None:
                continue
            
            # Find entry bar
            entry_idx = data.index.get_loc(signal.timestamp)
            
            if entry_idx >= len(data) - 1:
                continue
            
            # Calculate position size
            if self.use_compounding:
                current_balance = balance
            else:
                current_balance = self.initial_balance
            
            risk_amount = current_balance * (self.risk_pct / 100)
            stop_distance = abs(signal.entry_price - signal.stop_loss)
            
            if stop_distance == 0:
                continue
            
            # XAUUSD: $100 per $1 move per lot
            lots = risk_amount * 1.08 / (stop_distance * 100)  # EUR to USD
            lots = max(0.01, min(lots, 10.0))  # Clamp
            
            # Apply costs to entry
            entry_price = self.cost_model.apply_entry_cost(
                signal.entry_price,
                signal.signal_type == SignalType.LONG
            )
            
            # Simulate trade outcome
            exit_price, exit_reason, exit_time = self._simulate_trade_exit(
                data, entry_idx, signal, entry_price
            )
            
            if exit_price is None:
                continue
            
            # Calculate P&L
            if signal.signal_type == SignalType.LONG:
                pnl_points = exit_price - entry_price
            else:
                pnl_points = entry_price - exit_price
            
            pnl_usd = pnl_points * lots * 100
            commission = self.cost_model.commission_per_lot * lots
            pnl_usd -= commission
            pnl_eur = pnl_usd / 1.08
            
            # Update balance
            balance += pnl_eur
            
            trades.append({
                'entry_time': signal.timestamp,
                'exit_time': exit_time,
                'direction': signal.signal_type.name,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'lots': lots,
                'pnl': pnl_eur,
                'pnl_pct': pnl_eur / current_balance * 100,
                'exit_reason': exit_reason,
                'balance': balance,
                'atr': signal.atr_value,
                'probability': signal.probability,
            })
        
        return pd.DataFrame(trades)
    
    def _simulate_trade_exit(
        self,
        data: pd.DataFrame,
        entry_idx: int,
        signal: TradeSignal,
        entry_price: float
    ) -> Tuple[Optional[float], str, Optional[pd.Timestamp]]:
        """
        Simulate trade exit based on SL/TP/trailing.
        
        Returns:
            (exit_price, exit_reason, exit_time)
        """
        is_long = signal.signal_type == SignalType.LONG
        current_sl = signal.stop_loss
        
        for i in range(entry_idx + 1, len(data)):
            bar = data.iloc[i]
            bar_time = data.index[i]
            
            high = bar['high']
            low = bar['low']
            close = bar['close']
            
            # Check stop loss
            if is_long and low <= current_sl:
                exit_price = self.cost_model.apply_exit_cost(current_sl, True)
                return exit_price, 'stop_loss', bar_time
            elif not is_long and high >= current_sl:
                exit_price = self.cost_model.apply_exit_cost(current_sl, False)
                return exit_price, 'stop_loss', bar_time
            
            # Check take profit
            if is_long and high >= signal.take_profit:
                exit_price = self.cost_model.apply_exit_cost(signal.take_profit, True)
                return exit_price, 'take_profit', bar_time
            elif not is_long and low <= signal.take_profit:
                exit_price = self.cost_model.apply_exit_cost(signal.take_profit, False)
                return exit_price, 'take_profit', bar_time
            
            # Update trailing stop
            risk = abs(entry_price - signal.stop_loss)
            
            if is_long:
                current_pnl_r = (close - entry_price) / risk
                if current_pnl_r >= 1.0:  # Move to breakeven
                    current_sl = max(current_sl, entry_price)
                if current_pnl_r >= 1.5:  # Trail
                    trail_sl = close - (signal.atr_value * 0.8)
                    current_sl = max(current_sl, trail_sl)
            else:
                current_pnl_r = (entry_price - close) / risk
                if current_pnl_r >= 1.0:
                    current_sl = min(current_sl, entry_price)
                if current_pnl_r >= 1.5:
                    trail_sl = close + (signal.atr_value * 0.8)
                    current_sl = min(current_sl, trail_sl)
        
        # End of data, close at last price
        last_close = data['close'].iloc[-1]
        return last_close, 'end_of_data', data.index[-1]
    
    def _calculate_metrics(
        self,
        data: pd.DataFrame,
        trade_log: pd.DataFrame
    ) -> BacktestResult:
        """Calculate performance metrics from trade log."""
        # Calculate equity curve
        equity = [self.initial_balance]
        for _, trade in trade_log.iterrows():
            equity.append(trade['balance'])
        equity_curve = pd.Series(equity)
        
        # Calculate drawdown curve
        peak = equity_curve.cummax()
        drawdown = (peak - equity_curve) / peak * 100
        
        # Basic stats
        pnls = trade_log['pnl'].values
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        
        winning_trades = len(wins)
        losing_trades = len(losses)
        total_trades = len(pnls)
        
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        
        gross_profit = wins.sum() if len(wins) > 0 else 0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Returns
        final_balance = trade_log['balance'].iloc[-1] if len(trade_log) > 0 else self.initial_balance
        total_return = (final_balance - self.initial_balance) / self.initial_balance * 100
        
        # Calculate time period
        if len(trade_log) > 0:
            first_trade = trade_log['entry_time'].iloc[0]
            last_trade = trade_log['exit_time'].iloc[-1]
            days = (last_trade - first_trade).days
            months = max(1, days / 30)
            years = max(0.1, days / 365)
        else:
            months = 1
            years = 1
        
        monthly_return = (final_balance / self.initial_balance) ** (1 / months) - 1
        annual_return = (final_balance / self.initial_balance) ** (1 / years) - 1
        
        # Risk metrics
        returns = trade_log['pnl_pct'].values
        if len(returns) > 1:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 / months * 12) if np.std(returns) > 0 else 0
            
            neg_returns = returns[returns < 0]
            sortino = np.mean(returns) / np.std(neg_returns) * np.sqrt(252 / months * 12) if len(neg_returns) > 0 and np.std(neg_returns) > 0 else 0
        else:
            sharpe = 0
            sortino = 0
        
        # Trade duration
        if 'entry_time' in trade_log.columns and 'exit_time' in trade_log.columns:
            durations = (trade_log['exit_time'] - trade_log['entry_time']).dt.total_seconds() / 3600
            avg_duration = durations.mean()
        else:
            avg_duration = 0
        
        return BacktestResult(
            total_return_pct=total_return,
            monthly_return_pct=monthly_return * 100,
            annual_return_pct=annual_return * 100,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_pct=drawdown.max(),
            avg_drawdown_pct=drawdown.mean(),
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate_pct=win_rate,
            profit_factor=profit_factor,
            avg_trade_return=pnls.mean() if len(pnls) > 0 else 0,
            avg_winning_trade=wins.mean() if len(wins) > 0 else 0,
            avg_losing_trade=losses.mean() if len(losses) > 0 else 0,
            largest_win=wins.max() if len(wins) > 0 else 0,
            largest_loss=losses.min() if len(losses) > 0 else 0,
            avg_trade_duration_hours=avg_duration,
            initial_balance=self.initial_balance,
            final_balance=final_balance,
            peak_balance=equity_curve.max(),
            equity_curve=equity_curve,
            drawdown_curve=drawdown,
            trade_log=trade_log
        )
    
    def _empty_result(self) -> BacktestResult:
        """Return empty result when no trades."""
        return BacktestResult(
            total_return_pct=0, monthly_return_pct=0, annual_return_pct=0,
            sharpe_ratio=0, sortino_ratio=0, max_drawdown_pct=0, avg_drawdown_pct=0,
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate_pct=0, profit_factor=0,
            avg_trade_return=0, avg_winning_trade=0, avg_losing_trade=0,
            largest_win=0, largest_loss=0, avg_trade_duration_hours=0,
            initial_balance=self.initial_balance, final_balance=self.initial_balance,
            peak_balance=self.initial_balance,
            equity_curve=pd.Series([self.initial_balance]),
            drawdown_curve=pd.Series([0]),
            trade_log=pd.DataFrame()
        )
    
    def run_walk_forward(
        self,
        data: pd.DataFrame,
        strategy: LondonBreakoutStrategy,
        train_period: str = "6M",
        test_period: str = "1M",
    ) -> List[BacktestResult]:
        """
        Run walk-forward validation.
        
        Args:
            data: Full OHLC DataFrame
            strategy: Strategy instance
            train_period: Training period (e.g., "6M")
            test_period: Testing period (e.g., "1M")
        
        Returns:
            List of BacktestResult for each test period
        """
        logger.info(f"Running walk-forward: train={train_period}, test={test_period}")
        
        results = []
        
        train_td = pd.Timedelta(train_period)
        test_td = pd.Timedelta(test_period)
        
        start = data.index[0] + train_td
        end = data.index[-1]
        
        current = start
        fold = 1
        
        while current + test_td <= end:
            train_start = current - train_td
            train_end = current
            test_end = current + test_td
            
            train_data = data[train_start:train_end]
            test_data = data[train_end:test_end]
            
            logger.info(f"Fold {fold}: test period {train_end.date()} to {test_end.date()}")
            
            # Run backtest on test period
            result = self.run(test_data, strategy)
            results.append(result)
            
            current += test_td
            fold += 1
        
        logger.info(f"Walk-forward complete: {len(results)} folds")
        
        return results
    
    def print_summary(self, result: BacktestResult):
        """Print formatted backtest summary."""
        target_500 = "✅" if result.monthly_return_pct >= 200 else "❌"
        target_1000 = "✅" if result.monthly_return_pct >= 400 else "❌"
        
        print("\n" + "=" * 60)
        print("   BACKTEST RESULTS - AGGRESSIVE XAUUSD STRATEGY")
        print("=" * 60)
        print(f"  Initial Balance:  €{result.initial_balance:.2f}")
        print(f"  Final Balance:    €{result.final_balance:.2f}")
        print(f"  Peak Balance:     €{result.peak_balance:.2f}")
        print("-" * 60)
        print(f"  Total Return:     {result.total_return_pct:+.1f}%")
        print(f"  Monthly Return:   {result.monthly_return_pct:+.1f}%")
        print(f"  Annual Return:    {result.annual_return_pct:+.1f}%")
        print("-" * 60)
        print(f"  Sharpe Ratio:     {result.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio:    {result.sortino_ratio:.2f}")
        print(f"  Max Drawdown:     {result.max_drawdown_pct:.1f}%")
        print("-" * 60)
        print(f"  Total Trades:     {result.total_trades}")
        print(f"  Win Rate:         {result.win_rate_pct:.1f}%")
        print(f"  Profit Factor:    {result.profit_factor:.2f}")
        print(f"  Avg Trade:        €{result.avg_trade_return:.2f}")
        print(f"  Best Trade:       €{result.largest_win:.2f}")
        print(f"  Worst Trade:      €{result.largest_loss:.2f}")
        print("-" * 60)
        print(f"  €500/month Target (200%):  {target_500}")
        print(f"  €1000/month Target (400%): {target_1000}")
        print("=" * 60 + "\n")
