"""
Strategy optimization using Optuna.

Hyperparameter tuning for strategy and ML model.
"""

from typing import Optional, Dict, Callable, List
from pathlib import Path
import json
import pandas as pd
import numpy as np

from xauusd_strategy.strategy.london_breakout import LondonBreakoutStrategy
from xauusd_strategy.backtest.engine import BacktestEngine, BacktestResult
from xauusd_strategy.utils.logger import get_logger

logger = get_logger(__name__)


class StrategyOptimizer:
    """
    Optuna-based strategy optimizer.
    
    Optimizes strategy parameters to maximize risk-adjusted returns
    while respecting constraints (max drawdown, min trades, etc.).
    
    Example:
        >>> optimizer = StrategyOptimizer(train_data, test_data)
        >>> best_params = optimizer.optimize(n_trials=100)
        >>> strategy = LondonBreakoutStrategy(**best_params)
    """
    
    def __init__(
        self,
        train_data: pd.DataFrame,
        test_data: Optional[pd.DataFrame] = None,
        initial_balance: float = 250,
        leverage: int = 500,
        objective: str = "sharpe",
        constraints: Optional[Dict] = None
    ):
        """
        Initialize optimizer.
        
        Args:
            train_data: Training OHLC data
            test_data: Test data for validation (optional)
            initial_balance: Starting balance
            leverage: Account leverage
            objective: Optimization objective ("sharpe", "return", "calmar", "profit_factor")
            constraints: Dict of constraints (e.g., {"max_drawdown": 25, "min_trades": 30})
        """
        self.train_data = train_data
        self.test_data = test_data
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.objective = objective
        self.constraints = constraints or {
            "max_drawdown": 25,
            "min_trades": 30,
            "min_win_rate": 40
        }
        
        self.best_params: Optional[Dict] = None
        self.best_value: float = float('-inf')
        self.study = None
        
        logger.info(
            f"StrategyOptimizer initialized: objective={objective}, "
            f"train_size={len(train_data)}"
        )
    
    def optimize(
        self,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        show_progress: bool = True
    ) -> Dict:
        """
        Run optimization.
        
        Args:
            n_trials: Number of optimization trials
            timeout: Maximum time in seconds
            n_jobs: Number of parallel jobs
            show_progress: Show progress bar
        
        Returns:
            Best parameters dictionary
        """
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            raise ImportError("Optuna not installed. Run: pip install optuna")
        
        logger.info(f"Starting optimization: {n_trials} trials")
        
        # Create study
        sampler = TPESampler(seed=42)
        self.study = optuna.create_study(
            direction="maximize",
            sampler=sampler
        )
        
        # Run optimization
        self.study.optimize(
            self._objective_function,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=show_progress
        )
        
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value
        
        logger.info(f"Optimization complete: best_value={self.best_value:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return self.best_params
    
    def _objective_function(self, trial) -> float:
        """Optuna objective function."""
        # Sample parameters
        params = {
            'atr_period': trial.suggest_int('atr_period', 10, 20),
            'atr_min_multiplier': trial.suggest_float('atr_min_multiplier', 0.3, 0.8),
            'roc_period': trial.suggest_int('roc_period', 3, 10),
            'roc_threshold': trial.suggest_float('roc_threshold', 0.05, 0.30),
            'sl_atr_mult': trial.suggest_float('sl_atr_mult', 1.0, 2.0),
            'tp_atr_mult': trial.suggest_float('tp_atr_mult', 1.5, 3.5),
            'trailing_atr_mult': trial.suggest_float('trailing_atr_mult', 0.5, 1.2),
            'breakeven_at_rr': trial.suggest_float('breakeven_at_rr', 0.8, 1.5),
        }
        
        # Ensure TP > SL
        if params['tp_atr_mult'] <= params['sl_atr_mult']:
            return float('-inf')
        
        # Create strategy
        strategy = LondonBreakoutStrategy(**params)
        
        # Run backtest
        engine = BacktestEngine(
            initial_balance=self.initial_balance,
            leverage=self.leverage
        )
        
        result = engine.run(self.train_data, strategy)
        
        # Check constraints
        if result.total_trades < self.constraints.get('min_trades', 0):
            return float('-inf')
        
        if result.max_drawdown_pct > self.constraints.get('max_drawdown', 100):
            return float('-inf')
        
        if result.win_rate_pct < self.constraints.get('min_win_rate', 0):
            return float('-inf')
        
        # Calculate objective
        if self.objective == "sharpe":
            return result.sharpe_ratio
        elif self.objective == "return":
            return result.monthly_return_pct
        elif self.objective == "calmar":
            return result.monthly_return_pct / max(0.1, result.max_drawdown_pct)
        elif self.objective == "profit_factor":
            return result.profit_factor if result.profit_factor < 10 else 10
        else:
            return result.sharpe_ratio
    
    def optimize_risk_params(
        self,
        n_trials: int = 50
    ) -> Dict:
        """
        Optimize risk management parameters.
        
        Args:
            n_trials: Number of trials
        
        Returns:
            Best risk parameters
        """
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna not installed")
        
        def objective(trial):
            risk_pct = trial.suggest_float('risk_pct', 1.0, 5.0)
            kelly_fraction = trial.suggest_float('kelly_fraction', 0.2, 0.7)
            
            # Use default strategy with different risk params
            strategy = LondonBreakoutStrategy()
            
            engine = BacktestEngine(
                initial_balance=self.initial_balance,
                leverage=self.leverage,
                risk_pct=risk_pct
            )
            
            result = engine.run(self.train_data, strategy)
            
            # Maximize return while penalizing drawdown
            if result.max_drawdown_pct > 30:
                return float('-inf')
            
            return result.monthly_return_pct - (result.max_drawdown_pct * 0.5)
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        return study.best_params
    
    def validate_on_test(self, params: Dict) -> BacktestResult:
        """
        Validate parameters on test data.
        
        Args:
            params: Strategy parameters
        
        Returns:
            BacktestResult on test data
        """
        if self.test_data is None:
            raise ValueError("No test data provided")
        
        strategy = LondonBreakoutStrategy(**params)
        engine = BacktestEngine(
            initial_balance=self.initial_balance,
            leverage=self.leverage
        )
        
        return engine.run(self.test_data, strategy)
    
    def get_optimization_summary(self) -> Dict:
        """Get summary of optimization results."""
        if self.study is None:
            return {"error": "No optimization run yet"}
        
        return {
            "best_value": self.best_value,
            "best_params": self.best_params,
            "n_trials": len(self.study.trials),
            "n_completed": len([t for t in self.study.trials if t.state.name == 'COMPLETE']),
            "n_pruned": len([t for t in self.study.trials if t.state.name == 'PRUNED']),
        }
    
    def save_results(self, path: Path):
        """Save optimization results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        results = self.get_optimization_summary()
        
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {path}")
    
    def plot_optimization_history(self):
        """Plot optimization history."""
        if self.study is None:
            logger.warning("No study to plot")
            return
        
        try:
            import optuna.visualization as vis
            
            fig = vis.plot_optimization_history(self.study)
            fig.show()
        except ImportError:
            logger.warning("Plotly not installed for visualization")
    
    def plot_param_importances(self):
        """Plot parameter importance."""
        if self.study is None:
            logger.warning("No study to plot")
            return
        
        try:
            import optuna.visualization as vis
            
            fig = vis.plot_param_importances(self.study)
            fig.show()
        except ImportError:
            logger.warning("Plotly not installed for visualization")
