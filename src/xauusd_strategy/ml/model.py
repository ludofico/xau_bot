"""
XGBoost probability filter for trade signal validation.

Predicts probability that a trade signal will hit TP before SL.
"""

from pathlib import Path
from typing import Optional, List, Dict, Tuple
import pickle
import pandas as pd
import numpy as np

from xauusd_strategy.utils.logger import get_logger

logger = get_logger(__name__)


class MLProbabilityFilter:
    """
    XGBoost/LightGBM classifier to filter breakout signals.
    
    Predicts probability that a trade will hit Take Profit before Stop Loss.
    Used to filter low-quality signals and increase win rate.
    
    Features used:
    - Volatility: ATR, ATR ratio, Asian range
    - Momentum: ROC, momentum
    - Trend: EMA slopes, price relative to EMAs
    - Time: Hour, day of week, session flags
    - Entry quality: Distance from Asian box
    
    Example:
        >>> filter = MLProbabilityFilter()
        >>> filter.train(train_data, train_labels)
        >>> prob = filter.predict(signal_features)
        >>> if prob > 0.55:
        ...     take_trade()
    """
    
    FEATURE_COLUMNS = [
        'atr_14', 'atr_ratio', 'atr_pct',
        'roc_5', 'roc_10',
        'ema_20_slope', 'ema_50_slope',
        'price_to_ema20', 'price_to_ema50',
        'volatility_5', 'volatility_20',
        'candle_body_ratio',
        'hour', 'day_of_week',
        'is_london', 'is_overlap',
        'asian_range', 'distance_from_asian_high', 'distance_from_asian_low',
        'rsi_14', 'bb_position',
    ]
    
    def __init__(
        self,
        model_type: str = "xgboost",
        probability_threshold: float = 0.55,
        model_path: Optional[Path] = None
    ):
        """
        Initialize ML filter.
        
        Args:
            model_type: "xgboost" or "lightgbm"
            probability_threshold: Minimum probability to pass filter
            model_path: Path to saved model (optional)
        """
        self.model_type = model_type
        self.threshold = probability_threshold
        self.model = None
        self.feature_importance: Optional[pd.Series] = None
        
        if model_path and Path(model_path).exists():
            self.load(model_path)
    
    def _create_model(self, params: Optional[Dict] = None):
        """Create model instance with fallbacks."""
        # Try XGBoost first
        if self.model_type == "xgboost":
            try:
                import xgboost as xgb
                default_params = {
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'n_estimators': 100,
                    'min_child_weight': 3,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'use_label_encoder': False,
                    'random_state': 42
                }
                if params:
                    default_params.update(params)
                return xgb.XGBClassifier(**default_params)
            except (ImportError, Exception) as e:
                logger.warning(f"XGBoost not available ({e}), falling back to LightGBM")
                self.model_type = "lightgbm"
        
        # Try LightGBM
        if self.model_type == "lightgbm":
            try:
                import lightgbm as lgb
                default_params = {
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'n_estimators': 100,
                    'min_child_samples': 20,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'random_state': 42,
                    'verbose': -1
                }
                if params:
                    default_params.update(params)
                return lgb.LGBMClassifier(**default_params)
            except (ImportError, Exception) as e:
                logger.warning(f"LightGBM not available ({e}), falling back to Random Forest")
                self.model_type = "sklearn"
        
        # Fallback to sklearn Random Forest (always available)
        if self.model_type == "sklearn" or True:  # Ultimate fallback
            from sklearn.ensemble import RandomForestClassifier
            logger.info("Using sklearn RandomForestClassifier as ML model")
            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            }
            if params:
                default_params.update(params)
            self.model_type = "sklearn"
            return RandomForestClassifier(**default_params)
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2,
        early_stopping_rounds: int = 10
    ) -> Dict:
        """
        Train the probability filter.
        
        Args:
            X: Feature DataFrame
            y: Binary labels (1 = hit TP, 0 = hit SL)
            validation_split: Fraction for validation
            early_stopping_rounds: Early stopping patience
        
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training {self.model_type} model on {len(X)} samples")
        
        # Select features
        available_features = [f for f in self.FEATURE_COLUMNS if f in X.columns]
        X_train = X[available_features].copy()
        
        # Handle missing values
        X_train = X_train.fillna(0)
        
        # Split data (time-series aware)
        split_idx = int(len(X_train) * (1 - validation_split))
        X_val = X_train.iloc[split_idx:]
        y_val = y.iloc[split_idx:]
        X_train = X_train.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        
        # Create and train model
        self.model = self._create_model()
        
        if self.model_type == "xgboost":
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)]
            )
        
        # Calculate feature importance
        self.feature_importance = pd.Series(
            self.model.feature_importances_,
            index=available_features
        ).sort_values(ascending=False)
        
        # Evaluate
        train_proba = self.model.predict_proba(X_train)[:, 1]
        val_proba = self.model.predict_proba(X_val)[:, 1]
        
        train_acc = ((train_proba > self.threshold) == y_train).mean()
        val_acc = ((val_proba > self.threshold) == y_val).mean()
        
        # Calculate precision/recall at threshold
        val_pred = val_proba > self.threshold
        tp = ((val_pred == 1) & (y_val == 1)).sum()
        fp = ((val_pred == 1) & (y_val == 0)).sum()
        fn = ((val_pred == 0) & (y_val == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        metrics = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'precision': precision,
            'recall': recall,
            'samples_above_threshold': (val_proba > self.threshold).mean(),
            'top_features': self.feature_importance.head(10).to_dict()
        }
        
        logger.info(
            f"Training complete: val_acc={val_acc:.2%}, "
            f"precision={precision:.2%}, recall={recall:.2%}"
        )
        
        return metrics
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict probability for signals.
        
        Args:
            features: Feature DataFrame
        
        Returns:
            Array of probabilities
        """
        if self.model is None:
            logger.warning("Model not trained, returning 0.5")
            return np.full(len(features), 0.5)
        
        available_features = [f for f in self.FEATURE_COLUMNS if f in features.columns]
        X = features[available_features].fillna(0)
        
        return self.model.predict_proba(X)[:, 1]
    
    def predict_single(self, features: pd.Series) -> float:
        """Predict probability for a single signal."""
        df = pd.DataFrame([features])
        return self.predict(df)[0]
    
    def should_take_signal(
        self,
        features: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Get boolean array of signals to take.
        
        Args:
            features: Feature DataFrame
            threshold: Custom threshold (uses default if None)
        
        Returns:
            Boolean array
        """
        threshold = threshold or self.threshold
        proba = self.predict(features)
        return proba >= threshold
    
    def save(self, path: Path):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'model': self.model,
            'model_type': self.model_type,
            'threshold': self.threshold,
            'feature_importance': self.feature_importance,
            'feature_columns': self.FEATURE_COLUMNS
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: Path):
        """Load model from disk."""
        path = Path(path)
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.model_type = data['model_type']
        self.threshold = data['threshold']
        self.feature_importance = data.get('feature_importance')
        
        logger.info(f"Model loaded from {path}")
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance as Series."""
        if self.feature_importance is None:
            return pd.Series(dtype=float)
        return self.feature_importance


def create_labels_from_trades(
    signals: pd.DataFrame,
    trades: pd.DataFrame
) -> pd.Series:
    """
    Create binary labels from trade outcomes.
    
    Args:
        signals: DataFrame with signal timestamps
        trades: DataFrame with trade outcomes
    
    Returns:
        Series with 1 = hit TP, 0 = hit SL
    """
    labels = pd.Series(index=signals.index, dtype=int)
    
    for idx in signals.index:
        # Find matching trade
        matching_trades = trades[trades['entry_time'] == idx]
        
        if len(matching_trades) == 0:
            labels[idx] = 0  # No trade = unsuccessful
            continue
        
        trade = matching_trades.iloc[0]
        
        if trade['exit_reason'] == 'take_profit':
            labels[idx] = 1
        else:
            labels[idx] = 0
    
    return labels
