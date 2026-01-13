"""
Position sizing calculator for XAUUSD trading.

Calculates optimal lot size based on risk parameters,
account balance, and stop loss distance.
"""

from dataclasses import dataclass
from typing import Optional
import math

from xauusd_strategy.config.settings import Settings
from xauusd_strategy.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PositionSize:
    """Position sizing result."""
    lots: float
    risk_amount: float
    risk_pct: float
    position_value: float
    effective_leverage: float
    pip_value: float
    stop_distance: float
    
    def __repr__(self) -> str:
        return (
            f"PositionSize(lots={self.lots:.2f}, risk={self.risk_amount:.2f}, "
            f"leverage={self.effective_leverage:.0f}x)"
        )


class PositionSizer:
    """
    Calculate position sizes for XAUUSD trades.
    
    XAUUSD Contract Specifications (typical):
    - 1 standard lot = 100 oz gold
    - 1 mini lot = 10 oz gold  
    - 1 micro lot = 1 oz gold
    - Pip value for 1 lot ≈ $10 per $0.10 move (or $100 per $1 move)
    
    Example:
        >>> sizer = PositionSizer(balance=250, leverage=500)
        >>> pos = sizer.calculate(entry=2000, stop_loss=1990, risk_pct=2.5)
        >>> print(pos.lots)  # Output: 0.06
    """
    
    # XAUUSD constants
    CONTRACT_SIZE = 100  # 1 lot = 100 oz
    LOT_MIN = 0.01
    LOT_MAX = 100.0
    LOT_STEP = 0.01
    
    def __init__(
        self,
        balance: float = 250,
        leverage: int = 500,
        account_currency: str = "EUR",
        broker_min_lot: float = 0.01,
        broker_max_lot: float = 100.0
    ):
        """
        Initialize position sizer.
        
        Args:
            balance: Account balance in account currency
            leverage: Account leverage (e.g., 500 for 1:500)
            account_currency: Account base currency
            broker_min_lot: Broker's minimum lot size
            broker_max_lot: Broker's maximum lot size
        """
        self.balance = balance
        self.leverage = leverage
        self.account_currency = account_currency
        self.broker_min_lot = broker_min_lot
        self.broker_max_lot = broker_max_lot
        
        # EUR/USD rate for conversion (approximate, should be fetched live)
        self._eur_usd_rate = 1.08
    
    def update_balance(self, new_balance: float):
        """Update account balance."""
        self.balance = new_balance
        logger.debug(f"Balance updated to {new_balance:.2f}")
    
    def calculate(
        self,
        entry_price: float,
        stop_loss_price: float,
        risk_pct: float = 2.5,
        max_risk_pct: Optional[float] = None
    ) -> PositionSize:
        """
        Calculate position size based on risk.
        
        Args:
            entry_price: Entry price for XAUUSD
            stop_loss_price: Stop loss price
            risk_pct: Risk as percentage of balance (e.g., 2.5 for 2.5%)
            max_risk_pct: Maximum allowed risk percentage
        
        Returns:
            PositionSize with lot size and details
        """
        # Apply max risk cap
        if max_risk_pct:
            risk_pct = min(risk_pct, max_risk_pct)
        
        # Calculate stop distance
        stop_distance = abs(entry_price - stop_loss_price)
        
        if stop_distance == 0:
            logger.warning("Stop distance is 0, cannot calculate position size")
            return self._zero_position()
        
        # Risk amount in account currency
        risk_amount = self.balance * (risk_pct / 100)
        
        # Convert to USD if account is EUR
        if self.account_currency == "EUR":
            risk_amount_usd = risk_amount * self._eur_usd_rate
        else:
            risk_amount_usd = risk_amount
        
        # Calculate pip value per lot
        # For XAUUSD: 1 lot = 100 oz, so $1 price move = $100 P&L per lot
        pip_value_per_lot = self.CONTRACT_SIZE  # $100 per $1 move per lot
        
        # Calculate lots
        # lots = risk_amount / (stop_distance * pip_value_per_lot)
        lots = risk_amount_usd / (stop_distance * pip_value_per_lot)
        
        # Apply leverage constraint
        max_position_value = self.balance * self.leverage
        if self.account_currency == "EUR":
            max_position_value *= self._eur_usd_rate
        
        max_lots_by_leverage = max_position_value / (entry_price * self.CONTRACT_SIZE)
        
        # Use 50% of available margin at most
        max_lots_by_margin = max_lots_by_leverage * 0.5
        lots = min(lots, max_lots_by_margin)
        
        # Apply broker limits
        lots = max(self.broker_min_lot, min(lots, self.broker_max_lot))
        
        # Round to lot step
        lots = math.floor(lots / self.LOT_STEP) * self.LOT_STEP
        
        # Calculate actual values
        position_value = lots * entry_price * self.CONTRACT_SIZE
        pip_value = lots * pip_value_per_lot
        effective_leverage = position_value / (self.balance * self._eur_usd_rate)
        
        # Recalculate actual risk
        actual_risk_usd = stop_distance * pip_value
        actual_risk_eur = actual_risk_usd / self._eur_usd_rate
        actual_risk_pct = (actual_risk_eur / self.balance) * 100
        
        result = PositionSize(
            lots=lots,
            risk_amount=actual_risk_eur,
            risk_pct=actual_risk_pct,
            position_value=position_value,
            effective_leverage=effective_leverage,
            pip_value=pip_value,
            stop_distance=stop_distance
        )
        
        logger.debug(
            f"Position size: {lots:.2f} lots, risk={actual_risk_pct:.2f}%, "
            f"leverage={effective_leverage:.0f}x"
        )
        
        return result
    
    def calculate_with_kelly(
        self,
        entry_price: float,
        stop_loss_price: float,
        kelly_fraction: float,
        base_risk_pct: float = 2.5,
        max_risk_pct: float = 5.0
    ) -> PositionSize:
        """
        Calculate position size using Kelly-adjusted risk.
        
        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            kelly_fraction: Kelly fraction (0-1)
            base_risk_pct: Base risk percentage
            max_risk_pct: Maximum risk cap
        
        Returns:
            PositionSize
        """
        # Kelly-adjusted risk
        kelly_risk = base_risk_pct * kelly_fraction
        
        # Apply bounds
        risk_pct = max(0.5, min(kelly_risk, max_risk_pct))
        
        return self.calculate(entry_price, stop_loss_price, risk_pct, max_risk_pct)
    
    def calculate_pyramid_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        existing_lots: float,
        existing_avg_entry: float,
        max_total_risk_pct: float = 5.0
    ) -> PositionSize:
        """
        Calculate size for pyramid/add-on position.
        
        Args:
            entry_price: New entry price
            stop_loss_price: Current stop loss
            existing_lots: Existing position size
            existing_avg_entry: Average entry of existing position
            max_total_risk_pct: Maximum total risk for combined position
        
        Returns:
            PositionSize for additional position
        """
        # Calculate current risk
        existing_stop_dist = abs(existing_avg_entry - stop_loss_price)
        existing_risk_usd = existing_lots * existing_stop_dist * self.CONTRACT_SIZE
        existing_risk_eur = existing_risk_usd / self._eur_usd_rate
        existing_risk_pct = (existing_risk_eur / self.balance) * 100
        
        # Available risk for pyramid
        available_risk_pct = max_total_risk_pct - existing_risk_pct
        
        if available_risk_pct <= 0:
            logger.warning("No risk available for pyramid, max risk reached")
            return self._zero_position()
        
        return self.calculate(entry_price, stop_loss_price, available_risk_pct)
    
    def _zero_position(self) -> PositionSize:
        """Return zero position."""
        return PositionSize(
            lots=0,
            risk_amount=0,
            risk_pct=0,
            position_value=0,
            effective_leverage=0,
            pip_value=0,
            stop_distance=0
        )
    
    def validate_position(self, position: PositionSize) -> bool:
        """
        Validate position meets requirements.
        
        Args:
            position: Position to validate
        
        Returns:
            True if valid
        """
        if position.lots < self.broker_min_lot:
            logger.warning(f"Position size {position.lots} below minimum {self.broker_min_lot}")
            return False
        
        if position.lots > self.broker_max_lot:
            logger.warning(f"Position size {position.lots} above maximum {self.broker_max_lot}")
            return False
        
        if position.effective_leverage > self.leverage:
            logger.warning(
                f"Effective leverage {position.effective_leverage:.0f}x "
                f"exceeds max {self.leverage}x"
            )
            return False
        
        return True
    
    @classmethod
    def from_settings(cls, settings: Settings) -> "PositionSizer":
        """Create PositionSizer from Settings object."""
        return cls(
            balance=settings.account.initial_balance,
            leverage=settings.account.leverage,
            account_currency=settings.account.currency
        )
