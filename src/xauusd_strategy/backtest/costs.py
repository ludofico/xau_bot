"""
Cost model for realistic trade simulation.

Models spread, slippage, and commissions for XAUUSD trading.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CostModel:
    """
    Trading cost model for XAUUSD.
    
    Default values are for a typical ECN broker with:
    - Tight spreads (~$0.20-0.30)
    - Low slippage (~$0.10-0.15)  
    - Commission per lot (~$6-8 round trip)
    
    Example:
        >>> costs = CostModel(spread_usd=0.25, slippage_usd=0.15)
        >>> entry = costs.apply_entry_cost(2000.0, is_long=True)
        >>> # entry = 2000.40 (price + half spread + slippage)
    """
    
    # Spread in USD (bid-ask)
    spread_usd: float = 0.25
    
    # Slippage in USD (expected fill deviation)
    slippage_usd: float = 0.15
    
    # Commission per lot (round trip)
    commission_per_lot: float = 7.0
    
    # Swap rates (if holding overnight)
    swap_long_usd_per_lot: float = -0.50  # Cost for long positions
    swap_short_usd_per_lot: float = 0.30  # Credit for short positions
    
    def apply_entry_cost(self, price: float, is_long: bool) -> float:
        """
        Apply entry costs to price.
        
        Args:
            price: Theoretical entry price
            is_long: True for long, False for short
        
        Returns:
            Adjusted entry price
        """
        half_spread = self.spread_usd / 2
        
        if is_long:
            # Buy at ask price (higher) + slippage
            return price + half_spread + self.slippage_usd
        else:
            # Sell at bid price (lower) - slippage
            return price - half_spread - self.slippage_usd
    
    def apply_exit_cost(self, price: float, is_long: bool) -> float:
        """
        Apply exit costs to price.
        
        Args:
            price: Theoretical exit price
            is_long: True for closing long, False for closing short
        
        Returns:
            Adjusted exit price
        """
        half_spread = self.spread_usd / 2
        
        if is_long:
            # Sell at bid (lower) - slippage
            return price - half_spread - self.slippage_usd
        else:
            # Buy to cover at ask (higher) + slippage
            return price + half_spread + self.slippage_usd
    
    def calculate_trade_cost(
        self,
        lots: float,
        holding_days: int = 0,
        is_long: bool = True
    ) -> float:
        """
        Calculate total trade cost.
        
        Args:
            lots: Position size in lots
            holding_days: Number of overnight holds
            is_long: True for long position
        
        Returns:
            Total cost in USD
        """
        # Spread + slippage (entry and exit)
        spread_cost = (self.spread_usd + self.slippage_usd * 2) * lots * 100
        
        # Commission
        commission = self.commission_per_lot * lots
        
        # Swap
        if holding_days > 0:
            swap_rate = self.swap_long_usd_per_lot if is_long else self.swap_short_usd_per_lot
            swap_cost = swap_rate * lots * holding_days
        else:
            swap_cost = 0
        
        return spread_cost + commission + swap_cost
    
    def cost_as_pct(self, entry_price: float) -> float:
        """
        Calculate cost as percentage of entry.
        
        Args:
            entry_price: Entry price
        
        Returns:
            Cost as percentage
        """
        total_cost = self.spread_usd + self.slippage_usd * 2
        return total_cost / entry_price * 100
    
    def min_profit_to_break_even(self, lots: float) -> float:
        """
        Calculate minimum profit in points to break even.
        
        Args:
            lots: Position size
        
        Returns:
            Minimum profit in USD price points
        """
        total_cost = self.calculate_trade_cost(lots, 0, True)
        return total_cost / (lots * 100)
    
    @classmethod
    def from_broker_type(cls, broker_type: str) -> "CostModel":
        """
        Create cost model based on broker type.
        
        Args:
            broker_type: "ecn", "market_maker", "retail"
        
        Returns:
            CostModel with appropriate defaults
        """
        if broker_type.lower() == "ecn":
            return cls(
                spread_usd=0.20,
                slippage_usd=0.10,
                commission_per_lot=7.0
            )
        elif broker_type.lower() == "market_maker":
            return cls(
                spread_usd=0.45,
                slippage_usd=0.20,
                commission_per_lot=0.0  # Spread instead of commission
            )
        elif broker_type.lower() == "retail":
            return cls(
                spread_usd=0.35,
                slippage_usd=0.15,
                commission_per_lot=5.0
            )
        else:
            return cls()  # Default
    
    def __repr__(self) -> str:
        return (
            f"CostModel(spread=${self.spread_usd:.2f}, "
            f"slip=${self.slippage_usd:.2f}, "
            f"comm=${self.commission_per_lot:.2f}/lot)"
        )
