
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class Scenario:
    name: str
    win_rate: float
    risk_reward: float
    risk_per_trade: float # as decimal, e.g., 0.02 for 2%
    trades_per_day: float

def simulate_growth(starting_balance, target_monthly_income, scenario, months=12):
    balance = starting_balance
    days_per_month = 21
    
    results = []
    
    for month in range(1, months + 1):
        # Expected Value per trade = (Win% * Reward) - (Loss% * Risk)
        # But we need to compound.
        
        # Simple compounding simulation
        start_bal = balance
        
        # Monthly trades
        n_trades = int(scenario.trades_per_day * days_per_month)
        
        # Kelly Criterion (Aggressive but calculated)
        # f* = (bp - q) / b
        # b = odds received (Reward/Risk)
        # p = probability of winning
        # q = probability of losing (1-p)
        b = scenario.risk_reward
        p = scenario.win_rate
        q = 1 - p
        kelly = (b * p - q) / b
        
        # Buffett Safety Margin: Use Half-Kelly or fixed risk if Kelly is too high
        optimal_risk = min(kelly * 0.5, scenario.risk_per_trade)
        if optimal_risk < 0: optimal_risk = 0 # Don't trade if negative edge
        
        # Simulate trades roughly
        # For accurate geometric growth: Final = Initial * (1 + f*b)^W * (1 - f*)^L
        winning_trades = int(n_trades * scenario.win_rate)
        losing_trades = n_trades - winning_trades
        
        # Apply compounding trade by trade (approximation)
        current_bal = start_bal
        for _ in range(n_trades):
            # Monte carlo the sequence? No, let's use expected geometric growth for clarity
            # actually strict sequence matters for ruin, but let's assume average distribution
            pass

        # Geometric Mean Return Per Trade
        g_return = (1 + optimal_risk * b)**scenario.win_rate * (1 - optimal_risk)**(1 - scenario.win_rate) - 1
        
        # Monthly multiplier
        monthly_multiplier = (1 + g_return) ** n_trades
        
        end_bal = start_bal * monthly_multiplier
        profit = end_bal - start_bal
        
        # Can we withdraw?
        withdrawable = 0
        if profit > target_monthly_income:
            # If we made more than target, we can think about withdrawing
            # Buffett Rule: Retain earnings to compound unless capital is useless
            pass
        
        balance = end_bal
        
        results.append({
            "Month": month,
            "Start Balance": f"€{start_bal:.2f}",
            "End Balance": f"€{balance:.2f}",
            "Profit": f"€{profit:.2f}",
            "Return %": f"{(monthly_multiplier-1)*100:.1f}%",
            "Risk/Trade": f"{optimal_risk*100:.1f}%"
        })
        
    return pd.DataFrame(results)

# 1. The "Gambler" (Current State)
# High risk, low win rate, hoping for strings of luck
gambler = Scenario(
    name="The Speculator",
    win_rate=0.35,      # Current backtest
    risk_reward=2.0,    # Current
    risk_per_trade=0.05,# High risk
    trades_per_day=3    # Overtrading
)

# 2. The "Buffett Snowball" (Required State)
# High conviction (ML filtered), waiting for the fat pitch
buffett = Scenario(
    name="The Intelligent Investor",
    win_rate=0.60,      # ML Filtered Target
    risk_reward=2.0,    # Good R:R
    risk_per_trade=0.04,# Aggressive but calculated
    trades_per_day=1.5  # Selective
)

print("--- SCENARIO 1: CURRENT STATUS (The Speculator) ---")
print(simulate_growth(250, 500, gambler).to_string(index=False))

print("\n\n--- SCENARIO 2: REQUIRED MOAT (The Intelligent Investor) ---")
print(simulate_growth(250, 500, buffett).to_string(index=False))
