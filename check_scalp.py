"""Quick check for scalp conditions"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

mt5.initialize()
rates = mt5.copy_rates_from_pos('XAUUSD', mt5.TIMEFRAME_M5, 0, 50)
df = pd.DataFrame(rates)

# RSI
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss.replace(0, np.nan)
df['rsi'] = 100 - (100 / (1 + rs))

# BB
sma = df['close'].rolling(20).mean()
std = df['close'].rolling(20).std()
df['bb_upper'] = sma + (std * 2)
df['bb_lower'] = sma - (std * 2)

last = df.iloc[-1]
print('=== Current Market Conditions ===')
print(f'Price: {last.close:.2f}')
print(f'RSI(14): {last.rsi:.1f}')
print(f'BB Upper: {last.bb_upper:.2f}')
print(f'BB Lower: {last.bb_lower:.2f}')
print()
print('=== Scalp Signal Check ===')
print(f'BUY: RSI < 30 AND Price < BB_Lower')
print(f'  RSI={last.rsi:.1f} < 30? {last.rsi < 30}')
print(f'  Price={last.close:.2f} < BB={last.bb_lower:.2f}? {last.close < last.bb_lower}')
print(f'  >>> BUY SIGNAL: {last.rsi < 30 and last.close < last.bb_lower}')
print()
print(f'SELL: RSI > 70 AND Price > BB_Upper')
print(f'  RSI={last.rsi:.1f} > 70? {last.rsi > 70}')
print(f'  Price={last.close:.2f} > BB={last.bb_upper:.2f}? {last.close > last.bb_upper}')
print(f'  >>> SELL SIGNAL: {last.rsi > 70 and last.close > last.bb_upper}')
