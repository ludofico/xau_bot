
# 📈 Trading Strategy Documentation

The system operates a multi-strategy engine to maximize trade frequency and win rate simultaneously.

## 1. London Breakout (Logic Core)

Designed to capture the massive volatility during the London and New York market openings.

- **Session 1**: London Open (08:00 - 17:00 CET)
- **Session 2**: New York Open (14:30 - 20:00 CET)
- **Logic**:
    - Calculates the "Asian Box" (HL of 00:00 - 08:00).
    - If price breaks the box with momentum (ROC) and trend (EMA 200), a signal is generated.
- **Filtering**: Signals are verified by the XGBoost ML filter before execution.

## 2. Asian Scalper (Tactical)

Designed to generate high-win-rate income during low-volatility sessions.

- **Session**: Asian Night (01:00 - 08:00 CET)
- **Logic**: **Mean Reversion**.
    - Uses Bollinger Bands and RSI (14) to find overextended prices.
    - **Buy**: Price < Lower Band AND RSI < 30.
    - **Sell**: Price > Upper Band AND RSI > 70.
- **Targets**: Small, high-precision TP/SL clusters.

## 3. Position Management: Pyramiding

The system includes an aggressive compounding manager.
- **Logic**: If a trade is in profit by +$2.50 (on XAUUSD), it locks in profit by moving SL to breakeven and opens a second "layer."
- **Risk**: Up to 4 layers allowed per trade.
- **Goal**: Turn a 1R win into a 5-10R runner during big trends.
