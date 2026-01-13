
# 🤖 AI Model Documentation

This document provides technical details on the Machine Learning and Reinforcement Learning models used in the XAUUSD trading system.

## 1. XGBoost Probability Filter

The system uses an XGBoost classifier to filter signals from the `LondonBreakoutStrategy`.

### Model Specs
- **Algorithm**: XGBoost (Gradient Boosted Trees)
- **Target**: Binary Classification (1 = Win, 0 = Loss)
- **Features**: 108 indicators (ATR ratios, ROC, EMA slopes, RSI, Bollinger positions, Asian box distances).
- **Threshold**: 0.60 (Adjustable in `config/aggressive.yaml`)

### Performance
In backtests on XAUUSD M5 data (60-day window):
- **Accuracy**: ~60.5%
- **Role**: Prevents "fake breakouts" by analyzing market context.

---

## 2. DeepScalper (RL Agent)

An advanced Reinforcement Learning agent based on the **PPO (Proximal Policy Optimization)** algorithm.

### Architecture
- **Environment**: Custom `XauUsdEnv` (Gymnasium)
- **Observation Space**: Last 60 bars of OHLCV + Technical Indicators + Account Balance + Net Position.
- **Action Space**: 
    - `0`: Hold
    - `1`: Buy (Market order)
    - `2`: Sell (Market order)
    - `3`: Close All
- **Reward Function**: Scaled PnL with penalties for drawdown and time-decay.

### Training
The agent learns by "playing" 100,000+ simulated candles. 
- **Script**: `src/xauusd_strategy/rl/train_rl.py`
- **Output**: Saved to `models/rl_deepscalper/`

---

## 3. Future AI Roadmap

We have planned the transition from Stage 1 to Stage 3:
1.  **Stage 1 (Current)**: Hybrid XGBoost + Base RL.
2.  **Stage 2**: Integration of Time-Series Transformers (HuggingFace models like Chronos) for feature extraction.
3.  **Stage 3**: Full End-to-End PPO managing dynamic position sizing based on real-time volatility.
