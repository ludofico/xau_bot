
# 🗺️ AI Evolution Roadmap

This document outlines the planned evolutionary stages for the XAUUSD trading system.

## Stage 1: The Sniper (Current) 🎯
- **XGBoost Filter**: Standard binary classification (Win vs Loss).
- **Hard-coded Features**: RSI, ATR, ROC, EMA.
- **Rule-based Management**: Pyramiding and fixed TP/SL.

## Stage 2: The Linguist (Vision & Context) 👁️
- **Time-Series Transformers**: Integrate models like **Amazon Chronos** or **Informer** via HuggingFace.
- **Embeddings**: Replace manual features with learned "Price Embeddings."
- **Win**: The model identifies *patterns* (head and shoulders, flags) rather than just math ratios.

## Stage 3: The Manager (Deep RL) 🏆
- **PPO Expansion**: Transition from a discrete action space (Buy/Sell) to a continuous one. 
- **Dynamic Sizing**: The agent decides precisely how many lots to risk based on market confidence.
- **Unified Policy**: One single "Brain" that manages entry, sizing, exit, and hedging.

## Future Research Areas
- **Sentiment Analysis**: Scraping "ForexFactory" or "X" (Twitter) news sentiment to halt the bot before high-impact "Red Folder" events.
- **HFT Tick Analysis**: Using LSTMs on tick-level data (bid/ask volume) to predict the next 5-10 pips of price motion.
