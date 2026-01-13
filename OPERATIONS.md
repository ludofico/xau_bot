
# 🛠️ Operations & User Guide

How to run, monitor, and maintain your XAUUSD trading bot.

## 1. Security & Safety First 🛡️

The system includes multiple "Circuit Breakers":
- **Daily Loss Limit**: Configured in `config/aggressive.yaml`. If your daily drawdown hits 8%, the bot closes all positions and shuts down.
- **Spread Guard**: Skips any trade if the broker's spread exceeds $0.50 (50 points).
- **Equity Floor**: Stops trading if balance falls below €100.

## 2. Model Training Workflow 🧬

You should re-train models every 30 days to adapt to gold's changing regimes.

### XGBoost Filter
```bash
# Generate training data locally
PYTHONPATH=src python src/xauusd_strategy/ml/generator.py

# Train and save the filter
PYTHONPATH=src python src/xauusd_strategy/ml/trainer.py
```

### RL Agent (DeepScalper)
```bash
# Start the reinforcement learning process
PYTHONPATH=src python src/xauusd_strategy/rl/train_rl.py
```

## 3. Live Execution 🚀

Start the bot:
```bash
PYTHONPATH=src python src/xauusd_strategy/execution/live_trader.py
```

**Persistence**: The bot saves its state daily to `monitor/persistence.json`. If you stop and restart the bot, it will resume from its current daily trade count.

## 4. Monitoring (The Dashboard) 📊

- Check `monitor/state.json` for real-time P&L as seen by the bot.
- Check `logs/` for detailed execution logs.

## 5. Emergency Stop 🛑

To stop the bot immediately and prevent new trades, you can create a file:
`touch monitor/stop.signal`
The bot will see this file, halt, and shutdown.
