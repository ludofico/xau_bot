
# 🏆 XAUUSD Aggressive Trading System (v1.0 Prod)

This is a production-grade algorithmic trading system designed for aggressive growth on Gold (XAUUSD). It combines classic breakout strategies with modern Machine Learning filters and Reinforcement Learning agents.

## 📁 System Documentation

Explore the detailed documentation for every component:

- **🚀 [OPERATIONS.md](OPERATIONS.md)**: User guide, installation, training, and live execution.
- **📈 [STRATEGY.md](STRATEGY.md)**: Deep dive into the London Breakout and Asian Scalp logic.
- **🤖 [MODELS.md](MODELS.md)**: Technical breakdown of the XGBoost Filter and DeepScalper RL Agent.
- **🗺️ [ROADMAP.md](ROADMAP.md)**: Future AI evolution (Transformers & Advanced RL).

## ⚡ Quick Start

1. **Install**: `pip install -r requirements.txt`
2. **Setup**: Edit `config/aggressive.yaml`.
3. **Train AI**:
    ```bash
    PYTHONPATH=src python src/xauusd_strategy/ml/generator.py
    PYTHONPATH=src python src/xauusd_strategy/ml/trainer.py
    ```
4. **Launch**: `PYTHONPATH=src python src/xauusd_strategy/execution/live_trader.py`

## 🛡️ Safety & Reliability

- **Circuit Breakers**: Hard limits on daily drawdown and equity floors.
- **Persistence**: Daily state saved to `monitor/persistence.json` to handle power/internet outages.
- **Logging**: Comprehensive logs generated in the `logs/` directory.

---
**Build Goal**: €1,000/month recurring income with high precision (80% target win rate).
