#!/bin/bash
cd "$(dirname "$0")"

# Activate Venv
source venv/bin/activate

# Validations
if [ ! -f "config/aggressive.yaml" ]; then
    echo "Config missing!"
    exit 1
fi

echo "Starting XAUUSD Live Trader..."
python src/xauusd_strategy/execution/live_trader.py
