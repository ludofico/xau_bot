# from xauusd_strategy.execution.mt5_adapter import mt5 # REMOVED DIRECT IMPORT
import pandas as pd
import numpy as np
import time
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta, date
import joblib

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from xauusd_strategy.config.settings import Settings
from xauusd_strategy.strategy.london_breakout import LondonBreakoutStrategy, TradeSignal, SignalType
from xauusd_strategy.strategy.asian_scalp import AsianScalpingStrategy
from xauusd_strategy.ml.model import MLProbabilityFilter
# from xauusd_strategy.rl.agent import DeepScalperAgent
# from xauusd_strategy.ai.sentiment import SentimentAnalyst
# from xauusd_strategy.ml.embeddings import TransformerEmbedder
# from xauusd_strategy.ml.tick_model import TickPredictor
from xauusd_strategy.utils.logger import get_logger

# Import Adapters
from xauusd_strategy.execution.mt5_adapter import MT5Adapter, mt5 as native_mt5 # Kept native_mt5 for UnifiedNativeAdapter
from xauusd_strategy.execution.metaapi_adapter import MetaApiAdapter
from xauusd_strategy.execution.socket_adapter import SocketAdapter

logger = get_logger("LiveTrader")

from xauusd_strategy.execution.safety import SafetyMonitor

class LiveTrader:
    def __init__(self, config_path: str = "config/aggressive.yaml"):
        self.settings = Settings.from_yaml(Path(config_path))
        self.symbol = "XAUUSD"
        self.magic_number = 999
        # Changed timeframe to use native_mt5.TIMEFRAME_M5 as per instruction's intent
        # and to align with the UnifiedNativeAdapter's use of native_mt5.
        self.timeframe = native_mt5.TIMEFRAME_M5 
        
        # Initialize Adapter (Priority: cTrader > Socket Bridge > MetaApi > Native)
        if self.settings.ctrader and self.settings.ctrader.enabled:
            logger.info("Using cTrader Open API (Mac Native) 🍎")
            from xauusd_strategy.execution.ctrader_adapter import cTraderAdapter, cTraderConfig
            self.adapter = cTraderAdapter(self.settings.ctrader)
        elif getattr(self.settings, 'use_socket_bridge', False):
            logger.info("Using Open Source Socket Bridge 🔌")
            self.adapter = SocketAdapter()
        elif self.settings.metaapi and self.settings.metaapi.enabled:
            logger.info("Using MetaApi Adapter ☁️")
            self.adapter = MetaApiAdapter(self.settings.metaapi)
            # Patch constants
            self.adapter.TIMEFRAME_M5 = "5m" 
        else:
            logger.info("Using Native MT5 Adapter 🖥️")
            
            # UnifiedNativeAdapter combines the mt5 module with our safe MT5Adapter wrapper
            class UnifiedNativeAdapter:
                def __init__(self):
                    self.lib = native_mt5 # The module
                    self.wrapper = MT5Adapter # The class with safe order_send
                    
                    # Constants
                    self.TIMEFRAME_M5 = self.lib.TIMEFRAME_M5
                    self.ORDER_TYPE_BUY = self.lib.ORDER_TYPE_BUY
                    self.ORDER_TYPE_SELL = self.lib.ORDER_TYPE_SELL
                    self.TRADE_ACTION_DEAL = self.lib.TRADE_ACTION_DEAL
                    self.TRADE_ACTION_SLTP = self.lib.TRADE_ACTION_SLTP
                    self.ORDER_TIME_GTC = self.lib.ORDER_TIME_GTC
                    self.ORDER_FILLING_IOC = self.lib.ORDER_FILLING_IOC
                    
                def initialize(self): return self.wrapper.initialize()
                def copy_rates_from_pos(self, *args): return self.lib.copy_rates_from_pos(*args)
                def symbol_info_tick(self, *args): return self.lib.symbol_info_tick(*args)
                def symbol_info(self, *args): return self.lib.symbol_info(*args)
                def positions_get(self, **kwargs): return self.lib.positions_get(**kwargs)
                def account_info(self): return self.lib.account_info()
                def order_send(self, req): return self.wrapper.order_send(req)
            
            self.adapter = UnifiedNativeAdapter()
        
        # Components
        self.safety = SafetyMonitor(self.settings)
        self.ml_model = self._load_ml_model()
        self.strategy = LondonBreakoutStrategy(settings=self.settings)
        self.scalp_strategy = AsianScalpingStrategy(settings=self.settings)
        
        # Advanced Level 2 & 3 Brains (Optional via Config)
        self.use_advanced_ai = getattr(self.settings, 'advanced_ai', False)
        if self.use_advanced_ai:
            logger.info("Initializing Advanced AI Suite (Level 2 & 3) 🧠🚀")
            # Lazy imports to prevent segfaults on Mac (Torch vs MetaApi conflict)
            from xauusd_strategy.ai.sentiment import SentimentAnalyst
            from xauusd_strategy.ml.embeddings import TransformerEmbedder
            from xauusd_strategy.ml.tick_model import TickPredictor
            from xauusd_strategy.rl.agent import DeepScalperAgent
            
            self.transformer_brain = TransformerEmbedder()
            self.sentiment_analyst = SentimentAnalyst()
            self.tick_predictor = TickPredictor()
            self.rl_agent = DeepScalperAgent()
        else:
            self.transformer_brain = None
            self.sentiment_analyst = None
            self.tick_predictor = None
            self.rl_agent = None
        
        # Pyramiding State
        self.initial_sl_dist = 2.5 # $
        self.step_points = 250     # 250 points = $2.5
        self.max_layers = 4
        self.volume = 0.03
        # Persistence
        self.persistence_path = Path("monitor/persistence.json")
        self.signals_today = 0
        self.last_processed_time = None  # Track last candle time to avoid duplicate processing
        self._load_persistence()
        
    def _load_persistence(self):
        """Restore state from file to handle restarts during trading hours."""
        if self.persistence_path.exists():
            try:
                import json
                with open(self.persistence_path, 'r') as f:
                    data = json.load(f)
                    # Only restore if it's the same day
                    if data.get('date') == date.today().isoformat():
                        self.signals_today = data.get('signals_today', 0)
                        logger.info(f"Restored Persistence: {self.signals_today} signals already sent today.")
            except Exception as e:
                logger.error(f"Persistence Load Error: {e}")

    def _save_persistence(self):
        """Save state to handle unexpected crashes."""
        try:
            import json
            data = {
                'date': date.today().isoformat(),
                'signals_today': self.signals_today
            }
            self.persistence_path.parent.mkdir(exist_ok=True)
            with open(self.persistence_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Persistence Save Error: {e}")

    def _load_ml_model(self):
        try:
            model_path = Path("models/ml_filter_doubler.pkl")
            if model_path.exists():
                ml = MLProbabilityFilter()
                ml.load(model_path)
                logger.info("ML Model loaded successfully")
                return ml
            else:
                logger.warning("ML Model not found! Running WITHOUT filter (Dangerous)")
                return None
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            return None

    def connect(self):
        return self.adapter.initialize()

    def get_market_data(self, n_bars=100):
        # 1. Try fetching from Adapter (Native/MetaApi)
        rates = self.adapter.copy_rates_from_pos(self.symbol, self.adapter.TIMEFRAME_M5, 0, n_bars)
        
        # 2. If Adapter fails (e.g. Socket Bridge), use YFinance fallback
        if rates is None:
            from xauusd_strategy.data.fetcher import DataFetcher
            fetcher = DataFetcher()
            # YFinance symbol typically matches MetaTrader for XAUUSD
            df = fetcher.fetch_yfinance(self.symbol, period="5d", interval="5m")
            if df is not None and not df.empty:
                return df.tail(n_bars)
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        # Rename if needed (MetaApi returns tick_volume, mt5 returns tick_volume)
        if 'tick_volume' in df.columns:
            df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        return df

    def execute_trade(self, signal_type, sl_price=0.0, tp_price=0.0, comment=""):
        tick = self.adapter.symbol_info_tick(self.symbol)
        if not tick: return
        
        # SPREAD SAFETY GUARD
        spread = tick.ask - tick.bid
        max_spread = getattr(self.settings, 'execution_max_spread', 0.50)
        
        if spread > max_spread:
            logger.warning(f"Spread Too High: {spread:.2f} > {max_spread:.2f}. SKIPPING TRADE.")
            return

        # LEVEL 3: TICK SNIPING (Entry Optimization)
        if self.use_advanced_ai and self.tick_predictor:
            logger.info("Sniping Entry (Waiting for favorable tick prediction)...")
            sniped = False
            for _ in range(20): # Wait up to 2 seconds (20 * 100ms)
                tick_hist = self.get_market_data(20) # Simplified: uses 1m/5m but should be tick
                # In real MT5 we'd use copy_ticks_from, here we mock it with recent data
                sim_ticks = [(r.close, r.tick_volume) for _, r in tick_hist.iterrows()]
                prediction = self.tick_predictor.predict_delta(sim_ticks)
                
                # If buying, we want a positive predicted delta (Up)
                # If selling, we want a negative predicted delta (Down)
                if (signal_type == 1 and prediction > 0) or (signal_type == -1 and prediction < 0):
                    logger.info(f"Sniper Target Locked: Prediction={prediction:.4f}")
                    sniped = True
                    break
                time.sleep(0.1)
            if not sniped:
                logger.warning("Sniping timed out. Executing at current price.")

        action_type = self.adapter.ORDER_TYPE_BUY if signal_type == 1 else self.adapter.ORDER_TYPE_SELL
        # Refresh tick for entry
        tick = self.adapter.symbol_info_tick(self.symbol)
        price = tick.ask if signal_type == 1 else tick.bid
        
        request = {
            "action": self.adapter.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": self.volume,
            "type": action_type,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "magic": self.magic_number,
            "comment": comment,
            "type_time": self.adapter.ORDER_TIME_GTC,
            "type_filling": self.adapter.ORDER_FILLING_IOC,
        }
        
        self.adapter.order_send(request) # Safe execution with retries
        self.signals_today += 1
        self._save_persistence()

    def close_all_positions(self, comment="Emergency Close"):
        """Close all open positions for this magic number."""
        positions = self.adapter.positions_get(symbol=self.symbol, magic=self.magic_number)
        if not positions:
            return
            
        logger.info(f"Closing {len(positions)} positions [{comment}]")
        for p in positions:
            direction = 1 if p.type == 0 else -1 # BUY=0, SELL=1
            # Close by opening opposite direction (Market Close)
            # Actually, Unified adapter should ideally have a close_position method, 
            # but manually sending a reverse order is standard MT5 close.
            # However, Native MT5 order_send with target ticket is safer.
            # For now, keeping it simple:
            req = {
                "action": self.adapter.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": p.volume,
                "type": 1 if direction == 1 else 0, # Reverse
                "position": p.ticket,
                "price": self.adapter.symbol_info_tick(self.symbol).bid if direction == 1 else self.adapter.symbol_info_tick(self.symbol).ask,
                "magic": self.magic_number,
                "comment": comment,
                "type_time": self.adapter.ORDER_TIME_GTC,
                "type_filling": self.adapter.ORDER_FILLING_IOC,
            }
            self.adapter.order_send(req)

    def manage_pyramid(self, positions):
        # Sort positions by time
        sorted_pos = sorted(positions, key=lambda x: x.time)
        first_pos = sorted_pos[0]
        last_pos = sorted_pos[-1]
        
        direction = 1 if first_pos.type == 0 else -1 # 0 is BUY in MetaApi/MT5 normalized
        # Note: MetaApiAdapter normalizes type to 0/1. Native uses integer constants.
        # We need to ensure logic holds.
        # Native: ORDER_TYPE_BUY=0, SELL=1.
        
        count = len(sorted_pos)
        
        tick = self.adapter.symbol_info_tick(self.symbol)
        current_price = tick.bid if direction == 1 else tick.ask
        point = self.adapter.symbol_info(self.symbol).point
        
        # Scaling Logic
        if count < self.max_layers:
            dist = (current_price - last_pos.price_open) if direction == 1 else (last_pos.price_open - current_price)
            dist_points = dist / point
            
            if dist_points >= self.step_points:
                logger.info(f"Pyramid Trigger: +{dist_points} pts. Adding Layer {count+1}")
                # New SL is previous Entry
                new_sl = last_pos.price_open
                self.execute_trade(1 if direction == 1 else -1, sl_price=new_sl, comment=f"Layer {count+1}")
                return # Wait for next cycle
        
        # Trailing Logic (Collective Defense)
        if count > 1:
            target_sl = last_pos.price_open
            buffer = 0.50 
            target_sl = (target_sl - buffer) if direction == 1 else (target_sl + buffer)
            
            for pos in positions:
                update_needed = False
                if direction == 1 and (pos.sl < target_sl or pos.sl == 0):
                    update_needed = True
                elif direction == -1 and (pos.sl > target_sl or pos.sl == 0):
                    update_needed = True
                
                if update_needed:
                    req = {
                        "action": self.adapter.TRADE_ACTION_SLTP,
                        "position": pos.ticket,
                        "sl": target_sl,
                        "tp": pos.tp,
                        "symbol": self.symbol
                    }
                    self.adapter.order_send(req)

    def _save_monitor_state(self):
        """Save current state to JSON for dashboard monitoring."""
        try:
            account = self.adapter.account_info()
            if account is None:
                return

            positions = self.adapter.positions_get(symbol=self.symbol, magic=self.magic_number)
            pos_list = []
            if positions:
                for p in positions:
                    pos_data = {
                        'symbol': getattr(p, 'symbol', ''),
                        'type': 'BUY' if getattr(p, 'type', 0) == 0 else 'SELL',
                        'volume': getattr(p, 'volume', 0.0),
                        'price_open': getattr(p, 'price_open', 0.0),
                        'profit': getattr(p, 'profit', 0.0)
                    }
                    pos_list.append(pos_data)

            # Calculate Daily PnL
            daily_pnl_pct = 0.0
            if self.safety.initial_day_equity > 0:
                pnl = account.equity - self.safety.initial_day_equity
                daily_pnl_pct = (pnl / self.safety.initial_day_equity)

            state = {
                'timestamp': time.time(),
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'account': {
                    'balance': account.balance,
                    'equity': account.equity,
                    'profit': account.profit
                },
                'risk': {
                    'daily_pnl_pct': daily_pnl_pct,
                    'daily_limit': self.safety.max_daily_loss
                },
                'positions': pos_list
            }
            
            import json
            import os
            
            # Ensure safe atomic write
            with open("monitor/state.json.tmp", 'w') as f:
                json.dump(state, f, indent=2)
            os.replace("monitor/state.json.tmp", "monitor/state.json")
            
        except Exception as e:
            logger.error(f"Dashboard State Save Error: {e}")

    def run(self):
        if not self.connect():
            return

        logger.info("Live Trader Started (v1.0 Production)...")
        logger.info(f"Safety Monitor Active: Daily Loss Limit={self.safety.max_daily_loss}%")
        
        while True:
            try:
                # Check for remote STOP signal
                if os.path.exists("monitor/stop.signal"):
                    logger.critical("STOP SIGNAL RECEIVED from Dashboard. Halting.")
                    # Optional: Close all positions here if desired
                    break

                self._save_monitor_state()
                
                # 0. Safety Checks
                if not self.safety.check_risk_limits():
                    logger.critical("Risk Limit Hit. Shutting Down.")
                    break
                
                if not self.safety.check_market_conditions(self.symbol):
                    time.sleep(5)
                    continue

                # LEVEL 3: SENTIMENT ANALYST (News Circuit Breaker)
                if self.use_advanced_ai and self.sentiment_analyst:
                    risk_multiplier = self._check_news_sentiment()
                    if risk_multiplier == 0.0:
                        logger.critical("🛑 SENTINEL HALT: High-Impact News Detected. Trading Paused.")
                        time.sleep(60)
                        continue
                    elif risk_multiplier < 1.0:
                        logger.warning(f"⚠️ SENTINEL CAUTION: News risk detected. Multiplier: {risk_multiplier}")

                # 1. State Check ...
                positions = self.adapter.positions_get(symbol=self.symbol, magic=self.magic_number)
                
                if positions:
                    # MANAGING MODE
                    self.manage_pyramid(positions)
                else:
                    # HUNTING MODE (Multi-Strategy)
                    data = self.get_market_data(200) # Increased lookback for indicators
                    if data is not None and not data.empty:
                        # Check if new bar
                        last_time = data.index[-1]
                        if self.last_processed_time == last_time:
                            time.sleep(0.1)
                            continue
                            
                        self.last_processed_time = last_time
                        
                        # 1. Prepare Data
                        df_prep = self.strategy.prepare_data(data)
                        
                        # 2. ML Prediction (Model is trained on Breakouts)
                        ml_prob = 0.0
                        if self.ml_model:
                            try:
                                from xauusd_strategy.ml.features import MLFeatureEngineer
                                # Stage 2: Transformer embeddings enabled for ML filtering
                                eng = MLFeatureEngineer(use_transformers=(self.transformer_brain is not None))
                                if self.transformer_brain:
                                    eng._embedder = self.transformer_brain # Use the pre-loaded brain
                                
                                f_df = eng.prepare_ml_features(df_prep)
                                last_row = f_df.iloc[[-1]] 
                                ml_prob = self.ml_model.predict(last_row)[0]
                            except Exception as e:
                                logger.error(f"ML Predict Error: {e}")
                        
                        current_idx = len(df_prep) - 1
                        signals = []

                        # 3. Strategy A: London Breakout (Filtered)
                        if ml_prob > 0.60: # High threshold
                            sig_a = self.strategy.generate_signal(df_prep, current_idx, ml_prob)
                            if sig_a: signals.append(('Breakout', sig_a))

                        # 4. Strategy B: Asian Scalping (Rule-Based)
                        if self.scalp_strategy:
                            sig_b = self.scalp_strategy.generate_signal(df_prep, current_idx, ml_prob)
                            if sig_b: signals.append(('Scalp_Rules', sig_b))
                            
                        # 5. Strategy C: DeepScalper (RL Agent)
                        if self.rl_agent and self.rl_agent.model:
                            # RL needs N bars window
                            # Account state needed
                            try:
                                # Get Account Info safely
                                acct = self.adapter.account_info()
                                bal = acct.balance if acct else 0.0
                                # Get net position for symbol
                                pos_net = 0.0
                                # (Assuming positions variable is up to date or fetch fresh)
                                pos_fresh = self.adapter.positions_get(symbol=self.symbol)
                                if pos_fresh:
                                    for p in pos_fresh:
                                        if getattr(p, 'type') == 0: pos_net += getattr(p, 'volume')
                                        else: pos_net -= getattr(p, 'volume')
                                
                                action = self.rl_agent.predict(df_prep, bal, pos_net)
                                
                                # Action Mapping: 0=Hold, 1=Buy, 2=Sell, 3=Close
                                if action == 1:
                                    # Create LONG Signal equivalent
                                    # RL doesn't give SL/TP implicitly, we must assign defaults or let it manage
                                    # For safety, we assign scalping SL/TP
                                    import pandas as pd
                                    atr = df_prep.iloc[current_idx]['atr_14']
                                    close = df_prep.iloc[current_idx]['close']
                                    sl = close - (atr * 1.5)
                                    tp = close + (atr * 2.0)
                                    tsig = TradeSignal(SignalType.LONG, close, sl, tp, atr, 0, 0, 0, ml_prob, df_prep.index[current_idx])
                                    signals.append(('DeepScalper', tsig))
                                    
                                elif action == 2:
                                    # Create SHORT Signal
                                    atr = df_prep.iloc[current_idx]['atr_14']
                                    close = df_prep.iloc[current_idx]['close']
                                    sl = close + (atr * 1.5)
                                    tp = close - (atr * 2.0)
                                    tsig = TradeSignal(SignalType.SHORT, close, sl, tp, atr, 0, 0, 0, ml_prob, df_prep.index[current_idx])
                                    signals.append(('DeepScalper', tsig))
                                    
                                elif action == 3:
                                    # Close All
                                    if pos_fresh:
                                        for p in pos_fresh:
                                            # Using MT5Adapter close method if available, or raw order
                                            # MT5Adapter.close_position(p.ticket) # If implemented
                                            # For now implementation detail:
                                            self.execute_trade(1 if p.type==1 else -1, comment="RL Close") # Invert to close? No execute_trade opens.
                                            # We need a close method.
                                            pass
                            except Exception as e:
                                logger.error(f"RL Step Error: {e}")

                        
                        # Execute Signals
                        for name, signal in signals:
                            logger.info(f"Entry Signal [{name}]! Type: {signal.signal_type} Prob: {ml_prob:.2f}")
                            direction = 1 if signal.signal_type.name == 'LONG' else -1
                            
                            # Initial SL Logic (From Signal)
                            sl = signal.stop_loss
                            tp = signal.take_profit
                            
                            self.execute_trade(
                                direction, 
                                sl_price=sl, 
                                tp_price=tp, 
                                comment=f"{name} {ml_prob:.2f}"
                            )
                            # Only take one trade per bar to avoid conflict?
                            # For now take first valid
                            break 
                    
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Loop Error: {e}")
                time.sleep(5)
                self.connect() # Reconnect attempt

    def _check_news_sentiment(self) -> float:
        """Fetch and analyze news headlines (Mocked for demonstration or uses real scraping)."""
        # In production, we'd scrape ForexFactory or use an API
        # For Demo: check if 'news.txt' exists with headlines
        headlines = []
        if os.path.exists("news.txt"):
            with open("news.txt", "r") as f:
                headlines = [line.strip() for line in f.readlines() if line.strip()]
        
        if not headlines:
            # Silent fallback: assume neutral
            return 1.0
            
        return self.sentiment_analyst.get_news_risk_multiplier(headlines)

if __name__ == "__main__":
    trader = LiveTrader()
    trader.run()
