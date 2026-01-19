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
                    self.TIMEFRAME_M1 = self.lib.TIMEFRAME_M1
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
        self.safety = SafetyMonitor(self.settings, adapter=self.adapter)
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
            from xauusd_strategy.ml.online_trainer import OnlineRLTrainer
            
            self.transformer_brain = TransformerEmbedder()
            self.sentiment_analyst = SentimentAnalyst()
            self.tick_predictor = TickPredictor()
            self.rl_agent = DeepScalperAgent()
            
            # Online RL Training for real-time learning
            self.online_trainer = OnlineRLTrainer(
                self.rl_agent,
                buffer_size=1000,
                min_experiences=10,
                update_frequency=10  # Update every 10 closed trades
            )
            logger.info("🎓 Online RL Training enabled - agent will learn from live trades")
        else:
            self.transformer_brain = None
            self.sentiment_analyst = None
            self.tick_predictor = None
            self.rl_agent = None
            self.online_trainer = None
        
        # Pyramiding State
        self.initial_sl_dist = 2.5 # $
        self.step_points = 250     # 250 points = $2.5
        self.max_layers = 4
        
        # Dynamic Volume Calculation for Doubling Potential
        # With €250 balance, 3% risk = €7.5/trade
        # SL of $2.5 on XAUUSD = 250 points = €2.15 per 0.01 lot
        # Volume = Risk / (SL_points * point_value)
        self.base_risk_pct = getattr(self.settings.risk, 'risk_per_trade_pct', 3.0) / 100
        self.volume = self._calculate_lot_size()
        logger.info(f"💰 Dynamic Volume: {self.volume} lots (Risk: {self.base_risk_pct*100:.1f}%)")
        
        # Scalp Cooldown (avoid opening multiple trades on same signal)
        self.last_scalp_time = 0
        self.scalp_cooldown_seconds = 30  # Minimum 30 seconds between scalp entries
        
        # Breakeven Settings (from strategy or default)
        self.breakeven_at_rr = getattr(self.strategy, 'breakeven_at_rr', 1.0)
        self.breakeven_buffer = 0.30  # $0.30 buffer above entry for spread protection (was 0.10)
        self._breakeven_applied = set()  # Track tickets that already have breakeven applied
        self._last_state = None  # Last market state for Online RL Training
        # Persistence
        self.persistence_path = Path("monitor/persistence.json")
        self.signals_today = 0
        self.last_processed_time = None  # Track last candle time to avoid duplicate processing
        self._last_state_save = 0  # Throttle state saving
        self._state_save_interval = 30  # Save state every 30 seconds
        self._load_persistence()
        
        # Pre-initialize ML Feature Engineer (avoid recreation each loop)
        self.ml_feature_engineer = None
        if self.ml_model:
            from xauusd_strategy.ml.features import MLFeatureEngineer
            self.ml_feature_engineer = MLFeatureEngineer(
                use_transformers=(self.transformer_brain is not None)
            )
            if self.transformer_brain:
                self.ml_feature_engineer._embedder = self.transformer_brain
        
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

    def _calculate_lot_size(self, sl_distance: float = 2.5) -> float:
        """
        Calculate dynamic lot size based on account balance and risk percentage.
        
        For XAUUSD:
        - 1 lot = 100 oz, point value ~$1 per point per 0.01 lot
        - Risk = lot_size * sl_points * point_value
        
        Target: Aggressive sizing for doubling potential.
        """
        try:
            account = self.adapter.account_info()
            if account is None:
                return 0.05  # Default fallback
            
            balance = account.balance
            symbol_info = self.adapter.symbol_info(self.symbol)
            
            # Calculate risk amount in account currency
            risk_amount = balance * self.base_risk_pct
            
            # XAUUSD tick value (approx $0.86 per point per 0.01 lot for EUR account)
            # Using tick_value from symbol if available
            tick_value = getattr(symbol_info, 'trade_tick_value', 0.86)
            point = getattr(symbol_info, 'point', 0.01)
            
            # SL in points
            sl_points = sl_distance / point
            
            # Lot size = Risk / (SL_points * tick_value)
            lot_size = risk_amount / (sl_points * tick_value)
            
            # Round to 0.01 and apply limits
            lot_size = round(lot_size, 2)
            lot_size = max(0.01, min(lot_size, 1.0))  # Min 0.01, Max 1.0
            
            return lot_size
            
        except Exception as e:
            logger.error(f"Lot Size Calc Error: {e}")
            return 0.05

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

    def get_market_data(self, n_bars=100, timeframe=None):
        """Get market data. Default M5, but scalping uses M1."""
        if timeframe is None:
            timeframe = self.adapter.TIMEFRAME_M5
        
        # 1. Try fetching from Adapter (Native/MetaApi)
        rates = self.adapter.copy_rates_from_pos(self.symbol, timeframe, 0, n_bars)
        
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
                # Use 'volume' column (real_volume may not exist, tick_volume may not exist)
                vol_col = 'volume' if 'volume' in tick_hist.columns else 'real_volume' if 'real_volume' in tick_hist.columns else None
                if vol_col:
                    sim_ticks = [(r.close, r[vol_col]) for _, r in tick_hist.iterrows()]
                else:
                    sim_ticks = [(r.close, 100) for _, r in tick_hist.iterrows()]  # Fallback
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
        
        logger.info(f"📤 Sending Order: {action_type} {self.volume} @ {price:.2f} SL={sl_price:.2f} TP={tp_price:.2f}")
        result = self.adapter.order_send(request) # Safe execution with retries
        
        if result is None:
            logger.error("❌ Order send returned None!")
        elif hasattr(result, 'retcode'):
            if result.retcode == 10009:  # TRADE_RETCODE_DONE
                logger.info(f"✅ Order SUCCESS: Ticket={result.order} Deal={result.deal}")
                self.signals_today += 1
                self._save_persistence()
                
                # Record for Online RL Training
                if self.online_trainer and hasattr(self, '_last_state') and self._last_state is not None:
                    self.online_trainer.record_trade_open(
                        ticket=result.order,
                        state=self._last_state,
                        action=1 if signal_type == 1 else 2,  # 1=Buy, 2=Sell
                        entry_price=price,
                        direction=signal_type,
                        volume=self.volume
                    )
            else:
                logger.error(f"❌ Order FAILED: retcode={result.retcode} comment={result.comment}")
        else:
            # MetaApi or other adapter result format
            logger.info(f"Order result: {result}")
            self.signals_today += 1
            self._save_persistence()

    def close_all_positions(self, comment="Emergency Close"):
        """Close all open positions for this magic number."""
        positions = self.adapter.positions_get(symbol=self.symbol, magic=self.magic_number)
        if not positions:
            return
            
        logger.info(f"🔴 Closing {len(positions)} positions [{comment}]")
        for p in positions:
            direction = 1 if p.type == 0 else -1  # BUY=0, SELL=1
            tick = self.adapter.symbol_info_tick(self.symbol)
            
            req = {
                "action": self.adapter.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": p.volume,
                "type": 1 if direction == 1 else 0,  # Reverse to close
                "position": p.ticket,
                "price": tick.bid if direction == 1 else tick.ask,
                "magic": self.magic_number,
                "comment": comment,
                "type_time": self.adapter.ORDER_TIME_GTC,
                "type_filling": self.adapter.ORDER_FILLING_IOC,
            }
            result = self.adapter.order_send(req)
            
            if result and hasattr(result, 'retcode'):
                if result.retcode == 10009:
                    logger.info(f"✅ Closed Ticket {p.ticket}")
                else:
                    logger.error(f"❌ Failed to close Ticket {p.ticket}: {result.comment}")

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

    def _manage_breakeven(self, positions):
        """
        Move stop loss to breakeven (entry price + buffer) when position 
        reaches the configured R:R ratio (breakeven_at_rr).
        """
        if not positions:
            return
            
        tick = self.adapter.symbol_info_tick(self.symbol)
        if not tick:
            return
            
        for pos in positions:
            ticket = pos.ticket
            
            # Skip if breakeven already applied to this position
            if ticket in self._breakeven_applied:
                continue
                
            entry_price = pos.price_open
            current_sl = pos.sl
            current_tp = pos.tp
            
            # Skip if no SL set (shouldn't happen but safety check)
            if current_sl == 0:
                continue
                
            # Determine direction: BUY=0, SELL=1
            is_buy = (pos.type == 0)
            
            # Calculate initial risk (distance from entry to SL)
            if is_buy:
                initial_risk = entry_price - current_sl
                current_price = tick.bid
                current_profit_dist = current_price - entry_price
            else:
                initial_risk = current_sl - entry_price
                current_price = tick.ask
                current_profit_dist = entry_price - current_price
            
            # Avoid division by zero
            if initial_risk <= 0:
                continue
                
            # Calculate current R:R
            current_rr = current_profit_dist / initial_risk
            
            # Check if we've reached breakeven threshold
            if current_rr >= self.breakeven_at_rr:
                # Calculate new SL at breakeven + buffer
                if is_buy:
                    new_sl = entry_price + self.breakeven_buffer
                    # Only update if new SL is better (higher for buys)
                    if new_sl <= current_sl:
                        continue
                else:
                    new_sl = entry_price - self.breakeven_buffer
                    # Only update if new SL is better (lower for sells)
                    if new_sl >= current_sl:
                        continue
                
                logger.info(f"🔒 Breakeven Trigger: Ticket {ticket} at {current_rr:.2f}R. Moving SL from {current_sl:.2f} to {new_sl:.2f}")
                
                req = {
                    "action": self.adapter.TRADE_ACTION_SLTP,
                    "position": ticket,
                    "sl": new_sl,
                    "tp": current_tp,
                    "symbol": self.symbol
                }
                result = self.adapter.order_send(req)
                
                # Mark as breakeven applied (even if failed, to avoid spam)
                self._breakeven_applied.add(ticket)

    def _cleanup_breakeven_tracker(self, positions):
        """Remove closed positions from breakeven tracker and record experiences."""
        if not positions:
            # All positions closed - record experiences for any pending trades
            if self.online_trainer and hasattr(self.online_trainer, 'pending_trades'):
                for ticket in list(self.online_trainer.pending_trades.keys()):
                    self._record_closed_trade_experience(ticket)
            self._breakeven_applied.clear()
            return
            
        open_tickets = {pos.ticket for pos in positions}
        closed_tickets = self._breakeven_applied - open_tickets
        
        # Record experiences for closed trades
        if self.online_trainer:
            for ticket in closed_tickets:
                self._record_closed_trade_experience(ticket)
        
        self._breakeven_applied -= closed_tickets
        
    def _record_closed_trade_experience(self, ticket: int):
        """Record a closed trade for online RL training."""
        if not self.online_trainer:
            return
            
        if ticket not in self.online_trainer.pending_trades:
            return
            
        try:
            # Get deal history to find exit price and P&L
            # MT5: Use history_deals_get
            from datetime import datetime, timedelta
            now = datetime.now()
            from_time = now - timedelta(hours=24)
            
            deals = self.adapter.history_deals_get(from_time, now, position=ticket)
            if deals and len(deals) >= 2:
                # Find the closing deal (usually last one with this position)
                close_deal = deals[-1]
                exit_price = getattr(close_deal, 'price', 0)
                pnl = getattr(close_deal, 'profit', 0)
                
                # Get current market state for next_state
                data = self.get_market_data(50)
                if data is not None and len(data) >= 30:
                    df_prep = self.strategy.prepare_data(data)
                    next_state = df_prep.iloc[-30:].values.flatten().astype(np.float32)
                else:
                    next_state = np.zeros(150, dtype=np.float32)  # Fallback
                    
                self.online_trainer.record_trade_close(
                    ticket=ticket,
                    exit_price=exit_price,
                    pnl=pnl,
                    next_state=next_state
                )
                logger.info(f"📚 Recorded closed trade #{ticket}: PnL=${pnl:.2f}")
                
        except Exception as e:
            logger.warning(f"Could not record experience for trade {ticket}: {e}")

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
            import shutil
            
            # Ensure monitor directory exists
            os.makedirs("monitor", exist_ok=True)
            
            # Write to temp file
            tmp_path = "monitor/state.json.tmp"
            final_path = "monitor/state.json"
            
            with open(tmp_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            # Windows-safe file replace with retry
            for attempt in range(3):
                try:
                    if os.path.exists(final_path):
                        os.remove(final_path)
                    shutil.move(tmp_path, final_path)
                    break
                except (PermissionError, OSError):
                    time.sleep(0.1)
            
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

                # Throttled state save (every 30 seconds instead of every loop)
                current_time = time.time()
                if current_time - self._last_state_save >= self._state_save_interval:
                    self._save_monitor_state()
                    self._last_state_save = current_time
                
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
                
                # Cleanup breakeven tracker for closed positions
                self._cleanup_breakeven_tracker(positions)
                
                # Count current positions
                n_positions = len(positions) if positions else 0
                MAX_SCALP_POSITIONS = 5  # Allow up to 5 scalp positions simultaneously
                
                if positions:
                    # MANAGING MODE
                    # First: Check and apply breakeven for individual positions
                    self._manage_breakeven(positions)
                    # Then: Manage pyramid scaling and collective trailing
                    self.manage_pyramid(positions)
                
                # HUNTING MODE: Always check for scalp signals (even with positions open)
                # But limit total positions to MAX_SCALP_POSITIONS
                can_open_new = (n_positions < MAX_SCALP_POSITIONS)
                
                if can_open_new:
                    data = self.get_market_data(200) # M5 for breakout strategy
                    if data is not None and not data.empty:
                        # Check if new M5 bar for breakout strategy
                        last_time = data.index[-1]
                        new_m5_bar = (self.last_processed_time != last_time)
                        
                        if new_m5_bar:
                            self.last_processed_time = last_time
                        
                        # 1. Prepare Data for M5 strategies (Breakout only)
                        df_prep = self.strategy.prepare_data(data)
                        
                        # 2. ML Prediction (Model is trained on Breakouts) - only on new M5 bar
                        ml_prob = 0.0
                        if new_m5_bar and self.ml_model and self.ml_feature_engineer:
                            try:
                                f_df = self.ml_feature_engineer.prepare_ml_features(df_prep)
                                last_row = f_df.iloc[[-1]] 
                                ml_prob = self.ml_model.predict(last_row)[0]
                            except Exception as e:
                                logger.error(f"ML Predict Error: {e}")
                        
                        current_idx = len(df_prep) - 1
                        signals = []

                        # 3. Strategy A: London Breakout (ML Filtered) - only on new M5 bar, and only if NO positions
                        if n_positions == 0 and new_m5_bar and ml_prob > 0.55:
                            sig_a = self.strategy.generate_signal(df_prep, current_idx, ml_prob)
                            if sig_a: signals.append(('Breakout', sig_a))

                        # 4. Strategy B: Scalping on M1 (Rule-Based - Fast & Frequent)
                        # Check cooldown before looking for new scalp signals
                        scalp_cooldown_ok = (time.time() - self.last_scalp_time) >= self.scalp_cooldown_seconds
                        
                        if self.scalp_strategy and scalp_cooldown_ok:
                            # Get M1 data for faster scalping (check every loop iteration!)
                            try:
                                scalp_data = self.get_market_data(100, self.adapter.TIMEFRAME_M1)
                                if scalp_data is not None and not scalp_data.empty:
                                    scalp_prep = self.scalp_strategy.prepare_data(scalp_data)
                                    scalp_idx = len(scalp_prep) - 1
                                    sig_b = self.scalp_strategy.generate_signal(scalp_prep, scalp_idx, ml_prob)
                                    if sig_b: 
                                        signals.append(('Scalp', sig_b))
                                        logger.info(f"🎯 Scalp Signal Detected! RSI/BB Reversal on M1")
                            except Exception as e:
                                import traceback
                                logger.error(f"Scalp M1 Error: {e}\n{traceback.format_exc()}")
                            
                        # 5. Strategy C: DeepScalper (RL Agent)
                        # TEMPORARILY DISABLED: RL model needs retraining with new observation shape
                        # The model was trained with 154 features but now has 422 (with transformer embeddings)
                        # TODO: Retrain RL model with current feature set
                        rl_enabled = False  # Set to True after retraining
                        if rl_enabled and self.rl_agent and self.rl_agent.model:
                            # RL needs N bars window
                            # Account state needed
                            try:
                                # Get Account Info safely
                                acct = self.adapter.account_info()
                                bal = acct.balance if acct else 0.0
                                # Get net position for symbol (only our magic number)
                                pos_net = 0.0
                                pos_fresh = self.adapter.positions_get(symbol=self.symbol, magic=self.magic_number)
                                if pos_fresh:
                                    for p in pos_fresh:
                                        if getattr(p, 'type') == 0: pos_net += getattr(p, 'volume')
                                        else: pos_net -= getattr(p, 'volume')
                                
                                action = self.rl_agent.predict(df_prep, bal, pos_net)
                                
                                # Action Mapping: 0=Hold, 1=Buy, 2=Sell, 3=Close
                                if action == 1:
                                    # Create LONG Signal equivalent
                                    # RL doesn't give SL/TP implicitly, we must assign defaults
                                    # Use 'atr' from london_breakout (always present) or fallback to atr_14
                                    atr = df_prep.iloc[current_idx].get('atr', df_prep.iloc[current_idx].get('atr_14', 3.0))
                                    close = df_prep.iloc[current_idx]['close']
                                    sl = close - (atr * 1.5)
                                    tp = close + (atr * 2.0)
                                    tsig = TradeSignal(SignalType.LONG, close, sl, tp, atr, 0, 0, 0, ml_prob, df_prep.index[current_idx])
                                    signals.append(('DeepScalper', tsig))
                                    
                                elif action == 2:
                                    # Create SHORT Signal
                                    atr = df_prep.iloc[current_idx].get('atr', df_prep.iloc[current_idx].get('atr_14', 3.0))
                                    close = df_prep.iloc[current_idx]['close']
                                    sl = close + (atr * 1.5)
                                    tp = close - (atr * 2.0)
                                    tsig = TradeSignal(SignalType.SHORT, close, sl, tp, atr, 0, 0, 0, ml_prob, df_prep.index[current_idx])
                                    signals.append(('DeepScalper', tsig))
                                    
                                elif action == 3:
                                    # Close All Positions
                                    if pos_fresh:
                                        logger.info("🤖 RL Agent: CLOSE ALL signal received")
                                        self.close_all_positions(comment="RL Close")
                            except Exception as e:
                                logger.error(f"RL Step Error: {e}")

                        
                        # Capture state for Online RL Training BEFORE executing any trades
                        if self.online_trainer and signals:
                            try:
                                # State = last 30 rows flattened (features for RL)
                                self._last_state = df_prep.iloc[-30:].values.flatten().astype(np.float32)
                            except Exception as e:
                                logger.debug(f"Could not capture RL state: {e}")
                                self._last_state = None
                        
                        # Execute Signals
                        for name, signal in signals:
                            logger.info(f"Entry Signal [{name}]! Type: {signal.signal_type} Prob: {ml_prob:.2f}")
                            direction = 1 if signal.signal_type.name == 'LONG' else -1
                            
                            # Recalculate volume for compounding (uses current balance)
                            sl_distance = abs(signal.entry_price - signal.stop_loss)
                            self.volume = self._calculate_lot_size(sl_distance)
                            
                            # Initial SL Logic (From Signal)
                            sl = signal.stop_loss
                            tp = signal.take_profit
                            
                            self.execute_trade(
                                direction, 
                                sl_price=sl, 
                                tp_price=tp, 
                                comment=f"{name} {ml_prob:.2f}"
                            )
                            
                            # Update scalp cooldown after trade execution
                            if name == 'Scalp':
                                self.last_scalp_time = time.time()
                                logger.info(f"⏱️ Scalp cooldown started: {self.scalp_cooldown_seconds}s")
                            # Only take one trade per bar to avoid conflict
                            break 
                    
                time.sleep(1)
                
            except Exception as e:
                import traceback
                logger.error(f"Loop Error: {e}\n{traceback.format_exc()}")
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
