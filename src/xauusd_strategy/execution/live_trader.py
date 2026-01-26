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

# New Architecture Components
from xauusd_strategy.ai.news_calendar import NewsCalendar, NewsImpact
from xauusd_strategy.strategy.regime_detector import RegimeDetector, MarketRegime, RegimeAnalysis
from xauusd_strategy.strategy.multi_tf_aggregator import MultiTFAggregator, Timeframe
from xauusd_strategy.risk.risk_manager import RiskManager, RiskInfo
from xauusd_strategy.strategy.signal_aggregator import SignalAggregator, AggregatedSignal

from xauusd_strategy.utils.logger import get_logger

# Import Adapters
from xauusd_strategy.execution.mt5_adapter import MT5Adapter, mt5 as native_mt5 # Kept native_mt5 for UnifiedNativeAdapter
from xauusd_strategy.execution.metaapi_adapter import MetaApiAdapter
from xauusd_strategy.execution.socket_adapter import SocketAdapter

logger = get_logger("LiveTrader")

class LiveTrader:
    def __init__(self, config_path: str = "config/aggressive.yaml"):
        self.settings = Settings.from_yaml(Path(config_path))
        self.symbol = "XAUUSD"
        self.magic_number = 999
        # Changed timeframe to use native_mt5.TIMEFRAME_M5 as per instruction's intent
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
                    self.TIMEFRAME_H1 = self.lib.TIMEFRAME_H1
                    self.TIMEFRAME_D1 = self.lib.TIMEFRAME_D1
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
                def history_deals_get(self, *args, **kwargs): return self.lib.history_deals_get(*args, **kwargs)
                def account_info(self): return self.lib.account_info()
                def order_send(self, req): return self.wrapper.order_send(req)
            
            self.adapter = UnifiedNativeAdapter()
        
        # --- NEW ARCHITECTURE INITIALIZATION ---
        logger.info("Initializing Architecture 2.0 Components... 🚀")
        
        # 1. News Calendar (Sentiment Engine)
        self.news_calendar = NewsCalendar()

        # self.news_calendar.update() # Can be slow, do async or non-blocking in run
        
        # 2. Regime Detector
        self.regime_detector = RegimeDetector()
        
        # 3. Multi-TF Aggregator
        self.mtf_aggregator = MultiTFAggregator()
        
        # 4. Risk Manager v2 (Replaces SafetyMonitor)
        # Using settings from config if available, else defaults
        self.risk_manager = RiskManager(
            initial_balance=getattr(self.settings.risk, 'initial_balance', 250.0),
            base_risk_pct=getattr(self.settings.risk, 'risk_per_trade_pct', 3.0),
            max_daily_drawdown_pct=getattr(self.settings.risk, 'max_daily_loss_pct', 8.0),
            max_trades_per_day=getattr(self.settings.risk, 'max_daily_trades', 10),
        )
        
        # 5. Signal Aggregator
        self.signal_aggregator = SignalAggregator(
            risk_manager=self.risk_manager,
            min_confidence=60.0 # Configurable
        )
        
        # Strategies
        self.strategy_london = LondonBreakoutStrategy(settings=self.settings)
        self.strategy_asian = AsianScalpingStrategy(settings=self.settings)
        
        # Legacy/Support components
        self.ml_model = self._load_ml_model()
        self.ml_feature_engineer = None
        if self.ml_model:
            from xauusd_strategy.ml.features import MLFeatureEngineer
            self.ml_feature_engineer = MLFeatureEngineer(use_transformers=False)

        # Advanced AI placeholders (disabled by default in this refactor unless explicitly needed)
        self.use_advanced_ai = False 
        self.online_trainer = None

        # State tracking
        self.initial_sl_dist = 2.5 
        self.breakeven_at_rr = getattr(self.strategy_london, 'breakeven_at_rr', 1.0)
        self.breakeven_buffer = 0.30
        self._breakeven_applied = set()
        
        # Persistence
        self.persistence_path = Path("monitor/persistence.json")
        self.signals_today = 0
        self.last_processed_time = None
        self._last_state_save = 0
        self._state_save_interval = 30
        self.last_scalp_time = 0
        self.scalp_cooldown_seconds = 300
        
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

    def _fetch_multi_tf_data(self, n_bars=200):
        """Fetch data for multiple timeframes."""
        data_dict = {}
        
        # M1 (For Scalping)
        m1 = self.get_market_data(n_bars, self.adapter.TIMEFRAME_M1)
        if m1 is not None: data_dict['1m'] = m1
        
        # M5 (For London Breakout / Primary)
        m5 = self.get_market_data(n_bars, self.adapter.TIMEFRAME_M5)
        if m5 is not None: data_dict['5m'] = m5
        
        # H1 (For Trend Context)
        h1 = self.get_market_data(n_bars, self.adapter.TIMEFRAME_H1)
        if h1 is not None: data_dict['1h'] = h1
        
        # D1 (For Daily Bias)
        d1 = self.get_market_data(n_bars, self.adapter.TIMEFRAME_D1)
        if d1 is not None: data_dict['1d'] = d1
        
        return data_dict

    def execute_trade(self, signal_type, sl_price=0.0, tp_price=0.0, comment="", volume=None):
        tick = self.adapter.symbol_info_tick(self.symbol)
        if not tick: return
        
        # SPREAD SAFETY GUARD
        spread = tick.ask - tick.bid
        max_spread = getattr(self.settings, 'execution_max_spread', 0.50)
        
        if spread > max_spread:
            logger.warning(f"Spread Too High: {spread:.2f} > {max_spread:.2f}. SKIPPING TRADE.")
            return

        # Use passed volume or default to self.volume
        trade_volume = volume if volume else self.volume

        action_type = self.adapter.ORDER_TYPE_BUY if signal_type == 1 else self.adapter.ORDER_TYPE_SELL
        price = tick.ask if signal_type == 1 else tick.bid
        
        request = {
            "action": self.adapter.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": trade_volume,
            "type": action_type,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "magic": self.magic_number,
            "comment": comment,
            "type_time": self.adapter.ORDER_TIME_GTC,
            "type_filling": self.adapter.ORDER_FILLING_IOC,
        }
        
        logger.info(f"📤 Sending Order: {action_type} {trade_volume} @ {price:.2f} SL={sl_price:.2f} TP={tp_price:.2f}")
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
                        volume=trade_volume
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
                    df_prep = self.strategy_london.prepare_data(data)
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

            # Calculate Daily PnL using RiskManager
            dd_stats = self.risk_manager.get_drawdown_stats()
            
            state = {
                'timestamp': time.time(),
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'account': {
                    'balance': account.balance,
                    'equity': account.equity,
                    'profit': account.profit
                },
                'risk': {
                    'daily_pnl': dd_stats.daily_pnl,
                    'drawdown_pct': dd_stats.current_drawdown_pct,
                    'daily_limit': self.risk_manager.max_daily_dd
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

        logger.info("Live Trader Started (Architecture 2.0) 🚀")
        logger.info(f"Risk Manager Active: Daily Limit={self.risk_manager.max_daily_dd}%")
        
        while True:
            try:
                # 0. Check for remote STOP signal
                if os.path.exists("monitor/stop.signal"):
                    logger.critical("STOP SIGNAL RECEIVED from Dashboard. Halting.")
                    break

                # Throttled state save (every 30 seconds)
                current_time_sec = time.time()
                if current_time_sec - self._last_state_save >= self._state_save_interval:
                    self._save_monitor_state()
                    self._last_state_save = current_time_sec

                # 1. Update State & Risk
                account = self.adapter.account_info()
                if account:
                    self.risk_manager.update_equity(account.equity)
                
                positions = self.adapter.positions_get(symbol=self.symbol, magic=self.magic_number)
                self._cleanup_breakeven_tracker(positions)
                n_positions = len(positions) if positions else 0
                
                # Check Circuit Breaker
                # RiskManager tracks this internally but we should respect global halts
                status = self.risk_manager.get_status()
                if not status['can_trade']:
                    logger.warning(f"Circuit Breaker TRIPPED: {status['circuit_status']}. Pausing trading...")
                    time.sleep(60)
                    continue

                # 2. Manage Open Positions (Pyramiding, Breakeven)
                if positions:
                    self._manage_breakeven(positions)
                    self.manage_pyramid(positions)

                # 3. Fetch Data (Multi-Timeframe)
                # We fetch enough bars for indicators
                data_dict = self._fetch_multi_tf_data(n_bars=200)
                if not data_dict:
                    time.sleep(5)
                    continue
                    
                # 4. Update Context Analysis
                # News
                news_impact = self.news_calendar.assess_trading_impact()
                if news_impact.halt_trading:
                    logger.critical(f"NEWS HALT: {news_impact.notes}. Pausing...")
                    time.sleep(60)
                    continue
                
                # Regime (Update on Daily data)
                if '1d' in data_dict:
                    regime_analysis = self.regime_detector.detect(data_dict['1d'])
                else:
                    # Fallback default
                    regime_analysis = RegimeAnalysis(MarketRegime.RANGING, 0.5, 25, 50, 0, "normal")

                # Multi-TF Analysis
                mtf_data = {}
                if '1m' in data_dict: mtf_data[Timeframe.M1] = data_dict['1m']
                if '5m' in data_dict: mtf_data[Timeframe.M5] = data_dict['5m']
                if '1h' in data_dict: mtf_data[Timeframe.H1] = data_dict['1h']
                if '1d' in data_dict: mtf_data[Timeframe.D1] = data_dict['1d']

                mtf_analysis = self.mtf_aggregator.aggregate(mtf_data)
                
                # 5. Signal Generation
                raw_signals = []
                # current_dt = datetime.now()
                
                # A. London Breakout (M5)
                # Only run if we have fresh M5 bar
                if '5m' in data_dict and not data_dict['5m'].empty:
                    m5_df = data_dict['5m']
                    last_time = m5_df.index[-1]
                    
                    if self.last_processed_time != last_time:
                        self.last_processed_time = last_time
                        
                        # Prepare Data & Features
                        df_prep = self.strategy_london.prepare_data(m5_df)
                        
                        # ML Prediction
                        ml_prob = 0.0
                        if self.ml_model and self.ml_feature_engineer:
                            try:
                                f_df = self.ml_feature_engineer.prepare_ml_features(df_prep)
                                last_row = f_df.iloc[[-1]] 
                                ml_prob = self.ml_model.predict(last_row)[0]
                            except Exception as e:
                                logger.error(f"ML Predict Error: {e}")
                        
                        # Generate Signal
                        idx = len(df_prep) - 1
                        sig = self.strategy_london.generate_signal(df_prep, idx, ml_prob)
                        if sig:
                            raw_signals.append(sig)

                # B. Asian Scalp (M1)
                # Run more frequently (every loop, but check cooldown)
                current_ts = time.time()
                scalp_cooldown_ok = (current_ts - self.last_scalp_time) >= self.scalp_cooldown_seconds
                if '1m' in data_dict and scalp_cooldown_ok and self.strategy_asian:
                    m1_df = data_dict['1m']
                    m1_prep = self.strategy_asian.prepare_data(m1_df)
                    m1_idx = len(m1_prep) - 1
                    sig_scalp = self.strategy_asian.generate_signal(m1_prep, m1_idx, 0.0) # No ML for M1 yet
                    if sig_scalp:
                        raw_signals.append(sig_scalp)
                        self.last_scalp_time = current_ts # Reset cooldown
                
                # 6. Aggregation & Execution
                for signal in raw_signals:
                    # Process via Signal Aggregator
                    agg_signal = self.signal_aggregator.process_signal(
                        signal=signal,
                        regime_analysis=regime_analysis,
                        multi_tf_analysis=mtf_analysis,
                        news_impact=news_impact,
                        current_equity=account.equity,
                        current_positions=n_positions
                    )
                    
                    if agg_signal:
                        logger.info(f"✅ EXECUTING AGGREGATED SIGNAL: {agg_signal.direction} {agg_signal.position_size} lots")
                        logger.info(f"   Rationale: {agg_signal.rationale}")
                        logger.info(f"   Context: Regime={agg_signal.market_regime}, MTF={agg_signal.multi_tf_alignment}")
                        
                        # Execute
                        sig_type = 1 if agg_signal.direction == "BUY" else -1
                        self.execute_trade(
                            signal_type=sig_type,
                            sl_price=agg_signal.stop_loss,
                            tp_price=agg_signal.take_profit,
                            comment=f"Agg:{agg_signal.confidence_score:.0f}%",
                            volume=agg_signal.position_size
                        )

                # Sleep to prevent tight loop
                time.sleep(1) # 1 second tick

            except Exception as e:
                import traceback
                logger.error(f"Run Loop Error: {e}\n{traceback.format_exc()}")
                time.sleep(5)
                self.connect()


if __name__ == "__main__":
    trader = LiveTrader()
    trader.run()
