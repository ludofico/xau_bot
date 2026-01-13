
import asyncio
import nest_asyncio
from datetime import datetime
import pandas as pd
from typing import Optional, Dict, Any, List

from xauusd_strategy.utils.logger import get_logger
from xauusd_strategy.config.settings import MetaApiConfig

# Apply nest_asyncio to allow nested loops if we run in notebook or complex env
nest_asyncio.apply()

logger = get_logger("MetaApiAdapter")

# Mock constants to match MT5
TIMEFRAME_M1 = "1m"
TIMEFRAME_M5 = "5m"
TIMEFRAME_M15 = "15m"
TIMEFRAME_H1 = "1h"

ORDER_TYPE_BUY = "ORDER_TYPE_BUY"
ORDER_TYPE_SELL = "ORDER_TYPE_SELL"
TRADE_ACTION_DEAL = "TRADE_ACTION_DEAL"
TRADE_ACTION_SLTP = "TRADE_ACTION_SLTP"
ORDER_TIME_GTC = "ORDER_TIME_GTC"
ORDER_FILLING_IOC = "ORDER_FILLING_IOC"

class MetaApiAdapter:
    """
    Synchronous wrapper for MetaApi Cloud SDK.
    Mimics the `mt5` module interface used in LiveTrader.
    """
    
    def __init__(self, config: MetaApiConfig):
        self.token = config.token
        self.account_id = config.account_id
        self.domain = config.domain
        
        self.api = None
        self.account = None
        self.connection = None
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        # Mapping constants
        self.TIMEFRAME_M5 = "5m"
        self.ORDER_TYPE_BUY = "ORDER_TYPE_BUY"
        self.ORDER_TYPE_SELL = "ORDER_TYPE_SELL"
        self.TRADE_ACTION_DEAL = "TRADE_ACTION_DEAL"
        self.TRADE_ACTION_SLTP = "TRADE_ACTION_SLTP"
        self.ORDER_TIME_GTC = "ORDER_TIME_GTC"
        self.ORDER_FILLING_IOC = "ORDER_FILLING_IOC"
        
    def initialize(self) -> bool:
        """Connect to MetaApi."""
        try:
            from metaapi_cloud_sdk import MetaApi
            
            async def _init_and_connect():
                self.api = MetaApi(token=self.token)
                return await self._connect_async()
            
            return self._run_async(_init_and_connect())
        except ImportError:
            logger.error("metaapi_cloud_sdk not installed! Run: pip install metaapi-cloud-sdk")
            return False
        except Exception as e:
            logger.error(f"MetaApi Init Failed: {e}")
            return False

    async def _connect_async(self):
        try:
            self.account = await self.api.metatrader_account_api.get_account(self.account_id)
            
            # Wait for deployment if needed
            if self.account.state != 'DEPLOYED':
                 logger.info(f"Account state: {self.account.state}, waiting for deployment...")
                 await self.account.deploy()
                 
            logger.info("Connecting to RPC...")
            self.connection = self.account.get_rpc_connection()
            await self.connection.connect()
            await self.connection.wait_synchronized()
            logger.info("MetaApi Connected & Synchronized!")
            return True
        except Exception as e:
            logger.error(f"Async Connect Failed: {e}")
            return False

    def _run_async(self, coro):
        """Helper to run async in sync context."""
        return self._loop.run_until_complete(coro)

    # --- MT5 Interface Mimic ---

    def symbol_info_tick(self, symbol: str):
        """Get current tick."""
        # MetaApi doesn't have direct 'tick' always unless subscribed.
        # We use get_symbol_price
        try:
            price = self._run_async(self.connection.get_symbol_price(symbol))
            # Return object with ask/bid attributes
            class Tick:
                pass
            t = Tick()
            t.ask = price['ask']
            t.bid = price['bid']
            t.time = datetime.now() # Approx
            return t
        except Exception as e:
            logger.error(f"Tick Error: {e}")
            return None

    def symbol_info(self, symbol: str):
        """Get symbol info (point, etc)."""
        try:
            # MetaApi get_symbol_specification
            spec = self._run_async(self.connection.get_symbol_specification(symbol))
            class Info:
                pass
            i = Info()
            i.point = 10 ** -spec.get('digits', 2) # Approximation
            if 'point' in spec: i.point = spec['point']
            i.digits = spec.get('digits', 2)
            return i
        except Exception as e:
            logger.error(f"Symbol Info Error: {e}")
            return None

    def copy_rates_from_pos(self, symbol: str, timeframe: str, start_pos: int, count: int):
        """Get OHLCv."""
        # MetaApi uses get_candles(symbol, timeframe, startTime, limit)
        # timeframe format: '5m', '1h'
        try:
            # Note: start_pos 0 means 'latest'. 
            # We fetch 'count' candles ending now.
            candles = self._run_async(self.connection.get_recent_candles(symbol, timeframe, limit=count))
            if not candles: return None
            
            # Convert to configured list of dicts or tuples expected by pandas
            # MT5 returns numpy array of tuples usually, but LiveTrader converts to DataFrame immediately from dict
            # LiveTrader expects list of dicts or similar
            
            # MetaApi returns list of dicts: {'time': ..., 'open': ...}
            # We need to ensure keys match what LiveTrader expects: 'time', 'open', ...
            # MetaApi time is ISO string usually, we might need to parse? 
            # Actually get_recent_candles returns objects with 'time' as datetime usually in SDK
            
            # Corrections for LiveTrader df construction
            cleaned = []
            for c in candles:
                cleaned.append({
                    'time': c['time'].timestamp(), # LiveTrader expects timestamp for 's' unit conversion?
                    'open': c['open'],
                    'high': c['high'],
                    'low': c['low'],
                    'close': c['close'],
                    'tick_volume': c.get('tickVolume', 0),
                    'spread': c.get('spread', 0),
                    'real_volume': c.get('volume', 0)
                })
            return cleaned
        except Exception as e:
            logger.error(f"Copy Rates Error: {e}")
            return None

    def positions_get(self, symbol=None, magic=None):
        """Get open positions."""
        try:
            positions = self._run_async(self.connection.get_positions())
            
            result = []
            for p in positions:
                # Filter
                if symbol and p['symbol'] != symbol: continue
                # MetaApi usually doesn't show magic in simple get_positions? 
                # It does if we look close. assuming yes.
                if magic and p.get('magic', 0) != magic: continue
                
                class Pos:
                    pass
                obj = Pos()
                obj.ticket = p['id']
                obj.symbol = p['symbol']
                obj.type = 0 if p['type'] == 'POSITION_TYPE_BUY' else 1 # 0=Buy, 1=Sell
                obj.volume = p['volume']
                obj.price_open = p['openPrice']
                obj.sl = p.get('stopLoss', 0.0) or 0.0
                obj.tp = p.get('takeProfit', 0.0) or 0.0
                obj.profit = p.get('profit', 0.0)
                obj.time = p['time'].timestamp() # approx
                result.append(obj)
            return tuple(result)
        except Exception as e:
            logger.error(f"Positions Error: {e}")
            return None

    def account_info(self):
        """Get account info."""
        try:
            info = self._run_async(self.connection.get_account_information())
            class Acc:
                pass
            a = Acc()
            a.balance = info['balance']
            a.equity = info['equity']
            a.profit = info.get('profit', 0.0) # Might calculate from equity-balance
            return a
        except Exception as e:
            logger.error(f"Account Error: {e}")
            return None

    def order_send(self, request: Dict):
        """Execute order."""
        try:
            symbol = request['symbol']
            action = request['action']
            type_op = request['type']
            volume = request['volume']
            sl = request['sl']
            tp = request['tp']
            
            # Map MT5 request to MetaApi calls
            if action == TRADE_ACTION_DEAL:
                # Market Order
                action_type = 'ORDER_TYPE_BUY' if type_op == ORDER_TYPE_BUY or type_op == 0 else 'ORDER_TYPE_SELL'
                
                # MetaApi SDK uses create_market_buy_order / create_market_sell_order
                if action_type == 'ORDER_TYPE_BUY':
                    res = self._run_async(self.connection.create_market_buy_order(
                        symbol, volume, stop_loss=sl, take_profit=tp, options={'magic': request.get('magic')}
                    ))
                else:
                    res = self._run_async(self.connection.create_market_sell_order(
                        symbol, volume, stop_loss=sl, take_profit=tp, options={'magic': request.get('magic')}
                    ))
                    
                # Construct result object
                class Res:
                    pass
                r = Res()
                r.retcode = 10009 # DONE
                r.comment = "MetaApi Executed"
                return r

            elif action == TRADE_ACTION_SLTP:
                # Modify Position
                # request['position'] is ticket
                ticket = request['position']
                res = self._run_async(self.connection.modify_position(
                    ticket, stop_loss=sl, take_profit=tp
                ))
                class Res:
                    pass
                r = Res()
                r.retcode = 10009
                return r

        except Exception as e:
            logger.error(f"Order Send Error: {e}")
            class Res:
                pass
            r = Res()
            r.retcode = 0
            r.comment = str(e)
            return r

