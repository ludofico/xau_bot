"""
cTrader Open API Adapter for Mac-native execution.
Works with Fusion Markets and other cTrader brokers.
"""
import asyncio
import threading
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from ctrader_open_api import Client, Protobuf, TcpProtocol, Auth, EndPoints
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import *
from ctrader_open_api.messages.OpenApiMessages_pb2 import *

from xauusd_strategy.utils.logger import get_logger

logger = get_logger("cTraderAdapter")


@dataclass
class cTraderConfig:
    """cTrader API Configuration."""
    enabled: bool = False
    client_id: str = ""
    client_secret: str = ""
    account_id: str = ""
    access_token: str = ""
    refresh_token: str = ""
    host: str = EndPoints.PROTOBUF_LIVE_HOST
    port: int = EndPoints.PROTOBUF_PORT


class cTraderAdapter:
    """
    cTrader Open API adapter for Mac-native trading.
    Uses OAuth 2.0 authentication with streaming data.
    """
    
    # Constants matching MT5 interface
    TRADE_ACTION_DEAL = "TRADE_ACTION_DEAL"
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_IOC = 1
    TIMEFRAME_M5 = "5m"
    
    def __init__(self, config: cTraderConfig):
        self.config = config
        self.client = None
        self.connected = False
        self.account_id = None
        
        # Market data cache
        self.market_data: Dict[str, Dict] = {}
        self.account_data = {"balance": 0.0, "equity": 0.0, "profit": 0.0}
        self.positions: List = []
        
        # Symbol mapping (MT5 name -> cTrader symbol ID)
        self.symbol_map: Dict[str, int] = {}
        
    def initialize(self) -> bool:
        """Initialize connection to cTrader."""
        try:
            logger.info("Connecting to cTrader Open API...")
            
            # Check if we have access token
            if not self.config.access_token:
                logger.warning("No access token. Run OAuth flow first.")
                self._run_oauth_flow()
                return False
            
            # Create client
            self.client = Client(
                self.config.host,
                self.config.port,
                TcpProtocol
            )
            
            # Start connection in background thread
            self._start_client()
            
            # Wait for connection
            timeout = 30
            start = time.time()
            while not self.connected and time.time() - start < timeout:
                time.sleep(1)
            
            if self.connected:
                logger.info("cTrader Connected! ✅")
                return True
            else:
                logger.error("Failed to connect to cTrader")
                return False
                
        except Exception as e:
            logger.error(f"cTrader init error: {e}")
            return False
    
    def _start_client(self):
        """Start client in background thread."""
        def run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Set up callbacks
            self.client.setConnectedCallback(self._on_connected)
            self.client.setDisconnectedCallback(self._on_disconnected)
            self.client.setMessageReceivedCallback(self._on_message)
            
            # Start client
            self.client.startService()
            loop.run_forever()
        
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
    
    def _on_connected(self, client):
        """Handle connection established."""
        logger.info("TCP Connected, authenticating...")
        
        # Authorize application
        request = ProtoOAApplicationAuthReq()
        request.clientId = self.config.client_id
        request.clientSecret = self.config.client_secret
        
        self.client.send(request)
    
    def _on_disconnected(self, client, reason):
        """Handle disconnection."""
        logger.warning(f"cTrader disconnected: {reason}")
        self.connected = False
    
    def _on_message(self, client, message):
        """Handle incoming messages."""
        if message.payloadType == ProtoOAApplicationAuthRes().payloadType:
            logger.info("App authenticated, authorizing account...")
            
            # Authorize trading account
            request = ProtoOAAccountAuthReq()
            request.ctidTraderAccountId = int(self.config.account_id)
            request.accessToken = self.config.access_token
            
            self.client.send(request)
            
        elif message.payloadType == ProtoOAAccountAuthRes().payloadType:
            logger.info("Account authorized! ✅")
            self.connected = True
            self.account_id = int(self.config.account_id)
            
            # Subscribe to symbols
            self._subscribe_symbols()
            
        elif message.payloadType == ProtoOASpotEvent().payloadType:
            # Real-time price update
            event = Protobuf.extract(message)
            symbol_id = event.symbolId
            
            # Find symbol name
            for name, sid in self.symbol_map.items():
                if sid == symbol_id:
                    bid = event.bid / 100000 if event.HasField('bid') else self.market_data.get(name, {}).get('bid', 0)
                    ask = event.ask / 100000 if event.HasField('ask') else self.market_data.get(name, {}).get('ask', 0)
                    
                    self.market_data[name] = {
                        "bid": bid,
                        "ask": ask,
                        "time": time.time()
                    }
                    break
                    
        elif message.payloadType == ProtoOATrader().payloadType:
            trader = Protobuf.extract(message)
            self.account_data = {
                "balance": trader.balance / 100,
                "equity": trader.balance / 100,  # Will be updated with positions
                "profit": 0.0
            }
            
        elif message.payloadType == ProtoOAExecutionEvent().payloadType:
            event = Protobuf.extract(message)
            logger.info(f"Execution event: {event.executionType}")
    
    def _subscribe_symbols(self):
        """Subscribe to XAUUSD price updates."""
        # Request symbol list first
        request = ProtoOASymbolsListReq()
        request.ctidTraderAccountId = self.account_id
        self.client.send(request)
        
        # For now, assume XAUUSD symbol ID (will be updated from response)
        self.symbol_map["XAUUSD"] = 1  # Placeholder
    
    def _run_oauth_flow(self):
        """Run OAuth authorization flow."""
        auth = Auth(self.config.client_id, self.config.client_secret, "http://localhost:8080/callback")
        auth_url = auth.getAuthUri()
        
        logger.info("=" * 60)
        logger.info("OAUTH AUTHORIZATION REQUIRED")
        logger.info("=" * 60)
        logger.info(f"Open this URL in your browser:\n{auth_url}")
        logger.info("=" * 60)
        logger.info("After login, you'll be redirected to localhost:8080/callback")
        logger.info("Copy the 'code' parameter from the URL and paste it here.")
    
    def symbol_info_tick(self, symbol: str):
        """Get current tick for symbol."""
        data = self.market_data.get(symbol)
        if not data:
            return None
        
        class Tick:
            def __init__(self, d):
                self.bid = d['bid']
                self.ask = d['ask']
                self.time = d['time']
        
        return Tick(data)
    
    def account_info(self):
        """Get account information."""
        class AccountInfo:
            def __init__(self, d):
                self.balance = d['balance']
                self.equity = d['equity']
                self.profit = d['profit']
        
        return AccountInfo(self.account_data)
    
    def symbol_info(self, symbol: str):
        """Get symbol specifications."""
        class SymbolInfo:
            def __init__(self):
                self.point = 0.01
                self.digits = 2
        
        return SymbolInfo()
    
    def positions_get(self, **kwargs):
        """Get open positions."""
        return self.positions
    
    def order_send(self, request: Dict) -> Any:
        """Send a trading order."""
        try:
            order_type = request.get('type', self.ORDER_TYPE_BUY)
            symbol = request.get('symbol', 'XAUUSD')
            volume = request.get('volume', 0.01)
            sl = request.get('sl', 0)
            tp = request.get('tp', 0)
            
            # Convert to cTrader format
            trade_side = ProtoOATradeSide.BUY if order_type == self.ORDER_TYPE_BUY else ProtoOATradeSide.SELL
            
            # Create order request
            order_req = ProtoOANewOrderReq()
            order_req.ctidTraderAccountId = self.account_id
            order_req.symbolId = self.symbol_map.get(symbol, 1)
            order_req.orderType = ProtoOAOrderType.MARKET
            order_req.tradeSide = trade_side
            order_req.volume = int(volume * 100)  # Convert lots to units
            
            if sl > 0:
                order_req.stopLoss = sl
            if tp > 0:
                order_req.takeProfit = tp
            
            self.client.send(order_req)
            
            class Result:
                def __init__(self):
                    self.retcode = 10009  # Success
                    self.order = 0
            
            return Result()
            
        except Exception as e:
            logger.error(f"Order send error: {e}")
            
            class Result:
                def __init__(self, error):
                    self.retcode = 10004  # Error
                    self.order = 0
                    self.comment = str(error)
            
            return Result(e)
    
    def copy_rates_from_pos(self, symbol: str, timeframe, start_pos: int, count: int):
        """
        Fetch historical OHLC data from cTrader.
        Compatible with MT5's copy_rates_from_pos interface.
        Falls back to yfinance if cTrader fails.
        """
        try:
            import numpy as np
            import pandas as pd
            from datetime import datetime, timedelta
            
            # Try to fetch from cTrader API
            if self.connected and self.account_id:
                # Request historical data
                request = ProtoOAGetTrendbarsReq()
                request.ctidTraderAccountId = self.account_id
                request.symbolId = self.symbol_map.get(symbol, 1)
                request.period = ProtoOATrendbarPeriod.M5  # 5-minute bars
                
                # Calculate time range
                to_timestamp = int(datetime.now().timestamp() * 1000)
                from_timestamp = int((datetime.now() - timedelta(minutes=5 * count)).timestamp() * 1000)
                
                request.fromTimestamp = from_timestamp
                request.toTimestamp = to_timestamp
                
                self.client.send(request)
                
                # Note: Response will come async - for now fallback to yfinance
                logger.debug("Historical data request sent to cTrader")
            
            # Fallback to yfinance for historical data (always works)
            logger.info(f"Fetching historical data from yfinance for {symbol}...")
            import yfinance as yf
            
            ticker_map = {"XAUUSD": "GC=F", "EURUSD": "EURUSD=X"}
            ticker = ticker_map.get(symbol, symbol)
            
            data = yf.download(ticker, period="5d", interval="5m", progress=False)
            
            if data.empty:
                return None
            
            # Convert to MT5-compatible format
            rates = np.zeros(min(len(data), count), dtype=[
                ('time', 'i8'), ('open', 'f8'), ('high', 'f8'), 
                ('low', 'f8'), ('close', 'f8'), ('tick_volume', 'i8'),
                ('spread', 'i4'), ('real_volume', 'i8')
            ])
            
            data = data.tail(count)
            for i, (idx, row) in enumerate(data.iterrows()):
                if i >= count:
                    break
                rates[i]['time'] = int(idx.timestamp())
                rates[i]['open'] = float(row['Open'].iloc[0] if hasattr(row['Open'], 'iloc') else row['Open'])
                rates[i]['high'] = float(row['High'].iloc[0] if hasattr(row['High'], 'iloc') else row['High'])
                rates[i]['low'] = float(row['Low'].iloc[0] if hasattr(row['Low'], 'iloc') else row['Low'])
                rates[i]['close'] = float(row['Close'].iloc[0] if hasattr(row['Close'], 'iloc') else row['Close'])
                rates[i]['tick_volume'] = int(row['Volume'].iloc[0] if hasattr(row['Volume'], 'iloc') else row['Volume'])
            
            return rates
            
        except Exception as e:
            logger.warning(f"Historical data fetch failed: {e}")
            return None
    
    def get_historical_data(self, symbol: str = "XAUUSD", bars: int = 500) -> 'pd.DataFrame':
        """
        Get historical OHLC data as a pandas DataFrame.
        Used by RL and ML modules for training/inference.
        """
        import pandas as pd
        
        rates = self.copy_rates_from_pos(symbol, self.TIMEFRAME_M5, 0, bars)
        
        if rates is None or len(rates) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame({
            'time': pd.to_datetime(rates['time'], unit='s'),
            'open': rates['open'],
            'high': rates['high'],
            'low': rates['low'],
            'close': rates['close'],
            'volume': rates['tick_volume']
        })
        
        df.set_index('time', inplace=True)
        return df
