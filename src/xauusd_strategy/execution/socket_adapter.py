
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import time
from typing import Dict, Any, Optional, List
from xauusd_strategy.utils.logger import get_logger

logger = get_logger("HttpAdapterV3")

class SocketAdapter:
    """
    Open Source Bridge V3: HTTP Server.
    MT5 uses 'WebRequest' to poll this server.
    Extremely robust on Mac/Wine.
    """
    def __init__(self, host: str = "0.0.0.0", port: int = 5555):
        self.host = host
        self.port = port
        self.running = False
        
        # Consistent with Adapter interface
        self.TRADE_ACTION_DEAL = "TRADE_ACTION_DEAL"
        self.TRADE_ACTION_SLTP = "TRADE_ACTION_SLTP"
        self.ORDER_TYPE_BUY = 0
        self.ORDER_TYPE_SELL = 1
        self.ORDER_TIME_GTC = 0
        self.ORDER_FILLING_IOC = 1
        self.TIMEFRAME_M5 = "5m"

        # State Data
        self.market_data = {} 
        self.account_data = {"balance": 0.0, "equity": 0.0, "profit": 0.0}
        self.cmd_queue = []
        
        # Shared reference for the handler
        self._set_global_reference()
        self._start_server()

    def _set_global_reference(self):
        # We use a class variable to share the adapter instance with the handler
        type(self).instance = self

    def _start_server(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        logger.info(f"HTTP Bridge Server started on http://{self.host}:{self.port} 🌐")

    def _run_server(self):
        adapter_ref = self
        class BridgeHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return # Silence standard logs

            def do_POST(self):
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length).decode('utf-8')
                
                response = adapter_ref._handle_protocol(post_data)
                
                self.send_response(200)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(response.encode('utf-8'))

        server = HTTPServer((self.host, self.port), BridgeHandler)
        server.serve_forever()

    def _handle_protocol(self, data: str) -> str:
        """
        Parses POLL data: POLL|SYMBOL|BID|ASK|BALANCE|EQUITY
        """
        try:
            # Strip null bytes from MQL5's StringToCharArray
            clean_data = data.replace('\x00', '').strip()
            parts = clean_data.split('|')
            if parts[0] == "POLL":
                symbol = parts[1]
                bid = float(parts[2])
                ask = float(parts[3])
                balance = float(parts[4])
                equity = float(parts[5])
                
                # Update State
                self.market_data[symbol] = {"bid": bid, "ask": ask, "time": time.time()}
                self.account_data = {
                    "balance": balance, 
                    "equity": equity,
                    "profit": equity - balance
                }
                
                # Check for pending commands
                if self.cmd_queue:
                    return self.cmd_queue.pop(0)
                
                return "CMD|NONE"
            
            return "ERROR|UNKNOWN_ACTION"
        except Exception as e:
            return f"ERROR|{str(e)}"

    def initialize(self) -> bool:
        logger.info("Waiting for MT5 WebRequest Client to check in...")
        start_wait = time.time()
        while time.time() - start_wait < 30:
            if self.market_data:
                logger.info("MT5 HTTP Bridge Active! ✅")
                return True
            time.sleep(1)
        logger.warning("Bridge Offline. Ensure 'http://127.0.0.1:5555' is allowed in MT5.")
        return False

    def symbol_info_tick(self, symbol: str):
        data = self.market_data.get(symbol)
        if not data: return None
        class T:
            def __init__(self, d):
                self.bid, self.ask, self.time = d['bid'], d['ask'], d['time']
        return T(data)

    def account_info(self):
        class A:
            def __init__(self, d):
                self.balance, self.equity, self.profit = d['balance'], d['equity'], d['profit']
        return A(self.account_data)

    def order_send(self, request: Dict):
        action = "BUY" if request['type'] == self.ORDER_TYPE_BUY else "SELL"
        cmd = f"CMD|{action}|{request['symbol']}|{request['volume']}|{request['type']}|{request['sl']}|{request['tp']}|{request['magic']}"
        self.cmd_queue.append(cmd)
        class R:
            def __init__(self): self.retcode, self.order = 10009, 0
        return R()

    def copy_rates_from_pos(self, *args): return None
    def symbol_info(self, symbol: str):
        class I:
            def __init__(self): self.point, self.digits = 0.01, 2
        return I()
    def positions_get(self, **kwargs): return []
