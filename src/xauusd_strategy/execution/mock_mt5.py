
import sys
from unittest.mock import MagicMock

class MockMT5:
    """Mock MetaTrader5 for non-Windows environments"""
    TIMEFRAME_M5 = 5
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    TRADE_ACTION_DEAL = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_IOC = 1
    TRADE_ACTION_SLTP = 6
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_REQUOTE = 10004
    
    @staticmethod
    def initialize():
        print("[MOCK] MT5 Initialized")
        return True
        
    @staticmethod
    def last_error():
        return (1, "Success")
        
    @staticmethod
    def symbol_info(symbol):
        # Return a mock object with 'point' and 'visible' attributes
        m = MagicMock()
        m.point = 0.01
        m.visible = True
        return m
        
    @staticmethod
    def symbol_info_tick(symbol):
        # Return a mock tick
        m = MagicMock()
        m.bid = 2000.0
        m.ask = 2000.20
        return m
        
    @staticmethod
    def symbol_select(symbol, visible):
        return True
        
    @staticmethod
    def order_send(request):
        print(f"[MOCK] Order Send: {request}")
        res = MagicMock()
        res.retcode = 10009
        res.comment = "Mock Success"
        return res
    
    @staticmethod
    def copy_rates_from_pos(symbol, timeframe, start, count):
        # Return dummy rates
        # time, open, high, low, close, tick_volume, spread, real_volume
        import numpy as np
        dtype = [('time', 'i8'), ('open', 'f8'), ('high', 'f8'), ('low', 'f8'), ('close', 'f8'), ('tick_volume', 'i8'), ('spread', 'i4'), ('real_volume', 'i8')]
        rates = np.zeros(count, dtype=dtype)
        rates['close'] = 2000.0
        import time
        rates['time'] = int(time.time()) # Dynamic timestamp
        
        # Add some random movement to close price for ML to work on
        rates['close'] = 2000.0 + (np.random.random() * 10)
        rates['high'] = rates['close'] + 2.0
        rates['low'] = rates['close'] - 2.0
        rates['open'] = rates['close'] - 0.5

        return rates

    @staticmethod
    def account_info():
        m = MagicMock()
        m.equity = 1000.0
        m.balance = 1000.0
        m.profit = 0.0
        return m

    @staticmethod
    def positions_get(symbol=None, magic=None):
        return []

    # Add other constants as needed by your code
    
