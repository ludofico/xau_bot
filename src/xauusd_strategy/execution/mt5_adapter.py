
try:
    import MetaTrader5 as mt5
except ImportError:
    from xauusd_strategy.execution.mock_mt5 import MockMT5 as mt5
    print("WARNING: MetaTrader5 not found. Using MOCK for Simulation.")
from xauusd_strategy.utils.logger import get_logger
import time

logger = get_logger("MT5Adapter")

class MT5Adapter:
    """
    Robust wrapper for MetaTrader5 functions for production safety.
    """
    
    @staticmethod
    def initialize():
        if not mt5.initialize():
            logger.error(f"MT5 Init Failed: {mt5.last_error()}")
            return False
        return True

    @staticmethod
    def get_symbol_info(symbol):
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.error(f"Symbol {symbol} not found")
            return None
        if not info.visible:
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Symbol {symbol} selection failed")
                return None
        return info

    @staticmethod
    def order_send(request, retries=3):
        for i in range(retries):
            result = mt5.order_send(request)
            if result is None:
                logger.error("Order send returned None")
                time.sleep(0.5)
                continue
                
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                return result
            elif result.retcode == mt5.TRADE_RETCODE_REQUOTE:
                logger.warning(f"Requote (attempt {i+1}): {result.comment}")
                time.sleep(0.2)
                continue
            else:
                logger.error(f"Order failed: {result.comment} ({result.retcode})")
                return result
        return None
