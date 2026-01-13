
from xauusd_strategy.execution.mt5_adapter import mt5
from xauusd_strategy.utils.logger import get_logger
from datetime import datetime, date

logger = get_logger("SafetyMonitor")

class SafetyMonitor:
    def __init__(self, settings):
        self.max_spread = 200 # points, e.g. 20 cents
        self.eq_floor = 100.0 # Absolute Equity Floor in USD
        self.max_daily_loss = settings.risk.max_daily_drawdown_pct # e.g. 10.0%
        
        self.initial_day_equity = 0.0
        self.current_day = None
        
    def update_day_state(self, equity):
        today = date.today()
        if self.current_day != today:
            logger.info(f"New Day Detected: {today}. Resetting Daily Stats. Start Eq: {equity}")
            self.current_day = today
            self.initial_day_equity = equity

    def check_market_conditions(self, symbol):
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            logger.error("Safety: Tick not available!")
            return False
            
        spread = tick.ask - tick.bid
        point = mt5.symbol_info(symbol).point
        spread_points = spread / point
        
        if spread_points > self.max_spread:
            logger.warning(f"Safety: High Spread ({spread_points} pts). Trading Halted.")
            return False
            
        return True

    def check_risk_limits(self):
        account = mt5.account_info()
        if not account:
            return False

        current_equity = account.equity
        self.update_day_state(current_equity) # Ensure day is managed
        
        # 1. Absolute Floor
        if current_equity < self.eq_floor:
            logger.critical(f"FATAL: Equity ({current_equity}) below Floor ({self.eq_floor}). STOPPING.")
            return False

        # 2. Daily Drawdown
        if self.initial_day_equity > 0:
            daily_pnl = current_equity - self.initial_day_equity
            dd_pct = (abs(daily_pnl) / self.initial_day_equity) * 100
            
            if daily_pnl < 0 and dd_pct > self.max_daily_loss:
                logger.critical(f"FATAL: Daily Drawdown Limit Hit! (-{dd_pct:.2f}%). HALTING.")
                return False

        return True
