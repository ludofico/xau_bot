import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime
import sys

# Mock dependencies before imports
sys.modules['nest_asyncio'] = MagicMock()
sys.modules['MetaTrader5'] = MagicMock()
sys.modules['xauusd_strategy.execution.metaapi_adapter'] = MagicMock()
sys.modules['xauusd_strategy.execution.socket_adapter'] = MagicMock()

from xauusd_strategy.execution.live_trader import LiveTrader, TradeSignal, SignalType
from xauusd_strategy.strategy.london_breakout import LondonBreakoutStrategy
from xauusd_strategy.strategy.asian_scalp import AsianScalpingStrategy
from xauusd_strategy.strategy.signal_aggregator import AggregatedSignal

class MockAdapter:
    def __init__(self):
        self.TIMEFRAME_M1 = 1
        self.TIMEFRAME_M5 = 5
        self.TIMEFRAME_H1 = 16385
        self.TIMEFRAME_D1 = 16408
        self.ORDER_TYPE_BUY = 0
        self.ORDER_TYPE_SELL = 1
        self.TRADE_ACTION_DEAL = 1
        self.TRADE_ACTION_SLTP = 6
        self.ORDER_TIME_GTC = 0
        self.ORDER_FILLING_IOC = 1
        
        self.account_info_mock = MagicMock()
        self.account_info_mock.equity = 10000.0
        self.account_info_mock.balance = 10000.0
        self.account_info_mock.profit = 0.0
        
        self.positions_mock = []
        self.tick_mock = MagicMock()
        self.tick_mock.ask = 2000.50
        self.tick_mock.bid = 2000.40 # Spread 0.10
        
        self.orders = []

    def initialize(self): return True
    def account_info(self): return self.account_info_mock
    def positions_get(self, **kwargs): return self.positions_mock
    def symbol_info_tick(self, symbol): return self.tick_mock
    def symbol_info(self, symbol): 
        info = MagicMock()
        info.point = 0.01
        info.trade_tick_value = 1.0
        return info
    
    def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
        # Return structured array like MT5
        # time, open, high, low, close, tick_volume, spread, real_volume
        import numpy as np
        dtype = [('time', '<i8'), ('open', '<f8'), ('high', '<f8'), ('low', '<f8'), ('close', '<f8'), ('tick_volume', '<u8'), ('spread', '<i4'), ('real_volume', '<u8')]
        
        data = np.zeros(count, dtype=dtype)
        base_time = int(datetime.now().timestamp()) - (count * 60 * timeframe if timeframe < 10000 else count * 3600)
        
        for i in range(count):
            data[i]['time'] = base_time + (i * 60 * (timeframe if timeframe < 10000 else 60))
            data[i]['open'] = 2000.0 + (i * 0.1)
            data[i]['high'] = 2005.0 + (i * 0.1)
            data[i]['low'] = 1995.0 + (i * 0.1)
            data[i]['close'] = 2002.0 + (i * 0.1)
            data[i]['tick_volume'] = 100
            
        return data

    def order_send(self, req):
        self.orders.append(req)
        res = MagicMock()
        res.retcode = 10009 # DONE
        res.order = 12345
        res.deal = 67890
        return res
        
    def history_deals_get(self, *args, **kwargs):
        return []

@pytest.fixture
def mock_trader():
    with patch('xauusd_strategy.execution.live_trader.Settings') as MockSettings:
        # Configure settings
        settings = MagicMock()
        settings.ctrader.enabled = False
        settings.metaapi.enabled = False
        settings.use_socket_bridge = False
        settings.risk.initial_balance = 1000.0
        settings.risk.risk_per_trade_pct = 2.0
        settings.risk.max_daily_loss_pct = 5.0
        settings.risk.max_daily_trades = 10
        
        # Strategy Settings (Structure matches LondonBreakoutStrategy usage)
        settings.asian_session.start = "00:00"
        settings.asian_session.end = "08:00"
        settings.london_session.start = "08:00"
        settings.london_session.end = "16:00"
        
        settings.entry.atr_period = 14
        settings.entry.atr_min_multiplier = 0.5
        settings.entry.roc_period = 5
        settings.entry.roc_threshold = 0.1
        
        settings.exit.sl_atr_multiplier = 1.5
        settings.exit.tp_atr_multiplier = 2.0
        settings.exit.trailing_atr_multiplier = 1.5
        settings.exit.breakeven_after_rr = 1.0
        
        settings.ml.probability_threshold = 0.6
        
        settings.strategy.asian_start = "00:00" # Fallback/Legacy
        settings.strategy.asian_end = "08:00"
        settings.strategy.london_start = "08:00"
        settings.strategy.london_end = "16:00"
        settings.strategy.ny_start = "13:00"
        settings.strategy.ny_end = "22:00"
        
        settings.execution_max_spread = 0.50
        
        MockSettings.from_yaml.return_value = settings
        
        # Patch MT5 adapter checks in init to force fallback to our mock
        # Also patch _load_ml_model to avoid segfaults/IO
        with patch('xauusd_strategy.execution.live_trader.MT5Adapter') as MockMT5Wrapper, \
             patch.object(LiveTrader, '_load_ml_model', return_value=None):
            trader = LiveTrader(config_path="dummy.yaml")
            # Inject our MockAdapter
            trader.adapter = MockAdapter()
            # Explicitly set adapter map constants to match mock
            trader.adapter.TIMEFRAME_M5 = 5 
            
            return trader

def test_live_trader_initialization(mock_trader):
    assert mock_trader.risk_manager is not None
    assert mock_trader.signal_aggregator is not None
    assert mock_trader.news_calendar is not None
    assert mock_trader.regime_detector is not None

def test_fetch_multi_tf_data(mock_trader):
    data = mock_trader._fetch_multi_tf_data(n_bars=50)
    assert '1m' in data
    assert '5m' in data
    assert '1h' in data
    assert '1d' in data
    assert len(data['5m']) == 50
    assert isinstance(data['5m'], pd.DataFrame)

def test_execute_trade_integration(mock_trader):
    mock_trader.execute_trade(1, sl_price=1990.0, tp_price=2020.0, comment="Test", volume=0.1)
    assert len(mock_trader.adapter.orders) == 1
    order = mock_trader.adapter.orders[0]
    assert order['volume'] == 0.1
    assert order['type'] == 0 # BUY

class LoopBreaker(Exception):
    pass

def test_run_loop_flow(mock_trader):
    """Test one iteration of the run loop."""
    
    # Mock Risk Logic to allow trade
    mock_trader.risk_manager.get_status = MagicMock(return_value={'can_trade': True})
    
    # Mock Aggregator to return a signal
    agg_sig = AggregatedSignal(
        direction="BUY",
        confidence_score=90.0,
        position_size=0.12,
        stop_loss=1995.0,
        take_profit=2010.0,
        rationale=["Integration Test"],
        market_regime="TRENDING",
        multi_tf_alignment="BULLISH",
        expected_value=0.5,
        risk_reward=2.0,
        strategy_source="MOCK",
        timestamp="NOW",
        signal_id="123",
        news_status="SAFE",
        entry_price=2000.0,
        risk_info={}
    )
    mock_trader.signal_aggregator.process_signal = MagicMock(return_value=agg_sig)
    
    # Mock Strategies to emit a raw signal
    # We need London or Asian to emit signal.
    # Let's mock London generate_signal
    mock_trader.strategy_london.generate_signal = MagicMock(return_value=TradeSignal(
        SignalType.LONG, 2000.0, 1995.0, 2010.0, 5.0, 0.5, 2005.0, 1995.0, 0.8, datetime.now(), source="London"
    ))
    
    # Mock prepare_data to return dummy DF
    mock_trader.strategy_london.prepare_data = MagicMock(return_value=pd.DataFrame({'close': [2000]}, index=[datetime.now()]))
    
    # Force new bar detection
    mock_trader.last_processed_time = None
    
    # Mock sleep to break loop
    with patch('time.sleep', side_effect=LoopBreaker):
        try:
            mock_trader.run()
        except LoopBreaker:
            pass
            
    # Verification
    # 1. Did we fetch data?
    # _fetch_multi_tf_data calls get_market_data which calls adapter.copy_rates_from_pos
    # It should happen provided circuit breaker didn't trip.
    
    # 2. Did we generate signal?
    assert mock_trader.strategy_london.generate_signal.called
    
    # 3. Did we aggregate?
    assert mock_trader.signal_aggregator.process_signal.called
    
    # 4. Did we Execute?
    assert len(mock_trader.adapter.orders) >= 1
    last_order = mock_trader.adapter.orders[-1]
    assert last_order['volume'] == 0.12
