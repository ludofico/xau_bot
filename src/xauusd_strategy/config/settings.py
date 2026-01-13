"""
Centralized configuration management for XAUUSD trading strategy.

Supports loading from YAML files and environment variables.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os
import yaml


@dataclass
class SessionConfig:
    """Configuration for a trading session."""
    enabled: bool = True
    start: str = "00:00"
    end: str = "07:00"
    strategy: str = "breakout"
    max_trades: int = 3


@dataclass
class AccountConfig:
    """Account configuration."""
    initial_balance: float = 250.0
    currency: str = "EUR"
    leverage: int = 500
    broker_type: str = "ECN"


@dataclass
class RiskConfig:
    """Risk management configuration."""
    # Position Sizing
    risk_per_trade_pct: float = 2.5
    max_risk_per_trade_pct: float = 4.0
    kelly_fraction: float = 0.5
    
    # Daily Limits
    max_daily_drawdown_pct: float = 8.0
    max_daily_profit_pct: float = 15.0
    max_open_positions: int = 2
    max_trades_per_day: int = 8
    
    # Compounding
    compound_frequency: str = "daily"
    compound_after_profit_pct: float = 5.0


@dataclass
class EntryConfig:
    """Entry signal configuration."""
    atr_period: int = 14
    atr_min_multiplier: float = 0.5
    roc_period: int = 5
    roc_threshold: float = 0.15


@dataclass
class ExitConfig:
    """Exit management configuration."""
    sl_atr_multiplier: float = 1.2
    tp_atr_multiplier: float = 2.4
    trailing_atr_multiplier: float = 0.8
    breakeven_after_rr: float = 1.0


@dataclass
class CostConfig:
    """Trading costs configuration."""
    spread_usd: float = 0.25
    slippage_usd: float = 0.15
    commission_per_lot: float = 7.0


@dataclass
class MLConfig:
    """Machine learning filter configuration."""
    enabled: bool = True
    probability_threshold: float = 0.55
    retrain_frequency: str = "weekly"
    model_type: str = "xgboost"

@dataclass
class MetaApiConfig:
    """MetaApi Cloud Configuration."""
    enabled: bool = False
    token: str = ""
    account_id: str = ""
    domain: str = "agiliumtrade.agiliumtrade.ai"  # Default


@dataclass
class cTraderConfig:
    """cTrader Open API Configuration."""
    enabled: bool = False
    client_id: str = ""
    client_secret: str = ""
    account_id: str = ""
    access_token: str = ""
    refresh_token: str = ""



@dataclass
class Settings:
    """
    Main settings container for the trading strategy.
    
    Can be loaded from YAML config file or constructed programmatically.
    
    Example:
        >>> settings = Settings.from_yaml("config/aggressive.yaml")
        >>> print(settings.risk.risk_per_trade_pct)
        2.5
    """
    
    account: AccountConfig = field(default_factory=AccountConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    entry: EntryConfig = field(default_factory=EntryConfig)
    exit: ExitConfig = field(default_factory=ExitConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    metaapi: MetaApiConfig = field(default_factory=MetaApiConfig)
    ctrader: cTraderConfig = field(default_factory=cTraderConfig)
    
    # Sessions
    asian_session: SessionConfig = field(default_factory=lambda: SessionConfig(
        enabled=True, start="00:00", end="07:00", strategy="range_scalping", max_trades=2
    ))
    london_session: SessionConfig = field(default_factory=lambda: SessionConfig(
        enabled=True, start="08:00", end="12:00", strategy="breakout", max_trades=3
    ))
    ny_session: SessionConfig = field(default_factory=lambda: SessionConfig(
        enabled=True, start="14:00", end="18:00", strategy="momentum", max_trades=3
    ))
    
    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "Settings":
        """Load settings from a YAML configuration file."""
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path) as f:
            config = yaml.safe_load(f)
        
        return cls._from_dict(config)
    
    @classmethod
    def _from_dict(cls, config: dict) -> "Settings":
        """Create Settings from a dictionary."""
        settings = cls()
        
        if "account" in config:
            settings.account = AccountConfig(**config["account"])
        
        if "risk_management" in config:
            settings.risk = RiskConfig(**config["risk_management"])
        
        if "strategy" in config:
            strat = config["strategy"]
            if "entry" in strat:
                settings.entry = EntryConfig(**strat["entry"])
            if "exit" in strat:
                settings.exit = ExitConfig(**strat["exit"])
            
            # Sessions
            if "sessions" in strat:
                sessions = strat["sessions"]
                if "asian" in sessions:
                    settings.asian_session = SessionConfig(**sessions["asian"])
                if "london" in sessions:
                    settings.london_session = SessionConfig(**sessions["london"])
                if "ny" in sessions:
                    settings.ny_session = SessionConfig(**sessions["ny"])
        
        if "costs" in config:
            settings.costs = CostConfig(**config["costs"])
            
        if "execution" in config:
            # Simple dict access for now, or add ExecutionConfig class
            if "max_spread_usd" in config["execution"]:
                settings.execution_max_spread = config["execution"]["max_spread_usd"]
        else:
            settings.execution_max_spread = 0.35 # Default
        
        if "ml_filter" in config:
            settings.ml = MLConfig(**config["ml_filter"])
            
        if "metaapi" in config:
            settings.metaapi = MetaApiConfig(**config["metaapi"])
        
        if "ctrader" in config:
            settings.ctrader = cTraderConfig(**config["ctrader"])
        
        # Bridge and AI flags
        settings.use_socket_bridge = config.get("use_socket_bridge", False)
        settings.advanced_ai = config.get("advanced_ai", False)
        
        return settings
    
    @classmethod
    def aggressive(cls) -> "Settings":
        """Create aggressive settings preset for €500-700/month target."""
        return cls(
            account=AccountConfig(initial_balance=250, leverage=500),
            risk=RiskConfig(
                risk_per_trade_pct=2.5,
                max_risk_per_trade_pct=4.0,
                kelly_fraction=0.5,
                max_daily_drawdown_pct=8.0,
                max_trades_per_day=8
            ),
            entry=EntryConfig(atr_period=14, roc_threshold=0.15),
            exit=ExitConfig(sl_atr_multiplier=1.2, tp_atr_multiplier=2.4),
            ml=MLConfig(enabled=True, probability_threshold=0.55)
        )
    
    @classmethod
    def ultra_aggressive(cls) -> "Settings":
        """Create ultra-aggressive settings preset for €800-1000/month target."""
        return cls(
            account=AccountConfig(initial_balance=250, leverage=500),
            risk=RiskConfig(
                risk_per_trade_pct=3.5,
                max_risk_per_trade_pct=5.0,
                kelly_fraction=0.6,
                max_daily_drawdown_pct=10.0,
                max_trades_per_day=10
            ),
            entry=EntryConfig(atr_period=14, roc_threshold=0.12),
            exit=ExitConfig(sl_atr_multiplier=1.0, tp_atr_multiplier=2.0),
            ml=MLConfig(enabled=True, probability_threshold=0.50)
        )
    
    @classmethod
    def conservative(cls) -> "Settings":
        """Create conservative settings preset for €300-400/month target."""
        return cls(
            account=AccountConfig(initial_balance=250, leverage=500),
            risk=RiskConfig(
                risk_per_trade_pct=1.5,
                max_risk_per_trade_pct=2.5,
                kelly_fraction=0.25,
                max_daily_drawdown_pct=5.0,
                max_trades_per_day=5
            ),
            entry=EntryConfig(atr_period=14, roc_threshold=0.20),
            exit=ExitConfig(sl_atr_multiplier=1.5, tp_atr_multiplier=3.0),
            ml=MLConfig(enabled=True, probability_threshold=0.60)
        )
    
    def to_yaml(self, path: str | Path) -> None:
        """Save settings to a YAML file."""
        config = {
            "account": self.account.__dict__,
            "risk_management": self.risk.__dict__,
            "strategy": {
                "entry": self.entry.__dict__,
                "exit": self.exit.__dict__,
                "sessions": {
                    "asian": self.asian_session.__dict__,
                    "london": self.london_session.__dict__,
                    "ny": self.ny_session.__dict__,
                }
            },
            "costs": self.costs.__dict__,
            "execution": {
                 "max_spread_usd": 0.35 # Default safe limit
            },
            "ml_filter": self.ml.__dict__,
        }
        
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    def __repr__(self) -> str:
        return (
            f"Settings(\n"
            f"  account={self.account},\n"
            f"  risk={self.risk},\n"
            f"  entry={self.entry},\n"
            f"  exit={self.exit},\n"
            f"  costs={self.costs},\n"
            f"  ml={self.ml}\n"
            f")"
        )
