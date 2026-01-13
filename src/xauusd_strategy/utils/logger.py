"""
Logging configuration using Loguru.

Provides structured logging with file rotation and optional Telegram notifications.
"""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger

# Remove default logger
logger.remove()

# Global logger instance
_logger_configured = False


def setup_logger(
    log_dir: str | Path = "logs",
    log_level: str = "INFO",
    console: bool = True,
    file: bool = True,
    rotation: str = "10 MB",
    retention: str = "1 week",
) -> None:
    """
    Set up the logging configuration.
    
    Args:
        log_dir: Directory for log files
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console: Whether to log to console
        file: Whether to log to file
        rotation: Log file rotation size
        retention: How long to keep old logs
    """
    global _logger_configured
    
    if _logger_configured:
        return
    
    # Console handler with color
    if console:
        logger.add(
            sys.stderr,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
            level=log_level,
            colorize=True,
        )
    
    # File handler with rotation
    if file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # General log
        logger.add(
            log_path / "xauusd_strategy.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="zip",
        )
        
        # Trade log (only trades)
        logger.add(
            log_path / "trades.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
            level="INFO",
            filter=lambda record: "trade" in record["extra"],
            rotation="1 day",
            retention="1 month",
        )
        
        # Error log
        logger.add(
            log_path / "errors.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}\n{exception}",
            level="ERROR",
            rotation=rotation,
            retention="1 month",
            backtrace=True,
            diagnose=True,
        )
    
    _logger_configured = True


def get_logger(name: Optional[str] = None):
    """
    Get a logger instance.
    
    Args:
        name: Optional name for context
    
    Returns:
        Configured logger instance
    """
    if not _logger_configured:
        setup_logger()
    
    if name:
        return logger.bind(context=name)
    return logger


def log_trade(
    action: str,
    symbol: str = "XAUUSD",
    direction: str = "",
    entry_price: float = 0,
    stop_loss: float = 0,
    take_profit: float = 0,
    lots: float = 0,
    pnl: float = 0,
    **kwargs
) -> None:
    """
    Log a trade event.
    
    Args:
        action: Trade action (OPEN, CLOSE, MODIFY, etc.)
        symbol: Trading symbol
        direction: LONG or SHORT
        entry_price: Entry price
        stop_loss: Stop loss price
        take_profit: Take profit price
        lots: Position size in lots
        pnl: Profit/loss amount
        **kwargs: Additional fields
    """
    extra = {"trade": True}
    
    message_parts = [
        f"[{action}]",
        f"{symbol}",
    ]
    
    if direction:
        message_parts.append(direction)
    if entry_price:
        message_parts.append(f"Entry={entry_price:.2f}")
    if stop_loss:
        message_parts.append(f"SL={stop_loss:.2f}")
    if take_profit:
        message_parts.append(f"TP={take_profit:.2f}")
    if lots:
        message_parts.append(f"Lots={lots:.2f}")
    if pnl:
        message_parts.append(f"PnL={pnl:+.2f}")
    
    for key, value in kwargs.items():
        message_parts.append(f"{key}={value}")
    
    logger.bind(**extra).info(" | ".join(message_parts))


class PerformanceLogger:
    """Context manager for logging performance metrics."""
    
    def __init__(self, operation: str, log_level: str = "DEBUG"):
        self.operation = operation
        self.log_level = log_level
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        elapsed = time.perf_counter() - self.start_time
        
        log_func = getattr(logger, self.log_level.lower())
        log_func(f"{self.operation} completed in {elapsed:.3f}s")
        
        return False
