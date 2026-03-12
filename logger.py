"""
logger.py — Structured logging for all trading subsystems.

Provides separate loggers for signals, orders, errors, risk events.
Writes to logs/trading.log + console with rotation.
"""

import logging
import logging.handlers
import os
import json
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


def _create_handler(filename: str, max_bytes: int = 5_000_000, backup_count: int = 3):
    path = LOG_DIR / filename
    handler = logging.handlers.RotatingFileHandler(
        str(path), maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8'
    )
    handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(name)-12s | %(levelname)-5s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    return handler


def _create_console():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '[%(name)-10s] %(message)s'
    ))
    return handler


def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Get a named logger that writes to trading.log + console."""
    logger = logging.getLogger(f"qs.{name}")
    if logger.handlers:
        return logger
    logger.setLevel(level)
    logger.addHandler(_create_handler("trading.log"))
    logger.addHandler(_create_console())
    logger.propagate = False
    return logger


# Pre-configured loggers for each subsystem
signal_log = get_logger("signal")
order_log = get_logger("order")
risk_log = get_logger("risk")
error_log = get_logger("error", logging.ERROR)
system_log = get_logger("system")
ml_log = get_logger("ml")
data_log = get_logger("data")


def log_signal(symbol: str, direction: str, grade: str, score: float,
               quality: float = 0, accepted: bool = True, **extra):
    """Structured signal log entry."""
    msg = (f"{'ACCEPT' if accepted else 'REJECT'} {symbol} {direction} "
           f"grade={grade} score={score:.1f} quality={quality:.3f}")
    if extra:
        msg += f" | {json.dumps(extra, default=str)}"
    signal_log.info(msg)


def log_order(action: str, symbol: str, side: str, size: float, price: float, **extra):
    """Structured order log entry."""
    msg = f"{action} {symbol} {side} size=${size:.2f} price={price:.2f}"
    if extra:
        msg += f" | {json.dumps(extra, default=str)}"
    order_log.info(msg)


def log_risk(event: str, **details):
    """Structured risk event log entry."""
    msg = f"{event} | {json.dumps(details, default=str)}"
    risk_log.warning(msg)
