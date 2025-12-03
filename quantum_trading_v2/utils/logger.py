"""Central logging helper for quantum_trading_v2."""
from __future__ import annotations

import logging
from logging import Logger


def get_logger(name: str = "quantum_trading_v2", level: int = logging.INFO) -> Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(ch)
    logger.setLevel(level)
    return logger

__all__ = ["get_logger"]
