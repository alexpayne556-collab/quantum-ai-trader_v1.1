"""Utility helpers for Quantum AI Cockpit."""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Period to days mapping for data fetcher
_PERIOD_TO_DAYS = {
    "1mo": 30,
    "3mo": 90,
    "6mo": 180,
    "1y": 365,
    "2y": 730,
    "5y": 1825,
    "max": 3650,
}


def ensure_directory(path: Path) -> Path:
    """Create the directory if it does not already exist and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


@lru_cache(maxsize=128)
def fetch_price_history(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """
    Download adjusted OHLCV data with API fallback support.
    
    Uses DataFetcher (Polygon → FMP → EODHD → AlphaVantage → yfinance)
    for reliable data access across all configured API sources.
    """
    logger.debug("Fetching %s data (period=%s, interval=%s)", ticker, period, interval)
    
    # Try unified data fetcher first (uses all API keys with fallback)
    try:
        from .data_fetcher import get_fetcher
        days = _PERIOD_TO_DAYS.get(period, 180)
        fetcher = get_fetcher(verbose=False)
        data = fetcher.get_stock_data(ticker, days=days)
        if data is not None and not data.empty:
            logger.debug("Got %s data from %s", ticker, data.attrs.get("source", "unknown"))
            return data.dropna()
    except Exception as e:
        logger.debug("DataFetcher failed for %s: %s, falling back to yfinance", ticker, e)
    
    # Fallback to direct yfinance
    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError(f"No price history returned for {ticker}")
    return data.dropna()


def dataframe_from_records(records: Iterable[dict]) -> pd.DataFrame:
    """Convert iterable of dictionaries into a pandas DataFrame."""

    return pd.DataFrame(list(records))
