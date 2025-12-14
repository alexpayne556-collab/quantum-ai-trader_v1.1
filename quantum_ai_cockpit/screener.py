"""Live stock screener leveraging the validated Quantum AI signals."""
from __future__ import annotations

import logging
from typing import Iterable, List

import pandas as pd

from .config import MIN_VOLUME_RATIO, SIGNAL_THRESHOLD, STOCK_UNIVERSE
from .signals import snapshot_with_score

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _evaluate_ticker(ticker: str, period: str) -> dict | None:
    try:
        return snapshot_with_score(ticker)
    except Exception as exc:  # noqa: BLE001 - we want to log any upstream issue
        logger.warning("%s: unable to compute signals (%s)", ticker, exc)
        return None


def scan_universe(
    tickers: Iterable[str] | None = None,
    period: str = "6mo",
    score_threshold: float = SIGNAL_THRESHOLD,
    volume_ratio_threshold: float = MIN_VOLUME_RATIO,
) -> pd.DataFrame:
    """Scan the provided tickers and return qualified setups."""

    tickers = list(tickers or STOCK_UNIVERSE)
    qualified: List[dict] = []

    for ticker in tickers:
        snapshot = _evaluate_ticker(ticker, period=period)
        if not snapshot:
            continue

        if snapshot["score"] >= score_threshold and snapshot["volume_ratio"] >= volume_ratio_threshold:
            qualified.append(
                {
                    "Ticker": snapshot["ticker"],
                    "Price": round(snapshot["price"], 2),
                    "Score": round(snapshot["score"], 1),
                    "Volume Ratio": round(snapshot["volume_ratio"], 2),
                    "RSI": round(snapshot["rsi"], 1),
                    "MACD": round(snapshot["macd"], 3),
                    "Stop Loss": round(snapshot["stop_loss_price"], 2),
                    "Target": round(snapshot["target_price"], 2),
                }
            )

    df = pd.DataFrame(qualified)
    if not df.empty:
        df = df.sort_values(["Score", "Volume Ratio"], ascending=[False, False]).reset_index(drop=True)
    return df


__all__ = ["scan_universe"]
