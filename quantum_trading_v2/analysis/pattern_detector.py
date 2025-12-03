"""Simple pattern detection utilities used by the example system."""
from __future__ import annotations

from typing import Dict
import pandas as pd


class PatternDetector:
    def __init__(self):
        pass

    def detect(self, df: pd.DataFrame) -> Dict[str, float]:
        """Return a small dict of pattern scores (0-100)."""
        if df is None or df.empty:
            return {}
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else latest
        scores = {
            "gap": abs((latest["Open"] - prev["Close"]) / prev["Close"]) * 100 if prev["Close"] else 0,
            "range": float((latest["High"] - latest["Low"]) / latest["Close"] * 100) if latest["Close"] else 0,
        }
        return scores


__all__ = ["PatternDetector"]
