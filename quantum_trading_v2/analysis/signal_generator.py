"""Signal generator that uses pattern scores + simple heuristics."""
from __future__ import annotations

from typing import Dict, Optional
import pandas as pd
from .pattern_detector import PatternDetector


class SignalGenerator:
    def __init__(self, threshold: float = 80.0):
        self.detector = PatternDetector()
        self.threshold = threshold

    def generate(self, df: pd.DataFrame) -> Optional[Dict[str, float]]:
        scores = self.detector.detect(df)
        if not scores:
            return None
        avg = sum(scores.values()) / len(scores)
        if avg >= self.threshold:
            return {"score": avg, **scores}
        return None


__all__ = ["SignalGenerator"]
