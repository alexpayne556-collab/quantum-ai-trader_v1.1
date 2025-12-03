from typing import Any, Dict, List

import pandas as pd

from .elite_breakout_detector import EliteBreakoutDetector
from .elite_momentum_scanner import EliteMomentumScanner
from .elite_pattern_engine import ElitePatternEngine


class EliteOpportunityDetector:
    """Unified opportunity detection engine.

    Combines:
    - Chart patterns (Cup & Handle, Bull Flag, Ascending Triangle, Falling Wedge, Double Bottom)
    - Breakout setups (volume, consolidation, 52w high, gap continuation, resistance break)
    - Momentum scanning (price acceleration, volume surge, relative strength)
    """

    def __init__(self, lookback: int = 120) -> None:
        self.lookback = int(lookback)
        self._pattern_engine = ElitePatternEngine(lookback=lookback)
        self._breakout_detector = EliteBreakoutDetector(lookback_days=lookback)
        self._momentum_scanner = EliteMomentumScanner()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze_ticker(self, ticker: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Run full opportunity analysis for a single ticker.

        Returns a dict with opportunities list and momentum score.
        """
        ticker_data = data[data["ticker"] == ticker].sort_values("date").copy()
        if ticker_data.empty:
            return self._empty_analysis(ticker)

        opportunities: List[Dict[str, Any]] = []

        # 1) Breakout setups (all 5 types) using existing detector with
        # lowered thresholds.
        breakout_raw = self._breakout_detector.detect_all_breakouts(data)
        for b in breakout_raw:
            if b.get("ticker") != ticker:
                continue
            entry = float(b.get("entry_price", 0.0))
            target = float(b.get("target_price", entry))
            stop = float(b.get("stop_loss", entry * 0.98))
            expected_move = (target - entry) / entry if entry > 0 else 0.0
            opportunities.append(
                {
                    "type": str(b.get("breakout_type", "BREAKOUT")),
                    "confidence": float(b.get("confidence", 0.0)),
                    "expected_move": float(expected_move),
                    "entry": entry,
                    "target": target,
                    "stop": stop,
                    "catalyst": str(b.get("catalyst", "")),
                    "supporting_factors": list(b.get("supporting_factors", [])),
                }
            )

        # 2) Chart patterns (all 5 types) mapped into opportunity schema.
        pattern_raw = self._pattern_engine.detect_all_patterns(ticker_data)
        for p in pattern_raw:
            entry = float(p.get("entry_price", 0.0))
            target = float(p.get("target_price", entry))
            stop = float(p.get("stop_loss", entry * 0.97))
            expected_move = float(p.get("expected_move", 0.0))
            opportunities.append(
                {
                    "type": str(p.get("pattern_name", "PATTERN")).upper().replace(" ", "_"),
                    "confidence": float(p.get("confidence", 0.0)),
                    "expected_move": float(expected_move),
                    "entry": entry,
                    "target": target,
                    "stop": stop,
                    "catalyst": "Pattern completion and breakout",
                    "supporting_factors": list(p.get("supporting_factors", [])),
                }
            )

        # 3) Momentum score (single summary per ticker).
        momentum_raw = self._momentum_scanner.scan_for_momentum(data)
        momentum_score = 0
        for m in momentum_raw:
            if m.get("ticker") == ticker:
                momentum_score = int(m.get("momentum_score", 0))
                break

        # Sort opportunities by confidence descending
        opportunities.sort(key=lambda o: o.get("confidence", 0.0), reverse=True)

        return {
            "ticker": ticker,
            "opportunities": opportunities,
            "momentum_score": momentum_score,
            "total_opportunities": len(opportunities),
            "best_opportunity": opportunities[0] if opportunities else None,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _empty_analysis(self, ticker: str) -> Dict[str, Any]:
        return {
            "ticker": ticker,
            "opportunities": [],
            "momentum_score": 0,
            "total_opportunities": 0,
            "best_opportunity": None,
        }


if __name__ == "__main__":
    from backend.modules.elite.elite_data_fetcher import EliteDataFetcher

    fetcher = EliteDataFetcher()
    tickers = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA"]
    data = fetcher.fetch_data(tickers, days=90)

    detector = EliteOpportunityDetector()
    for t in tickers:
        analysis = detector.analyze_ticker(t, data)
        print("=" * 80)
        print(f"{t} - Opportunities: {analysis['total_opportunities']}, Momentum: {analysis['momentum_score']}")
        if analysis["best_opportunity"]:
            best = analysis["best_opportunity"]
            print(
                f"Best: {best['type']} (conf {best['confidence']:.0%}, "
                f"exp {best['expected_move']*100:.1f}%)"
            )
