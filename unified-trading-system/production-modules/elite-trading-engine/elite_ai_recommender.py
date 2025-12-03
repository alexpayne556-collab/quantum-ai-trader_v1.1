from typing import Any, Dict, List

from .elite_data_fetcher import EliteDataFetcher
from .elite_signal_generator import EliteSignalGenerator
from .elite_opportunity_detector import EliteOpportunityDetector
from .elite_forecaster import EliteForecaster


class EliteAIRecommender:
    """Master module that combines signals, patterns, and data into a final view."""

    def __init__(self) -> None:
        self.signal_generator = EliteSignalGenerator(min_confidence=0.70, backtest_mode=True)
        self.data_fetcher = EliteDataFetcher()
        self.opportunity_detector = EliteOpportunityDetector()
        self.forecaster = EliteForecaster()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze_ticker(self, ticker: str, days: int = 90) -> Dict[str, Any]:
        """Run a full analysis pipeline for a single ticker.

        Returns a recommendation dict including confluence, entry/exit, and R:R.
        """
        data = self.data_fetcher.fetch_data([ticker], days=days)
        if data.empty:
            return self._empty_recommendation(ticker, reason="No data fetched")

        # STEP 2: Multi-strategy signals (15 strategies)
        signals = self.signal_generator.generate_signals(data)
        ticker_signals = [s for s in signals if s.get("ticker") == ticker]

        # STEP 3: Unified opportunity detection (patterns + breakouts + momentum)
        opp = self.opportunity_detector.analyze_ticker(ticker, data)
        opp_list = opp.get("opportunities", [])
        momentum_score = int(opp.get("momentum_score", 0))

        # STEP 4: 14-day hybrid forecast
        forecast = self.forecaster.forecast_ticker(ticker, data, days=14)
        forecast_direction = str(forecast.get("overall_direction", "FLAT"))
        forecast_14d_return = float(forecast.get("expected_return_14d", 0.0))
        forecast_conf = float(forecast.get("overall_confidence", 0.0))
        # Map forecast to a 0-100 score (magnitude * confidence, scaled)
        raw_fs = abs(forecast_14d_return) * 100.0 * forecast_conf * 2.0
        forecast_score = max(0.0, min(100.0, raw_fs))

        # STEP 5: Confluence score (signals + opportunities + momentum + forecast)
        confluence_score = self._calculate_confluence(
            signals=ticker_signals,
            opportunities=opp_list,
            momentum_score=momentum_score,
            forecast_score=forecast_score,
        )

        # STEP 6: Recommendation based on more permissive confluence rules
        total_detections = len(ticker_signals) + len(opp_list)
        if confluence_score >= 0.45 and (
            len(ticker_signals) >= 1
            or len(opp_list) >= 1
            or momentum_score >= 60
        ):
            recommendation = "STRONG_BUY"
            confidence = confluence_score
        elif confluence_score >= 0.30 and len(ticker_signals) >= 1:
            recommendation = "BUY"
            confidence = confluence_score
        else:
            recommendation = "PASS"
            confidence = 0.0

        # STEP 7: Entry / target / stop / R:R using best signal or opportunity
        entry, target, stop_loss, rr = self._calculate_trade_levels(
            ticker_signals, opp_list
        )

        return {
            "ticker": ticker,
            "recommendation": recommendation,
            "confidence": float(confidence),
            "confluence_score": float(confluence_score),
            "signals_detected": len(ticker_signals),
            "opportunities_detected": len(opp_list),
            "momentum_score": momentum_score,
            "forecast_direction": forecast_direction,
            "forecast_14d_return": float(forecast_14d_return),
            "forecast_overall_confidence": float(forecast_conf),
            "forecast_score": float(forecast_score),
            "entry": float(entry),
            "target": float(target),
            "stop_loss": float(stop_loss),
            "risk_reward_ratio": float(rr),
            "supporting_evidence": self._build_evidence(ticker_signals, opp_list),
            "forecast": forecast,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _calculate_confluence(
        self,
        signals: List[Dict[str, Any]],
        opportunities: List[Dict[str, Any]],
        momentum_score: int,
        forecast_score: float,
    ) -> float:
        """Calculate confluence score (0-1) from signals, unified opportunities, and momentum.

        Max points: 120
        - Each signal: 5 points (max 25)
        - Each opportunity: 10 points (max 50)
        - Momentum score: momentum_score / 4, capped at 25
        - Forecast score: forecast_score * 0.2, capped at 20
        """

        points = 0.0
        points += min(25.0, len(signals) * 5.0)
        points += min(50.0, len(opportunities) * 10.0)
        points += min(25.0, float(momentum_score) / 4.0)
        points += min(20.0, float(forecast_score) * 0.2)

        confluence_score = points / 120.0
        return float(max(0.0, min(1.0, confluence_score)))

    def _calculate_trade_levels(
        self,
        signals: List[Dict[str, Any]],
        opportunities: List[Dict[str, Any]],
    ) -> tuple[float, float, float, float]:
        """Derive entry, target, stop, and risk-reward from signals and opportunities."""
        if not signals and not opportunities:
            return 0.0, 0.0, 0.0, 0.0

        entry = 0.0
        target = 0.0
        stop_loss = 0.0

        if signals:
            best_signal = max(signals, key=lambda s: s.get("confidence", 0.0))
            entry = float(best_signal.get("entry_price", 0.0))
        elif opportunities:
            best_opp = max(opportunities, key=lambda o: o.get("confidence", 0.0))
            entry = float(best_opp.get("entry", 0.0))

        if entry <= 0:
            return 0.0, 0.0, 0.0, 0.0

        # Targets: from signals and opportunities
        targets: List[float] = []
        targets.extend(float(s.get("target_price", 0.0)) for s in signals if s.get("target_price"))
        targets.extend(float(o.get("target", 0.0)) for o in opportunities if o.get("target"))
        target = max(targets) if targets else entry * 1.08

        # Stops: choose tightest valid stop
        stops: List[float] = []
        stops.extend(float(s.get("stop_loss", 0.0)) for s in signals if s.get("stop_loss"))
        stops.extend(float(o.get("stop", 0.0)) for o in opportunities if o.get("stop"))
        stop_candidates = [s for s in stops if 0 < s < entry]
        stop_loss = min(stop_candidates) if stop_candidates else entry * 0.98

        risk = entry - stop_loss
        reward = target - entry
        rr = reward / risk if risk > 0 else 0.0

        return entry, target, stop_loss, rr

    def _build_evidence(self, signals: List[Dict[str, Any]], patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize evidence from signals and patterns for explainability."""
        signal_summaries = []
        for s in signals:
            signal_summaries.append(
                {
                    "strategy": s.get("strategy"),
                    "confidence": float(s.get("confidence", 0.0)),
                    "expected_return_pct": float(s.get("expected_return_pct", 0.0)),
                    "entry_price": float(s.get("entry_price", 0.0)),
                    "target_price": float(s.get("target_price", 0.0)),
                }
            )

        pattern_summaries = []
        for p in patterns:
            pattern_summaries.append(
                {
                    "pattern_name": p.get("pattern_name"),
                    "confidence": float(p.get("confidence", 0.0)),
                    "expected_move": float(p.get("expected_move", 0.0)),
                    "entry_price": float(p.get("entry_price", 0.0)),
                    "target_price": float(p.get("target_price", 0.0)),
                }
            )

        return {
            "signals": signal_summaries,
            "patterns": pattern_summaries,
        }

    def _empty_recommendation(self, ticker: str, reason: str) -> Dict[str, Any]:
        return {
            "ticker": ticker,
            "recommendation": "PASS",
            "confidence": 0.0,
            "confluence_score": 0.0,
            "signals_detected": 0,
            "patterns_detected": 0,
            "entry": 0.0,
            "target": 0.0,
            "stop_loss": 0.0,
            "risk_reward_ratio": 0.0,
            "supporting_evidence": {"signals": [], "patterns": []},
            "reason": reason,
        }


if __name__ == "__main__":
    import json

    from datetime import datetime

    print("=" * 80)
    print("ELITE AI RECOMMENDER - SINGLE TICKER ANALYSIS")
    print("=" * 80)

    ticker = "SPY"
    days = 120

    engine = EliteAIRecommender()
    result = engine.analyze_ticker(ticker, days=days)

    print(f"\nAnalysis for {ticker} over last {days} trading days (as of {datetime.now():%Y-%m-%d}):")
    print(json.dumps(result, indent=2, default=str))
