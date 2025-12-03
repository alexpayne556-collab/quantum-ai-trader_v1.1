import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from .elite_data_fetcher import EliteDataFetcher
from .elite_signal_generator import EliteSignalGenerator
from .elite_risk_manager import EliteRiskManager


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@dataclass
class FusionConfig:
    """Configuration for the EliteFusionEngine."""

    min_confidence: float = 0.70
    account_value: float = 3000.0
    risk_per_trade: float = 0.02
    days: int = 250
    backtest_mode: bool = True
    pump_volume_multiple: float = 4.0
    pump_price_z: float = 3.0


class EliteFusionEngine:
    """Fusion engine that combines mean reversion, trend context, and pump-risk filters.

    This orchestrates the existing Elite modules:
    - EliteDataFetcher: fetches OHLCV + indicators
    - EliteSignalGenerator: generates mean reversion signals
    - EliteRiskManager: sizes positions

    And adds:
    - EMA ribbon-based trend confirmation
    - Simple pump-and-dump style price/volume anomaly filter
    """

    def __init__(self, config: FusionConfig | None = None) -> None:
        """Initialize the fusion engine with configuration."""
        self.config = config or FusionConfig()
        self.fetcher = EliteDataFetcher()
        self.signal_generator = EliteSignalGenerator(
            min_confidence=self.config.min_confidence,
            backtest_mode=self.config.backtest_mode,
        )
        self.risk_manager = EliteRiskManager(
            account_value=self.config.account_value,
            risk_per_trade=self.config.risk_per_trade,
        )

    def run_pipeline(self, tickers: List[str]) -> List[Dict[str, Any]]:
        """Run the full fusion pipeline for a list of tickers.

        Steps:
        1. Fetch data with EliteDataFetcher
        2. Compute EMA ribbon context
        3. Generate mean reversion signals
        4. Apply trend and pump-risk filters
        5. Size positions with EliteRiskManager
        6. Rank and return final candidates
        """
        data = self.fetcher.fetch_data(tickers, days=self.config.days)
        if data.empty:
            logger.warning("Fusion pipeline: no data fetched for tickers %s", tickers)
            return []

        data = self._add_ema_ribbon(data)
        data = self._add_pump_risk_features(data)

        signals = self.signal_generator.generate_signals(data)
        if not signals:
            logger.info("Fusion pipeline: no base signals generated.")
            return []

        enriched_signals: List[Dict[str, Any]] = []
        for sig in signals:
            ticker = sig["ticker"]
            signal_date = pd.to_datetime(sig["date"]).normalize()

            # Get row for that ticker/date
            row = data[(data["ticker"] == ticker) & (pd.to_datetime(data["date"]).dt.normalize() == signal_date)]
            if row.empty:
                continue
            row_s = row.iloc[-1]

            trend_score = float(row_s.get("ema_trend_score", 0.0))
            pump_flag = bool(row_s.get("pump_flag", False))
            pump_score = float(row_s.get("pump_score", 0.0))

            # Simple fusion logic: require non-negative trend, avoid pump-risk
            if trend_score < 0:
                continue

            if pump_flag:
                # For now, skip extremely anomalous pump/dump candidates entirely
                logger.info(
                    "Fusion pipeline: skipping %s on %s due to pump-risk flag.",
                    ticker,
                    signal_date.date(),
                )
                continue

            # Boost confidence slightly based on trend_score and lack of pump risk
            base_conf = float(sig.get("confidence", 0.0))
            adjusted_conf = base_conf + 0.05 * trend_score + 0.05 * (1.0 - pump_score)
            # Clip to [0, 0.99]
            adjusted_conf = max(0.0, min(0.99, adjusted_conf))

            sig_enriched = dict(sig)
            sig_enriched["confidence"] = adjusted_conf
            sig_enriched["edge_components"] = {
                "trend_score": trend_score,
                "pump_score": pump_score,
            }

            # Size position
            try:
                sized = self.risk_manager.calculate_position_size(sig_enriched)
            except ValueError:
                continue

            enriched_signals.append(sized)

        # Rank by confidence * expected_return_pct
        enriched_signals.sort(
            key=lambda s: s["confidence"] * s.get("expected_return_pct", 0.0),
            reverse=True,
        )

        logger.info("Fusion pipeline: generated %d sized candidates.", len(enriched_signals))
        return enriched_signals

    def _add_ema_ribbon(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add simple EMA ribbon trend context to the data.

        Computes a set of EMAs and derives a trend_score based on their ordering.
        """
        df = data.copy()
        df = df.sort_values(["ticker", "date"])  # ensure correct order

        ema_periods = [5, 8, 13, 21, 34, 55]
        for p in ema_periods:
            df[f"ema_{p}"] = df.groupby("ticker")["close"].transform(
                lambda s, span=p: s.ewm(span=span, adjust=False).mean(),
                span=p,
            )

        def _trend_score(row: pd.Series) -> float:
            values = [row.get(f"ema_{p}") for p in ema_periods]
            if any(v is None for v in values):
                return 0.0
            vals = np.array(values, dtype=float)
            # Uptrend if EMAs are ordered from shortest > longest (prices rising)
            is_up = np.all(np.diff(vals) <= 0)
            # Downtrend if EMAs are ordered from shortest < longest (prices falling)
            is_down = np.all(np.diff(vals) >= 0)
            if is_up:
                return 1.0
            if is_down:
                return -1.0
            return 0.0

        df["ema_trend_score"] = df.apply(_trend_score, axis=1)
        return df

    def _add_pump_risk_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add simple pump-and-dump style risk features.

        Uses 20-day EWMA of price and volume, flags extreme deviations.
        """
        df = data.copy()
        df = df.sort_values(["ticker", "date"])

        def _compute_features(group: pd.DataFrame) -> pd.DataFrame:
            g = group.copy()
            g["price_ewm20"] = g["close"].ewm(span=20, adjust=False).mean()
            g["price_std20"] = g["close"].ewm(span=20, adjust=False).std()
            g["vol_ewm20"] = g["volume"].ewm(span=20, adjust=False).mean()

            # Avoid division by zero
            g["price_std20"] = g["price_std20"].replace(0, np.nan)
            g["vol_ewm20"] = g["vol_ewm20"].replace(0, np.nan)

            g["price_z"] = (g["close"] - g["price_ewm20"]) / g["price_std20"]
            g["volume_multiple"] = g["volume"] / g["vol_ewm20"]

            # Pump flag and score
            pump_flag = (
                (g["price_z"] > self.config.pump_price_z)
                & (g["volume_multiple"] > self.config.pump_volume_multiple)
            )
            g["pump_flag"] = pump_flag.fillna(False)

            # Pump score between 0 and 1 based on how extreme the anomaly is
            z_norm = (g["price_z"] / (self.config.pump_price_z * 2)).clip(lower=0, upper=1)
            vol_norm = (
                g["volume_multiple"] / (self.config.pump_volume_multiple * 2)
            ).clip(lower=0, upper=1)
            g["pump_score"] = (z_norm + vol_norm) / 2.0

            return g

        df = df.groupby("ticker", group_keys=False).apply(_compute_features)
        return df


if __name__ == "__main__":
    # Simple test harness for the fusion engine
    tickers = ["SPY", "QQQ", "AMZN", "MU", "TQQQ", "SQQQ"]

    engine = EliteFusionEngine()
    candidates = engine.run_pipeline(tickers)

    print(f"Generated {len(candidates)} fused candidates")
    for c in candidates[:10]:
        print("\nTicker:", c["ticker"])
        print("  Date:", c["date"])
        print(f"  Confidence: {c['confidence']:.0%}")
        print(f"  Expected return: {c.get('expected_return_pct', 0.0):.1f}%")
        print(f"  Shares: {c['shares']}")
        print(f"  Position value: ${c['position_value']:.2f}")
        print("  Edge components:", c.get("edge_components"))
