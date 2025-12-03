from typing import Any, Dict, List

import numpy as np
import pandas as pd


class EliteMomentumScanner:
    """Scan for high-momentum stocks using price, volume, and relative strength."""

    def __init__(self, min_score: int = 40) -> None:
        # Default threshold lowered further to 40 to surface more opportunities
        self.min_score = int(min_score)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def scan_for_momentum(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        if data.empty:
            return []

        data = data.sort_values(["ticker", "date"]).copy()

        # Compute SPY benchmark if present
        spy_mask = data["ticker"] == "SPY"
        spy_close = None
        if spy_mask.any():
            spy_df = data[spy_mask].set_index("date").sort_index()
            spy_close = spy_df["close"]

        movers: List[Dict[str, Any]] = []

        for ticker in data["ticker"].unique():
            if ticker == "SPY":
                continue
            df_t = data[data["ticker"] == ticker].set_index("date").sort_index()
            if len(df_t) < 10:
                continue

            res = self._evaluate_ticker(df_t, spy_close)
            if res and res["momentum_score"] >= self.min_score:
                movers.append(res)

        return sorted(movers, key=lambda x: x["momentum_score"], reverse=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _evaluate_ticker(self, df: pd.DataFrame, spy_close: pd.Series | None) -> Dict[str, Any] | None:
        close = df["close"]
        volume = df["volume"]

        # Volume ratio: last volume vs 20-day avg
        vol_ma20 = volume.rolling(window=20, min_periods=5).mean()
        vol_ratio = float(volume.iloc[-1] / vol_ma20.iloc[-1]) if vol_ma20.iloc[-1] else 1.0

        # Price acceleration
        price_change_1d = float(close.iloc[-1] / close.iloc[-2] - 1.0)
        if len(close) >= 4:
            price_change_3d = float(close.iloc[-1] / close.iloc[-4] - 1.0)
        else:
            price_change_3d = 0.0
        if len(close) >= 6:
            price_change_5d = float(close.iloc[-1] / close.iloc[-6] - 1.0)
        else:
            price_change_5d = 0.0

        # Relative strength vs SPY
        relative_strength = 0.0
        if spy_close is not None:
            aligned = pd.concat([close, spy_close], axis=1, join="inner").dropna()
            if len(aligned) >= 6:
                stock_ret = aligned.iloc[-1, 0] / aligned.iloc[-6, 0] - 1.0
                spy_ret = aligned.iloc[-1, 1] / aligned.iloc[-6, 1] - 1.0
                relative_strength = float(stock_ret - spy_ret)

        # 10-day high breakout
        new_10day_high = False
        breakout_confirmed = False
        if len(close) >= 10:
            recent = close.iloc[-10:]
            prev_max = float(recent.iloc[:-1].max())
            last = float(recent.iloc[-1])
            if last >= prev_max and price_change_1d > 0:
                new_10day_high = True
                breakout_confirmed = vol_ratio >= 3.0 and price_change_1d >= 0.03

        # Momentum score calculation with more permissive thresholds
        score = 0
        # Volume surge
        if vol_ratio > 1.8:
            score += 30
        elif vol_ratio > 1.3:
            score += 20
        elif vol_ratio > 1.1:
            score += 10

        # 1-day acceleration
        if price_change_1d > 0.015:  # >1.5%
            score += 25
        elif price_change_1d > 0.008:  # >0.8%
            score += 15

        # 3-day acceleration
        if price_change_3d > 0.03:  # >3%
            score += 20
        elif price_change_3d > 0.015:  # >1.5%
            score += 15

        if relative_strength > 0.02:
            score += 15
        if new_10day_high:
            score += 10

        score = min(100, score)
        if score < self.min_score:
            return None

        # Expected continuation and confidence
        expected_continuation = 0.05
        if score >= 90:
            expected_continuation = 0.10
        elif score >= 80:
            expected_continuation = 0.08

        confidence = min(0.95, 0.5 + score / 200.0)

        entry = float(close.iloc[-1])
        target = entry * (1 + expected_continuation)
        stop_loss = entry * 0.97

        momentum_factors = {
            "volume_surge": vol_ratio,
            "price_acceleration_1d": price_change_1d,
            "price_acceleration_3d": price_change_3d,
            "price_acceleration_5d": price_change_5d,
            "relative_strength_vs_spy": relative_strength,
            "new_highs": new_10day_high,
            "breakout_confirmed": breakout_confirmed,
        }

        supporting_evidence = [
            f"Volume {vol_ratio:.1f}x average",
            f"1-day move {price_change_1d:.2%}",
            f"3-day move {price_change_3d:.2%}",
            f"Relative strength vs SPY {relative_strength:.2%}",
        ]
        if new_10day_high:
            supporting_evidence.append("New 10-day high with volume confirmation" if breakout_confirmed else "New 10-day high")

        return {
            "ticker": str(df["ticker"].iloc[0]) if "ticker" in df.columns else "",
            "momentum_score": int(score),
            "confidence": float(confidence),
            "expected_continuation": float(expected_continuation),
            "timeframe": "1-3 days",
            "entry_price": entry,
            "target_price": target,
            "stop_loss": stop_loss,
            "momentum_factors": momentum_factors,
            "supporting_evidence": supporting_evidence,
        }


if __name__ == "__main__":
    from backend.modules.elite.elite_data_fetcher import EliteDataFetcher

    fetcher = EliteDataFetcher()
    tickers = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META", "GOOGL", "AMZN"]
    data = fetcher.fetch_data(tickers, days=30)

    scanner = EliteMomentumScanner(min_score=70)
    movers = scanner.scan_for_momentum(data)

    print(f"Found {len(movers)} high-momentum stocks")
    for m in movers[:5]:
        print(
            f"{m['ticker']}: Score {m['momentum_score']}/100 "
            f"({m['confidence']:.0%} confidence, +{m['expected_continuation']:.0%} expected)"
        )
