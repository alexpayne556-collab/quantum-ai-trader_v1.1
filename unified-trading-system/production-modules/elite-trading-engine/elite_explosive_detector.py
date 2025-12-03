from typing import Any, Dict, List

import numpy as np
import pandas as pd


class EliteExplosiveDetector:
    """Detector focused on catching explosive movers *before* they make big runs.

    Uses only real, quantitative signals available from EliteDataFetcher:
    - Accumulation via ADL and volume trend
    - Gap + early-session strength as a proxy for pre-market interest
    - Breakout confirmation via volume spike, resistance break, and momentum

    Expected input columns (per ticker):
    ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume',
     'adl', 'volume_ratio', 'rsi_14']  (and other indicators if available).
    """

    def __init__(self, lookback_days: int = 60) -> None:
        self.lookback_days = int(lookback_days)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze_universe(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze all tickers in a DataFrame and return explosive candidates.

        Returns list of dicts sorted by combined_score descending:
        {
            'ticker': str,
            'accumulation_score': 0-100,
            'premarket_score': 0-100,
            'breakout_score': 0-100,
            'combined_score': 0-100,
            'signals': [...],  # textual explanations
        }
        Only tickers with combined_score >= 70 are returned.
        """
        if data.empty:
            return []

        data = data.sort_values(["ticker", "date"]).copy()
        results: List[Dict[str, Any]] = []

        for ticker in data["ticker"].unique():
            df_t = data[data["ticker"] == ticker].tail(self.lookback_days).reset_index(drop=True)
            if len(df_t) < 20:
                continue

            scores = self._analyze_ticker_df(df_t)
            if scores["combined_score"] >= 70:
                results.append(scores)

        # Sort by combined score descending
        results.sort(key=lambda x: x["combined_score"], reverse=True)
        return results

    # ------------------------------------------------------------------
    # Per-ticker analysis
    # ------------------------------------------------------------------
    def _analyze_ticker_df(self, df: pd.DataFrame) -> Dict[str, Any]:
        ticker = str(df["ticker"].iloc[0])
        df = df.copy()

        accumulation_score, acc_signals = self._accumulation_score(df)
        premarket_score, pre_signals = self._premarket_score(df)
        breakout_score, brk_signals = self._breakout_score(df)

        # Weighted combination: accumulation (30) + premarket (30) + breakout (40)
        combined = (
            0.30 * accumulation_score + 0.30 * premarket_score + 0.40 * breakout_score
        )
        combined = float(np.clip(combined, 0.0, 100.0))

        signals: List[str] = []
        signals.extend(acc_signals)
        signals.extend(pre_signals)
        signals.extend(brk_signals)

        return {
            "ticker": ticker,
            "accumulation_score": accumulation_score,
            "premarket_score": premarket_score,
            "breakout_score": breakout_score,
            "combined_score": combined,
            "signals": signals,
        }

    # ------------------------------------------------------------------
    # 1. Accumulation detection
    # ------------------------------------------------------------------
    def _accumulation_score(self, df: pd.DataFrame) -> tuple[int, List[str]]:
        signals: List[str] = []

        # Require ADL for true accumulation; if missing, approximate with OBV if present
        if "adl" in df.columns:
            adl = df["adl"].astype(float)
        elif "obv" in df.columns:
            adl = df["obv"].astype(float)
        else:
            return 0, ["No ADL/OBV data for accumulation"]

        # Use last 30 bars for accumulation
        recent = adl.tail(30)
        if len(recent) < 10:
            return 0, ["Insufficient history for accumulation"]

        x = np.arange(len(recent))
        coef = np.polyfit(x, recent.values, 1)
        adl_slope = coef[0]

        # Volume trend
        vol = df["volume"].astype(float).tail(30)
        vol_ma_20 = vol.rolling(20, min_periods=10).mean()
        vol_trend = 0.0
        if vol_ma_20.notna().all():
            vol_trend = float(vol_ma_20.iloc[-1] / vol_ma_20.iloc[0] - 1.0)

        # Price consolidation: last 10 days range vs price
        closes = df["close"].astype(float).tail(10)
        if len(closes) >= 5:
            price_range_pct = (closes.max() - closes.min()) / closes.mean()
        else:
            price_range_pct = 1.0

        # Scoring rules
        score = 0
        if adl_slope > 0:
            score += 30
        if vol_trend > 0.1:
            score += 30
        elif vol_trend > 0.05:
            score += 20
        elif vol_trend > 0.0:
            score += 10

        if price_range_pct < 0.03:
            score += 40
        elif price_range_pct < 0.05:
            score += 25
        elif price_range_pct < 0.08:
            score += 10

        score = int(min(100, max(0, score)))

        if adl_slope > 0:
            signals.append(f"ADL rising (slope {adl_slope:.2f}) - accumulation")
        if vol_trend > 0:
            signals.append(f"Volume trend +{vol_trend*100:.1f}% over 30 bars")
        signals.append(f"10-day price range {price_range_pct*100:.1f}% (consolidation)")

        return score, signals

    # ------------------------------------------------------------------
    # 2. Pre-market / gap strength proxy
    # ------------------------------------------------------------------
    def _premarket_score(self, df: pd.DataFrame) -> tuple[int, List[str]]:
        """Approximate pre-market excitement via gaps and early-session strength.

        With only daily bars available, we treat a strong gap + day that holds
        as a proxy for pre-market accumulation and news catalysts.
        """
        signals: List[str] = []
        if len(df) < 3:
            return 0, ["Not enough data for gap analysis"]

        prev = df.iloc[-2]
        last = df.iloc[-1]

        prev_close = float(prev["close"])
        gap_open = float(last["open"])
        gap_pct = (gap_open - prev_close) / prev_close

        # Basic gap threshold 2%+
        score = 0
        if gap_pct >= 0.02:
            score += 40
            signals.append(f"Gap up {gap_pct*100:.1f}% vs prior close")

            # Holds gap? low stays above part of gap
            low = float(last["low"])
            if low > prev_close * 0.99:
                score += 20
                signals.append("Gap largely held (low near/above prior close)")

            # Volume confirmation vs 20-day average if available
            if "volume_ratio" in df.columns:
                vol_ratio = float(last["volume_ratio"])
                if vol_ratio > 2.0:
                    score += 25
                elif vol_ratio > 1.5:
                    score += 15
                signals.append(f"Volume {vol_ratio:.1f}x 20-day average")

        score = int(min(100, max(0, score)))
        if not signals:
            signals.append("No significant gap or volume pre-market proxy")

        return score, signals

    # ------------------------------------------------------------------
    # 3. Breakout confirmation
    # ------------------------------------------------------------------
    def _breakout_score(self, df: pd.DataFrame) -> tuple[int, List[str]]:
        signals: List[str] = []
        if len(df) < 20:
            return 0, ["Not enough history for breakout confirmation"]

        df = df.copy()
        df["resistance"] = df["high"].rolling(20).max()

        last = df.iloc[-1]
        prev = df.iloc[-2]

        close = float(last["close"])
        prev_close = float(prev["close"])
        resistance = float(last["resistance"])

        # Volume spike: use volume_ratio if present, else raw volume vs 20-day avg
        if "volume_ratio" in df.columns:
            vol_ratio = float(last["volume_ratio"])
        else:
            vol = df["volume"].astype(float)
            vol_ma20 = vol.rolling(20, min_periods=10).mean()
            vol_ratio = float(vol.iloc[-1] / vol_ma20.iloc[-1]) if vol_ma20.iloc[-1] else 1.0

        price_gain = (close - prev_close) / prev_close

        score = 0
        # 4x+ volume spike
        if vol_ratio >= 4.0:
            score += 40
        elif vol_ratio >= 2.5:
            score += 25
        elif vol_ratio >= 1.8:
            score += 15

        # Above resistance
        if close > resistance:
            score += 30
            signals.append(f"Breakout above 20-day resistance ${resistance:.2f}")

        # Momentum acceleration
        if price_gain >= 0.04:
            score += 30
        elif price_gain >= 0.025:
            score += 20
        elif price_gain >= 0.015:
            score += 10

        if "rsi_14" in df.columns:
            rsi = float(last["rsi_14"])
            if rsi >= 65:
                score += 10
                signals.append(f"RSI14 strong at {rsi:.1f}")

        score = int(min(100, max(0, score)))

        signals.append(f"Volume ratio {vol_ratio:.1f}x, 1-day gain {price_gain*100:.1f}%")

        return score, signals


if __name__ == "__main__":
    from backend.modules.elite.elite_data_fetcher import EliteDataFetcher

    fetcher = EliteDataFetcher()
    tickers = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META", "GOOGL", "AMZN"]
    data = fetcher.fetch_data(tickers, days=60)

    detector = EliteExplosiveDetector(lookback_days=60)
    explosive = detector.analyze_universe(data)

    print(f"Found {len(explosive)} explosive candidates")
    for e in explosive[:10]:
        print(
            f"{e['ticker']}: Acc {e['accumulation_score']}, Pre {e['premarket_score']}, "
            f"Brk {e['breakout_score']}, Combined {e['combined_score']:.1f}"
        )
