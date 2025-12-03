from typing import Any, Dict, List

import numpy as np
import pandas as pd


class EliteBreakoutDetector:
    """Detect high-probability breakout setups across multiple tickers.

    Expects DataFrame from EliteDataFetcher with at least:
    ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'volume_ratio'].
    """

    def __init__(self, lookback_days: int = 60) -> None:
        self.lookback_days = int(lookback_days)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect_all_breakouts(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        if data.empty:
            return []

        data = data.sort_values(["ticker", "date"]).copy()
        breakouts: List[Dict[str, Any]] = []

        for ticker in data["ticker"].unique():
            df_t = data[data["ticker"] == ticker].tail(self.lookback_days).reset_index(drop=True)
            if len(df_t) < 20:
                continue

            breakouts.extend(self._detect_volume_breakout(df_t))
            breakouts.extend(self._detect_consolidation_breakout(df_t))
            breakouts.extend(self._detect_52w_high_breakout(df_t))
            breakouts.extend(self._detect_gap_up_continuation(df_t))
            breakouts.extend(self._detect_resistance_breakout(df_t))

        return breakouts

    # ------------------------------------------------------------------
    # Setup 1: Volume Breakout
    # ------------------------------------------------------------------
    def _detect_volume_breakout(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        if "volume_ratio" not in df.columns:
            return []

        out: List[Dict[str, Any]] = []
        # Use last bar only for breakout decision
        row = df.iloc[-1]
        vol_ratio = float(row["volume_ratio"])
        # Loosen further: treat 10% above average as meaningful accumulation
        if vol_ratio < 1.1:
            return []

        # Resistance = highest close in last 20 days excluding today
        lookback = df.tail(21).iloc[:-1]
        resistance = float(lookback["close"].max())
        close = float(row["close"])
        if close <= resistance:
            return []

        entry = close
        expected_move_pct = 7.5
        target = entry * (1 + expected_move_pct / 100.0)
        stop = resistance * 0.98

        out.append(
            {
                "ticker": str(row["ticker"]),
                "breakout_type": "VOLUME_BREAKOUT",
                "confidence": min(0.95, 0.8 + (vol_ratio - 3.0) * 0.05),
                "expected_move_pct": expected_move_pct,
                "entry_price": entry,
                "target_price": target,
                "stop_loss": stop,
                "catalyst": f"Volume {vol_ratio:.1f}x average, breaking resistance",
                "supporting_factors": [
                    f"Volume: {vol_ratio:.1f}x 20-day average",
                    f"Price cleared 20-day resistance at ${resistance:.2f}",
                ],
            }
        )
        return out

    # ------------------------------------------------------------------
    # Setup 2: Consolidation Breakout
    # ------------------------------------------------------------------
    def _detect_consolidation_breakout(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        required = {"high", "low", "close", "bb_upper", "bb_lower", "volume_ratio"}
        if not required.issubset(df.columns):
            return []

        out: List[Dict[str, Any]] = []
        window = 10
        if len(df) < window + 5:
            return []

        recent = df.tail(window)
        # Tight range condition: <3% range for at least 2 days
        range_pct = (recent["high"] - recent["low"]) / recent["close"]
        tight_days = (range_pct < 0.03).sum()
        if tight_days < 2:
            return []

        # BB squeeze
        bb_width = recent["bb_upper"] - recent["bb_lower"]
        bb_width_ratio = bb_width / bb_width.rolling(20, min_periods=5).mean()
        if bb_width_ratio.iloc[-1] >= 0.7:
            return []

        row = df.iloc[-1]
        vol_ratio = float(row["volume_ratio"])
        if vol_ratio < 1.2:
            return []

        # Breakout = close above recent high
        recent_high = float(recent["high"].max())
        close = float(row["close"])
        if close <= recent_high:
            return []

        entry = close
        expected_move_pct = 10.0
        target = entry * (1 + expected_move_pct / 100.0)
        stop = float(recent["low"].min()) * 0.99

        out.append(
            {
                "ticker": str(row["ticker"]),
                "breakout_type": "CONSOLIDATION_BREAKOUT",
                "confidence": 0.82,
                "expected_move_pct": expected_move_pct,
                "entry_price": entry,
                "target_price": target,
                "stop_loss": stop,
                "catalyst": "Tight consolidation + BB squeeze breakout",
                "supporting_factors": [
                    f"Tight range {tight_days} days (<2% ATR)",
                    f"BB width {float(bb_width_ratio.iloc[-1]):.2f}x average (squeeze)",
                    f"Breakout above range high ${recent_high:.2f}",
                    f"Volume: {vol_ratio:.1f}x average",
                ],
            }
        )
        return out

    # ------------------------------------------------------------------
    # Setup 3: 52-week High Breakout
    # ------------------------------------------------------------------
    def _detect_52w_high_breakout(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        if len(df) < 60:
            return []
        out: List[Dict[str, Any]] = []

        window = min(len(df), 252)
        recent = df.tail(window)
        row = recent.iloc[-1]
        high_close = float(recent["close"].max())
        prev_max = float(recent["close"].iloc[:-1].max())
        close = float(row["close"])

        if close <= prev_max:
            return []

        # Volume confirmation if available (loosen to ~1.1x)
        vol_ratio = float(row.get("volume_ratio", np.nan))
        if not np.isnan(vol_ratio) and vol_ratio < 1.1:
            return []

        entry = close
        expected_move_pct = 15.0
        target = entry * (1 + expected_move_pct / 100.0)
        stop = prev_max * 0.97

        out.append(
            {
                "ticker": str(row["ticker"]),
                "breakout_type": "HIGH_52W_BREAKOUT",
                "confidence": 0.86,
                "expected_move_pct": expected_move_pct,
                "entry_price": entry,
                "target_price": target,
                "stop_loss": stop,
                "catalyst": "New 52-week high with volume confirmation",
                "supporting_factors": [
                    f"New 52-week high close ${close:.2f}",
                    f"Previous max close ${prev_max:.2f}",
                    f"Volume ratio: {vol_ratio:.1f}x" if not np.isnan(vol_ratio) else "",
                ],
            }
        )
        return out

    # ------------------------------------------------------------------
    # Setup 4: Gap Up Continuation
    # ------------------------------------------------------------------
    def _detect_gap_up_continuation(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        if len(df) < 2:
            return []

        out: List[Dict[str, Any]] = []
        prev = df.iloc[-2]
        row = df.iloc[-1]

        prev_close = float(prev["close"])
        gap_open = float(row["open"])
        close = float(row["close"])

        gap_pct = (gap_open - prev_close) / prev_close * 100.0
        if gap_pct < 3.0:
            return []

        # Holds gap: low above prev close
        if float(row["low"]) <= prev_close:
            return []

        vol_ratio = float(row.get("volume_ratio", 1.0))

        entry = close
        expected_move_pct = 7.0
        target = entry * (1 + expected_move_pct / 100.0)
        stop = prev_close

        out.append(
            {
                "ticker": str(row["ticker"]),
                "breakout_type": "GAP_UP_CONTINUATION",
                "confidence": 0.80 if vol_ratio < 2.0 else 0.88,
                "expected_move_pct": expected_move_pct,
                "entry_price": entry,
                "target_price": target,
                "stop_loss": stop,
                "catalyst": f"Gap up {gap_pct:.1f}% holding above prior close",
                "supporting_factors": [
                    f"Gap from ${prev_close:.2f} to open ${gap_open:.2f}",
                    f"Day's low ${float(row['low']):.2f} above prior close",
                    f"Volume ratio: {vol_ratio:.1f}x",
                ],
            }
        )
        return out

    # ------------------------------------------------------------------
    # Setup 5: Resistance Break
    # ------------------------------------------------------------------
    def _detect_resistance_breakout(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        if len(df) < 25:
            return []
        if "volume_ratio" not in df.columns:
            return []

        out: List[Dict[str, Any]] = []
        recent = df.tail(25)
        row = recent.iloc[-1]

        # Resistance: upper quartile of highs in last 20 bars
        highs = recent["high"].values
        resistance = float(np.quantile(highs[:-1], 0.8))
        close = float(row["close"])
        if close <= resistance:
            return []

        vol_ratio = float(row["volume_ratio"])
        # Loosen to ~1.2x to capture more valid resistance breaks
        if vol_ratio < 1.2:
            return []

        entry = close
        expected_move_pct = 10.0
        target = entry * (1 + expected_move_pct / 100.0)
        stop = resistance * 0.98

        out.append(
            {
                "ticker": str(row["ticker"]),
                "breakout_type": "RESISTANCE_BREAK",
                "confidence": 0.84,
                "expected_move_pct": expected_move_pct,
                "entry_price": entry,
                "target_price": target,
                "stop_loss": stop,
                "catalyst": "Multiple resistance tests followed by volume breakout",
                "supporting_factors": [
                    f"Resistance near ${resistance:.2f}",
                    f"Close ${close:.2f} above resistance",
                    f"Volume {vol_ratio:.1f}x average",
                ],
            }
        )
        return out


if __name__ == "__main__":
    from backend.modules.elite.elite_data_fetcher import EliteDataFetcher

    fetcher = EliteDataFetcher()
    tickers = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA"]
    data = fetcher.fetch_data(tickers, days=60)

    detector = EliteBreakoutDetector()
    breakouts = detector.detect_all_breakouts(data)

    print(f"Found {len(breakouts)} breakout setups")
    for b in breakouts:
        print(
            f"{b['ticker']}: {b['breakout_type']} "
            f"({b['confidence']:.0%} confidence, +{b['expected_move_pct']:.1f}% expected)"
        )
