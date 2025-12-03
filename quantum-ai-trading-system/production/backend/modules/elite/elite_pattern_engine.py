import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class DetectedPattern:
    pattern_name: str
    confidence: float
    expected_move: float
    entry_price: float
    target_price: float
    stop_loss: float
    success_rate: float
    supporting_factors: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "pattern_name": self.pattern_name,
            "confidence": float(self.confidence),
            "expected_move": float(self.expected_move),
            "entry_price": float(self.entry_price),
            "target_price": float(self.target_price),
            "stop_loss": float(self.stop_loss),
            "success_rate": float(self.success_rate),
            "supporting_factors": list(self.supporting_factors),
        }


class ElitePatternEngine:
    """Detect high-value classical chart patterns on end-of-day data.

    All detection assumes a single-ticker DataFrame with at least
    ['date', 'open', 'high', 'low', 'close', 'volume'].
    """

    def __init__(self, lookback: int = 120) -> None:
        self.lookback = int(lookback)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect_all_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Run all pattern detectors and return a list of pattern dicts."""
        if df.empty:
            return []

        data = df.sort_values("date").tail(self.lookback).reset_index(drop=True)
        patterns: List[DetectedPattern] = []

        cup = self._detect_cup_and_handle(data)
        if cup:
            patterns.append(cup)

        flag = self._detect_bull_flag(data)
        if flag:
            patterns.append(flag)

        tri = self._detect_ascending_triangle(data)
        if tri:
            patterns.append(tri)

        wedge = self._detect_falling_wedge(data)
        if wedge:
            patterns.append(wedge)

        dbl = self._detect_double_bottom(data)
        if dbl:
            patterns.append(dbl)

        return [p.as_dict() for p in patterns]

    # ------------------------------------------------------------------
    # Cup & Handle
    # ------------------------------------------------------------------
    def _detect_cup_and_handle(self, df: pd.DataFrame) -> Optional[DetectedPattern]:
        close = df["close"].values
        volume = df["volume"].values
        n = len(close)
        if n < 40:
            return None

        # Use last ~80 bars for pattern, or all if shorter
        window = min(80, n)
        sub = close[-window:]
        idx_offset = n - window

        # Approx: cup low is global min, rim near first/last 20% of window
        low_idx = int(np.argmin(sub))
        low_price = float(sub[low_idx])
        left_rim_idx = int(np.argmax(sub[: max(5, window // 5)]))
        right_rim_idx = int(np.argmax(sub[window - max(5, window // 5) :])) + (window - max(5, window // 5))
        left_rim = float(sub[left_rim_idx])
        right_rim = float(sub[right_rim_idx])

        if low_idx <= left_rim_idx or low_idx >= right_rim_idx:
            return None

        rim_level = (left_rim + right_rim) / 2.0
        cup_depth_pct = (rim_level - low_price) / rim_level * 100.0 if rim_level > 0 else 0.0
        # Relax cup depth to 8–25% (was up to 35%) to better match common patterns
        if not (8.0 <= cup_depth_pct <= 25.0):
            return None

        # Rough U-shape: midpoints should be closer to low than rims
        mid_left = int((left_rim_idx + low_idx) / 2)
        mid_right = int((low_idx + right_rim_idx) / 2)
        if mid_left <= 0 or mid_right >= window:
            return None
        u_shape_ok = (
            abs(sub[mid_left] - low_price) < abs(left_rim - low_price)
            and abs(sub[mid_right] - low_price) < abs(right_rim - low_price)
        )
        if not u_shape_ok:
            return None

        # Handle: last ~20% after right rim, small pullback
        handle_start = right_rim_idx
        handle_window = max(5, window // 10)
        if handle_start + 3 >= window:
            return None
        handle_slice = sub[handle_start: handle_start + handle_window]
        handle_high = float(handle_slice[0])
        handle_low = float(handle_slice.min())
        handle_depth_pct = (handle_high - handle_low) / handle_high * 100.0 if handle_high > 0 else 0.0
        # Relax handle depth to 2–8% (was 1–8%)
        if not (2.0 <= handle_depth_pct <= 8.0):
            return None

        # Breakout: last close above rim with volume spike
        breakout_price = float(sub[-1])
        if breakout_price <= rim_level:
            return None

        vol_series = volume[-window:]
        vol_ma = pd.Series(vol_series).rolling(window=20, min_periods=5).mean().iloc[-1]
        breakout_vol = float(vol_series[-1])
        vol_ratio = breakout_vol / vol_ma if vol_ma and not math.isnan(vol_ma) else 1.0
        if vol_ratio < 1.5:
            return None

        expected_move = 0.25  # +25% typical
        entry = breakout_price
        target = entry * (1 + expected_move)
        stop = min(handle_low, low_price * 0.97)

        supporting = [
            f"Cup depth: {cup_depth_pct:.1f}% (ideal 12-20%)",
            f"Handle depth: {handle_depth_pct:.1f}% (ideal 3-5%)",
            f"Breakout volume: {vol_ratio:.1f}x 20-day avg",
            f"Rim level: {rim_level:.2f}, breakout: {breakout_price:.2f}",
        ]

        confidence = 0.90
        if 12.0 <= cup_depth_pct <= 20.0:
            confidence += 0.03
        if 3.0 <= handle_depth_pct <= 6.0:
            confidence += 0.02
        confidence = min(confidence, 0.97)

        return DetectedPattern(
            pattern_name="Cup & Handle",
            confidence=confidence,
            expected_move=expected_move,
            entry_price=entry,
            target_price=target,
            stop_loss=stop,
            success_rate=0.95,
            supporting_factors=supporting,
        )

    # ------------------------------------------------------------------
    # Bull Flag
    # ------------------------------------------------------------------
    def _detect_bull_flag(self, df: pd.DataFrame) -> Optional[DetectedPattern]:
        close = df["close"].values
        volume = df["volume"].values
        n = len(close)
        if n < 30:
            return None

        window = min(60, n)
        sub = close[-window:]
        vol_sub = volume[-window:]

        # Flagpole: strong up move over ~10 bars
        pole_len = 10
        pole_start = window - (pole_len + 10)
        if pole_start < 0:
            pole_start = 0
        pole_end = pole_start + pole_len
        if pole_end >= window:
            return None

        pole_start_price = float(sub[pole_start])
        pole_end_price = float(sub[pole_end])
        pole_ret = (pole_end_price - pole_start_price) / pole_start_price * 100.0
        # Relax flagpole requirement from 10% to 5%
        if pole_ret < 5.0:
            return None

        # Flag: mild downward/sideways consolidation after pole
        flag_slice = sub[pole_end:]
        if len(flag_slice) < 5:
            return None
        flag_high = float(flag_slice.max())
        flag_low = float(flag_slice.min())
        flag_depth_pct = (flag_high - flag_low) / flag_high * 100.0 if flag_high > 0 else 0.0
        # Allow flags up to ~5% depth (was 10%) to focus on tighter consolidations
        if not (1.0 <= flag_depth_pct <= 5.0):
            return None

        # Breakout: last close near top of flag range
        breakout_price = float(flag_slice[-1])
        if breakout_price < flag_high * 0.995:
            return None

        vol_ma = pd.Series(vol_sub).rolling(window=20, min_periods=5).mean().iloc[-1]
        breakout_vol = float(vol_sub[-1])
        vol_ratio = breakout_vol / vol_ma if vol_ma and not math.isnan(vol_ma) else 1.0
        if vol_ratio < 1.3:
            return None

        expected_move = pole_ret / 100.0  # continuation roughly equal to pole
        expected_move = max(0.10, min(expected_move, 0.30))
        entry = breakout_price
        target = entry * (1 + expected_move)
        stop = flag_low * 0.99

        supporting = [
            f"Flagpole gain: {pole_ret:.1f}%",
            f"Flag depth: {flag_depth_pct:.1f}%",
            f"Breakout volume: {vol_ratio:.1f}x avg",
        ]

        confidence = 0.78
        return DetectedPattern(
            pattern_name="Bull Flag",
            confidence=confidence,
            expected_move=expected_move,
            entry_price=entry,
            target_price=target,
            stop_loss=stop,
            success_rate=0.78,
            supporting_factors=supporting,
        )

    # ------------------------------------------------------------------
    # Ascending Triangle
    # ------------------------------------------------------------------
    def _detect_ascending_triangle(self, df: pd.DataFrame) -> Optional[DetectedPattern]:
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        n = len(close)
        if n < 40:
            return None

        window = min(80, n)
        sub_high = high[-window:]
        sub_low = low[-window:]

        # Resistance: approximate by upper quartile of highs
        res_level = float(np.quantile(sub_high, 0.8))
        res_tolerance = res_level * 0.01
        res_touches = np.where(abs(sub_high - res_level) <= res_tolerance)[0]
        if len(res_touches) < 3:
            return None

        # Support: trend of rising lows via simple linear regression
        x = np.arange(window)
        coef = np.polyfit(x, sub_low, 1)
        slope = coef[0]
        if slope <= 0:
            return None

        # Breakout: last close above resistance
        breakout_price = float(close[-1])
        if breakout_price <= res_level:
            return None

        expected_move = 0.15
        entry = breakout_price
        target = entry * (1 + expected_move)
        stop = res_level * 0.97

        supporting = [
            f"Flat resistance near {res_level:.2f} with {len(res_touches)} touches",
            f"Rising lows slope: {slope:.4f}",
            f"Breakout close: {breakout_price:.2f}",
        ]

        confidence = 0.85
        return DetectedPattern(
            pattern_name="Ascending Triangle",
            confidence=confidence,
            expected_move=expected_move,
            entry_price=entry,
            target_price=target,
            stop_loss=stop,
            success_rate=0.85,
            supporting_factors=supporting,
        )

    # ------------------------------------------------------------------
    # Falling Wedge
    # ------------------------------------------------------------------
    def _detect_falling_wedge(self, df: pd.DataFrame) -> Optional[DetectedPattern]:
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        n = len(close)
        if n < 40:
            return None

        window = min(80, n)
        sub_high = high[-window:]
        sub_low = low[-window:]
        x = np.arange(window)

        # Fit lines to highs and lows
        coef_high = np.polyfit(x, sub_high, 1)
        coef_low = np.polyfit(x, sub_low, 1)
        slope_high, intercept_high = coef_high
        slope_low, intercept_low = coef_low

        # Both slopes downward, lows falling slower than highs (converging)
        if not (slope_high < 0 and slope_low < 0 and abs(slope_low) < abs(slope_high)):
            return None

        # Distance between lines shrinking
        dist_start = (slope_high * 0 + intercept_high) - (slope_low * 0 + intercept_low)
        dist_end = (slope_high * (window - 1) + intercept_high) - (
            slope_low * (window - 1) + intercept_low
        )
        if not (dist_start > dist_end > 0):
            return None

        # Breakout: last close above projected upper trendline
        last_x = window - 1
        proj_high_last = slope_high * last_x + intercept_high
        breakout_price = float(close[-1])
        if breakout_price <= proj_high_last * 1.01:
            return None

        expected_move = 0.18
        entry = breakout_price
        target = entry * (1 + expected_move)
        stop = float(sub_low.min()) * 0.97

        supporting = [
            f"Upper slope: {slope_high:.4f}, lower slope: {slope_low:.4f}",
            f"Wedge width from {dist_start:.2f} to {dist_end:.2f}",
            f"Breakout close: {breakout_price:.2f} above upper trendline {proj_high_last:.2f}",
        ]

        confidence = 0.82
        return DetectedPattern(
            pattern_name="Falling Wedge",
            confidence=confidence,
            expected_move=expected_move,
            entry_price=entry,
            target_price=target,
            stop_loss=stop,
            success_rate=0.82,
            supporting_factors=supporting,
        )

    # ------------------------------------------------------------------
    # Double Bottom
    # ------------------------------------------------------------------
    def _detect_double_bottom(self, df: pd.DataFrame) -> Optional[DetectedPattern]:
        close = df["close"].values
        volume = df["volume"].values
        n = len(close)
        if n < 40:
            return None

        window = min(80, n)
        sub = close[-window:]

        # Find two local minima separated by at least 5 bars
        from scipy.signal import argrelextrema  # type: ignore

        arr = np.array(sub)
        mins = argrelextrema(arr, np.less_equal, order=3)[0]
        if len(mins) < 2:
            return None

        # Choose last two minima
        first, second = mins[-2], mins[-1]
        if second - first < 5:
            return None

        low1 = float(arr[first])
        low2 = float(arr[second])
        if abs(low1 - low2) / max(low1, low2) > 0.03:
            return None

        mid_slice = arr[first:second]
        neck = float(mid_slice.max())
        if neck <= 0:
            return None

        # Breakout: last close above neckline
        breakout_price = float(sub[-1])
        if breakout_price <= neck:
            return None

        depth = (neck - min(low1, low2)) / neck * 100.0
        expected_move = min(0.30, max(0.10, depth / 100.0 * 2.0))
        entry = breakout_price
        target = entry * (1 + expected_move)
        stop = min(low1, low2) * 0.98

        vol_sub = volume[-window:]
        vol_ma = pd.Series(vol_sub).rolling(window=20, min_periods=5).mean().iloc[-1]
        breakout_vol = float(vol_sub[-1])
        vol_ratio = breakout_vol / vol_ma if vol_ma and not math.isnan(vol_ma) else 1.0

        supporting = [
            f"Low1: {low1:.2f}, Low2: {low2:.2f}",
            f"Neckline: {neck:.2f}, breakout: {breakout_price:.2f}",
            f"Pattern depth: {depth:.1f}%",
            f"Breakout volume: {vol_ratio:.1f}x avg",
        ]

        confidence = 0.80
        if vol_ratio > 1.5:
            confidence += 0.03
        confidence = min(confidence, 0.85)

        return DetectedPattern(
            pattern_name="Double Bottom",
            confidence=confidence,
            expected_move=expected_move,
            entry_price=entry,
            target_price=target,
            stop_loss=stop,
            success_rate=0.80,
            supporting_factors=supporting,
        )


if __name__ == "__main__":
    # Simple smoke test using random data (will usually return no patterns).
    dates = pd.date_range(end=pd.Timestamp.today(), periods=120, freq="B")
    prices = np.cumsum(np.random.normal(0, 1, size=len(dates))) + 100
    highs = prices + np.random.uniform(0, 1, size=len(dates))
    lows = prices - np.random.uniform(0, 1, size=len(dates))
    opens = prices + np.random.uniform(-0.5, 0.5, size=len(dates))
    vols = np.random.randint(100_000, 200_000, size=len(dates))

    df_test = pd.DataFrame(
        {
            "date": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": vols,
        }
    )

    engine = ElitePatternEngine()
    patterns = engine.detect_all_patterns(df_test)
    print(f"Detected {len(patterns)} patterns")
    for p in patterns:
        print(p)
