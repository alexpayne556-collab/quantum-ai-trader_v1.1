import logging
from collections import defaultdict
from datetime import datetime, date, time
from typing import List, Dict, Any

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


class EliteSignalGenerator:
    """Generate institutional-grade multi-strategy trading signals.

    Strategies implemented (all long-only for small-account swing trading):
    1. Mean reversion (Connors-style RSI(2) pullback)
    2. Momentum breakout (20-day high + trend filters)
    3. Ichimoku cloud trend continuation
    4. Volume spike + mean reversion reversal
    5. Bollinger Band squeeze breakout
    6. RSI + Bollinger Band confluence
    7. Donchian channel breakout
    8. Williams %R mean reversion
    9. EMA golden cross
    10. Supply/demand zone reversal
    11. Gap fill mean reversion
    12. Three white soldiers continuation
    13. Hammer at support
    14. Inside bar breakout
    15. VWAP reversion
    """

    def __init__(self, min_confidence: float = 0.60, backtest_mode: bool = False) -> None:
        """Initialize signal generator.

        Args:
            min_confidence: Minimum confidence threshold (default 0.70).
            backtest_mode: If True, slightly relax some entry filters to
                surface more signals for backtesting while keeping live logic
                strict.
        """
        self.min_confidence = float(min_confidence)
        self.backtest_mode = bool(backtest_mode)
        self.mr_exit_rules = "RSI(2) > 40 OR 2 consecutive up closes OR 5 days max hold"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate signals from all strategies and combine.

        Returns
        -------
        List[Dict[str, Any]]
            Signals sorted by confidence * expected_return_pct.
        """
        if data.empty:
            logger.info("Empty DataFrame passed to generate_signals; returning no signals.")
            return []

        logger.info("Generating signals from 15 strategies...")

        # Strategy 1: Mean Reversion (existing behaviour)
        mr_signals = self._generate_mean_reversion_signals(data)
        logger.info("  Mean Reversion: %d signals", len(mr_signals))

        # Strategy 2: Momentum Breakout
        momentum_signals = self._generate_momentum_signals(data)
        logger.info("  Momentum Breakout: %d signals", len(momentum_signals))

        # Strategy 3: Ichimoku Cloud
        ichimoku_signals = self._generate_ichimoku_signals(data)
        logger.info("  Ichimoku Cloud: %d signals", len(ichimoku_signals))

        # Strategy 4: Volume Spike Reversal
        volume_spike_signals = self._generate_volume_spike_signals(data)
        logger.info("  Volume Spike: %d signals", len(volume_spike_signals))

        # Strategy 5: Bollinger Squeeze
        bb_squeeze_signals = self._generate_bb_squeeze_signals(data)
        logger.info("  Bollinger Squeeze: %d signals", len(bb_squeeze_signals))

        # Strategy 6: RSI + Bollinger Band confluence
        rsi_bb_signals = self._generate_rsi_bb_confluence_signals(data)
        logger.info("  RSI+BB Confluence: %d signals", len(rsi_bb_signals))

        # Strategy 7: Donchian breakout
        donchian_signals = self._generate_donchian_breakout_signals(data)
        logger.info("  Donchian Breakout: %d signals", len(donchian_signals))

        # Strategy 8: Williams %R reversal
        williams_signals = self._generate_williams_r_signals(data)
        logger.info("  Williams %%R Reversal: %d signals", len(williams_signals))

        # Strategy 9: EMA golden cross
        ema_signals = self._generate_ema_crossover_signals(data)
        logger.info("  EMA Golden Cross: %d signals", len(ema_signals))

        # Strategy 10: Supply/demand zones
        supply_demand_signals = self._generate_supply_demand_signals(data)
        logger.info("  Supply/Demand Zones: %d signals", len(supply_demand_signals))

        # Strategy 11: Gap fill
        gap_fill_signals = self._generate_gap_fill_signals(data)
        logger.info("  Gap Fill: %d signals", len(gap_fill_signals))

        # Strategy 12: Three white soldiers
        soldiers_signals = self._generate_three_soldiers_signals(data)
        logger.info("  Three Soldiers: %d signals", len(soldiers_signals))

        # Strategy 13: Hammer at support
        hammer_signals = self._generate_hammer_signals(data)
        logger.info("  Hammer at Support: %d signals", len(hammer_signals))

        # Strategy 14: Inside bar breakout
        inside_signals = self._generate_inside_bar_signals(data)
        logger.info("  Inside Bar: %d signals", len(inside_signals))

        # Strategy 15: VWAP reversion
        vwap_signals = self._generate_vwap_reversion_signals(data)
        logger.info("  VWAP Reversion: %d signals", len(vwap_signals))

        all_signals: List[Dict[str, Any]] = (
            mr_signals
            + momentum_signals
            + ichimoku_signals
            + volume_spike_signals
            + bb_squeeze_signals
            + rsi_bb_signals
            + donchian_signals
            + williams_signals
            + ema_signals
            + supply_demand_signals
            + gap_fill_signals
            + soldiers_signals
            + hammer_signals
            + inside_signals
            + vwap_signals
        )

        # Resolve conflicts and boost agreement
        all_signals = self._resolve_conflicts(all_signals)

        # Sort by quality score (confidence * expected return)
        if all_signals:
            all_signals.sort(
                key=lambda s: s["confidence"] * s.get("expected_return_pct", 0.0),
                reverse=True,
            )

        logger.info("Total signals generated: %d", len(all_signals))
        return all_signals

    # ------------------------------------------------------------------
    # Strategy 1: Mean Reversion (existing logic, refactored)
    # ------------------------------------------------------------------
    def _generate_mean_reversion_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        required_columns = [
            "date",
            "ticker",
            "close",
            "rsi_2",
            "bb_lower",
            "bb_middle",
            "sma_200",
            "volume_ratio",
        ]

        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            logger.warning("Mean reversion missing columns: %s", missing)
            return []

        cleaned = data.dropna(subset=required_columns).copy()
        if cleaned.empty:
            return []

        signals: List[Dict[str, Any]] = []
        for _, row in cleaned.iterrows():
            if not self._check_mr_entry_conditions(row):
                continue

            confidence = self._calculate_mr_confidence(row)
            if confidence < self.min_confidence:
                continue

            close_price = float(row["close"])
            target_price = float(row["bb_middle"])
            expected_return_pct = float((target_price - close_price) / close_price * 100.0)

            # ATR-based stop: 2x ATR14 below entry, with a minimum 4% stop.
            atr_14 = float(row.get("atr_14", close_price * 0.02))
            stop_loss = close_price - (2.0 * atr_14)
            stop_loss = max(stop_loss, close_price * 0.96)

            signal_date = self._normalize_date(row["date"])
            below_bb_pct = float((row["bb_lower"] - close_price) / close_price * 100.0)
            volume_ratio = float(row["volume_ratio"])
            rsi_2 = float(row["rsi_2"])

            supporting_factors = [
                f"RSI(2): {rsi_2:.1f} (< 30 = oversold)",
                f"Below BB by: {below_bb_pct:.1f}%",
                f"Volume: {volume_ratio:.1f}x average",
                "Above 200 SMA (uptrend confirmed)",
                f"ATR-based stop at ${stop_loss:.2f}",
            ]

            signals.append(
                {
                    "ticker": str(row["ticker"]),
                    "date": signal_date,
                    "signal": "BUY",
                    "entry_price": close_price,
                    "target_price": target_price,
                    "stop_loss": stop_loss,
                    "confidence": confidence,
                    "expected_return_pct": expected_return_pct,
                    "timeframe": "swing",
                    "strategy": "MEAN_REVERSION",
                    "exit_rules": self.mr_exit_rules,
                    "supporting_factors": supporting_factors,
                }
            )

        return signals

    def _check_mr_entry_conditions(self, row: pd.Series) -> bool:
        """Entry rules for mean reversion strategy."""
        try:
            rsi_2 = float(row["rsi_2"])
            close_price = float(row["close"])
            bb_lower = float(row["bb_lower"])
            sma_200 = float(row["sma_200"])
            volume_ratio = float(row["volume_ratio"])
        except (KeyError, TypeError, ValueError):
            return False

        if self.backtest_mode:
            cond_rsi = rsi_2 < 40.0
            cond_bb = close_price < bb_lower
            cond_trend = close_price > sma_200
            cond_volume = volume_ratio > 1.1
        else:
            cond_rsi = rsi_2 < 30.0
            cond_bb = close_price < bb_lower
            cond_trend = close_price > sma_200
            cond_volume = volume_ratio > 1.1

        return cond_rsi and cond_bb and cond_trend and cond_volume

    def _calculate_mr_confidence(self, row: pd.Series) -> float:
        """Confidence score for mean reversion signal.

        Base: 0.70
        +0.05 if rsi_2 < 20
        +0.05 if rsi_2 < 10
        +0.05 if (bb_lower - close) / close > 0.02
        +0.05 if volume_ratio > 2.0
        Capped at 0.95.
        """
        try:
            rsi_2 = float(row["rsi_2"])
            close_price = float(row["close"])
            bb_lower = float(row["bb_lower"])
            volume_ratio = float(row["volume_ratio"])
        except (KeyError, TypeError, ValueError):
            return 0.0

        confidence = 0.70
        if rsi_2 < 20.0:
            confidence += 0.05
        if rsi_2 < 10.0:
            confidence += 0.05

        below_bb_pct = (bb_lower - close_price) / close_price
        if below_bb_pct > 0.02:
            confidence += 0.05
        if volume_ratio > 2.0:
            confidence += 0.05

        return float(np.clip(confidence, 0.0, 0.95))

    # ------------------------------------------------------------------
    # Strategy 2: Momentum Breakout
    # ------------------------------------------------------------------
    def _generate_momentum_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Momentum breakout strategy using 20-day high and ADX confirmation."""
        required_columns = [
            "date",
            "ticker",
            "close",
            "high",
            "volume_ratio",
            "adx_14",
        ]

        missing = [c for c in required_columns if c not in data.columns]
        if missing:
            logger.warning("Momentum breakout missing columns: %s", missing)
            return []

        df = data.copy()
        # 20-day high and 10-day SMA per ticker
        df["high_20"] = df.groupby("ticker")["high"].transform(lambda x: x.rolling(20).max())
        df["sma_10"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(10).mean())
        df["sma_50"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(50).mean())

        signals: List[Dict[str, Any]] = []
        for _, row in df.dropna(subset=["high_20", "sma_10", "sma_50"]).iterrows():
            breakout = float(row["close"]) > float(row["high_20"])
            volume_conf = float(row["volume_ratio"]) > 1.2
            adx_val = float(row.get("adx_14", 0.0))
            strong_trend = adx_val > 25.0
            uptrend = float(row["close"]) > float(row["sma_50"])

            if not (breakout and volume_conf and strong_trend and uptrend):
                continue

            confidence = 0.60
            if adx_val > 30.0:
                confidence += 0.05
            if float(row["volume_ratio"]) > 2.0:
                confidence += 0.05
            if "sma_200" in row and float(row["close"]) > float(row["sma_200"]):
                confidence += 0.05

            confidence = min(0.90, confidence)
            if confidence < self.min_confidence:
                continue

            entry_price = float(row["close"])
            target_price = entry_price * 1.15
            stop_loss = entry_price * 0.93
            expected_return = 15.0

            signals.append(
                {
                    "ticker": str(row["ticker"]),
                    "date": self._normalize_date(row["date"]),
                    "signal": "BUY",
                    "strategy": "MOMENTUM_BREAKOUT",
                    "entry_price": entry_price,
                    "target_price": target_price,
                    "stop_loss": stop_loss,
                    "confidence": confidence,
                    "expected_return_pct": expected_return,
                    "timeframe": "swing",
                    "exit_rules": "Close < 10-SMA OR +15% target OR -7% stop",
                    "supporting_factors": [
                        "20-day high breakout confirmed",
                        f"ADX: {adx_val:.1f} (strong trend)",
                        f"Volume: {float(row['volume_ratio']):.1f}x average",
                        f"Price above SMA(50): ${float(row['sma_50']):.2f}",
                    ],
                }
            )

        return signals

    # ------------------------------------------------------------------
    # Strategy 3: Ichimoku Cloud
    # ------------------------------------------------------------------
    def _generate_ichimoku_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Ichimoku cloud trend continuation strategy."""
        required_columns = [
            "date",
            "ticker",
            "close",
            "ichimoku_conversion",
            "ichimoku_base",
            "ichimoku_span_a",
            "ichimoku_span_b",
            "sma_200",
            "volume_ratio",
        ]

        missing = [c for c in required_columns if c not in data.columns]
        if missing:
            logger.warning("Ichimoku strategy missing columns: %s", missing)
            return []

        signals: List[Dict[str, Any]] = []
        for ticker in data["ticker"].unique():
            ticker_data = data[data["ticker"] == ticker].sort_values("date")
            if len(ticker_data) < 27:
                continue

            # In backtest_mode we only need to evaluate the *latest* bar per
            # ticker, because generate_signals() is called repeatedly on
            # expanding history and run_backtest() filters signals by date.
            if self.backtest_mode:
                i = len(ticker_data) - 1
                idx_range = [i]
            else:
                idx_range = range(26, len(ticker_data))

            for i in idx_range:
                current = ticker_data.iloc[i]
                prev_26 = ticker_data.iloc[i - 26]

                span_a = float(current["ichimoku_span_a"])
                span_b = float(current["ichimoku_span_b"])
                cloud_top = max(span_a, span_b)
                cloud_bottom = min(span_a, span_b)

                above_cloud = float(current["close"]) > cloud_top
                bullish_cross = float(current["ichimoku_conversion"]) > float(current["ichimoku_base"])
                lagging_confirm = float(current["close"]) > float(prev_26["close"])
                green_cloud = span_a > span_b

                if not (above_cloud and bullish_cross and lagging_confirm and green_cloud):
                    continue

                confidence = 0.65
                confidence += 0.05  # all four aligned by construction
                if float(current["close"]) > float(current["sma_200"]):
                    confidence += 0.05
                if float(current["volume_ratio"]) > 1.5:
                    confidence += 0.05
                confidence = min(0.90, confidence)
                if confidence < self.min_confidence:
                    continue

                entry_price = float(current["close"])
                target_price = entry_price * 1.10
                expected_return = 10.0

                signals.append(
                    {
                        "ticker": str(current["ticker"]),
                        "date": self._normalize_date(current["date"]),
                        "signal": "BUY",
                        "strategy": "ICHIMOKU_CLOUD",
                        "entry_price": entry_price,
                        "target_price": target_price,
                        "stop_loss": float(cloud_bottom),
                        "confidence": confidence,
                        "expected_return_pct": expected_return,
                        "timeframe": "swing",
                        "exit_rules": "Price < cloud OR Conversion < Base OR +10% target",
                        "supporting_factors": [
                            "Price above cloud (bullish)",
                            f"Conversion ({float(current['ichimoku_conversion']):.2f}) > Base ({float(current['ichimoku_base']):.2f})",
                            "Lagging span confirms momentum",
                            "Cloud is green (uptrend)",
                        ],
                    }
                )

        return signals

    # ------------------------------------------------------------------
    # Strategy 4: Volume Spike Mean Reversion
    # ------------------------------------------------------------------
    def _generate_volume_spike_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Volume spike + mean reversion combo."""
        required_columns = [
            "date",
            "ticker",
            "close",
            "low",
            "rsi_2",
            "volume_ratio",
            "obv",
            "sma_200",
        ]

        missing = [c for c in required_columns if c not in data.columns]
        if missing:
            logger.warning("Volume spike strategy missing columns: %s", missing)
            return []

        df = data.copy()
        df["low_52w"] = df.groupby("ticker")["low"].transform(
            lambda x: x.rolling(252, min_periods=50).min()
        )
        df["obv_3d_change"] = df.groupby("ticker")["obv"].transform(lambda x: x.diff(3))

        signals: List[Dict[str, Any]] = []
        for _, row in df.dropna(subset=["low_52w"]).iterrows():
            volume_spike = float(row["volume_ratio"]) > 1.8
            extreme_oversold = float(row["rsi_2"]) < 15.0
            near_low = float(row["close"]) < float(row["low_52w"]) * 1.05
            obv_rising = float(row.get("obv_3d_change", 0.0)) > 0.0

            if not (volume_spike and extreme_oversold and near_low and obv_rising):
                continue

            confidence = 0.70
            if float(row["volume_ratio"]) > 5.0:
                confidence += 0.05
            if float(row["rsi_2"]) < 10.0:
                confidence += 0.05
            if float(row["close"]) > float(row["sma_200"]):
                confidence += 0.05
            confidence = min(0.90, confidence)
            if confidence < self.min_confidence:
                continue

            entry_price = float(row["close"])
            target_price = entry_price * 1.08
            expected_return = 8.0

            signals.append(
                {
                    "ticker": str(row["ticker"]),
                    "date": self._normalize_date(row["date"]),
                    "signal": "BUY",
                    "strategy": "VOLUME_SPIKE_REVERSAL",
                    "entry_price": entry_price,
                    "target_price": target_price,
                    "stop_loss": None,
                    "confidence": confidence,
                    "expected_return_pct": expected_return,
                    "timeframe": "swing",
                    "exit_rules": "RSI(2) > 50 OR 3 up closes OR +8% target",
                    "supporting_factors": [
                        f"Volume spike: {float(row['volume_ratio']):.1f}x average (unusual activity)",
                        f"RSI(2): {float(row['rsi_2']):.1f} (extreme panic)",
                        f"Near 52-week low: ${float(row['low_52w']):.2f}",
                        "OBV rising (accumulation despite price drop)",
                    ],
                }
            )

        return signals

    # ------------------------------------------------------------------
    # Strategy 5: Bollinger Band Squeeze
    # ------------------------------------------------------------------
    def _generate_bb_squeeze_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Bollinger Band squeeze breakout strategy."""
        required_columns = [
            "date",
            "ticker",
            "close",
            "bb_upper",
            "bb_lower",
            "volume_ratio",
            "adx_14",
        ]

        missing = [c for c in required_columns if c not in data.columns]
        if missing:
            logger.warning("BB squeeze strategy missing columns: %s", missing)
            return []

        df = data.copy()
        df["bb_width"] = df["bb_upper"] - df["bb_lower"]
        df["bb_width_avg"] = df.groupby("ticker")["bb_width"].transform(
            lambda x: x.rolling(20).mean()
        )
        df["bb_width_ratio"] = df["bb_width"] / df["bb_width_avg"]

        signals: List[Dict[str, Any]] = []
        for _, row in df.dropna(subset=["bb_width_ratio"]).iterrows():
            squeeze = float(row["bb_width_ratio"]) < 0.50
            breakout = (float(row["close"]) > float(row["bb_upper"])) or (
                float(row["close"]) < float(row["bb_lower"])
            )
            volume_conf = float(row["volume_ratio"]) > 1.5
            adx_val = float(row.get("adx_14", 0.0))
            trend_starting = adx_val > 20.0

            bullish_breakout = float(row["close"]) > float(row["bb_upper"])
            if not (squeeze and bullish_breakout and volume_conf and trend_starting and breakout):
                continue

            confidence = 0.68
            if float(row["bb_width_ratio"]) < 0.30:
                confidence += 0.05
            if float(row["volume_ratio"]) > 2.0:
                confidence += 0.05
            if adx_val > 25.0:
                confidence += 0.05
            confidence = min(0.90, confidence)
            if confidence < self.min_confidence:
                continue

            entry_price = float(row["close"])
            target_price = entry_price * 1.12
            stop_loss = entry_price * 0.95
            expected_return = 12.0

            signals.append(
                {
                    "ticker": str(row["ticker"]),
                    "date": self._normalize_date(row["date"]),
                    "signal": "BUY",
                    "strategy": "BOLLINGER_SQUEEZE",
                    "entry_price": entry_price,
                    "target_price": target_price,
                    "stop_loss": stop_loss,
                    "confidence": confidence,
                    "expected_return_pct": expected_return,
                    "timeframe": "swing",
                    "exit_rules": "Close inside bands OR +12% target OR -5% stop",
                    "supporting_factors": [
                        f"Bollinger squeeze: {float(row['bb_width_ratio']):.1%} of average width",
                        f"Upward breakout above ${float(row['bb_upper']):.2f}",
                        f"Volume confirmation: {float(row['volume_ratio']):.1f}x",
                        f"Trend developing: ADX {adx_val:.1f}",
                    ],
                }
            )

        return signals

    # ------------------------------------------------------------------
    # Strategy 6: RSI + Bollinger Band Confluence
    # ------------------------------------------------------------------
    def _generate_rsi_bb_confluence_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """RSI(30) trend + RSI(2) oversold + BB lower confluence."""
        required_columns = [
            "date",
            "ticker",
            "close",
            "rsi_2",
            "bb_lower",
            "volume_ratio",
        ]

        missing = [c for c in required_columns if c not in data.columns]
        if missing:
            logger.warning("RSI+BB confluence missing columns: %s", missing)
            return []

        df = data.copy()

        def _rsi(series: pd.Series, period: int) -> pd.Series:
            delta = series.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=period, min_periods=period).mean()
            avg_loss = loss.rolling(window=period, min_periods=period).mean()
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))

        df["rsi_30"] = df.groupby("ticker")["close"].transform(lambda x: _rsi(x, 30))

        signals: List[Dict[str, Any]] = []
        for _, row in df.dropna(subset=["rsi_30"]).iterrows():
            long_rsi_bullish = float(row.get("rsi_30", 0.0)) > 50.0
            short_rsi_oversold = float(row["rsi_2"]) < 15.0
            below_bb = float(row["close"]) < float(row["bb_lower"])
            volume_conf = float(row["volume_ratio"]) > 1.2

            if not (long_rsi_bullish and short_rsi_oversold and below_bb and volume_conf):
                continue

            confidence = 0.85
            if float(row["rsi_2"]) < 10.0:
                confidence += 0.05
            if float(row["volume_ratio"]) > 2.0:
                confidence += 0.05
            confidence = min(0.95, confidence)
            if confidence < self.min_confidence:
                continue

            entry_price = float(row["close"])
            target_price = entry_price * 1.08

            signals.append(
                {
                    "ticker": str(row["ticker"]),
                    "date": self._normalize_date(row["date"]),
                    "signal": "BUY",
                    "strategy": "RSI_BB_CONFLUENCE",
                    "entry_price": entry_price,
                    "target_price": target_price,
                    "stop_loss": None,
                    "confidence": confidence,
                    "expected_return_pct": 8.0,
                    "timeframe": "swing",
                    "exit_rules": "RSI(2) > 85 OR 3 consecutive up closes",
                    "supporting_factors": [
                        f"RSI(30): {float(row.get('rsi_30', 0.0)):.1f} (bullish trend)",
                        f"RSI(2): {float(row['rsi_2']):.1f} (extreme oversold)",
                        f"Price ${float(row['close']):.2f} < BB Lower ${float(row['bb_lower']):.2f}",
                        f"Volume: {float(row['volume_ratio']):.1f}x confirmation",
                    ],
                }
            )

        return signals

    # ------------------------------------------------------------------
    # Strategy 7: Donchian Channel Breakout
    # ------------------------------------------------------------------
    def _generate_donchian_breakout_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Donchian 20-day breakout with ADX and volume confirmation."""
        required_columns = [
            "date",
            "ticker",
            "close",
            "high",
            "low",
            "volume_ratio",
            "adx_14",
        ]

        missing = [c for c in required_columns if c not in data.columns]
        if missing:
            logger.warning("Donchian breakout missing columns: %s", missing)
            return []

        df = data.copy()
        df["donchian_high_20"] = df.groupby("ticker")["high"].transform(lambda x: x.rolling(20).max())
        df["donchian_low_10"] = df.groupby("ticker")["low"].transform(lambda x: x.rolling(10).min())

        signals: List[Dict[str, Any]] = []
        for _, row in df.dropna(subset=["donchian_high_20", "donchian_low_10"]).iterrows():
            breakout = float(row["close"]) > float(row["donchian_high_20"])
            volume_conf = float(row["volume_ratio"]) > 1.5
            trend_present = float(row.get("adx_14", 0.0)) > 20.0

            if not (breakout and volume_conf and trend_present):
                continue

            adx_val = float(row.get("adx_14", 0.0))
            confidence = 0.70
            if adx_val > 25.0:
                confidence += 0.05
            if float(row["volume_ratio"]) > 2.0:
                confidence += 0.05
            confidence = min(0.80, confidence)
            if confidence < self.min_confidence:
                continue

            entry_price = float(row["close"])
            target_price = entry_price * 1.12
            stop_loss = float(row["donchian_low_10"])

            signals.append(
                {
                    "ticker": str(row["ticker"]),
                    "date": self._normalize_date(row["date"]),
                    "signal": "BUY",
                    "strategy": "DONCHIAN_BREAKOUT",
                    "entry_price": entry_price,
                    "target_price": target_price,
                    "stop_loss": stop_loss,
                    "confidence": confidence,
                    "expected_return_pct": 12.0,
                    "timeframe": "swing",
                    "exit_rules": "Price < 10-day low OR +12% target",
                    "supporting_factors": [
                        f"20-day high breakout: ${float(row['donchian_high_20']):.2f}",
                        f"ADX: {adx_val:.1f} (trend confirmed)",
                        f"Volume: {float(row['volume_ratio']):.1f}x average",
                        f"Stop at 10-day low: ${float(row['donchian_low_10']):.2f}",
                    ],
                }
            )

        return signals

    # ------------------------------------------------------------------
    # Strategy 8: Williams %R Mean Reversion
    # ------------------------------------------------------------------
    def _generate_williams_r_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Williams %R < -80 in uptrend with volume confirmation."""
        required_columns = [
            "date",
            "ticker",
            "close",
            "sma_200",
            "volume_ratio",
        ]

        missing = [c for c in required_columns if c not in data.columns]
        if missing:
            logger.warning("Williams %%R strategy missing columns: %s", missing)
            return []

        if "williams_r" not in data.columns:
            logger.warning("Williams %%R column not present; skip Williams strategy.")
            return []

        df = data.copy()
        signals: List[Dict[str, Any]] = []
        for _, row in df.dropna(subset=["williams_r"]).iterrows():
            wr = float(row.get("williams_r", 0.0))
            oversold = wr < -80.0
            uptrend = float(row["close"]) > float(row["sma_200"])
            volume_conf = float(row["volume_ratio"]) > 1.5

            if not (oversold and uptrend and volume_conf):
                continue

            confidence = 0.70
            if wr < -90.0:
                confidence += 0.05
            if float(row["volume_ratio"]) > 2.0:
                confidence += 0.05
            confidence = min(0.80, confidence)
            if confidence < self.min_confidence:
                continue

            entry_price = float(row["close"])
            target_price = entry_price * 1.06

            signals.append(
                {
                    "ticker": str(row["ticker"]),
                    "date": self._normalize_date(row["date"]),
                    "signal": "BUY",
                    "strategy": "WILLIAMS_R_REVERSAL",
                    "entry_price": entry_price,
                    "target_price": target_price,
                    "stop_loss": None,
                    "confidence": confidence,
                    "expected_return_pct": 6.0,
                    "timeframe": "swing",
                    "exit_rules": "Williams %R > -20 OR +6% target",
                    "supporting_factors": [
                        f"Williams %R: {wr:.1f} (oversold)",
                        f"Above 200-SMA: ${float(row['sma_200']):.2f}",
                        f"Volume: {float(row['volume_ratio']):.1f}x confirmation",
                    ],
                }
            )

        return signals

    # ------------------------------------------------------------------
    # Strategy 9: EMA Golden Cross
    # ------------------------------------------------------------------
    def _generate_ema_crossover_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """EMA(12) crossing above EMA(26) with EMA(50) and volume filters."""
        required_columns = [
            "date",
            "ticker",
            "close",
            "volume_ratio",
            "sma_200",
        ]

        missing = [c for c in required_columns if c not in data.columns]
        if missing:
            logger.warning("EMA crossover missing columns: %s", missing)
            return []

        df = data.copy()
        df["ema_12"] = df.groupby("ticker")["close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
        df["ema_26"] = df.groupby("ticker")["close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())
        df["ema_50"] = df.groupby("ticker")["close"].transform(lambda x: x.ewm(span=50, adjust=False).mean())

        signals: List[Dict[str, Any]] = []
        for ticker in df["ticker"].unique():
            ticker_data = df[df["ticker"] == ticker].sort_values("date")
            if len(ticker_data) < 2:
                continue

            # In backtest_mode evaluate only the latest bar to save time
            if self.backtest_mode:
                idx_iter = [len(ticker_data) - 1]
            else:
                idx_iter = range(1, len(ticker_data))

            for i in idx_iter:
                current = ticker_data.iloc[i]
                prev = ticker_data.iloc[i - 1]

                golden_cross = (
                    float(prev["ema_12"]) <= float(prev["ema_26"]) and float(current["ema_12"]) > float(current["ema_26"])
                )
                uptrend = float(current["close"]) > float(current["ema_50"])
                volume_conf = float(current["volume_ratio"]) > 1.3

                if not (golden_cross and uptrend and volume_conf):
                    continue

                confidence = 0.65
                if float(current["volume_ratio"]) > 2.0:
                    confidence += 0.05
                if float(current["close"]) > float(current["sma_200"]):
                    confidence += 0.05
                confidence = min(0.75, confidence)
                if confidence < self.min_confidence:
                    continue

                entry_price = float(current["close"])
                target_price = entry_price * 1.15
                stop_loss = float(current["ema_50"])

                signals.append(
                    {
                        "ticker": str(current["ticker"]),
                        "date": self._normalize_date(current["date"]),
                        "signal": "BUY",
                        "strategy": "EMA_GOLDEN_CROSS",
                        "entry_price": entry_price,
                        "target_price": target_price,
                        "stop_loss": stop_loss,
                        "confidence": confidence,
                        "expected_return_pct": 15.0,
                        "timeframe": "position",
                        "exit_rules": "EMA(12) < EMA(26) OR +15% target",
                        "supporting_factors": [
                            f"Golden cross: EMA(12) ${float(current['ema_12']):.2f} > EMA(26) ${float(current['ema_26']):.2f}",
                            f"Above EMA(50): ${float(current['ema_50']):.2f}",
                            f"Volume: {float(current['volume_ratio']):.1f}x confirmation",
                        ],
                    }
                )

        return signals

    # ------------------------------------------------------------------
    # Strategy 10: Supply/Demand Zone Reversal (simplified)
    # ------------------------------------------------------------------
    def _generate_supply_demand_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Simplified supply/demand zone detection based on consolidation + rally."""
        required_columns = [
            "date",
            "ticker",
            "open",
            "high",
            "low",
            "close",
            "volume_ratio",
        ]

        missing = [c for c in required_columns if c not in data.columns]
        if missing:
            logger.warning("Supply/demand strategy missing columns: %s", missing)
            return []

        signals: List[Dict[str, Any]] = []
        for ticker in data["ticker"].unique():
            ticker_data = data[data["ticker"] == ticker].sort_values("date").reset_index(drop=True)
            if len(ticker_data) < 30:
                continue

            for i in range(20, len(ticker_data) - 5):
                consolidation = ticker_data.iloc[i - 3 : i]
                if consolidation.empty:
                    continue
                cons_close_mean = float(consolidation["close"].mean())
                cons_range = (float(consolidation["high"].max()) - float(consolidation["low"].min())) / cons_close_mean
                if cons_range > 0.02:
                    continue

                rally = ticker_data.iloc[i : i + 5]
                rally_pct = (float(rally["close"].max()) - cons_close_mean) / cons_close_mean
                if rally_pct < 0.05:
                    continue

                demand_low = float(consolidation["low"].min())
                demand_high = float(consolidation["high"].max())

                future_data = ticker_data.iloc[i + 5 :]
                for _, row in future_data.iterrows():
                    if not (demand_low <= float(row["low"]) <= demand_high):
                        continue

                    candle_range = float(row["high"]) - float(row["low"])
                    if candle_range <= 0:
                        continue
                    close_position = (float(row["close"]) - float(row["low"])) / candle_range
                    if close_position < 0.5:
                        continue
                    if float(row["volume_ratio"]) < 1.5:
                        continue

                    confidence = 0.75
                    if close_position > 0.75:
                        confidence += 0.05
                    if float(row["volume_ratio"]) > 2.0:
                        confidence += 0.05
                    confidence = min(0.85, confidence)
                    if confidence < self.min_confidence:
                        continue

                    entry_price = float(row["close"])
                    target_price = entry_price * 1.10
                    stop_loss = demand_low * 0.98

                    signals.append(
                        {
                            "ticker": str(row["ticker"]),
                            "date": self._normalize_date(row["date"]),
                            "signal": "BUY",
                            "strategy": "SUPPLY_DEMAND_ZONE",
                            "entry_price": entry_price,
                            "target_price": target_price,
                            "stop_loss": float(stop_loss),
                            "confidence": confidence,
                            "expected_return_pct": 10.0,
                            "timeframe": "swing",
                            "exit_rules": "Reaches supply zone OR +10% target OR stop",
                            "supporting_factors": [
                                f"Demand zone: ${demand_low:.2f} - ${demand_high:.2f}",
                                f"Rejection candle: close at {close_position:.0%} of range",
                                f"Volume spike: {float(row['volume_ratio']):.1f}x",
                                f"Stop below zone: ${stop_loss:.2f}",
                            ],
                        }
                    )
                    break

        return signals

    # ------------------------------------------------------------------
    # Strategy 11: Gap Fill
    # ------------------------------------------------------------------
    def _generate_gap_fill_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Gap-down open with recovery toward prior close."""
        required_columns = [
            "date",
            "ticker",
            "open",
            "close",
            "high",
            "low",
        ]

        missing = [c for c in required_columns if c not in data.columns]
        if missing:
            logger.warning("Gap fill strategy missing columns: %s", missing)
            return []

        signals: List[Dict[str, Any]] = []
        for ticker in data["ticker"].unique():
            ticker_data = data[data["ticker"] == ticker].sort_values("date").reset_index(drop=True)
            if len(ticker_data) < 2:
                continue

            if self.backtest_mode:
                idx_iter = [len(ticker_data) - 1]
            else:
                idx_iter = range(1, len(ticker_data))

            for i in idx_iter:
                prev = ticker_data.iloc[i - 1]
                current = ticker_data.iloc[i]
                prev_close = float(prev["close"])
                gap_down = float(current["open"]) < prev_close * 0.98
                recovering = float(current["close"]) > float(current["open"])
                if not (gap_down and recovering):
                    continue

                confidence = 0.65
                entry_price = float(current["close"])
                target_price = min(prev_close, entry_price * 1.04)

                signals.append(
                    {
                        "ticker": str(current["ticker"]),
                        "date": self._normalize_date(current["date"]),
                        "signal": "BUY",
                        "strategy": "GAP_FILL",
                        "entry_price": entry_price,
                        "target_price": target_price,
                        "stop_loss": float(current["low"]),
                        "confidence": confidence,
                        "expected_return_pct": 4.0,
                        "timeframe": "swing",
                        "exit_rules": "Gap 80% filled OR +4% target OR break of low",
                        "supporting_factors": [
                            f"Gap down from ${prev_close:.2f} to open ${float(current['open']):.2f}",
                            "Price recovering intraday",
                        ],
                    }
                )

        return signals

    # ------------------------------------------------------------------
    # Strategy 12: Three White Soldiers
    # ------------------------------------------------------------------
    def _generate_three_soldiers_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Three consecutive strong up candles pattern."""
        required_columns = [
            "date",
            "ticker",
            "open",
            "close",
        ]

        missing = [c for c in required_columns if c not in data.columns]
        if missing:
            logger.warning("Three soldiers strategy missing columns: %s", missing)
            return []

        signals: List[Dict[str, Any]] = []
        for ticker in data["ticker"].unique():
            ticker_data = data[data["ticker"] == ticker].sort_values("date").reset_index(drop=True)
            if len(ticker_data) < 3:
                continue

            if self.backtest_mode:
                idx_iter = [len(ticker_data) - 1]
            else:
                idx_iter = range(2, len(ticker_data))

            for i in idx_iter:
                c1, c2, c3 = ticker_data.iloc[i - 2 : i + 1].itertuples(index=False)
                if not (
                    c1.close > c1.open
                    and c2.close > c2.open
                    and c3.close > c3.open
                    and c2.close > c1.close * 1.01
                    and c3.close > c2.close * 1.01
                ):
                    continue

                confidence = 0.70
                entry_price = float(c3.close)
                target_price = entry_price * 1.05

                signals.append(
                    {
                        "ticker": str(c3.ticker),
                        "date": self._normalize_date(c3.date),
                        "signal": "BUY",
                        "strategy": "THREE_WHITE_SOLDIERS",
                        "entry_price": entry_price,
                        "target_price": target_price,
                        "stop_loss": float(c2.close),
                        "confidence": confidence,
                        "expected_return_pct": 5.0,
                        "timeframe": "swing",
                        "exit_rules": "First down candle OR +5% target",
                        "supporting_factors": [
                            "Three consecutive strong up candles",
                        ],
                    }
                )

        return signals

    # ------------------------------------------------------------------
    # Strategy 13: Hammer at Support (approx. 20-day low)
    # ------------------------------------------------------------------
    def _generate_hammer_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Hammer candle near 20-day support level."""
        required_columns = [
            "date",
            "ticker",
            "open",
            "high",
            "low",
            "close",
        ]

        missing = [c for c in required_columns if c not in data.columns]
        if missing:
            logger.warning("Hammer strategy missing columns: %s", missing)
            return []

        df = data.copy()
        df["low_20"] = df.groupby("ticker")["low"].transform(lambda x: x.rolling(20).min())

        signals: List[Dict[str, Any]] = []
        for _, row in df.dropna(subset=["low_20"]).iterrows():
            support = float(row["low_20"])
            near_support = float(row["low"]) <= support * 1.01
            body = abs(float(row["close"]) - float(row["open"]))
            lower_wick = float(row["close"]) - float(row["low"]) if float(row["close"]) >= float(row["open"]) else float(row["open"]) - float(row["low"])
            upper_wick = float(row["high"]) - max(float(row["close"]), float(row["open"]))
            is_hammer = lower_wick > body * 2 and lower_wick > upper_wick

            if not (near_support and is_hammer):
                continue

            confidence = 0.70
            entry_price = float(row["close"])
            target_price = entry_price * 1.03
            stop_loss = float(row["low"]) * 0.99

            signals.append(
                {
                    "ticker": str(row["ticker"]),
                    "date": self._normalize_date(row["date"]),
                    "signal": "BUY",
                    "strategy": "HAMMER_AT_SUPPORT",
                    "entry_price": entry_price,
                    "target_price": target_price,
                    "stop_loss": stop_loss,
                    "confidence": confidence,
                    "expected_return_pct": 3.0,
                    "timeframe": "swing",
                    "exit_rules": "+3% target OR break below support",
                    "supporting_factors": [
                        f"Hammer near 20-day low support ${support:.2f}",
                    ],
                }
            )

        return signals

    # ------------------------------------------------------------------
    # Strategy 14: Inside Bar Breakout
    # ------------------------------------------------------------------
    def _generate_inside_bar_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Inside bar followed by bullish breakout above mother bar high."""
        required_columns = [
            "date",
            "ticker",
            "high",
            "low",
            "close",
        ]

        missing = [c for c in required_columns if c not in data.columns]
        if missing:
            logger.warning("Inside bar strategy missing columns: %s", missing)
            return []

        signals: List[Dict[str, Any]] = []
        for ticker in data["ticker"].unique():
            ticker_data = data[data["ticker"] == ticker].sort_values("date").reset_index(drop=True)
            if len(ticker_data) < 2:
                continue

            if self.backtest_mode:
                idx_iter = [len(ticker_data) - 1]
            else:
                idx_iter = range(1, len(ticker_data))

            for i in idx_iter:
                prev = ticker_data.iloc[i - 1]
                current = ticker_data.iloc[i]
                inside = float(current["high"]) <= float(prev["high"]) and float(current["low"]) >= float(prev["low"])
                breakout = float(current["close"]) > float(prev["high"])
                if not (inside and breakout):
                    continue

                confidence = 0.70
                entry_price = float(current["close"])
                target_price = entry_price * 1.05

                signals.append(
                    {
                        "ticker": str(current["ticker"]),
                        "date": self._normalize_date(current["date"]),
                        "signal": "BUY",
                        "strategy": "INSIDE_BAR_BREAKOUT",
                        "entry_price": entry_price,
                        "target_price": target_price,
                        "stop_loss": float(prev["low"]),
                        "confidence": confidence,
                        "expected_return_pct": 5.0,
                        "timeframe": "swing",
                        "exit_rules": "+5% target OR back inside range",
                        "supporting_factors": [
                            "Inside bar then bullish breakout",
                        ],
                    }
                )

        return signals

    # ------------------------------------------------------------------
    # Strategy 15: VWAP Reversion
    # ------------------------------------------------------------------
    def _generate_vwap_reversion_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Price extended below VWAP with bounce back toward VWAP."""
        required_columns = [
            "date",
            "ticker",
            "close",
            "vwap",
        ]

        missing = [c for c in required_columns if c not in data.columns]
        if missing:
            logger.warning("VWAP reversion strategy missing columns: %s", missing)
            return []

        signals: List[Dict[str, Any]] = []
        for ticker in data["ticker"].unique():
            ticker_data = data[data["ticker"] == ticker].sort_values("date").reset_index(drop=True)
            if len(ticker_data) < 2:
                continue

            if self.backtest_mode:
                idx_iter = [len(ticker_data) - 1]
            else:
                idx_iter = range(1, len(ticker_data))

            for i in idx_iter:
                prev = ticker_data.iloc[i - 1]
                current = ticker_data.iloc[i]
                if pd.isna(current.get("vwap")):
                    continue
                below_vwap = float(current["close"]) < float(current["vwap"]) * 0.98
                bouncing = float(current["close"]) > float(prev["close"])
                if not (below_vwap and bouncing):
                    continue

                confidence = 0.70
                entry_price = float(current["close"])
                target_price = min(float(current["vwap"]), entry_price * 1.03)

                signals.append(
                    {
                        "ticker": str(current["ticker"]),
                        "date": self._normalize_date(current["date"]),
                        "signal": "BUY",
                        "strategy": "VWAP_REVERSION",
                        "entry_price": entry_price,
                        "target_price": target_price,
                        "stop_loss": float(current["close"]) * 0.97,
                        "confidence": confidence,
                        "expected_return_pct": 3.0,
                        "timeframe": "swing",
                        "exit_rules": "Return to VWAP OR +3% target",
                        "supporting_factors": [
                            f"Price ${entry_price:.2f} extended below VWAP ${float(current['vwap']):.2f}",
                        ],
                    }
                )

        return signals

    # ------------------------------------------------------------------
    # Conflict resolution
    # ------------------------------------------------------------------
    def _resolve_conflicts(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect and resolve conflicts when multiple strategies hit same ticker/date."""
        if not signals:
            return []

        ticker_date_signals: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
        for sig in signals:
            key = (sig["ticker"], sig["date"].date())
            ticker_date_signals[key].append(sig)

        resolved: List[Dict[str, Any]] = []
        for key, sigs in ticker_date_signals.items():
            if len(sigs) == 1:
                resolved.append(sigs[0])
                continue

            directions = {s.get("signal", "BUY") for s in sigs}
            if len(directions) == 1:
                # All agree – boost best
                best = max(sigs, key=lambda s: s.get("confidence", 0.0))
                best["confidence"] = min(0.95, best.get("confidence", 0.0) + 0.10)
                best.setdefault("supporting_factors", []).append(
                    f"✅ {len(sigs)} strategies agree (confidence boosted)"
                )
                resolved.append(best)
            else:
                # Conflicting directions – flag but keep
                for s in sigs:
                    s["conflict_detected"] = True
                    s.setdefault("supporting_factors", []).append(
                        "⚠️ CONFLICT: Other strategies disagree on direction"
                    )
                resolved.extend(sigs)

        return resolved

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_date(value: Any) -> datetime:
        """Normalize various date types to a datetime instance."""
        if isinstance(value, datetime):
            return value
        if isinstance(value, date):
            return datetime.combine(value, time.min)
        return pd.to_datetime(value).to_pydatetime()


if __name__ == "__main__":
    # Simple smoke test for manual runs
    now = datetime.now()
    test_data = pd.DataFrame(
        {
            "date": [now] * 3,
            "ticker": ["TEST1", "TEST2", "TEST3"],
            "close": [95.0, 92.0, 88.0],
            "high": [96.0, 93.0, 89.0],
            "low": [94.0, 91.0, 87.0],
            "rsi_2": [25.0, 18.0, 8.0],
            "bb_lower": [96.0, 93.0, 89.0],
            "bb_middle": [100.0, 100.0, 100.0],
            "bb_upper": [104.0, 104.0, 104.0],
            "sma_200": [90.0, 90.0, 90.0],
            "volume_ratio": [1.8, 2.5, 1.6],
            "adx_14": [30.0, 28.0, 22.0],
            "ichimoku_conversion": [96.0, 93.0, 89.0],
            "ichimoku_base": [95.0, 92.0, 88.0],
            "ichimoku_span_a": [97.0, 94.0, 90.0],
            "ichimoku_span_b": [96.0, 93.0, 89.0],
            "obv": [1000.0, 1100.0, 1200.0],
        }
    )

    generator = EliteSignalGenerator(min_confidence=0.70, backtest_mode=True)
    signals = generator.generate_signals(test_data)

    print(f"Generated {len(signals)} signals")
    for sig in signals:
        print(f"\n{sig['ticker']} - {sig['strategy']}")
        print(f"  Confidence: {sig['confidence']:.0%}")
        print(f"  Expected return: {sig.get('expected_return_pct', 0):.1f}%")
        print(f"  Entry: ${sig['entry_price']:.2f}")
        print(f"  Target: ${sig['target_price']:.2f}")
