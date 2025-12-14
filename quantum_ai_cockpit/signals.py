"""Indicator calculations and scoring utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from .config import STOP_LOSS_PCT, TARGET_PCT
from .utils import fetch_price_history


@dataclass(frozen=True)
class SignalSnapshot:
    """Encapsulates the latest technical indicator values for a ticker."""

    ticker: str
    price: float
    rsi: float
    macd: float
    signal: float
    histogram: float
    sma20: float
    sma50: float
    volume_ratio: float
    momentum_pct: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "ticker": self.ticker,
            "price": self.price,
            "rsi": self.rsi,
            "macd": self.macd,
            "signal": self.signal,
            "histogram": self.histogram,
            "sma20": self.sma20,
            "sma50": self.sma50,
            "volume_ratio": self.volume_ratio,
            "momentum_pct": self.momentum_pct,
        }


def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method="bfill").fillna(50)


def _compute_macd(close: pd.Series) -> pd.DataFrame:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "histogram": histogram})


def calculate_signals(ticker: str, period: str = "6mo") -> SignalSnapshot:
    """Calculate RSI, MACD, SMAs, and volume ratio for the provided ticker."""

    data = fetch_price_history(ticker, period=period, interval="1d")
    if len(data) < 60:
        raise ValueError(f"Not enough data to compute indicators for {ticker}")

    close = data["Close"]
    volume = data["Volume"]

    rsi = _compute_rsi(close)
    macd_df = _compute_macd(close)
    sma20 = close.rolling(window=20).mean()
    sma50 = close.rolling(window=50).mean()
    avg_volume = volume.rolling(window=20).mean()

    latest = data.index[-1]
    price = float(close.loc[latest])
    volume_ratio = float(volume.loc[latest] / avg_volume.loc[latest])
    momentum_pct = float((close.loc[latest] / close.shift(1).loc[latest] - 1) * 100)

    return SignalSnapshot(
        ticker=ticker,
        price=price,
        rsi=float(rsi.loc[latest]),
        macd=float(macd_df.loc[latest, "macd"]),
        signal=float(macd_df.loc[latest, "signal"]),
        histogram=float(macd_df.loc[latest, "histogram"]),
        sma20=float(sma20.loc[latest]),
        sma50=float(sma50.loc[latest]),
        volume_ratio=volume_ratio,
        momentum_pct=momentum_pct,
    )


def score_setup(snapshot: SignalSnapshot) -> float:
    """Score a setup using the validated rubric (0-100)."""

    score = 0.0

    # Volume spike
    vr = snapshot.volume_ratio
    if vr > 3:
        score += 35
    elif vr > 2:
        score += 25
    elif vr > 1.5:
        score += 15
    else:
        score -= 10

    # RSI
    rsi = snapshot.rsi
    if rsi < 40:
        score += 20
    elif 40 <= rsi < 60:
        score += 10
    elif 60 <= rsi <= 75:
        score += 15
    else:  # > 75
        score -= 5

    # MACD
    macd = snapshot.macd
    signal = snapshot.signal
    histogram = snapshot.histogram
    if histogram > 0 and macd > signal:
        score += 25
    elif histogram > 0:
        score += 15
    else:
        score -= 5

    # Price vs SMA
    price = snapshot.price
    above_sma20 = price > snapshot.sma20
    above_sma50 = price > snapshot.sma50
    if above_sma20 and above_sma50:
        score += 15
    elif above_sma20:
        score += 10
    elif not above_sma20 and not above_sma50:
        score -= 5

    # Momentum
    momentum = snapshot.momentum_pct
    if momentum > 3:
        score += 10
    elif momentum > 1:
        score += 5

    return max(0.0, min(100.0, score))


def snapshot_with_score(ticker: str) -> Dict[str, float]:
    """Convenience helper returning indicators, score, and trade levels."""

    snapshot = calculate_signals(ticker)
    score = score_setup(snapshot)
    return {
        **snapshot.as_dict(),
        "score": score,
        "stop_loss_price": snapshot.price * (1 - STOP_LOSS_PCT),
        "target_price": snapshot.price * (1 + TARGET_PCT),
    }
