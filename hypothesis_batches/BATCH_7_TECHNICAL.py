#!/usr/bin/env python3
"""
HYPOTHESIS BATCH 7: TECHNICAL PATTERNS (H88-H99)
=================================================
52-week highs/lows, Donchian channels, volume signals.
Est. time: ~10 minutes for 12 tests
API calls: 0 (all calculations from price data)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict


# ============================================================================
# SIGNAL FUNCTIONS - TECHNICAL PATTERNS
# ============================================================================

def signal_52w_high_proximity(data: pd.DataFrame, threshold: float = 0.95, **kwargs) -> pd.Series:
    """H88: 52-Week High Proximity."""
    high_52w = data['high'].rolling(252).max()
    proximity = data['close'] / high_52w
    
    # Near 52w high = strong momentum
    return (proximity > threshold).astype(int)


def signal_52w_low_reversal(data: pd.DataFrame, threshold: float = 0.1, **kwargs) -> pd.Series:
    """H89: 52-Week Low Reversal Signal."""
    low_52w = data['low'].rolling(252).min()
    dist_from_low = (data['close'] - low_52w) / low_52w
    
    # Bounced from 52w low = potential reversal
    was_at_low = dist_from_low.shift(5) < 0.05
    bounced = dist_from_low > threshold
    
    return (was_at_low & bounced).astype(int)


def signal_donchian_breakout(data: pd.DataFrame, lookback: int = 20, **kwargs) -> pd.Series:
    """H90: Donchian Channel Breakout."""
    high_channel = data['high'].rolling(lookback).max()
    low_channel = data['low'].rolling(lookback).min()
    
    # Breakout above channel = bullish
    breakout_up = data['close'] > high_channel.shift(1)
    breakdown = data['close'] < low_channel.shift(1)
    
    signal = pd.Series(1, index=data.index)
    signal[breakdown] = 0
    return signal


def signal_donchian_middle(data: pd.DataFrame, lookback: int = 20, **kwargs) -> pd.Series:
    """H91: Donchian Middle Band Signal."""
    high_channel = data['high'].rolling(lookback).max()
    low_channel = data['low'].rolling(lookback).min()
    middle = (high_channel + low_channel) / 2
    
    # Above middle = uptrend, below = downtrend
    return (data['close'] > middle).astype(int)


def signal_volume_breakout(data: pd.DataFrame, vol_mult: float = 2.0, **kwargs) -> pd.Series:
    """H92: Volume Breakout Signal."""
    if 'volume' not in data.columns:
        return pd.Series(1, index=data.index)
    
    avg_vol = data['volume'].rolling(20).mean()
    high_vol = data['volume'] > avg_vol * vol_mult
    
    price_up = data['close'].pct_change() > 0.01
    
    # High volume + price up = confirmation
    return (high_vol & price_up).astype(int)


def signal_volume_dry_up(data: pd.DataFrame, vol_pct: float = 0.5, **kwargs) -> pd.Series:
    """H93: Volume Dry-Up Signal."""
    if 'volume' not in data.columns:
        return pd.Series(1, index=data.index)
    
    avg_vol = data['volume'].rolling(20).mean()
    low_vol = data['volume'] < avg_vol * vol_pct
    
    price_down = data['close'].pct_change() < 0
    
    # Low volume on decline = weak selling
    return (low_vol & price_down).astype(int)


def signal_obv_trend(data: pd.DataFrame, lookback: int = 21, **kwargs) -> pd.Series:
    """H94: On-Balance Volume Trend."""
    if 'volume' not in data.columns:
        return pd.Series(1, index=data.index)
    
    # Calculate OBV
    price_direction = np.sign(data['close'].diff())
    obv = (price_direction * data['volume']).cumsum()
    
    # OBV trend
    obv_ma = obv.rolling(lookback).mean()
    return (obv > obv_ma).astype(int)


def signal_vwap_trend(data: pd.DataFrame, **kwargs) -> pd.Series:
    """H95: VWAP-based Signal."""
    if 'volume' not in data.columns:
        return pd.Series(1, index=data.index)
    
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    vwap = (typical_price * data['volume']).rolling(20).sum() / data['volume'].rolling(20).sum()
    
    return (data['close'] > vwap).astype(int)


def signal_price_channel(data: pd.DataFrame, lookback: int = 50, pct: float = 0.9, **kwargs) -> pd.Series:
    """H96: Price Channel Position."""
    high = data['high'].rolling(lookback).max()
    low = data['low'].rolling(lookback).min()
    
    channel_position = (data['close'] - low) / (high - low)
    
    # In upper portion of channel = bullish
    return (channel_position > pct).astype(int) | ((channel_position > 0.3) & (channel_position < 0.7)).astype(int)


def signal_atr_expansion(data: pd.DataFrame, lookback: int = 14, mult: float = 1.5, **kwargs) -> pd.Series:
    """H97: ATR Expansion Signal."""
    high_low = data['high'] - data['low']
    high_close = abs(data['high'] - data['close'].shift())
    low_close = abs(data['low'] - data['close'].shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(lookback).mean()
    atr_ma = atr.rolling(50).mean()
    
    # ATR expansion = high volatility regime
    expansion = atr > atr_ma * mult
    
    # During expansion, use mean reversion
    return (~expansion).astype(int)


def signal_keltner_squeeze(data: pd.DataFrame, ema_len: int = 20, atr_mult: float = 1.5, **kwargs) -> pd.Series:
    """H98: Keltner Channel Squeeze."""
    # EMA
    ema = data['close'].ewm(span=ema_len).mean()
    
    # ATR
    high_low = data['high'] - data['low']
    high_close = abs(data['high'] - data['close'].shift())
    low_close = abs(data['low'] - data['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    
    # Keltner bands
    upper = ema + atr_mult * atr
    lower = ema - atr_mult * atr
    
    # Bollinger bands (for squeeze detection)
    bb_std = data['close'].rolling(20).std()
    bb_upper = ema + 2 * bb_std
    bb_lower = ema - 2 * bb_std
    
    # Squeeze = BB inside Keltner
    squeeze = (bb_upper < upper) & (bb_lower > lower)
    
    # Breakout from squeeze = strong move
    return squeeze.astype(int)


def signal_pivot_points(data: pd.DataFrame, **kwargs) -> pd.Series:
    """H99: Pivot Point Support/Resistance."""
    # Calculate pivot from previous day
    pivot = (data['high'].shift() + data['low'].shift() + data['close'].shift()) / 3
    r1 = 2 * pivot - data['low'].shift()
    s1 = 2 * pivot - data['high'].shift()
    
    # Above pivot = bullish bias
    above_pivot = data['close'] > pivot
    above_r1 = data['close'] > r1
    below_s1 = data['close'] < s1
    
    signal = pd.Series(1, index=data.index)
    signal[below_s1] = 0  # Below support
    return signal


# ============================================================================
# HYPOTHESIS DEFINITIONS
# ============================================================================

BATCH_7_HYPOTHESES = [
    {
        'id': 'H88',
        'name': '52W High Proximity',
        'category': 'Technical',
        'description': 'Near 52-week high = strong momentum',
        'signal_func': signal_52w_high_proximity,
        'tickers': ['SPY', 'QQQ'],
        'hold_period': 21,
        'priority': 1,
    },
    {
        'id': 'H89',
        'name': '52W Low Reversal',
        'category': 'Technical',
        'description': 'Bounce from 52-week low',
        'signal_func': signal_52w_low_reversal,
        'tickers': ['SPY', 'QQQ'],
        'hold_period': 10,
        'priority': 2,
    },
    {
        'id': 'H90',
        'name': 'Donchian Breakout',
        'category': 'Technical',
        'description': '20-day channel breakout',
        'signal_func': signal_donchian_breakout,
        'tickers': ['SPY', 'QQQ'],
        'hold_period': 10,
        'priority': 2,
    },
    {
        'id': 'H91',
        'name': 'Donchian Middle',
        'category': 'Technical',
        'description': 'Above/below middle band',
        'signal_func': signal_donchian_middle,
        'tickers': ['SPY', 'QQQ'],
        'hold_period': 5,
        'priority': 3,
    },
    {
        'id': 'H92',
        'name': 'Volume Breakout',
        'category': 'Technical',
        'description': 'High volume + price up',
        'signal_func': signal_volume_breakout,
        'tickers': ['SPY', 'QQQ'],
        'hold_period': 5,
        'priority': 2,
    },
    {
        'id': 'H93',
        'name': 'Volume Dry-Up',
        'category': 'Technical',
        'description': 'Low volume on decline = weak',
        'signal_func': signal_volume_dry_up,
        'tickers': ['SPY', 'QQQ'],
        'hold_period': 5,
        'priority': 3,
    },
    {
        'id': 'H94',
        'name': 'OBV Trend',
        'category': 'Technical',
        'description': 'On-Balance Volume momentum',
        'signal_func': signal_obv_trend,
        'tickers': ['SPY', 'QQQ'],
        'hold_period': 10,
        'priority': 2,
    },
    {
        'id': 'H95',
        'name': 'VWAP Signal',
        'category': 'Technical',
        'description': 'Price vs VWAP',
        'signal_func': signal_vwap_trend,
        'tickers': ['SPY', 'QQQ'],
        'hold_period': 5,
        'priority': 3,
    },
    {
        'id': 'H96',
        'name': 'Price Channel Position',
        'category': 'Technical',
        'description': '50-day channel position',
        'signal_func': signal_price_channel,
        'tickers': ['SPY', 'QQQ'],
        'hold_period': 10,
        'priority': 2,
    },
    {
        'id': 'H97',
        'name': 'ATR Expansion',
        'category': 'Technical',
        'description': 'High volatility regime filter',
        'signal_func': signal_atr_expansion,
        'tickers': ['SPY', 'QQQ'],
        'hold_period': 10,
        'priority': 2,
    },
    {
        'id': 'H98',
        'name': 'Keltner Squeeze',
        'category': 'Technical',
        'description': 'BB inside Keltner = squeeze',
        'signal_func': signal_keltner_squeeze,
        'tickers': ['SPY', 'QQQ'],
        'hold_period': 10,
        'priority': 2,
    },
    {
        'id': 'H99',
        'name': 'Pivot Points',
        'category': 'Technical',
        'description': 'Daily pivot S/R levels',
        'signal_func': signal_pivot_points,
        'tickers': ['SPY', 'QQQ'],
        'hold_period': 1,
        'priority': 3,
    },
]


def get_batch_7_hypotheses():
    """Return all Batch 7 hypotheses."""
    return BATCH_7_HYPOTHESES


if __name__ == "__main__":
    print(f"Batch 7: Technical Patterns - {len(BATCH_7_HYPOTHESES)} hypotheses")
    for h in BATCH_7_HYPOTHESES:
        print(f"  {h['id']}: {h['name']} (Priority: {h['priority']})")
