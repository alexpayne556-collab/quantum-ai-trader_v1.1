#!/usr/bin/env python3
"""
HYPOTHESIS BATCH 4: MEAN REVERSION (H16-H27)
=============================================
Technical mean reversion signals.
Est. time: ~10 minutes for 12 tests
API calls: 1 (SPY, QQQ, IWM)
"""

import pandas as pd
import numpy as np
from typing import Optional


# ============================================================================
# SIGNAL FUNCTIONS - MEAN REVERSION
# ============================================================================

def signal_weekly_reversal(data: pd.DataFrame, lookback: int = 5, 
                           threshold_pct: float = 20, **kwargs) -> pd.Series:
    """H16: Weekly Reversal - Fade extreme 5-day moves."""
    ret_5d = data['close'].pct_change(lookback)
    
    # Percentile ranking
    percentile = ret_5d.rolling(252).rank(pct=True)
    
    # Fade extremes
    signal = pd.Series(0, index=data.index)
    signal[percentile < threshold_pct / 100] = 1   # Oversold = buy
    signal[percentile > (100 - threshold_pct) / 100] = -1  # Overbought = sell
    return signal


def signal_rsi_mean_reversion(data: pd.DataFrame, period: int = 14,
                               oversold: int = 30, overbought: int = 70, **kwargs) -> pd.Series:
    """H17: RSI Mean Reversion - Buy < 30, Sell > 70."""
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    signal = pd.Series(0, index=data.index)
    signal[rsi < oversold] = 1
    signal[rsi > overbought] = -1
    return signal


def signal_rsi_extreme(data: pd.DataFrame, period: int = 14,
                        oversold: int = 20, overbought: int = 80, **kwargs) -> pd.Series:
    """H18: RSI Extreme (20/80) - More extreme thresholds."""
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    signal = pd.Series(0, index=data.index)
    signal[rsi < oversold] = 1
    signal[rsi > overbought] = -1
    return signal


def signal_bollinger_mean_reversion(data: pd.DataFrame, period: int = 20,
                                     std_dev: float = 2.0, **kwargs) -> pd.Series:
    """H19: Bollinger Band Mean Reversion."""
    sma = data['close'].rolling(period).mean()
    std = data['close'].rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    signal = pd.Series(0, index=data.index)
    signal[data['close'] < lower] = 1   # Below lower band = buy
    signal[data['close'] > upper] = -1  # Above upper band = sell
    return signal


def signal_sector_relative_value(data: pd.DataFrame, spy_data: pd.DataFrame = None,
                                  lookback: int = 63, threshold: float = 2.0, **kwargs) -> pd.Series:
    """H23: Sector Relative Value - Z-score of relative performance."""
    if spy_data is None:
        return pd.Series(0, index=data.index)
    
    spy_price = spy_data['close'].reindex(data.index).ffill()
    ratio = data['close'] / spy_price
    
    ratio_mean = ratio.rolling(lookback).mean()
    ratio_std = ratio.rolling(lookback).std()
    zscore = (ratio - ratio_mean) / ratio_std
    
    # Mean reversion: buy underperformers, sell outperformers
    signal = pd.Series(0, index=data.index)
    signal[zscore < -threshold] = 1   # Underperforming = buy
    signal[zscore > threshold] = -1   # Outperforming = sell
    return signal


def signal_country_mean_reversion(data: pd.DataFrame, efa_data: pd.DataFrame = None,
                                   lookback: int = 63, threshold: float = 2.0, **kwargs) -> pd.Series:
    """H24: Country Mean Reversion - Country vs EFA deviation."""
    if efa_data is None:
        return pd.Series(0, index=data.index)
    
    efa_price = efa_data['close'].reindex(data.index).ffill()
    ratio = data['close'] / efa_price
    
    ratio_mean = ratio.rolling(lookback).mean()
    ratio_std = ratio.rolling(lookback).std()
    zscore = (ratio - ratio_mean) / ratio_std
    
    signal = pd.Series(0, index=data.index)
    signal[zscore < -threshold] = 1
    signal[zscore > threshold] = -1
    return signal


def signal_gap_fill(data: pd.DataFrame, threshold: float = 0.01, **kwargs) -> pd.Series:
    """H26: Gap Fill - Trade toward gap fill."""
    if 'open' not in data.columns:
        return pd.Series(0, index=data.index)
    
    gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Gap down > 1% = expect fill (buy), Gap up > 1% = expect fill (sell)
    signal = pd.Series(0, index=data.index)
    signal[gap < -threshold] = 1   # Gap down = buy for fill
    signal[gap > threshold] = -1   # Gap up = sell for fill
    return signal


def signal_post_large_move_reversal(data: pd.DataFrame, lookback: int = 21,
                                     std_threshold: float = 2.0, **kwargs) -> pd.Series:
    """H27: Post-Large-Move Reversal - Fade 2-sigma moves."""
    daily_ret = data['close'].pct_change()
    rolling_std = daily_ret.rolling(lookback).std()
    zscore = daily_ret / rolling_std
    
    # Fade large moves
    signal = pd.Series(0, index=data.index)
    signal[zscore < -std_threshold] = 1   # Large down move = buy
    signal[zscore > std_threshold] = -1   # Large up move = sell
    return signal


def signal_zscore_mean_reversion(data: pd.DataFrame, lookback: int = 60,
                                  threshold: float = 2.0, **kwargs) -> pd.Series:
    """General Z-score mean reversion."""
    rolling_mean = data['close'].rolling(lookback).mean()
    rolling_std = data['close'].rolling(lookback).std()
    zscore = (data['close'] - rolling_mean) / rolling_std
    
    signal = pd.Series(0, index=data.index)
    signal[zscore < -threshold] = 1
    signal[zscore > threshold] = -1
    return signal


def signal_ma_distance(data: pd.DataFrame, period: int = 50, 
                        threshold: float = 0.05, **kwargs) -> pd.Series:
    """Distance from moving average mean reversion."""
    ma = data['close'].rolling(period).mean()
    distance = (data['close'] - ma) / ma
    
    signal = pd.Series(0, index=data.index)
    signal[distance < -threshold] = 1   # Far below MA = buy
    signal[distance > threshold] = -1   # Far above MA = sell
    return signal


def signal_consecutive_down_reversal(data: pd.DataFrame, n_days: int = 3, **kwargs) -> pd.Series:
    """Buy after N consecutive down days."""
    down = (data['close'] < data['close'].shift(1)).astype(int)
    consecutive_down = down.rolling(n_days).sum()
    return (consecutive_down >= n_days).astype(int)


def signal_oversold_breadth(data: pd.DataFrame, rsi_period: int = 14,
                            oversold: int = 30, **kwargs) -> pd.Series:
    """Multiple indicators oversold."""
    # RSI
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_oversold = rsi < oversold
    
    # Bollinger
    sma = data['close'].rolling(20).mean()
    std = data['close'].rolling(20).std()
    lower_band = sma - 2 * std
    bb_oversold = data['close'] < lower_band
    
    # 5-day return
    ret_5d = data['close'].pct_change(5)
    ret_oversold = ret_5d < ret_5d.rolling(252).quantile(0.1)
    
    # Signal when multiple indicators agree
    oversold_count = rsi_oversold.astype(int) + bb_oversold.astype(int) + ret_oversold.astype(int)
    return (oversold_count >= 2).astype(int)


# ============================================================================
# HYPOTHESIS DEFINITIONS
# ============================================================================

BATCH_4_HYPOTHESES = [
    {
        'id': 'H16',
        'name': 'Weekly Reversal',
        'category': 'Mean Reversion',
        'description': 'Fade extreme 5-day moves',
        'signal_func': signal_weekly_reversal,
        'tickers': ['SPY', 'QQQ'],
        'hold_period': 5,
        'priority': 2,
    },
    {
        'id': 'H17',
        'name': 'RSI Mean Reversion (30/70)',
        'category': 'Mean Reversion',
        'description': 'Buy RSI < 30, Sell > 70',
        'signal_func': signal_rsi_mean_reversion,
        'tickers': ['SPY', 'QQQ', 'IWM'],
        'parameters': {'period': 14, 'oversold': 30, 'overbought': 70},
        'hold_period': 5,
        'priority': 1,
    },
    {
        'id': 'H18',
        'name': 'RSI Extreme (20/80)',
        'category': 'Mean Reversion',
        'description': 'More extreme RSI thresholds',
        'signal_func': signal_rsi_extreme,
        'tickers': ['SPY', 'QQQ', 'IWM'],
        'parameters': {'period': 14, 'oversold': 20, 'overbought': 80},
        'hold_period': 5,
        'priority': 2,
    },
    {
        'id': 'H19',
        'name': 'Bollinger Band Mean Reversion',
        'category': 'Mean Reversion',
        'description': 'Buy at lower band, sell at upper',
        'signal_func': signal_bollinger_mean_reversion,
        'tickers': ['SPY', 'IWM'],
        'parameters': {'period': 20, 'std_dev': 2.0},
        'hold_period': 5,
        'priority': 1,
    },
    {
        'id': 'H23',
        'name': 'Sector Relative Value',
        'category': 'Mean Reversion',
        'description': 'Z-score of sector vs SPY',
        'signal_func': signal_sector_relative_value,
        'tickers': ['XLF', 'XLK', 'XLE', 'XLV', 'XLI'],
        'hold_period': 21,
        'priority': 2,
    },
    {
        'id': 'H24',
        'name': 'Country Mean Reversion',
        'category': 'Mean Reversion',
        'description': 'Country vs EFA deviation',
        'signal_func': signal_country_mean_reversion,
        'tickers': ['EWJ', 'EWG', 'EWU', 'FXI'],
        'hold_period': 21,
        'priority': 3,
    },
    {
        'id': 'H26',
        'name': 'Gap Fill',
        'category': 'Mean Reversion',
        'description': 'Trade toward gap fill',
        'signal_func': signal_gap_fill,
        'tickers': ['SPY', 'QQQ'],
        'hold_period': 1,
        'priority': 3,
    },
    {
        'id': 'H27',
        'name': 'Post-Large-Move Reversal',
        'category': 'Mean Reversion',
        'description': 'Fade 2-sigma daily moves',
        'signal_func': signal_post_large_move_reversal,
        'tickers': ['SPY', 'QQQ'],
        'hold_period': 1,
        'priority': 2,
    },
    {
        'id': 'H27B',
        'name': 'Z-Score Mean Reversion',
        'category': 'Mean Reversion',
        'description': 'Buy at -2 Z-score, sell at +2',
        'signal_func': signal_zscore_mean_reversion,
        'tickers': ['SPY', 'QQQ', 'IWM'],
        'hold_period': 10,
        'priority': 2,
    },
    {
        'id': 'H27C',
        'name': 'MA Distance',
        'category': 'Mean Reversion',
        'description': 'Revert when 5% from 50-day MA',
        'signal_func': signal_ma_distance,
        'tickers': ['SPY', 'QQQ'],
        'hold_period': 10,
        'priority': 3,
    },
    {
        'id': 'H27D',
        'name': 'Consecutive Down Reversal',
        'category': 'Mean Reversion',
        'description': 'Buy after 3 down days',
        'signal_func': signal_consecutive_down_reversal,
        'tickers': ['SPY', 'QQQ'],
        'hold_period': 3,
        'priority': 3,
    },
    {
        'id': 'H27E',
        'name': 'Multi-Indicator Oversold',
        'category': 'Mean Reversion',
        'description': 'RSI + BB + 5d return all oversold',
        'signal_func': signal_oversold_breadth,
        'tickers': ['SPY', 'QQQ', 'IWM'],
        'hold_period': 5,
        'priority': 2,
    },
]


def get_batch_4_hypotheses():
    """Return all Batch 4 hypotheses."""
    return BATCH_4_HYPOTHESES


if __name__ == "__main__":
    print(f"Batch 4: Mean Reversion - {len(BATCH_4_HYPOTHESES)} hypotheses")
    for h in BATCH_4_HYPOTHESES:
        print(f"  {h['id']}: {h['name']} (Priority: {h['priority']})")
