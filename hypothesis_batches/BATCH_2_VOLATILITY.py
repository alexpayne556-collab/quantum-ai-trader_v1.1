#!/usr/bin/env python3
"""
HYPOTHESIS BATCH 2: VOLATILITY (H20-H41)
=========================================
VIX-based signals - your core strength area.
Est. time: ~12 minutes for 14 tests
API calls: 3 (^VIX, ^VIX3M, ^SKEW)
"""

import pandas as pd
import numpy as np
from typing import Optional


# ============================================================================
# SIGNAL FUNCTIONS - VOLATILITY
# ============================================================================

def signal_vix_mean_reversion(data: pd.DataFrame, vix_data: pd.Series = None, 
                               threshold_high: float = 25, threshold_low: float = 12, **kwargs) -> pd.Series:
    """H20: VIX Mean Reversion - Buy when high, reduce when low."""
    if vix_data is None:
        return pd.Series(0, index=data.index)
    
    vix = vix_data.reindex(data.index).ffill()
    
    # Signal: 1 = buy (high VIX), 0 = neutral, -1 = reduce (low VIX)
    signal = pd.Series(0, index=data.index)
    signal[vix > threshold_high] = 1
    signal[vix < threshold_low] = -1
    return signal


def signal_vix_percentile(data: pd.DataFrame, vix_data: pd.Series = None,
                          percentile_threshold: float = 90, lookback: int = 252, **kwargs) -> pd.Series:
    """H21: VIX Percentile - Buy at 90th+ percentile VIX."""
    if vix_data is None:
        return pd.Series(0, index=data.index)
    
    vix = vix_data.reindex(data.index).ffill()
    percentile = vix.rolling(lookback).rank(pct=True) * 100
    
    return (percentile >= percentile_threshold).astype(int)


def signal_vix_spike_reversal(data: pd.DataFrame, vix_data: pd.Series = None,
                               spike_pct: float = 20, **kwargs) -> pd.Series:
    """H22: VIX Spike Reversal - Buy after 20%+ VIX spike."""
    if vix_data is None:
        return pd.Series(0, index=data.index)
    
    vix = vix_data.reindex(data.index).ffill()
    vix_change = vix.pct_change() * 100
    
    return (vix_change > spike_pct).astype(int)


def signal_vix_term_structure(data: pd.DataFrame, vix: pd.Series = None, 
                               vix3m: pd.Series = None, **kwargs) -> pd.Series:
    """H28: VIX Term Structure - Long when contango, reduce in backwardation."""
    if vix is None or vix3m is None:
        return pd.Series(0, index=data.index)
    
    vix_aligned = vix.reindex(data.index).ffill()
    vix3m_aligned = vix3m.reindex(data.index).ffill()
    
    ratio = vix_aligned / vix3m_aligned
    
    # Contango (ratio < 0.9) = bullish, Backwardation (ratio > 1) = bearish
    signal = pd.Series(0, index=data.index)
    signal[ratio < 0.9] = 1   # Strong contango = long
    signal[ratio > 1.0] = -1  # Backwardation = reduce
    return signal


def signal_vix_term_extreme(data: pd.DataFrame, vix: pd.Series = None,
                             vix3m: pd.Series = None, **kwargs) -> pd.Series:
    """H29: VIX/VIX3M Extreme - Contrarian at extreme readings."""
    if vix is None or vix3m is None:
        return pd.Series(0, index=data.index)
    
    vix_aligned = vix.reindex(data.index).ffill()
    vix3m_aligned = vix3m.reindex(data.index).ffill()
    
    ratio = vix_aligned / vix3m_aligned
    
    # Panic (> 1.15) = contrarian buy, Complacency (< 0.85) = reduce
    signal = pd.Series(0, index=data.index)
    signal[ratio > 1.15] = 1   # Panic = buy
    signal[ratio < 0.85] = -1  # Complacency = cautious
    return signal


def signal_skew_index(data: pd.DataFrame, skew_data: pd.Series = None,
                      threshold: float = 150, **kwargs) -> pd.Series:
    """H31: SKEW Index Signal - Reduce when SKEW > 150."""
    if skew_data is None:
        return pd.Series(0, index=data.index)
    
    skew = skew_data.reindex(data.index).ffill()
    return (skew < threshold).astype(int)


def signal_volatility_targeting(data: pd.DataFrame, target_vol: float = 0.10,
                                 lookback: int = 21, **kwargs) -> pd.Series:
    """H32/H33: Volatility Targeting - Scale exposure by vol."""
    returns = data['close'].pct_change()
    realized_vol = returns.rolling(lookback).std() * np.sqrt(252)
    
    # Position size = target / realized (capped at 2x)
    leverage = target_vol / realized_vol.clip(lower=0.05)
    leverage = leverage.clip(upper=2.0)
    
    return leverage


def signal_vol_regime_filter(data: pd.DataFrame, vix_data: pd.Series = None,
                              threshold: float = 20, **kwargs) -> pd.Series:
    """H34: Volatility Regime Filter - Reduce in high vol."""
    if vix_data is None:
        # Use realized vol as fallback
        returns = data['close'].pct_change()
        vol = returns.rolling(21).std() * np.sqrt(252) * 100
        return (vol < threshold).astype(int)
    
    vix = vix_data.reindex(data.index).ffill()
    return (vix < threshold).astype(int)


def signal_low_vol_anomaly(data: pd.DataFrame, **kwargs) -> pd.Series:
    """H35: Low Volatility Anomaly - Low vol outperforms risk-adjusted."""
    returns = data['close'].pct_change()
    vol = returns.rolling(60).std()
    median_vol = vol.rolling(252).median()
    
    return (vol < median_vol).astype(int)


def signal_vol_breakout(data: pd.DataFrame, vol_lookback: int = 21, 
                        compression_lookback: int = 126, **kwargs) -> pd.Series:
    """H36: Volatility Breakout - Vol at 6-month low predicts expansion."""
    returns = data['close'].pct_change()
    vol = returns.rolling(vol_lookback).std()
    vol_min = vol.rolling(compression_lookback).min()
    
    # At compression (vol near its low)
    compression = vol <= vol_min * 1.05
    
    # Add momentum direction
    mom = data['close'].pct_change(21)
    
    # Long if compressed + positive momentum
    return (compression & (mom > 0)).astype(int)


def signal_atr_breakout(data: pd.DataFrame, lookback: int = 14, 
                        multiplier: float = 2.0, **kwargs) -> pd.Series:
    """H37: ATR Breakout - Daily move > 2× ATR."""
    high = data['high'] if 'high' in data.columns else data['close']
    low = data['low'] if 'low' in data.columns else data['close']
    close = data['close']
    
    # True Range
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(lookback).mean()
    daily_move = (close - close.shift(1)).abs()
    
    # Breakout = move > 2× ATR
    breakout = daily_move > multiplier * atr
    direction = close > close.shift(1)
    
    return (breakout & direction).astype(int)


def signal_vrp(data: pd.DataFrame, vix_data: pd.Series = None,
               lookback: int = 21, **kwargs) -> pd.Series:
    """H38: Implied vs Realized Vol Spread (VRP) - Large spread = sell vol."""
    if vix_data is None:
        return pd.Series(0, index=data.index)
    
    vix = vix_data.reindex(data.index).ffill()
    
    # Realized vol (annualized)
    returns = data['close'].pct_change()
    realized_vol = returns.rolling(lookback).std() * np.sqrt(252) * 100
    
    # VRP = implied - realized
    vrp = vix - realized_vol
    
    # Large positive VRP = sell vol / be long equity
    return (vrp > 5).astype(int)


def signal_cross_asset_vol(data: pd.DataFrame, vix_data: pd.Series = None,
                            gld_data: pd.DataFrame = None, 
                            tlt_data: pd.DataFrame = None, **kwargs) -> pd.Series:
    """H39: Cross-Asset Vol Signal - Equity vol spiking alone = fear."""
    if vix_data is None:
        return pd.Series(0, index=data.index)
    
    vix = vix_data.reindex(data.index).ffill()
    vix_spike = vix.pct_change(5) > 0.20  # 20% spike in 5 days
    
    # Check if other assets stable
    other_stable = True
    if gld_data is not None:
        gld_vol = gld_data['close'].pct_change().rolling(21).std()
        gld_vol_change = gld_vol.pct_change(5).reindex(data.index).ffill()
        other_stable = other_stable & (gld_vol_change < 0.20)
    
    # Equity-specific fear = contrarian buy
    return (vix_spike & other_stable).astype(int)


def signal_correlation_regime(data: pd.DataFrame, sector_data: dict = None,
                               threshold: float = 0.8, **kwargs) -> pd.Series:
    """H40: Correlation Regime - High correlation = crisis."""
    # Use rolling correlation with SPY as proxy
    if sector_data is None or len(sector_data) < 3:
        # Fallback: use price momentum dispersion
        mom = data['close'].pct_change(21)
        return (mom > 0).astype(int)
    
    # Calculate average pairwise correlation
    returns_df = pd.DataFrame()
    for ticker, df in sector_data.items():
        returns_df[ticker] = df['close'].pct_change()
    
    # Rolling correlation matrix
    def avg_corr(window):
        if len(window) < 20:
            return np.nan
        corr_matrix = window.corr()
        n = len(corr_matrix)
        if n < 2:
            return np.nan
        # Average off-diagonal correlation
        mask = ~np.eye(n, dtype=bool)
        return corr_matrix.values[mask].mean()
    
    avg_correlation = returns_df.rolling(63).apply(
        lambda x: x.corr().values[~np.eye(len(x.corr()), dtype=bool)].mean() if len(x) > 20 else np.nan
    ).mean(axis=1)
    
    # Low correlation = diversification works = long
    return (avg_correlation < threshold).astype(int)


def signal_vol_of_vol(data: pd.DataFrame, vix_data: pd.Series = None,
                      lookback: int = 21, **kwargs) -> pd.Series:
    """H41: Vol of Vol (VVIX Proxy) - High vol-of-vol = uncertainty."""
    if vix_data is None:
        return pd.Series(0, index=data.index)
    
    vix = vix_data.reindex(data.index).ffill()
    vix_changes = vix.pct_change()
    vol_of_vol = vix_changes.rolling(lookback).std()
    vol_of_vol_median = vol_of_vol.rolling(252).median()
    
    # High vol-of-vol = reduce
    return (vol_of_vol < vol_of_vol_median).astype(int)


# ============================================================================
# HYPOTHESIS DEFINITIONS
# ============================================================================

BATCH_2_HYPOTHESES = [
    {
        'id': 'H20',
        'name': 'VIX Mean Reversion',
        'category': 'Volatility',
        'description': 'VIX > 25 buy, < 12 reduce',
        'signal_func': signal_vix_mean_reversion,
        'tickers': ['SPY'],
        'requires_macro': ['VIX'],
        'hold_period': 5,
        'priority': 1,
    },
    {
        'id': 'H21',
        'name': 'VIX Percentile',
        'category': 'Volatility',
        'description': 'Buy at 90th+ percentile VIX',
        'signal_func': signal_vix_percentile,
        'tickers': ['SPY'],
        'requires_macro': ['VIX'],
        'hold_period': 10,
        'priority': 1,
    },
    {
        'id': 'H22',
        'name': 'VIX Spike Reversal',
        'category': 'Volatility',
        'description': 'Buy after 20%+ VIX spike',
        'signal_func': signal_vix_spike_reversal,
        'tickers': ['SPY'],
        'requires_macro': ['VIX'],
        'hold_period': 5,
        'priority': 2,
    },
    {
        'id': 'H28',
        'name': 'VIX Term Structure',
        'category': 'Volatility',
        'description': 'Contango bullish, backwardation bearish',
        'signal_func': signal_vix_term_structure,
        'tickers': ['SPY'],
        'requires_macro': ['VIX', 'VIX3M'],
        'hold_period': 5,
        'priority': 1,
    },
    {
        'id': 'H29',
        'name': 'VIX/VIX3M Extreme',
        'category': 'Volatility',
        'description': 'Contrarian at panic/complacency',
        'signal_func': signal_vix_term_extreme,
        'tickers': ['SPY'],
        'requires_macro': ['VIX', 'VIX3M'],
        'hold_period': 10,
        'priority': 2,
    },
    {
        'id': 'H31',
        'name': 'SKEW Index Signal',
        'category': 'Volatility',
        'description': 'Reduce when SKEW > 150',
        'signal_func': signal_skew_index,
        'tickers': ['SPY'],
        'requires_macro': ['SKEW'],
        'hold_period': 10,
        'priority': 3,
    },
    {
        'id': 'H32',
        'name': 'Volatility Targeting (10%)',
        'category': 'Volatility',
        'description': 'Scale by 10% / realized vol',
        'signal_func': signal_volatility_targeting,
        'tickers': ['SPY'],
        'parameters': {'target_vol': 0.10},
        'hold_period': 21,
        'priority': 1,  # Well-documented
    },
    {
        'id': 'H33',
        'name': 'Volatility Targeting (15%)',
        'category': 'Volatility',
        'description': 'Scale by 15% / realized vol',
        'signal_func': signal_volatility_targeting,
        'tickers': ['SPY'],
        'parameters': {'target_vol': 0.15},
        'hold_period': 21,
        'priority': 1,
    },
    {
        'id': 'H34',
        'name': 'Volatility Regime Filter',
        'category': 'Volatility',
        'description': 'VIX > 20 = reduce position',
        'signal_func': signal_vol_regime_filter,
        'tickers': ['SPY', 'QQQ'],
        'requires_macro': ['VIX'],
        'hold_period': 5,
        'priority': 2,
    },
    {
        'id': 'H35',
        'name': 'Low Volatility Anomaly',
        'category': 'Volatility',
        'description': 'Low vol outperforms risk-adjusted',
        'signal_func': signal_low_vol_anomaly,
        'tickers': ['SPLV', 'SPHB'],
        'hold_period': 21,
        'priority': 2,
    },
    {
        'id': 'H36',
        'name': 'Volatility Breakout',
        'category': 'Volatility',
        'description': 'Vol at 6-month low = expect expansion',
        'signal_func': signal_vol_breakout,
        'tickers': ['SPY', 'QQQ'],
        'hold_period': 10,
        'priority': 3,
    },
    {
        'id': 'H37',
        'name': 'ATR Breakout',
        'category': 'Volatility',
        'description': 'Daily move > 2× ATR',
        'signal_func': signal_atr_breakout,
        'tickers': ['SPY', 'QQQ', 'IWM'],
        'hold_period': 5,
        'priority': 3,
    },
    {
        'id': 'H38',
        'name': 'Implied vs Realized Vol (VRP)',
        'category': 'Volatility',
        'description': 'Large VRP = sell vol / long equity',
        'signal_func': signal_vrp,
        'tickers': ['SPY'],
        'requires_macro': ['VIX'],
        'hold_period': 21,
        'priority': 2,
    },
    {
        'id': 'H39',
        'name': 'Cross-Asset Vol Signal',
        'category': 'Volatility',
        'description': 'Equity vol spike alone = fear',
        'signal_func': signal_cross_asset_vol,
        'tickers': ['SPY'],
        'requires_macro': ['VIX'],
        'hold_period': 5,
        'priority': 3,
    },
    {
        'id': 'H40',
        'name': 'Correlation Regime',
        'category': 'Volatility',
        'description': 'High sector correlation = crisis',
        'signal_func': signal_correlation_regime,
        'tickers': ['SPY'],
        'hold_period': 21,
        'priority': 3,
    },
    {
        'id': 'H41',
        'name': 'Vol of Vol (VVIX Proxy)',
        'category': 'Volatility',
        'description': 'High vol-of-vol = uncertainty',
        'signal_func': signal_vol_of_vol,
        'tickers': ['SPY'],
        'requires_macro': ['VIX'],
        'hold_period': 10,
        'priority': 3,
    },
]


def get_batch_2_hypotheses():
    """Return all Batch 2 hypotheses."""
    return BATCH_2_HYPOTHESES


if __name__ == "__main__":
    print(f"Batch 2: Volatility - {len(BATCH_2_HYPOTHESES)} hypotheses")
    for h in BATCH_2_HYPOTHESES:
        print(f"  {h['id']}: {h['name']} (Priority: {h['priority']})")
