#!/usr/bin/env python3
"""
HYPOTHESIS BATCH 9: CREATIVE/NOVEL (H128-H145)
===============================================
18 NEW hypotheses: VIX turbulence, divergences, events, regime detection.
Est. time: ~18 minutes for 18 tests
API calls: Minimal (most use existing data)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict


# ============================================================================
# SIGNAL FUNCTIONS - CREATIVE/NOVEL
# ============================================================================

def signal_vix_turbulence(data: pd.DataFrame, vix_data: pd.Series = None,
                          lookback: int = 14, threshold: float = 2.0, **kwargs) -> pd.Series:
    """H128: Volatility of VIX (VIX Turbulence)."""
    if vix_data is None:
        return pd.Series(1, index=data.index)
    
    vix = vix_data.reindex(data.index).ffill()
    vix_changes = vix.diff()
    
    # Rolling std of VIX changes
    vix_vol = vix_changes.rolling(lookback).std()
    vix_vol_mean = vix_vol.rolling(252).mean()
    vix_vol_std = vix_vol.rolling(252).std()
    
    z_score = (vix_vol - vix_vol_mean) / vix_vol_std
    
    # Extreme VIX turbulence = buying opportunity
    extreme_turbulence = z_score > threshold
    
    return extreme_turbulence.astype(int)


def signal_credit_equity_divergence(data: pd.DataFrame, hyg_data: pd.DataFrame = None,
                                     lookback: int = 20, **kwargs) -> pd.Series:
    """H129: Credit-Equity Divergence."""
    if hyg_data is None:
        return pd.Series(1, index=data.index)
    
    spy_ret = data['close'].pct_change(lookback)
    hyg_ret = hyg_data['close'].pct_change(lookback).reindex(data.index).ffill()
    
    # Divergence: HYG down but SPY up (or flat)
    divergence = (hyg_ret < -0.02) & (spy_ret > 0)
    
    # Reduce when diverging (credit leads)
    return (~divergence).astype(int)


def signal_gold_tech_divergence(data: pd.DataFrame, gld_data: pd.DataFrame = None,
                                 qqq_data: pd.DataFrame = None, lookback: int = 5, **kwargs) -> pd.Series:
    """H130: Gold-Tech Divergence (Flight to Quality)."""
    if gld_data is None or qqq_data is None:
        return pd.Series(1, index=data.index)
    
    gld_ret = gld_data['close'].pct_change(lookback).reindex(data.index).ffill()
    qqq_ret = qqq_data['close'].pct_change(lookback).reindex(data.index).ffill()
    
    # Gold up 2%+ while QQQ flat/down = flight to quality
    flight_to_quality = (gld_ret > 0.02) & (qqq_ret < 0.01)
    
    return (~flight_to_quality).astype(int)


def signal_earnings_season(data: pd.DataFrame, **kwargs) -> pd.Series:
    """H131: Earnings Season Seasonality."""
    # Earnings season: mid-Jan, mid-Apr, mid-Jul, mid-Oct (± 3 weeks)
    month = data.index.month
    day = data.index.day
    
    # Earnings windows
    jan_earnings = (month == 1) & (day >= 10) | (month == 2) & (day <= 15)
    apr_earnings = (month == 4) & (day >= 10) | (month == 5) & (day <= 15)
    jul_earnings = (month == 7) & (day >= 10) | (month == 8) & (day <= 15)
    oct_earnings = (month == 10) & (day >= 10) | (month == 11) & (day <= 15)
    
    earnings_season = jan_earnings | apr_earnings | jul_earnings | oct_earnings
    
    # During earnings: higher vol, mean reversion works
    # Signal=1 during earnings (use mean reversion), Signal=0 outside (use momentum)
    return earnings_season.astype(int)


def signal_fomc_drift(data: pd.DataFrame, fomc_dates: list = None, **kwargs) -> pd.Series:
    """H132: FOMC Pre-Announcement Drift."""
    # Key FOMC dates 2024 (example - would need to update annually)
    if fomc_dates is None:
        fomc_dates = [
            '2024-01-31', '2024-03-20', '2024-05-01', '2024-06-12',
            '2024-07-31', '2024-09-18', '2024-11-07', '2024-12-18',
            '2025-01-29', '2025-03-19', '2025-05-07', '2025-06-18',
        ]
    
    fomc_dates = pd.to_datetime(fomc_dates)
    signal = pd.Series(0, index=data.index)
    
    for fomc_date in fomc_dates:
        # Pre-FOMC window: 3 days before
        pre_start = fomc_date - pd.Timedelta(days=5)
        pre_end = fomc_date - pd.Timedelta(days=1)
        
        mask = (data.index >= pre_start) & (data.index <= pre_end)
        signal[mask] = 1
    
    return signal


def signal_rsi_50_cross(data: pd.DataFrame, period: int = 14, **kwargs) -> pd.Series:
    """H133: RSI 50 Crossover (Regime Shift)."""
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # RSI crossing 50 = regime change
    cross_above = (rsi > 50) & (rsi.shift(1) <= 50)
    above_50 = rsi > 50
    
    return above_50.astype(int)


def signal_vwpc(data: pd.DataFrame, threshold: float = 1.5, **kwargs) -> pd.Series:
    """H134: Volume-Weighted Price Change."""
    if 'volume' not in data.columns:
        return pd.Series(1, index=data.index)
    
    price_change = data['close'].pct_change()
    avg_volume = data['volume'].rolling(20).mean()
    
    # VWPC = (Price Change × Volume) / Avg Volume
    vwpc = (price_change * data['volume']) / avg_volume
    vwpc_std = vwpc.rolling(20).std()
    
    # Strong VWPC (volume confirms move)
    strong_up = vwpc > threshold * vwpc_std
    strong_down = vwpc < -threshold * vwpc_std
    
    signal = pd.Series(1, index=data.index)
    signal[strong_down] = 0
    return signal


def signal_sector_breadth_momentum(data: pd.DataFrame, sector_data: Dict[str, pd.DataFrame] = None,
                                    lookback: int = 63, threshold: int = 7, **kwargs) -> pd.Series:
    """H135: Sector Breadth Momentum."""
    # Count sectors with positive 3-month momentum
    if sector_data is None:
        return pd.Series(1, index=data.index)
    
    positive_count = pd.Series(0, index=data.index)
    
    for sector, sdata in sector_data.items():
        if sdata is not None and 'close' in sdata.columns:
            mom = sdata['close'].pct_change(lookback).reindex(data.index).ffill()
            positive_count += (mom > 0).astype(int)
    
    total_sectors = len(sector_data)
    
    # > 7/11 sectors positive = broad strength
    return (positive_count >= threshold).astype(int)


def signal_gap_reversal(data: pd.DataFrame, gap_threshold: float = 0.01, **kwargs) -> pd.Series:
    """H137: Intraday Gap Reversal Pattern."""
    # Gap = difference between open and previous close
    gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Reversal: Gap up but close near low, or gap down but close near high
    day_range = data['high'] - data['low']
    close_position = (data['close'] - data['low']) / day_range
    
    # Gap up reversal: gap > 1%, close in lower 30% of range
    gap_up_reversal = (gap > gap_threshold) & (close_position < 0.3)
    
    # Gap down reversal: gap < -1%, close in upper 30% of range
    gap_down_reversal = (gap < -gap_threshold) & (close_position > 0.7)
    
    # After gap down reversal = buy
    return gap_down_reversal.shift(1).fillna(0).astype(int)


def signal_drawdown_recovery(data: pd.DataFrame, dd_threshold: float = 0.05, **kwargs) -> pd.Series:
    """H139: Drawdown Recovery Pattern."""
    rolling_max = data['close'].expanding().max()
    drawdown = (data['close'] - rolling_max) / rolling_max
    
    # 5%+ drawdown = buying opportunity
    in_drawdown = drawdown < -dd_threshold
    
    # Buy on first day of 5%+ drawdown
    entering_dd = in_drawdown & ~in_drawdown.shift(1).fillna(False)
    
    return entering_dd.astype(int)


def signal_sector_divergence_index(data: pd.DataFrame, sector_data: Dict[str, pd.DataFrame] = None,
                                    lookback: int = 252, **kwargs) -> pd.Series:
    """H140: Sector Divergence Index (Leadership Concentration)."""
    if sector_data is None:
        return pd.Series(1, index=data.index)
    
    returns = {}
    for sector, sdata in sector_data.items():
        if sdata is not None and 'close' in sdata.columns:
            returns[sector] = sdata['close'].pct_change(lookback).reindex(data.index).ffill()
    
    if not returns:
        return pd.Series(1, index=data.index)
    
    returns_df = pd.DataFrame(returns)
    
    # Std dev of sector returns = concentration measure
    divergence = returns_df.std(axis=1)
    divergence_pct = divergence.rolling(252).rank(pct=True)
    
    # High divergence (>75th) = narrow leadership = vulnerable
    return (divergence_pct < 0.75).astype(int)


def signal_dollar_valuation(data: pd.DataFrame, uup_data: pd.DataFrame = None,
                            lookback: int = 252, **kwargs) -> pd.Series:
    """H142: Currency Valuation vs Equity."""
    if uup_data is None:
        return pd.Series(1, index=data.index)
    
    uup = uup_data['close'].reindex(data.index).ffill()
    uup_pct = uup.rolling(lookback).rank(pct=True)
    
    # Weak dollar (<20th percentile) = bullish for equities
    return (uup_pct < 0.5).astype(int)


def signal_commodity_momentum(data: pd.DataFrame, dbc_data: pd.DataFrame = None,
                               lookback: int = 50, **kwargs) -> pd.Series:
    """H143: Commodity Futures Momentum (Broad)."""
    if dbc_data is None:
        return pd.Series(1, index=data.index)
    
    dbc = dbc_data['close'].reindex(data.index).ffill()
    dbc_ma = dbc.rolling(lookback).mean()
    
    # DBC above 50 DMA = commodities in uptrend
    return (dbc > dbc_ma).astype(int)


def signal_volatility_skew(data: pd.DataFrame, lookback: int = 21, **kwargs) -> pd.Series:
    """H144: Volatility Asymmetry (Skew)."""
    daily_ret = data['close'].pct_change()
    
    # Separate up and down day volatility
    up_days = daily_ret[daily_ret > 0]
    down_days = daily_ret[daily_ret < 0]
    
    up_vol = daily_ret.where(daily_ret > 0, np.nan).rolling(lookback, min_periods=5).std()
    down_vol = (-daily_ret).where(daily_ret < 0, np.nan).rolling(lookback, min_periods=5).std()
    
    # Skew = down vol / up vol
    skew_ratio = down_vol / up_vol
    skew_pct = skew_ratio.rolling(252).rank(pct=True)
    
    # High skew = risk priced in = bullish
    return (skew_pct > 0.5).astype(int)


def signal_market_regime_hmm(data: pd.DataFrame, vix_data: pd.Series = None,
                              lookback: int = 63, **kwargs) -> pd.Series:
    """H145: Market Regime Identification (Simple Clustering)."""
    daily_ret = data['close'].pct_change()
    realized_vol = daily_ret.rolling(21).std() * np.sqrt(252)
    
    # Simple regime detection (no HMM, just vol-based)
    vol_percentile = realized_vol.rolling(252).rank(pct=True)
    
    # Regime 1: Low vol (<30th percentile) - momentum works
    # Regime 2: High vol (>70th percentile) - mean reversion works
    # Regime 3: Transition (30-70th)
    
    regime = pd.Series(0, index=data.index)
    regime[vol_percentile < 0.3] = 1  # Low vol
    regime[(vol_percentile >= 0.3) & (vol_percentile <= 0.7)] = 2  # Transition
    regime[vol_percentile > 0.7] = 3  # High vol
    
    # In low vol regime, be fully invested
    # In high vol regime, reduce (but contrarian opportunities)
    return (regime <= 2).astype(int)


def signal_treasury_equity_spread(data: pd.DataFrame, spy_pe: pd.Series = None,
                                   dgs10: pd.Series = None, threshold: float = 0.03, **kwargs) -> pd.Series:
    """H138: Treasury-Equity Spread (Equity Risk Premium)."""
    # This requires external P/E data - simplified version
    if dgs10 is None:
        return pd.Series(1, index=data.index)
    
    # Approximate earnings yield from price
    # Would need actual P/E data for proper calculation
    dgs10_aligned = dgs10.reindex(data.index, method='ffill') / 100
    
    # Placeholder: assume fixed equity yield ~5%
    # In production, get from external source
    equity_yield = 0.05
    
    erp = equity_yield - dgs10_aligned
    
    # ERP > 3% = stocks cheap vs bonds
    return (erp > threshold).astype(int)


def signal_put_call_extreme(data: pd.DataFrame, pcall_data: pd.Series = None,
                            threshold: float = 1.2, **kwargs) -> pd.Series:
    """H136: Put/Call Ratio Extreme."""
    if pcall_data is None:
        return pd.Series(1, index=data.index)
    
    pc_ratio = pcall_data.reindex(data.index, method='ffill')
    
    # P/C > 1.2 = extreme fear = contrarian buy
    extreme_fear = pc_ratio > threshold
    
    return extreme_fear.astype(int)


def signal_economic_surprise(data: pd.DataFrame, surprise_data: pd.Series = None,
                             threshold: float = 50, **kwargs) -> pd.Series:
    """H141: Economic Surprise Index."""
    if surprise_data is None:
        return pd.Series(1, index=data.index)
    
    surprise = surprise_data.reindex(data.index, method='ffill')
    
    # Positive surprises > 50 = economy better than expected
    return (surprise > threshold).astype(int)


# ============================================================================
# HYPOTHESIS DEFINITIONS
# ============================================================================

BATCH_9_HYPOTHESES = [
    {
        'id': 'H128',
        'name': 'VIX Turbulence',
        'category': 'Volatility Meta',
        'description': 'Vol of VIX extreme = buy opportunity',
        'signal_func': signal_vix_turbulence,
        'tickers': ['SPY'],
        'requires_macro': ['VIX'],
        'hold_period': 10,
        'priority': 1,
    },
    {
        'id': 'H129',
        'name': 'Credit-Equity Divergence',
        'category': 'Cross-Asset',
        'description': 'HYG declining + SPY rising = warning',
        'signal_func': signal_credit_equity_divergence,
        'tickers': ['SPY', 'HYG'],
        'hold_period': 10,
        'priority': 1,
    },
    {
        'id': 'H130',
        'name': 'Gold-Tech Divergence',
        'category': 'Cross-Asset',
        'description': 'Flight to quality signal',
        'signal_func': signal_gold_tech_divergence,
        'tickers': ['SPY', 'GLD', 'QQQ'],
        'hold_period': 5,
        'priority': 2,
    },
    {
        'id': 'H131',
        'name': 'Earnings Season',
        'category': 'Calendar',
        'description': 'Higher vol during earnings',
        'signal_func': signal_earnings_season,
        'tickers': ['SPY'],
        'hold_period': 21,
        'priority': 2,
    },
    {
        'id': 'H132',
        'name': 'FOMC Pre-Drift',
        'category': 'Events',
        'description': 'Pre-FOMC bullish drift',
        'signal_func': signal_fomc_drift,
        'tickers': ['SPY'],
        'hold_period': 3,
        'priority': 2,
    },
    {
        'id': 'H133',
        'name': 'RSI 50 Crossover',
        'category': 'Technical',
        'description': 'RSI regime shift signal',
        'signal_func': signal_rsi_50_cross,
        'tickers': ['SPY', 'QQQ'],
        'hold_period': 10,
        'priority': 2,
    },
    {
        'id': 'H134',
        'name': 'Vol-Weighted Price',
        'category': 'Technical',
        'description': 'VWPC confirmation',
        'signal_func': signal_vwpc,
        'tickers': ['SPY', 'QQQ'],
        'hold_period': 5,
        'priority': 2,
    },
    {
        'id': 'H135',
        'name': 'Sector Breadth',
        'category': 'Breadth',
        'description': '>7/11 sectors positive = bullish',
        'signal_func': signal_sector_breadth_momentum,
        'tickers': ['SPY'],
        'hold_period': 21,
        'priority': 1,
    },
    {
        'id': 'H136',
        'name': 'Put/Call Extreme',
        'category': 'Sentiment',
        'description': 'P/C > 1.2 = contrarian buy',
        'signal_func': signal_put_call_extreme,
        'tickers': ['SPY'],
        'hold_period': 5,
        'priority': 2,
    },
    {
        'id': 'H137',
        'name': 'Gap Reversal',
        'category': 'Technical',
        'description': 'Intraday gap reversal pattern',
        'signal_func': signal_gap_reversal,
        'tickers': ['SPY', 'QQQ'],
        'hold_period': 1,
        'priority': 3,
    },
    {
        'id': 'H138',
        'name': 'Equity Risk Premium',
        'category': 'Valuation',
        'description': 'ERP vs history',
        'signal_func': signal_treasury_equity_spread,
        'tickers': ['SPY'],
        'requires_fred': ['DGS10'],
        'hold_period': 63,
        'priority': 2,
    },
    {
        'id': 'H139',
        'name': 'Drawdown Recovery',
        'category': 'Psychology',
        'description': 'Buy 5%+ drawdowns',
        'signal_func': signal_drawdown_recovery,
        'tickers': ['SPY', 'QQQ'],
        'hold_period': 21,
        'priority': 1,
    },
    {
        'id': 'H140',
        'name': 'Sector Divergence',
        'category': 'Breadth',
        'description': 'Leadership concentration warning',
        'signal_func': signal_sector_divergence_index,
        'tickers': ['SPY'],
        'hold_period': 21,
        'priority': 2,
    },
    {
        'id': 'H141',
        'name': 'Economic Surprise',
        'category': 'Macro',
        'description': 'Positive surprises = bullish',
        'signal_func': signal_economic_surprise,
        'tickers': ['SPY'],
        'hold_period': 21,
        'priority': 2,
    },
    {
        'id': 'H142',
        'name': 'Dollar Valuation',
        'category': 'FX',
        'description': 'Weak dollar = equity tailwind',
        'signal_func': signal_dollar_valuation,
        'tickers': ['SPY', 'UUP'],
        'hold_period': 21,
        'priority': 2,
    },
    {
        'id': 'H143',
        'name': 'Commodity Momentum',
        'category': 'Cross-Asset',
        'description': 'DBC trend for equity signal',
        'signal_func': signal_commodity_momentum,
        'tickers': ['SPY', 'DBC'],
        'hold_period': 21,
        'priority': 2,
    },
    {
        'id': 'H144',
        'name': 'Volatility Skew',
        'category': 'Volatility',
        'description': 'Down vol vs up vol ratio',
        'signal_func': signal_volatility_skew,
        'tickers': ['SPY'],
        'hold_period': 10,
        'priority': 2,
    },
    {
        'id': 'H145',
        'name': 'Market Regime HMM',
        'category': 'Regime',
        'description': 'Vol-based regime detection',
        'signal_func': signal_market_regime_hmm,
        'tickers': ['SPY'],
        'hold_period': 21,
        'priority': 1,
    },
]


def get_batch_9_hypotheses():
    """Return all Batch 9 hypotheses."""
    return BATCH_9_HYPOTHESES


if __name__ == "__main__":
    print(f"Batch 9: Creative/Novel - {len(BATCH_9_HYPOTHESES)} hypotheses")
    for h in BATCH_9_HYPOTHESES:
        print(f"  {h['id']}: {h['name']} (Priority: {h['priority']})")
