#!/usr/bin/env python3
"""
HYPOTHESIS BATCH 3: MOMENTUM - ASSET CLASS & CROSS-SECTIONAL (H11-H15, H70-H79)
================================================================================
Cross-asset rotation and momentum strategies.
Est. time: ~10 minutes for 10 tests
API calls: 4 (SPY, TLT, GLD, EFA, EEM, IWM + sectors)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


# ============================================================================
# SIGNAL FUNCTIONS - MOMENTUM
# ============================================================================

def signal_roc_momentum(data: pd.DataFrame, lookback: int = 63, **kwargs) -> pd.Series:
    """H11: Rate of Change (ROC) Momentum."""
    roc = (data['close'] - data['close'].shift(lookback)) / data['close'].shift(lookback)
    return roc


def signal_momentum_breadth(data: pd.DataFrame, sector_data: Dict[str, pd.DataFrame] = None,
                            breadth_threshold: float = 0.6, lookback: int = 252, skip: int = 21, **kwargs) -> pd.Series:
    """H12: Momentum Breadth - % of sectors with positive 12-1m momentum."""
    if sector_data is None or len(sector_data) < 5:
        return pd.Series(0, index=data.index)
    
    # Calculate 12-1 month momentum for each sector
    sector_mom = pd.DataFrame()
    for ticker, df in sector_data.items():
        ret_12m = df['close'].pct_change(lookback)
        ret_1m = df['close'].pct_change(skip)
        sector_mom[ticker] = (ret_12m - ret_1m) > 0
    
    # Align to main data index
    sector_mom = sector_mom.reindex(data.index).ffill()
    
    # Breadth = % of sectors with positive momentum
    breadth = sector_mom.mean(axis=1)
    
    signal = pd.Series(0, index=data.index)
    signal[breadth > breadth_threshold] = 1
    signal[breadth < 0.4] = -1
    return signal


def signal_sector_momentum_top_n(data: pd.DataFrame, sector_data: Dict[str, pd.DataFrame] = None,
                                  top_n: int = 3, lookback: int = 252, skip: int = 21, **kwargs) -> pd.Series:
    """H13: Sector Momentum (Top N) - Long top 3 sectors by 12-1m momentum."""
    if sector_data is None:
        return pd.Series(1, index=data.index)
    
    # Calculate momentum scores
    mom_scores = pd.DataFrame()
    for ticker, df in sector_data.items():
        ret_12m = df['close'].pct_change(lookback)
        ret_1m = df['close'].pct_change(skip)
        mom_scores[ticker] = ret_12m - ret_1m
    
    mom_scores = mom_scores.reindex(data.index).ffill()
    
    # Rank and select top N
    ranks = mom_scores.rank(axis=1, ascending=False)
    is_top_n = (ranks <= top_n).astype(int)
    
    # Return average signal for the universe (or use specific ticker)
    return is_top_n.mean(axis=1)


def signal_international_momentum(data: pd.DataFrame, intl_data: Dict[str, pd.DataFrame] = None,
                                   top_n: int = 3, lookback: int = 126, **kwargs) -> pd.Series:
    """H14: International Momentum - Long top 3 regions by 6-month momentum."""
    if intl_data is None:
        return pd.Series(1, index=data.index)
    
    mom_scores = pd.DataFrame()
    for ticker, df in intl_data.items():
        mom_scores[ticker] = df['close'].pct_change(lookback)
    
    mom_scores = mom_scores.reindex(data.index).ffill()
    ranks = mom_scores.rank(axis=1, ascending=False)
    is_top_n = (ranks <= top_n).astype(int)
    
    return is_top_n.mean(axis=1)


def signal_asset_class_momentum(data: pd.DataFrame, spy_data: pd.DataFrame = None,
                                 tlt_data: pd.DataFrame = None, gld_data: pd.DataFrame = None,
                                 top_n: int = 2, lookback: int = 63, **kwargs) -> pd.Series:
    """H15: Asset Class Momentum - Long top 2 asset classes by 3-month momentum."""
    mom_scores = pd.DataFrame(index=data.index)
    
    # Add each asset's momentum
    assets = {'SPY': spy_data, 'TLT': tlt_data, 'GLD': gld_data}
    for name, asset_data in assets.items():
        if asset_data is not None:
            mom = asset_data['close'].pct_change(lookback).reindex(data.index).ffill()
            mom_scores[name] = mom
    
    if mom_scores.empty:
        return pd.Series(1, index=data.index)
    
    # Rank and select top N
    ranks = mom_scores.rank(axis=1, ascending=False)
    is_top_n = (ranks <= top_n).astype(int)
    
    # Return signal for SPY (or whichever asset is being tested)
    return is_top_n.get('SPY', is_top_n.mean(axis=1))


def signal_us_vs_international(data: pd.DataFrame, efa_data: pd.DataFrame = None,
                                lookback: int = 126, **kwargs) -> pd.Series:
    """H70: US vs International Momentum - Long the leader."""
    if efa_data is None:
        return pd.Series(1, index=data.index)
    
    spy_mom = data['close'].pct_change(lookback)
    efa_mom = efa_data['close'].pct_change(lookback).reindex(data.index).ffill()
    
    # Long SPY if US winning, else reduce/go international
    return (spy_mom > efa_mom).astype(int)


def signal_dm_vs_em(data: pd.DataFrame, efa_data: pd.DataFrame = None, 
                    eem_data: pd.DataFrame = None, lookback: int = 63, **kwargs) -> pd.Series:
    """H71: Developed vs Emerging - Trend follow the DM/EM ratio."""
    if efa_data is None or eem_data is None:
        return pd.Series(1, index=data.index)
    
    efa_price = efa_data['close'].reindex(data.index).ffill()
    eem_price = eem_data['close'].reindex(data.index).ffill()
    
    # DM/EM ratio
    ratio = efa_price / eem_price
    ratio_mom = ratio.pct_change(lookback)
    
    # If DM outperforming (ratio rising), prefer developed
    return (ratio_mom > 0).astype(int)


def signal_growth_vs_value(data: pd.DataFrame, growth_data: pd.DataFrame = None,
                            value_data: pd.DataFrame = None, lookback: int = 63, **kwargs) -> pd.Series:
    """H77: Growth vs Value Rotation - Trend follow the ratio."""
    if growth_data is None or value_data is None:
        return pd.Series(1, index=data.index)
    
    growth = growth_data['close'].reindex(data.index).ffill()
    value = value_data['close'].reindex(data.index).ffill()
    
    ratio = growth / value
    ratio_mom = ratio.pct_change(lookback)
    
    return (ratio_mom > 0).astype(int)


def signal_growth_value_mean_reversion(data: pd.DataFrame, growth_data: pd.DataFrame = None,
                                        value_data: pd.DataFrame = None, lookback: int = 252,
                                        threshold: float = 2.0, **kwargs) -> pd.Series:
    """H78: Growth vs Value Mean Reversion - Mean revert at extremes."""
    if growth_data is None or value_data is None:
        return pd.Series(0, index=data.index)
    
    growth = growth_data['close'].reindex(data.index).ffill()
    value = value_data['close'].reindex(data.index).ffill()
    
    ratio = growth / value
    ratio_mean = ratio.rolling(lookback).mean()
    ratio_std = ratio.rolling(lookback).std()
    zscore = (ratio - ratio_mean) / ratio_std
    
    # Mean revert: buy value when growth extended, vice versa
    signal = pd.Series(0, index=data.index)
    signal[zscore > threshold] = -1   # Growth extended, favor value
    signal[zscore < -threshold] = 1   # Value extended, favor growth
    return signal


def signal_large_vs_small_cap(data: pd.DataFrame, iwm_data: pd.DataFrame = None,
                               lookback: int = 63, **kwargs) -> pd.Series:
    """H79: Large vs Small Cap Rotation - Trend follow size factor."""
    if iwm_data is None:
        return pd.Series(1, index=data.index)
    
    spy_mom = data['close'].pct_change(lookback)
    iwm_mom = iwm_data['close'].pct_change(lookback).reindex(data.index).ffill()
    
    # Long whichever is stronger
    return (spy_mom > iwm_mom).astype(int)


def signal_dual_momentum(data: pd.DataFrame, spy_data: pd.DataFrame = None,
                         tlt_data: pd.DataFrame = None, lookback: int = 252, **kwargs) -> pd.Series:
    """Dual Momentum - Absolute + relative momentum."""
    spy_ret = data['close'].pct_change(lookback) if spy_data is None else spy_data['close'].pct_change(lookback).reindex(data.index).ffill()
    
    # Absolute momentum: positive return?
    abs_mom = spy_ret > 0
    
    # Relative momentum: beating bonds?
    if tlt_data is not None:
        tlt_ret = tlt_data['close'].pct_change(lookback).reindex(data.index).ffill()
        rel_mom = spy_ret > tlt_ret
    else:
        rel_mom = abs_mom
    
    return (abs_mom & rel_mom).astype(int)


# ============================================================================
# HYPOTHESIS DEFINITIONS
# ============================================================================

BATCH_3_HYPOTHESES = [
    {
        'id': 'H11',
        'name': 'ROC Momentum',
        'category': 'Momentum',
        'description': 'Rate of Change - long top quintile',
        'signal_func': signal_roc_momentum,
        'tickers': ['XLK', 'XLV', 'XLF', 'XLE', 'XLI', 'XLU', 'XLP', 'XLY', 'XLB'],
        'hold_period': 21,
        'priority': 2,
    },
    {
        'id': 'H12',
        'name': 'Momentum Breadth',
        'category': 'Momentum',
        'description': '% of sectors with positive 12-1m momentum',
        'signal_func': signal_momentum_breadth,
        'tickers': ['SPY'],
        'hold_period': 21,
        'priority': 2,
    },
    {
        'id': 'H13',
        'name': 'Sector Momentum (Top 3)',
        'category': 'Momentum',
        'description': 'Long top 3 sectors by 12-1m return',
        'signal_func': signal_sector_momentum_top_n,
        'tickers': ['XLK', 'XLV', 'XLF', 'XLE', 'XLI', 'XLU', 'XLP', 'XLY', 'XLB'],
        'parameters': {'top_n': 3},
        'hold_period': 21,
        'priority': 1,
    },
    {
        'id': 'H14',
        'name': 'International Momentum',
        'category': 'Momentum',
        'description': 'Long top 3 regions by 6-month momentum',
        'signal_func': signal_international_momentum,
        'tickers': ['EFA', 'EEM', 'VEA', 'VWO'],
        'parameters': {'top_n': 3},
        'hold_period': 21,
        'priority': 2,
    },
    {
        'id': 'H15',
        'name': 'Asset Class Momentum',
        'category': 'Momentum',
        'description': 'Long top 2 asset classes by 3-month momentum',
        'signal_func': signal_asset_class_momentum,
        'tickers': ['SPY', 'TLT', 'GLD'],
        'parameters': {'top_n': 2},
        'hold_period': 21,
        'priority': 1,
    },
    {
        'id': 'H70',
        'name': 'US vs International',
        'category': 'Momentum',
        'description': 'Long the leader (SPY vs EFA)',
        'signal_func': signal_us_vs_international,
        'tickers': ['SPY', 'EFA'],
        'hold_period': 21,
        'priority': 2,
    },
    {
        'id': 'H71',
        'name': 'Developed vs Emerging',
        'category': 'Momentum',
        'description': 'Trend follow DM/EM ratio',
        'signal_func': signal_dm_vs_em,
        'tickers': ['EFA', 'EEM'],
        'hold_period': 21,
        'priority': 2,
    },
    {
        'id': 'H77',
        'name': 'Growth vs Value Rotation',
        'category': 'Momentum',
        'description': 'Trend follow IWF/IWD ratio',
        'signal_func': signal_growth_vs_value,
        'tickers': ['IWF', 'IWD'],
        'hold_period': 21,
        'priority': 2,
    },
    {
        'id': 'H78',
        'name': 'Growth vs Value Mean Reversion',
        'category': 'Momentum',
        'description': 'Mean revert at Z-score extremes',
        'signal_func': signal_growth_value_mean_reversion,
        'tickers': ['IWF', 'IWD'],
        'hold_period': 21,
        'priority': 3,
    },
    {
        'id': 'H79',
        'name': 'Large vs Small Cap',
        'category': 'Momentum',
        'description': 'Trend follow size factor',
        'signal_func': signal_large_vs_small_cap,
        'tickers': ['SPY', 'IWM'],
        'hold_period': 21,
        'priority': 2,
    },
    {
        'id': 'H05',
        'name': 'Dual Momentum',
        'category': 'Momentum',
        'description': 'Absolute + relative momentum',
        'signal_func': signal_dual_momentum,
        'tickers': ['SPY', 'TLT'],
        'hold_period': 21,
        'priority': 1,  # Published track record
    },
]


def get_batch_3_hypotheses():
    """Return all Batch 3 hypotheses."""
    return BATCH_3_HYPOTHESES


if __name__ == "__main__":
    print(f"Batch 3: Momentum - {len(BATCH_3_HYPOTHESES)} hypotheses")
    for h in BATCH_3_HYPOTHESES:
        print(f"  {h['id']}: {h['name']} (Priority: {h['priority']})")
