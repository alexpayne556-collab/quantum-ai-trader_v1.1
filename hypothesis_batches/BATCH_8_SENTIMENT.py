#!/usr/bin/env python3
"""
HYPOTHESIS BATCH 8: SENTIMENT PROXIES (H119-H127)
==================================================
Speculative/defensive ratios, risk appetite proxies.
Est. time: ~10 minutes for 9 tests
API calls: 2-3 (ARKK, XBI, SPHB, SPLV)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict


# ============================================================================
# SIGNAL FUNCTIONS - SENTIMENT PROXIES
# ============================================================================

def signal_arkk_risk_appetite(data: pd.DataFrame, arkk_data: pd.DataFrame = None,
                               lookback: int = 21, **kwargs) -> pd.Series:
    """H119: ARKK as Risk Appetite Proxy."""
    if arkk_data is None:
        return pd.Series(1, index=data.index)
    
    arkk = arkk_data['close'].reindex(data.index).ffill()
    spy = data['close']
    
    # ARKK relative strength
    arkk_mom = arkk.pct_change(lookback)
    spy_mom = spy.pct_change(lookback)
    
    relative_strength = arkk_mom - spy_mom
    
    # ARKK outperforming = risk-on
    return (relative_strength > 0).astype(int)


def signal_arkk_mean_reversion(data: pd.DataFrame, arkk_data: pd.DataFrame = None,
                                lookback: int = 63, **kwargs) -> pd.Series:
    """H120: ARKK Mean Reversion."""
    if arkk_data is None:
        return pd.Series(1, index=data.index)
    
    arkk = arkk_data['close'].reindex(data.index).ffill()
    arkk_ret = arkk.pct_change(lookback)
    
    # Extreme selloff in ARKK = contrarian buy for risk assets
    extreme_low = arkk_ret < arkk_ret.rolling(252).quantile(0.1)
    
    return extreme_low.astype(int) | (arkk_ret > arkk_ret.rolling(252).quantile(0.5)).astype(int)


def signal_xbi_biotech(data: pd.DataFrame, xbi_data: pd.DataFrame = None,
                       lookback: int = 21, **kwargs) -> pd.Series:
    """H121: XBI Biotech Risk Appetite."""
    if xbi_data is None:
        return pd.Series(1, index=data.index)
    
    xbi = xbi_data['close'].reindex(data.index).ffill()
    xbi_mom = xbi.pct_change(lookback)
    
    # XBI positive momentum = risk-on
    return (xbi_mom > 0).astype(int)


def signal_xbi_extreme(data: pd.DataFrame, xbi_data: pd.DataFrame = None,
                       lookback: int = 63, **kwargs) -> pd.Series:
    """H122: XBI Extreme Reading."""
    if xbi_data is None:
        return pd.Series(1, index=data.index)
    
    xbi = xbi_data['close'].reindex(data.index).ffill()
    xbi_ret = xbi.pct_change(lookback)
    
    # Extreme XBI moves = contrarian
    z_score = (xbi_ret - xbi_ret.rolling(252).mean()) / xbi_ret.rolling(252).std()
    
    # Extreme low = buy, extreme high = cautious
    extreme_low = z_score < -2
    extreme_high = z_score > 2
    
    signal = pd.Series(1, index=data.index)
    signal[extreme_low] = 1  # Buy
    signal[extreme_high] = 0  # Cautious
    return signal


def signal_high_beta_low_vol(data: pd.DataFrame, sphb_data: pd.DataFrame = None,
                              splv_data: pd.DataFrame = None, lookback: int = 21, **kwargs) -> pd.Series:
    """H123: High Beta vs Low Vol Ratio."""
    if sphb_data is None or splv_data is None:
        return pd.Series(1, index=data.index)
    
    sphb = sphb_data['close'].reindex(data.index).ffill()
    splv = splv_data['close'].reindex(data.index).ffill()
    
    ratio = sphb / splv
    ratio_mom = ratio.pct_change(lookback)
    
    # SPHB outperforming SPLV = risk-on
    return (ratio_mom > 0).astype(int)


def signal_defensive_rotation(data: pd.DataFrame, xlu_data: pd.DataFrame = None,
                               xlp_data: pd.DataFrame = None, lookback: int = 21, **kwargs) -> pd.Series:
    """H124: Defensive Sector Rotation."""
    if xlu_data is None or xlp_data is None:
        return pd.Series(1, index=data.index)
    
    xlu = xlu_data['close'].reindex(data.index).ffill()
    xlp = xlp_data['close'].reindex(data.index).ffill()
    spy = data['close']
    
    # Defensive relative strength
    def_avg = (xlu + xlp) / 2
    def_rel = def_avg.pct_change(lookback) - spy.pct_change(lookback)
    
    # Defensives outperforming = risk-off
    return (def_rel < 0).astype(int)


def signal_junk_quality_spread(data: pd.DataFrame, hyg_data: pd.DataFrame = None,
                                lqd_data: pd.DataFrame = None, lookback: int = 21, **kwargs) -> pd.Series:
    """H125: Junk vs Quality Bond Spread."""
    if hyg_data is None or lqd_data is None:
        return pd.Series(1, index=data.index)
    
    hyg = hyg_data['close'].reindex(data.index).ffill()
    lqd = lqd_data['close'].reindex(data.index).ffill()
    
    ratio = hyg / lqd
    ratio_mom = ratio.pct_change(lookback)
    
    # HYG outperforming LQD = risk appetite
    return (ratio_mom > 0).astype(int)


def signal_small_large_spread(data: pd.DataFrame, iwm_data: pd.DataFrame = None,
                               lookback: int = 21, **kwargs) -> pd.Series:
    """H126: Small Cap vs Large Cap."""
    if iwm_data is None:
        return pd.Series(1, index=data.index)
    
    iwm = iwm_data['close'].reindex(data.index).ffill()
    spy = data['close']
    
    rel_strength = iwm.pct_change(lookback) - spy.pct_change(lookback)
    
    # Small caps outperforming = risk-on
    return (rel_strength > 0).astype(int)


def signal_sentiment_composite(data: pd.DataFrame, arkk_data: pd.DataFrame = None,
                                xbi_data: pd.DataFrame = None, sphb_data: pd.DataFrame = None,
                                splv_data: pd.DataFrame = None, lookback: int = 21, **kwargs) -> pd.Series:
    """H127: Sentiment Composite Index."""
    signals = []
    
    # ARKK signal
    if arkk_data is not None:
        arkk = arkk_data['close'].reindex(data.index).ffill()
        arkk_mom = arkk.pct_change(lookback)
        signals.append((arkk_mom > 0).astype(int))
    
    # XBI signal
    if xbi_data is not None:
        xbi = xbi_data['close'].reindex(data.index).ffill()
        xbi_mom = xbi.pct_change(lookback)
        signals.append((xbi_mom > 0).astype(int))
    
    # High beta / Low vol
    if sphb_data is not None and splv_data is not None:
        sphb = sphb_data['close'].reindex(data.index).ffill()
        splv = splv_data['close'].reindex(data.index).ffill()
        ratio_mom = (sphb / splv).pct_change(lookback)
        signals.append((ratio_mom > 0).astype(int))
    
    if not signals:
        return pd.Series(1, index=data.index)
    
    composite = pd.concat(signals, axis=1).mean(axis=1)
    return (composite > 0.5).astype(int)


# ============================================================================
# HYPOTHESIS DEFINITIONS
# ============================================================================

BATCH_8_HYPOTHESES = [
    {
        'id': 'H119',
        'name': 'ARKK Risk Appetite',
        'category': 'Sentiment',
        'description': 'ARKK relative strength',
        'signal_func': signal_arkk_risk_appetite,
        'tickers': ['SPY', 'ARKK'],
        'hold_period': 10,
        'priority': 2,
    },
    {
        'id': 'H120',
        'name': 'ARKK Mean Reversion',
        'category': 'Sentiment',
        'description': 'Extreme ARKK selloff = buy',
        'signal_func': signal_arkk_mean_reversion,
        'tickers': ['SPY', 'ARKK'],
        'hold_period': 21,
        'priority': 2,
    },
    {
        'id': 'H121',
        'name': 'XBI Risk Appetite',
        'category': 'Sentiment',
        'description': 'Biotech momentum',
        'signal_func': signal_xbi_biotech,
        'tickers': ['SPY', 'XBI'],
        'hold_period': 10,
        'priority': 2,
    },
    {
        'id': 'H122',
        'name': 'XBI Extreme',
        'category': 'Sentiment',
        'description': 'Biotech extreme readings',
        'signal_func': signal_xbi_extreme,
        'tickers': ['SPY', 'XBI'],
        'hold_period': 21,
        'priority': 2,
    },
    {
        'id': 'H123',
        'name': 'High Beta vs Low Vol',
        'category': 'Sentiment',
        'description': 'SPHB/SPLV ratio momentum',
        'signal_func': signal_high_beta_low_vol,
        'tickers': ['SPY', 'SPHB', 'SPLV'],
        'hold_period': 10,
        'priority': 1,
    },
    {
        'id': 'H124',
        'name': 'Defensive Rotation',
        'category': 'Sentiment',
        'description': 'XLU/XLP relative strength',
        'signal_func': signal_defensive_rotation,
        'tickers': ['SPY', 'XLU', 'XLP'],
        'hold_period': 21,
        'priority': 2,
    },
    {
        'id': 'H125',
        'name': 'Junk vs Quality',
        'category': 'Sentiment',
        'description': 'HYG/LQD ratio',
        'signal_func': signal_junk_quality_spread,
        'tickers': ['SPY', 'HYG', 'LQD'],
        'hold_period': 10,
        'priority': 2,
    },
    {
        'id': 'H126',
        'name': 'Small vs Large Cap',
        'category': 'Sentiment',
        'description': 'IWM/SPY relative strength',
        'signal_func': signal_small_large_spread,
        'tickers': ['SPY', 'IWM'],
        'hold_period': 21,
        'priority': 2,
    },
    {
        'id': 'H127',
        'name': 'Sentiment Composite',
        'category': 'Sentiment',
        'description': 'Combined risk appetite signals',
        'signal_func': signal_sentiment_composite,
        'tickers': ['SPY'],
        'hold_period': 10,
        'priority': 1,
    },
]


def get_batch_8_hypotheses():
    """Return all Batch 8 hypotheses."""
    return BATCH_8_HYPOTHESES


if __name__ == "__main__":
    print(f"Batch 8: Sentiment Proxies - {len(BATCH_8_HYPOTHESES)} hypotheses")
    for h in BATCH_8_HYPOTHESES:
        print(f"  {h['id']}: {h['name']} (Priority: {h['priority']})")
