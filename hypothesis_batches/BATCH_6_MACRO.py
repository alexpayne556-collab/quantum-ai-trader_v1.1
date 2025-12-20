#!/usr/bin/env python3
"""
HYPOTHESIS BATCH 6: YIELD CURVE & MACRO (H65-H68, H100-H110)
=============================================================
Interest rates, yield curve, economic indicators.
Est. time: ~15 minutes for 14 tests
API calls: FRED (DGS10, DGS2, T10YIE), manual data
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict


# ============================================================================
# SIGNAL FUNCTIONS - YIELD CURVE
# ============================================================================

def signal_yield_curve_slope(data: pd.DataFrame, dgs10: pd.Series = None,
                              dgs2: pd.Series = None, **kwargs) -> pd.Series:
    """H65: Yield Curve Slope (10Y-2Y)."""
    if dgs10 is None or dgs2 is None:
        return pd.Series(1, index=data.index)
    
    spread = (dgs10 - dgs2).reindex(data.index).ffill()
    
    # Normal slope = bullish, inverted = warning
    return (spread > 0).astype(int)


def signal_curve_inversion_lag(data: pd.DataFrame, dgs10: pd.Series = None,
                                dgs2: pd.Series = None, lag_days: int = 252, **kwargs) -> pd.Series:
    """H66: Curve Inversion with Lag - Recession signal delayed."""
    if dgs10 is None or dgs2 is None:
        return pd.Series(1, index=data.index)
    
    spread = (dgs10 - dgs2).reindex(data.index).ffill()
    inverted = spread < 0
    
    # Inversion started signal (first day inverted after being normal)
    inversion_start = inverted & ~inverted.shift(1).fillna(False)
    
    # Stay cautious for lag_days after inversion
    cautious = inversion_start.rolling(lag_days).sum() > 0
    
    return (~cautious).astype(int)


def signal_real_yields(data: pd.DataFrame, dgs10: pd.Series = None,
                       t10yie: pd.Series = None, threshold: float = 1.5, **kwargs) -> pd.Series:
    """H67: Real Yields Signal - High real yields = equity headwind."""
    if dgs10 is None or t10yie is None:
        return pd.Series(1, index=data.index)
    
    # Real yield = nominal - inflation expectations
    real_yield = (dgs10 - t10yie).reindex(data.index).ffill()
    
    # High real yields = competition for equities
    return (real_yield < threshold).astype(int)


def signal_breakeven_inflation(data: pd.DataFrame, t10yie: pd.Series = None,
                                lookback: int = 63, **kwargs) -> pd.Series:
    """H68: Breakeven Inflation Momentum."""
    if t10yie is None:
        return pd.Series(1, index=data.index)
    
    inflation_exp = t10yie.reindex(data.index).ffill()
    inf_mom = inflation_exp.diff(lookback)
    
    # Rising inflation expectations = cautious on growth
    # But moderate rise = healthy
    moderate_rise = (inf_mom > 0) & (inf_mom < 0.5)
    return moderate_rise.astype(int) | (inf_mom <= 0).astype(int)


# ============================================================================
# SIGNAL FUNCTIONS - MACRO
# ============================================================================

def signal_pmi_momentum(data: pd.DataFrame, pmi: pd.Series = None,
                         threshold: float = 50, **kwargs) -> pd.Series:
    """H100: PMI > 50 = expansion."""
    if pmi is None:
        return pd.Series(1, index=data.index)
    
    pmi_aligned = pmi.reindex(data.index, method='ffill')
    return (pmi_aligned > threshold).astype(int)


def signal_pmi_acceleration(data: pd.DataFrame, pmi: pd.Series = None, **kwargs) -> pd.Series:
    """H101: PMI Rate of Change."""
    if pmi is None:
        return pd.Series(1, index=data.index)
    
    pmi_aligned = pmi.reindex(data.index, method='ffill')
    pmi_change = pmi_aligned.diff(3)  # 3-month change
    
    # Rising PMI = bullish
    return (pmi_change > 0).astype(int)


def signal_consumer_sentiment(data: pd.DataFrame, umcsent: pd.Series = None,
                               lookback: int = 63, **kwargs) -> pd.Series:
    """H102: Consumer Sentiment Level."""
    if umcsent is None:
        return pd.Series(1, index=data.index)
    
    sent = umcsent.reindex(data.index, method='ffill')
    sent_percentile = sent.rolling(252).rank(pct=True)
    
    # High sentiment = cautious (contrarian)
    # Low sentiment = bullish
    return (sent_percentile < 0.7).astype(int)


def signal_sentiment_extremes(data: pd.DataFrame, umcsent: pd.Series = None, **kwargs) -> pd.Series:
    """H103: Consumer Sentiment Extreme Readings."""
    if umcsent is None:
        return pd.Series(1, index=data.index)
    
    sent = umcsent.reindex(data.index, method='ffill')
    sent_z = (sent - sent.rolling(252).mean()) / sent.rolling(252).std()
    
    # Extreme low sentiment = contrarian buy
    extreme_low = sent_z < -2
    return extreme_low.astype(int) | (sent_z > -1).astype(int)


def signal_unemployment_rate(data: pd.DataFrame, unrate: pd.Series = None, **kwargs) -> pd.Series:
    """H104: Unemployment Rate Level."""
    if unrate is None:
        return pd.Series(1, index=data.index)
    
    unemp = unrate.reindex(data.index, method='ffill')
    unemp_ma = unemp.rolling(12).mean()
    
    # Unemployment below average = healthy economy
    return (unemp <= unemp_ma).astype(int)


def signal_unemployment_change(data: pd.DataFrame, unrate: pd.Series = None, **kwargs) -> pd.Series:
    """H105: Unemployment Rate of Change."""
    if unrate is None:
        return pd.Series(1, index=data.index)
    
    unemp = unrate.reindex(data.index, method='ffill')
    unemp_change = unemp.diff(3)  # 3-month change
    
    # Rising unemployment = warning
    return (unemp_change < 0.3).astype(int)


def signal_initial_claims(data: pd.DataFrame, icsa: pd.Series = None,
                          threshold: int = 300000, **kwargs) -> pd.Series:
    """H106: Initial Jobless Claims Level."""
    if icsa is None:
        return pd.Series(1, index=data.index)
    
    claims = icsa.reindex(data.index, method='ffill')
    return (claims < threshold).astype(int)


def signal_claims_spike(data: pd.DataFrame, icsa: pd.Series = None, **kwargs) -> pd.Series:
    """H107: Jobless Claims Spike."""
    if icsa is None:
        return pd.Series(1, index=data.index)
    
    claims = icsa.reindex(data.index, method='ffill')
    claims_ma = claims.rolling(26).mean()
    claims_spike = claims > claims_ma * 1.3  # 30% above average
    
    return (~claims_spike).astype(int)


def signal_cpi_level(data: pd.DataFrame, cpi: pd.Series = None,
                     threshold: float = 3.0, **kwargs) -> pd.Series:
    """H108: CPI Inflation Level."""
    if cpi is None:
        return pd.Series(1, index=data.index)
    
    # Assuming CPI is already YoY change
    inflation = cpi.reindex(data.index, method='ffill')
    
    # Moderate inflation good for stocks
    moderate = (inflation > 0) & (inflation < threshold)
    return moderate.astype(int)


def signal_cpi_momentum(data: pd.DataFrame, cpi: pd.Series = None, **kwargs) -> pd.Series:
    """H109: CPI Rate of Change."""
    if cpi is None:
        return pd.Series(1, index=data.index)
    
    inflation = cpi.reindex(data.index, method='ffill')
    inf_change = inflation.diff(3)
    
    # Accelerating inflation = headwind
    return (inf_change < 0.5).astype(int)


def signal_macro_composite(data: pd.DataFrame, pmi: pd.Series = None,
                           umcsent: pd.Series = None, unrate: pd.Series = None,
                           icsa: pd.Series = None, **kwargs) -> pd.Series:
    """H110: Macro Composite Index."""
    signals = []
    
    if pmi is not None:
        pmi_aligned = pmi.reindex(data.index, method='ffill')
        signals.append((pmi_aligned > 50).astype(int))
    
    if unrate is not None:
        unemp = unrate.reindex(data.index, method='ffill')
        unemp_change = unemp.diff(3)
        signals.append((unemp_change < 0.3).astype(int))
    
    if icsa is not None:
        claims = icsa.reindex(data.index, method='ffill')
        signals.append((claims < 300000).astype(int))
    
    if not signals:
        return pd.Series(1, index=data.index)
    
    composite = pd.concat(signals, axis=1).mean(axis=1)
    return (composite > 0.5).astype(int)


# ============================================================================
# HYPOTHESIS DEFINITIONS
# ============================================================================

BATCH_6_HYPOTHESES = [
    # Yield Curve
    {
        'id': 'H65',
        'name': 'Yield Curve Slope',
        'category': 'Rates',
        'description': '10Y-2Y spread direction',
        'signal_func': signal_yield_curve_slope,
        'tickers': ['SPY'],
        'requires_fred': ['DGS10', 'DGS2'],
        'hold_period': 21,
        'priority': 1,
    },
    {
        'id': 'H66',
        'name': 'Curve Inversion Lag',
        'category': 'Rates',
        'description': 'Inversion predicts recession 12mo out',
        'signal_func': signal_curve_inversion_lag,
        'tickers': ['SPY'],
        'requires_fred': ['DGS10', 'DGS2'],
        'hold_period': 63,
        'priority': 1,
    },
    {
        'id': 'H67',
        'name': 'Real Yields',
        'category': 'Rates',
        'description': 'High real yields = equity headwind',
        'signal_func': signal_real_yields,
        'tickers': ['SPY'],
        'requires_fred': ['DGS10', 'T10YIE'],
        'hold_period': 21,
        'priority': 2,
    },
    {
        'id': 'H68',
        'name': 'Breakeven Inflation',
        'category': 'Rates',
        'description': 'Inflation expectations momentum',
        'signal_func': signal_breakeven_inflation,
        'tickers': ['SPY'],
        'requires_fred': ['T10YIE'],
        'hold_period': 21,
        'priority': 2,
    },
    # Macro
    {
        'id': 'H100',
        'name': 'PMI Level',
        'category': 'Macro',
        'description': 'PMI > 50 = expansion',
        'signal_func': signal_pmi_momentum,
        'tickers': ['SPY'],
        'requires_fred': ['MANEMP'],  # or ISM PMI
        'hold_period': 21,
        'priority': 1,
    },
    {
        'id': 'H101',
        'name': 'PMI Acceleration',
        'category': 'Macro',
        'description': 'PMI rate of change',
        'signal_func': signal_pmi_acceleration,
        'tickers': ['SPY'],
        'requires_fred': ['MANEMP'],
        'hold_period': 21,
        'priority': 2,
    },
    {
        'id': 'H102',
        'name': 'Consumer Sentiment Level',
        'category': 'Macro',
        'description': 'UMCSENT percentile',
        'signal_func': signal_consumer_sentiment,
        'tickers': ['SPY'],
        'requires_fred': ['UMCSENT'],
        'hold_period': 21,
        'priority': 2,
    },
    {
        'id': 'H103',
        'name': 'Sentiment Extremes',
        'category': 'Macro',
        'description': 'Contrarian sentiment signal',
        'signal_func': signal_sentiment_extremes,
        'tickers': ['SPY'],
        'requires_fred': ['UMCSENT'],
        'hold_period': 21,
        'priority': 2,
    },
    {
        'id': 'H104',
        'name': 'Unemployment Level',
        'category': 'Macro',
        'description': 'UNRATE vs average',
        'signal_func': signal_unemployment_rate,
        'tickers': ['SPY'],
        'requires_fred': ['UNRATE'],
        'hold_period': 63,
        'priority': 2,
    },
    {
        'id': 'H105',
        'name': 'Unemployment Change',
        'category': 'Macro',
        'description': 'Rising unemployment = warning',
        'signal_func': signal_unemployment_change,
        'tickers': ['SPY'],
        'requires_fred': ['UNRATE'],
        'hold_period': 21,
        'priority': 1,
    },
    {
        'id': 'H106',
        'name': 'Initial Claims Level',
        'category': 'Macro',
        'description': 'ICSA below 300K',
        'signal_func': signal_initial_claims,
        'tickers': ['SPY'],
        'requires_fred': ['ICSA'],
        'hold_period': 5,
        'priority': 2,
    },
    {
        'id': 'H107',
        'name': 'Claims Spike',
        'category': 'Macro',
        'description': 'Jobless claims spike detection',
        'signal_func': signal_claims_spike,
        'tickers': ['SPY'],
        'requires_fred': ['ICSA'],
        'hold_period': 5,
        'priority': 1,
    },
    {
        'id': 'H108',
        'name': 'CPI Level',
        'category': 'Macro',
        'description': 'Moderate inflation (0-3%)',
        'signal_func': signal_cpi_level,
        'tickers': ['SPY'],
        'requires_fred': ['CPIAUCSL'],
        'hold_period': 21,
        'priority': 2,
    },
    {
        'id': 'H109',
        'name': 'CPI Momentum',
        'category': 'Macro',
        'description': 'Accelerating inflation = headwind',
        'signal_func': signal_cpi_momentum,
        'tickers': ['SPY'],
        'requires_fred': ['CPIAUCSL'],
        'hold_period': 21,
        'priority': 2,
    },
    {
        'id': 'H110',
        'name': 'Macro Composite',
        'category': 'Macro',
        'description': 'Combined macro signals',
        'signal_func': signal_macro_composite,
        'tickers': ['SPY'],
        'requires_fred': ['MANEMP', 'UNRATE', 'ICSA'],
        'hold_period': 21,
        'priority': 1,
    },
]


def get_batch_6_hypotheses():
    """Return all Batch 6 hypotheses."""
    return BATCH_6_HYPOTHESES


if __name__ == "__main__":
    print(f"Batch 6: Yield Curve & Macro - {len(BATCH_6_HYPOTHESES)} hypotheses")
    for h in BATCH_6_HYPOTHESES:
        print(f"  {h['id']}: {h['name']} (Priority: {h['priority']})")
