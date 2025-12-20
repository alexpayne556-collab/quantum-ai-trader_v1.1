#!/usr/bin/env python3
"""
HYPOTHESIS BATCH 5: CROSS-ASSET & CREDIT (H56-H73)
===================================================
Credit spreads, intermarket signals, flight to quality.
Est. time: ~12 minutes for 14 tests
API calls: 4 (HYG, LQD, TLT, GLD, FXY, USO, COPX)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict


# ============================================================================
# SIGNAL FUNCTIONS - CROSS-ASSET
# ============================================================================

def signal_stock_bond_correlation(data: pd.DataFrame, tlt_data: pd.DataFrame = None,
                                   lookback: int = 63, threshold: float = 0.3, **kwargs) -> pd.Series:
    """H56: Stock-Bond Correlation Regime."""
    if tlt_data is None:
        return pd.Series(1, index=data.index)
    
    spy_ret = data['close'].pct_change()
    tlt_ret = tlt_data['close'].pct_change().reindex(data.index).ffill()
    
    rolling_corr = spy_ret.rolling(lookback).corr(tlt_ret)
    
    # When correlation > 0.3 (both falling together), reduce exposure
    return (rolling_corr < threshold).astype(int)


def signal_bond_leading(data: pd.DataFrame, tlt_data: pd.DataFrame = None,
                        lookback: int = 5, **kwargs) -> pd.Series:
    """H57: Bond Leading Indicator - TLT decline warns SPY."""
    if tlt_data is None:
        return pd.Series(1, index=data.index)
    
    tlt_ret = tlt_data['close'].pct_change(lookback).reindex(data.index).ffill()
    
    # Positive TLT return = bullish for SPY
    return (tlt_ret > 0).astype(int)


def signal_credit_spread(data: pd.DataFrame, hyg_data: pd.DataFrame = None,
                         lqd_data: pd.DataFrame = None, lookback: int = 21, **kwargs) -> pd.Series:
    """H58: Credit Spread Signal - HYG/LQD declining = risk-off."""
    if hyg_data is None or lqd_data is None:
        return pd.Series(1, index=data.index)
    
    hyg = hyg_data['close'].reindex(data.index).ffill()
    lqd = lqd_data['close'].reindex(data.index).ffill()
    
    ratio = hyg / lqd
    ratio_mom = ratio.pct_change(lookback)
    
    # Declining ratio = spreads widening = bearish
    return (ratio_mom > 0).astype(int)


def signal_high_yield_momentum(data: pd.DataFrame, hyg_data: pd.DataFrame = None,
                                lookback: int = 21, **kwargs) -> pd.Series:
    """H59: High Yield Momentum - HYG trending predicts risk appetite."""
    if hyg_data is None:
        return pd.Series(1, index=data.index)
    
    hyg_mom = hyg_data['close'].pct_change(lookback).reindex(data.index).ffill()
    hyg_ma = hyg_data['close'].rolling(50).mean().reindex(data.index).ffill()
    hyg_price = hyg_data['close'].reindex(data.index).ffill()
    
    # HYG above MA and positive momentum = risk-on
    return ((hyg_price > hyg_ma) & (hyg_mom > 0)).astype(int)


def signal_copper_gold_ratio(data: pd.DataFrame, copx_data: pd.DataFrame = None,
                              gld_data: pd.DataFrame = None, lookback: int = 21, **kwargs) -> pd.Series:
    """H60: Copper/Gold Ratio - Rising = risk-on."""
    if copx_data is None or gld_data is None:
        return pd.Series(1, index=data.index)
    
    copx = copx_data['close'].reindex(data.index).ffill()
    gld = gld_data['close'].reindex(data.index).ffill()
    
    ratio = copx / gld
    ratio_mom = ratio.pct_change(lookback)
    
    return (ratio_mom > 0).astype(int)


def signal_gold_fear(data: pd.DataFrame, gld_data: pd.DataFrame = None,
                     spike_threshold: float = 0.02, lookback: int = 5, **kwargs) -> pd.Series:
    """H61: Gold as Fear Indicator - GLD spiking = defensive."""
    if gld_data is None:
        return pd.Series(1, index=data.index)
    
    gld_ret = gld_data['close'].pct_change(lookback).reindex(data.index).ffill()
    
    # Gold spike = fear = reduce equity
    return (gld_ret < spike_threshold).astype(int)


def signal_oil_equity(data: pd.DataFrame, uso_data: pd.DataFrame = None,
                      lookback: int = 21, **kwargs) -> pd.Series:
    """H62: Oil-Equity Relationship - Healthy when rising together."""
    if uso_data is None:
        return pd.Series(1, index=data.index)
    
    spy_mom = data['close'].pct_change(lookback)
    uso_mom = uso_data['close'].pct_change(lookback).reindex(data.index).ffill()
    
    # Both positive = healthy market
    both_up = (spy_mom > 0) & (uso_mom > 0)
    divergence = (spy_mom > 0) & (uso_mom < -0.05)  # Stocks up, oil down
    
    signal = pd.Series(1, index=data.index)
    signal[divergence] = 0  # Warning
    return signal


def signal_dollar_strength(data: pd.DataFrame, uup_data: pd.DataFrame = None,
                           lookback: int = 21, **kwargs) -> pd.Series:
    """H63: Dollar Strength Signal - Strong dollar = equity headwind."""
    if uup_data is None:
        return pd.Series(1, index=data.index)
    
    uup_mom = uup_data['close'].pct_change(lookback).reindex(data.index).ffill()
    
    # Weak dollar = bullish for equities
    return (uup_mom < 0).astype(int)


def signal_yen_carry(data: pd.DataFrame, fxy_data: pd.DataFrame = None,
                     lookback: int = 5, threshold: float = 0.02, **kwargs) -> pd.Series:
    """H64: Yen Carry Indicator - Yen spike = risk-off."""
    if fxy_data is None:
        return pd.Series(1, index=data.index)
    
    fxy_ret = fxy_data['close'].pct_change(lookback).reindex(data.index).ffill()
    
    # Yen strengthening rapidly = risk-off
    return (fxy_ret < threshold).astype(int)


def signal_risk_composite(data: pd.DataFrame, vix_data: pd.Series = None,
                          hyg_data: pd.DataFrame = None, fxy_data: pd.DataFrame = None,
                          gld_data: pd.DataFrame = None, lookback: int = 21, **kwargs) -> pd.Series:
    """H72: Risk-On/Risk-Off Composite - Multiple signals."""
    signals = []
    
    # VIX signal
    if vix_data is not None:
        vix = vix_data.reindex(data.index).ffill()
        vix_percentile = vix.rolling(252).rank(pct=True)
        signals.append((vix_percentile < 0.7).astype(int))
    
    # Credit signal
    if hyg_data is not None:
        hyg_mom = hyg_data['close'].pct_change(lookback).reindex(data.index).ffill()
        signals.append((hyg_mom > 0).astype(int))
    
    # Yen signal
    if fxy_data is not None:
        fxy_ret = fxy_data['close'].pct_change(lookback).reindex(data.index).ffill()
        signals.append((fxy_ret < 0.02).astype(int))
    
    # Gold signal
    if gld_data is not None:
        gld_ret = gld_data['close'].pct_change(lookback).reindex(data.index).ffill()
        signals.append((gld_ret < 0.02).astype(int))
    
    if not signals:
        return pd.Series(1, index=data.index)
    
    # Composite: average of all signals
    composite = pd.concat(signals, axis=1).mean(axis=1)
    return (composite > 0.5).astype(int)


def signal_flight_to_quality(data: pd.DataFrame, tlt_data: pd.DataFrame = None,
                              gld_data: pd.DataFrame = None, lookback: int = 5, **kwargs) -> pd.Series:
    """H73: Flight to Quality - TLT+GLD up, SPY down pattern."""
    if tlt_data is None or gld_data is None:
        return pd.Series(1, index=data.index)
    
    spy_ret = data['close'].pct_change(lookback)
    tlt_ret = tlt_data['close'].pct_change(lookback).reindex(data.index).ffill()
    gld_ret = gld_data['close'].pct_change(lookback).reindex(data.index).ffill()
    
    # Flight to quality: bonds and gold up, stocks down
    ftq = (tlt_ret > 0.01) & (gld_ret > 0.01) & (spy_ret < -0.01)
    
    # After FTQ, expect bounce
    return ftq.shift(1).fillna(0).astype(int)


# ============================================================================
# HYPOTHESIS DEFINITIONS
# ============================================================================

BATCH_5_HYPOTHESES = [
    {
        'id': 'H56',
        'name': 'Stock-Bond Correlation',
        'category': 'Cross-Asset',
        'description': 'Reduce when correlation > 0.3',
        'signal_func': signal_stock_bond_correlation,
        'tickers': ['SPY', 'TLT'],
        'hold_period': 21,
        'priority': 1,
    },
    {
        'id': 'H57',
        'name': 'Bond Leading Indicator',
        'category': 'Cross-Asset',
        'description': 'TLT 5-day return warns SPY',
        'signal_func': signal_bond_leading,
        'tickers': ['SPY', 'TLT'],
        'hold_period': 5,
        'priority': 2,
    },
    {
        'id': 'H58',
        'name': 'Credit Spread Signal',
        'category': 'Credit',
        'description': 'HYG/LQD declining = risk-off',
        'signal_func': signal_credit_spread,
        'tickers': ['SPY', 'HYG', 'LQD'],
        'hold_period': 10,
        'priority': 1,
    },
    {
        'id': 'H59',
        'name': 'High Yield Momentum',
        'category': 'Credit',
        'description': 'HYG trend predicts risk appetite',
        'signal_func': signal_high_yield_momentum,
        'tickers': ['SPY', 'HYG'],
        'hold_period': 21,
        'priority': 2,
    },
    {
        'id': 'H60',
        'name': 'Copper/Gold Ratio',
        'category': 'Cross-Asset',
        'description': 'Rising ratio = risk-on',
        'signal_func': signal_copper_gold_ratio,
        'tickers': ['SPY', 'COPX', 'GLD'],
        'hold_period': 21,
        'priority': 2,
    },
    {
        'id': 'H61',
        'name': 'Gold Fear Indicator',
        'category': 'Cross-Asset',
        'description': 'GLD spike = defensive',
        'signal_func': signal_gold_fear,
        'tickers': ['SPY', 'GLD'],
        'hold_period': 5,
        'priority': 2,
    },
    {
        'id': 'H62',
        'name': 'Oil-Equity Relationship',
        'category': 'Cross-Asset',
        'description': 'Healthy when rising together',
        'signal_func': signal_oil_equity,
        'tickers': ['SPY', 'USO'],
        'hold_period': 21,
        'priority': 3,
    },
    {
        'id': 'H63',
        'name': 'Dollar Strength',
        'category': 'FX',
        'description': 'Strong dollar = equity headwind',
        'signal_func': signal_dollar_strength,
        'tickers': ['SPY', 'UUP'],
        'hold_period': 21,
        'priority': 2,
    },
    {
        'id': 'H64',
        'name': 'Yen Carry Indicator',
        'category': 'FX',
        'description': 'Yen spike = risk-off',
        'signal_func': signal_yen_carry,
        'tickers': ['SPY', 'FXY'],
        'hold_period': 5,
        'priority': 2,
    },
    {
        'id': 'H72',
        'name': 'Risk-On/Off Composite',
        'category': 'Cross-Asset',
        'description': 'Composite of VIX, credit, yen, gold',
        'signal_func': signal_risk_composite,
        'tickers': ['SPY'],
        'requires_macro': ['VIX'],
        'hold_period': 10,
        'priority': 1,
    },
    {
        'id': 'H73',
        'name': 'Flight to Quality',
        'category': 'Cross-Asset',
        'description': 'TLT+GLD up, SPY down = contrarian buy',
        'signal_func': signal_flight_to_quality,
        'tickers': ['SPY', 'TLT', 'GLD'],
        'hold_period': 5,
        'priority': 2,
    },
]


def get_batch_5_hypotheses():
    """Return all Batch 5 hypotheses."""
    return BATCH_5_HYPOTHESES


if __name__ == "__main__":
    print(f"Batch 5: Cross-Asset & Credit - {len(BATCH_5_HYPOTHESES)} hypotheses")
    for h in BATCH_5_HYPOTHESES:
        print(f"  {h['id']}: {h['name']} (Priority: {h['priority']})")
