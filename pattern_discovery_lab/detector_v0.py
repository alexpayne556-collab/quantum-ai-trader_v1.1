#!/usr/bin/env python3
"""
Baseline detector for Pattern Discovery Lab V0.

Pure functions, deterministic, no randomness.
"""
import numpy as np
import pandas as pd


def momentum_detector(prices: pd.Series, lookback: int = 20) -> pd.Series:
    """
    Baseline momentum detector.
    
    Signal = (price / price.shift(lookback) - 1)
    
    Args:
        prices: Price series (index should be dates/timestamps)
        lookback: Lookback period in bars
    
    Returns:
        Momentum signal series (same index as prices, first lookback values are NaN)
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series")
    
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    
    signal = prices / prices.shift(lookback) - 1.0
    return signal
