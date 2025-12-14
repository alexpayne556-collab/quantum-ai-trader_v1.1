#!/usr/bin/env python3
"""
Pattern Discovery Lab V0 - Core evaluation logic.

Pure functions for walk-forward evaluation, RankIC computation, and gate checks.
NO printing - all outputs through return values.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Iterator
from scipy import stats


# ============================================================================
# CONSTANTS
# ============================================================================

NUMERIC_PRECISION = 6  # Decimal places for rounding floats in outputs


# ============================================================================
# WALK-FORWARD SPLITS
# ============================================================================

def walk_forward_splits(
    dates: pd.DatetimeIndex,
    train_len: int,
    test_len: int,
    embargo: int = 1
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate walk-forward train/test splits with embargo.
    
    Embargo prevents information leakage by excluding `embargo` bars
    between train and test sets.
    
    Args:
        dates: DatetimeIndex of all dates
        train_len: Training window length in bars
        test_len: Test window length in bars
        embargo: Number of bars to skip between train and test (default: 1)
    
    Yields:
        (train_indices, test_indices) tuples
    
    Example:
        dates = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        train_len=3, test_len=2, embargo=1
        
        Split 1: train=[0,1,2], embargo=[3], test=[4,5]
        Split 2: train=[1,2,3], embargo=[4], test=[5,6]
        etc.
    """
    n = len(dates)
    
    if train_len < 1:
        raise ValueError("train_len must be >= 1")
    if test_len < 1:
        raise ValueError("test_len must be >= 1")
    if embargo < 0:
        raise ValueError("embargo must be >= 0")
    
    min_required = train_len + embargo + test_len
    if n < min_required:
        raise ValueError(
            f"Insufficient data: need {min_required} bars "
            f"(train={train_len} + embargo={embargo} + test={test_len}), "
            f"but only have {n}"
        )
    
    # Generate splits
    start = 0
    while start + train_len + embargo + test_len <= n:
        train_end = start + train_len
        test_start = train_end + embargo
        test_end = test_start + test_len
        
        train_idx = np.arange(start, train_end)
        test_idx = np.arange(test_start, test_end)
        
        yield train_idx, test_idx
        
        # Move forward by test_len for next split
        start += test_len


# ============================================================================
# FORWARD RETURNS
# ============================================================================

def compute_forward_returns(prices: pd.Series, horizon: int = 1) -> pd.Series:
    """
    Compute forward returns.
    
    fwd_return[t] = (price[t+horizon] / price[t]) - 1
    
    Args:
        prices: Price series
        horizon: Forward horizon in bars
    
    Returns:
        Forward returns series (last `horizon` values are NaN)
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series")
    
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    
    fwd_returns = prices.shift(-horizon) / prices - 1.0
    return fwd_returns


# ============================================================================
# RANK IC COMPUTATION
# ============================================================================

def compute_rank_ic(signal: pd.Series, fwd_returns: pd.Series) -> float:
    """
    Compute Rank IC (Spearman correlation between signal and forward returns).
    
    Args:
        signal: Signal values
        fwd_returns: Forward returns
    
    Returns:
        Spearman correlation coefficient (float)
        Returns NaN if insufficient valid data
    """
    if not isinstance(signal, pd.Series):
        raise TypeError("signal must be a pandas Series")
    if not isinstance(fwd_returns, pd.Series):
        raise TypeError("fwd_returns must be a pandas Series")
    
    # Align and drop NaN
    df = pd.DataFrame({'signal': signal, 'fwd_ret': fwd_returns})
    df = df.dropna()
    
    if len(df) < 2:
        return np.nan
    
    # Compute Spearman correlation
    corr, _ = stats.spearmanr(df['signal'], df['fwd_ret'])
    
    return float(corr)


def compute_ic_tstat(ic_values: List[float]) -> Tuple[float, float]:
    """
    Compute t-statistic and p-value for IC time series.
    
    H0: mean IC = 0
    
    Args:
        ic_values: List of IC values across time
    
    Returns:
        (t_statistic, p_value) tuple
        Returns (NaN, NaN) if insufficient data
    """
    # Filter out NaN values
    valid_ics = [ic for ic in ic_values if not np.isnan(ic)]
    
    if len(valid_ics) < 2:
        return (np.nan, np.nan)
    
    t_stat, p_val = stats.ttest_1samp(valid_ics, 0.0)
    
    return (float(t_stat), float(p_val))


# ============================================================================
# NEGATIVE CONTROLS
# ============================================================================

def negative_control_random(
    length: int,
    seed: int = 42,
    index=None
) -> pd.Series:
    """
    Generate random signal as negative control.
    
    Expected: RankIC near 0
    
    Args:
        length: Length of signal
        seed: Random seed for reproducibility
        index: Index to use (for alignment with price data)
    
    Returns:
        Random signal series
    """
    rng = np.random.RandomState(seed)
    signal = pd.Series(rng.randn(length), index=index)
    return signal


def placebo_time_shift(signal: pd.Series, shift: int = 5) -> pd.Series:
    """
    Time-shift signal as placebo control.
    
    Expected: RankIC should degrade (signal misaligned with future)
    
    Args:
        signal: Original signal
        shift: Number of bars to shift forward (positive = look into future incorrectly)
    
    Returns:
        Shifted signal
    """
    if not isinstance(signal, pd.Series):
        raise TypeError("signal must be a pandas Series")
    
    return signal.shift(shift)


# ============================================================================
# GATE CHECKS
# ============================================================================

def check_no_nan_inf(value: float, name: str = "value") -> None:
    """
    Check that a value is not NaN or Inf.
    
    Args:
        value: Value to check
        name: Name for error message
    
    Raises:
        ValueError: If value is NaN or Inf
    """
    if np.isnan(value):
        raise ValueError(f"{name} is NaN")
    if np.isinf(value):
        raise ValueError(f"{name} is Inf")


def is_finite(value: float) -> bool:
    """
    Check if value is finite (not NaN or Inf).
    
    Args:
        value: Value to check
    
    Returns:
        True if finite, False otherwise
    """
    return not (np.isnan(value) or np.isinf(value))


def check_finite_metrics_gate(metrics: dict) -> Tuple[bool, str]:
    """
    Gate: All metrics must be finite (no NaN/Inf).
    
    Args:
        metrics: Dictionary of metric name -> value
    
    Returns:
        (passed, reason) tuple
    """
    non_finite = []
    
    for name, value in metrics.items():
        if value is None:
            continue  # null is acceptable
        if isinstance(value, (int, float)):
            if not is_finite(value):
                non_finite.append(name)
        elif isinstance(value, dict):
            # Recursive check
            sub_passed, sub_reason = check_finite_metrics_gate(value)
            if not sub_passed:
                non_finite.append(f"{name} ({sub_reason})")
    
    if non_finite:
        return (False, f"Non-finite metrics: {', '.join(non_finite)}")
    return (True, "All metrics are finite")


def check_oos_degradation_gate(
    is_ic_mean: float,
    oos_ic_mean: float,
    threshold: float = 0.60
) -> Tuple[bool, str]:
    """
    OOS degradation gate: OOS IC must be >= threshold * IS IC.
    
    Args:
        is_ic_mean: In-sample IC mean
        oos_ic_mean: Out-of-sample IC mean
        threshold: Minimum ratio (default: 0.60 = 60%)
    
    Returns:
        (passed, reason) tuple
    """
    # Check for NaN/Inf
    if np.isnan(is_ic_mean) or np.isnan(oos_ic_mean):
        return (False, "IC values contain NaN")
    if np.isinf(is_ic_mean) or np.isinf(oos_ic_mean):
        return (False, "IC values contain Inf")
    
    # Gate logic
    min_required = threshold * is_ic_mean
    
    if oos_ic_mean >= min_required:
        return (True, f"OOS IC {oos_ic_mean:.4f} >= {threshold:.0%} * IS IC {is_ic_mean:.4f}")
    else:
        return (False, f"OOS IC {oos_ic_mean:.4f} < {threshold:.0%} * IS IC {is_ic_mean:.4f}")


def ensure_deterministic_order(items: List, key=None) -> List:
    """
    Ensure list is sorted deterministically.
    
    Args:
        items: List to sort
        key: Sort key function (optional)
    
    Returns:
        Sorted list
    """
    return sorted(items, key=key)


def round_numeric(value: float, decimals: int = NUMERIC_PRECISION):
    """
    Round numeric value to fixed decimals.
    
    Returns None for NaN/Inf (strict JSON compliance).
    
    Args:
        value: Value to round
        decimals: Number of decimal places
    
    Returns:
        Rounded value or None if NaN/Inf
    """
    if value is None:
        return None
    if np.isnan(value) or np.isinf(value):
        return None  # Convert to null for JSON
    
    return round(value, decimals)
