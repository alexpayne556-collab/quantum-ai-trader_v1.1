"""
TEST FIXTURES
=============
Fast, synthetic data generation for unit tests.
No network calls, no external dependencies.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_mock_market_data(days=252, trend='sideways', volatility='normal'):
    """
    Create synthetic market data for fast testing.
    
    Args:
        days: Number of trading days to generate
        trend: 'bull', 'bear', or 'sideways'
        volatility: 'low', 'normal', 'high', or 'extreme'
    
    Returns:
        DataFrame with columns: close, high, low, vix
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
    
    # Base price movement
    if trend == 'bull':
        base = np.cumsum(np.random.normal(0.0008, 0.01, days)) + 100
    elif trend == 'bear':
        base = 100 - np.cumsum(np.random.normal(0.0005, 0.012, days))
    else:  # sideways
        base = 100 + np.cumsum(np.random.normal(0, 0.015, days))
    
    # VIX levels
    vix_levels = {
        'low': np.random.normal(15, 2, days),
        'normal': np.random.normal(20, 3, days),
        'high': np.random.normal(30, 5, days),
        'extreme': np.random.normal(40, 8, days)
    }
    vix = np.clip(vix_levels[volatility], 10, 80)
    
    return pd.DataFrame({
        'close': np.abs(base),  # Ensure positive prices
        'high': np.abs(base * 1.02),
        'low': np.abs(base * 0.98),
        'vix': vix
    }, index=dates)

def create_crisis_scenario():
    """2020 COVID crash scenario."""
    return create_mock_market_data(days=60, trend='bear', volatility='extreme')

def create_bull_market_scenario():
    """2021 bull market scenario."""
    return create_mock_market_data(days=252, trend='bull', volatility='low')

def create_range_bound_scenario():
    """Sideways choppy market."""
    return create_mock_market_data(days=252, trend='sideways', volatility='normal')

def create_volatile_choppy_scenario():
    """High volatility but no clear trend."""
    return create_mock_market_data(days=252, trend='sideways', volatility='high')

def create_mock_signals(num_signals=5, bias='neutral'):
    """
    Create mock signal values for testing.
    
    Args:
        num_signals: Number of signals to generate
        bias: 'bullish', 'bearish', or 'neutral'
    
    Returns:
        Dict of {signal_name: value}
    """
    signal_names = ['H16', 'H19', 'H20', 'H21', 'H27E', 'H128', 'H62'][:num_signals]
    
    if bias == 'bullish':
        values = np.random.uniform(0.3, 0.9, num_signals)
    elif bias == 'bearish':
        values = np.random.uniform(-0.9, -0.3, num_signals)
    else:  # neutral
        values = np.random.uniform(-0.5, 0.5, num_signals)
    
    return {name: float(val) for name, val in zip(signal_names, values)}

def create_mock_news_event(event_type='fomc'):
    """Create a mock news event for testing."""
    return {
        'type': event_type,
        'timestamp': datetime.now(),
        'description': f'Mock {event_type} event'
    }
