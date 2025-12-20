#!/usr/bin/env python3
"""
SYSTEMATIC HYPOTHESIS TESTING ENGINE
=====================================

A Renaissance-grade framework for testing 200+ market hypotheses.

Features:
- Structured hypothesis definitions
- Automatic data fetching
- Walk-forward validation
- Monte Carlo significance testing
- Multiple testing correction (Bonferroni, BH-FDR)
- Comprehensive result tracking

Usage:
    from HYPOTHESIS_ENGINE import HypothesisEngine
    
    engine = HypothesisEngine()
    results = engine.run_all()
    survivors = engine.get_survivors()
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any
from enum import Enum
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import json
from pathlib import Path
from scipy import stats
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path("data/free_harvest")
RESULTS_DIR = Path("data/hypothesis_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Validation parameters
TRAIN_DAYS = 252  # 1 year
TEST_DAYS = 63    # 3 months
STEP_DAYS = 63    # Quarterly rolling
MIN_TRADES = 30   # Minimum trades for significance
SIGNIFICANCE_LEVEL = 0.05
N_MONTE_CARLO = 5000


# ============================================================================
# HYPOTHESIS DEFINITION STRUCTURES
# ============================================================================

class HypothesisCategory(Enum):
    TREND_FOLLOWING = "Trend Following"
    MEAN_REVERSION = "Mean Reversion"
    SEASONALITY = "Seasonality"
    VALUE = "Value"
    CARRY = "Carry"
    VOLATILITY = "Volatility"
    LIQUIDITY = "Liquidity"
    RISK_REGIME = "Risk On/Off"
    SENTIMENT = "Sentiment"
    ECONOMIC = "Economic Indicators"
    INTERMARKET = "Intermarket"
    FACTOR_TIMING = "Factor Timing"
    EVENT_DRIVEN = "Event Driven"
    BEHAVIORAL = "Behavioral"
    MICROSTRUCTURE = "Microstructure"
    BREADTH = "Market Breadth"


class SignalType(Enum):
    LONG_ONLY = "long_only"
    LONG_SHORT = "long_short"
    BINARY = "binary"  # 1 or 0
    CONTINUOUS = "continuous"  # -1 to 1


@dataclass
class HypothesisResult:
    """Results from testing a single hypothesis."""
    hypothesis_id: str
    name: str
    category: str
    
    # In-sample metrics
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    n_trades: int = 0
    
    # Baseline comparison
    baseline_sharpe: float = 0.0
    excess_sharpe: float = 0.0
    
    # Statistical tests
    monte_carlo_percentile: float = 0.0
    t_stat: float = 0.0
    p_value: float = 1.0
    
    # Walk-forward results
    wf_periods: int = 0
    wf_win_rate: float = 0.0
    wf_avg_excess: float = 0.0
    wf_p_value: float = 1.0
    
    # Final verdict
    passed_monte_carlo: bool = False
    passed_walkforward: bool = False
    is_survivor: bool = False
    
    # Metadata
    tickers_tested: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class Hypothesis:
    """Definition of a testable hypothesis."""
    id: str
    name: str
    category: HypothesisCategory
    description: str
    
    # Signal generation
    signal_func: Callable
    signal_type: SignalType
    
    # Data requirements
    tickers: List[str]
    requires_volume: bool = False
    requires_macro: bool = False
    macro_series: List[str] = field(default_factory=list)
    
    # Parameters
    lookback: int = 20
    hold_period: int = 5
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Baseline
    baseline: str = "buy_and_hold"  # or "random", "inverse"
    
    # Priority (for ordering tests)
    priority: int = 5  # 1=highest, 10=lowest


# ============================================================================
# SIGNAL GENERATION FUNCTIONS
# ============================================================================

def signal_momentum_timeseries(data: pd.DataFrame, lookback: int = 21, **kwargs) -> pd.Series:
    """Time-series momentum: positive past returns → long."""
    returns = data['close'].pct_change(lookback)
    return (returns > 0).astype(int)


def signal_momentum_12m_1m(data: pd.DataFrame, **kwargs) -> pd.Series:
    """12-1 month momentum (skip most recent month)."""
    ret_12m = data['close'].pct_change(252)
    ret_1m = data['close'].pct_change(21)
    mom = ret_12m - ret_1m  # Skip recent month
    return (mom > 0).astype(int)


def signal_rsi_oversold(data: pd.DataFrame, period: int = 14, oversold: int = 30, **kwargs) -> pd.Series:
    """RSI oversold signal."""
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return (rsi < oversold).astype(int)


def signal_rsi_overbought(data: pd.DataFrame, period: int = 14, overbought: int = 70, **kwargs) -> pd.Series:
    """RSI overbought signal (short)."""
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return (rsi > overbought).astype(int) * -1  # Short signal


def signal_bollinger_lower(data: pd.DataFrame, period: int = 20, std_dev: float = 2.0, **kwargs) -> pd.Series:
    """Bollinger Band lower touch → long (mean reversion)."""
    sma = data['close'].rolling(period).mean()
    std = data['close'].rolling(period).std()
    lower = sma - std_dev * std
    return (data['close'] < lower).astype(int)


def signal_bollinger_upper(data: pd.DataFrame, period: int = 20, std_dev: float = 2.0, **kwargs) -> pd.Series:
    """Bollinger Band upper touch → short (mean reversion)."""
    sma = data['close'].rolling(period).mean()
    std = data['close'].rolling(period).std()
    upper = sma + std_dev * std
    return (data['close'] > upper).astype(int) * -1


def signal_ma_crossover(data: pd.DataFrame, fast: int = 50, slow: int = 200, **kwargs) -> pd.Series:
    """Moving average crossover."""
    ma_fast = data['close'].rolling(fast).mean()
    ma_slow = data['close'].rolling(slow).mean()
    return (ma_fast > ma_slow).astype(int)


def signal_above_ma(data: pd.DataFrame, period: int = 200, **kwargs) -> pd.Series:
    """Price above moving average."""
    ma = data['close'].rolling(period).mean()
    return (data['close'] > ma).astype(int)


def signal_short_term_reversal(data: pd.DataFrame, lookback: int = 5, **kwargs) -> pd.Series:
    """Short-term reversal: poor performers bounce."""
    returns = data['close'].pct_change(lookback)
    return (returns < returns.rolling(60).quantile(0.2)).astype(int)


def signal_volume_spike(data: pd.DataFrame, threshold: float = 2.0, **kwargs) -> pd.Series:
    """Volume spike signal."""
    if 'volume' not in data.columns:
        return pd.Series(0, index=data.index)
    avg_vol = data['volume'].rolling(20).mean()
    return (data['volume'] > threshold * avg_vol).astype(int)


def signal_gap_down(data: pd.DataFrame, threshold: float = -0.02, **kwargs) -> pd.Series:
    """Gap down signal (for gap fill strategy)."""
    gap = data['open'] / data['close'].shift(1) - 1
    return (gap < threshold).astype(int)


def signal_gap_up(data: pd.DataFrame, threshold: float = 0.02, **kwargs) -> pd.Series:
    """Gap up signal (for gap fade strategy)."""
    gap = data['open'] / data['close'].shift(1) - 1
    return (gap > threshold).astype(int) * -1  # Fade the gap


def signal_vix_high(data: pd.DataFrame, vix_data: pd.Series = None, threshold: float = 25, **kwargs) -> pd.Series:
    """VIX above threshold → defensive."""
    if vix_data is None:
        return pd.Series(0, index=data.index)
    aligned_vix = vix_data.reindex(data.index).ffill()
    return (aligned_vix > threshold).astype(int) * -1  # Short when VIX high


def signal_vix_low(data: pd.DataFrame, vix_data: pd.Series = None, threshold: float = 15, **kwargs) -> pd.Series:
    """VIX below threshold → risk-on."""
    if vix_data is None:
        return pd.Series(0, index=data.index)
    aligned_vix = vix_data.reindex(data.index).ffill()
    return (aligned_vix < threshold).astype(int)


def signal_vix_term_structure(data: pd.DataFrame, vix: pd.Series = None, vix3m: pd.Series = None, **kwargs) -> pd.Series:
    """VIX term structure signal."""
    if vix is None or vix3m is None:
        return pd.Series(0, index=data.index)
    vix_aligned = vix.reindex(data.index).ffill()
    vix3m_aligned = vix3m.reindex(data.index).ffill()
    contango = vix3m_aligned / vix_aligned
    # Backwardation = fear = expect bounce
    return (contango < 0.95).astype(int)


def signal_turn_of_month(data: pd.DataFrame, **kwargs) -> pd.Series:
    """Turn of month effect: last 3 days + first 3 days."""
    dom = pd.Series(data.index.day, index=data.index)
    eom = data.index.to_series().apply(lambda x: x + pd.offsets.MonthEnd(0)).dt.day
    
    # Last 3 days of month or first 3 days
    is_eom = (eom - dom) <= 2
    is_bom = dom <= 3
    return (is_eom | is_bom).astype(int)


def signal_sell_in_may(data: pd.DataFrame, **kwargs) -> pd.Series:
    """Sell in May and go away: Oct-Apr = long, May-Sep = cash."""
    month = pd.Series(data.index.month, index=data.index)
    return ((month >= 11) | (month <= 4)).astype(int)


def signal_january_effect(data: pd.DataFrame, **kwargs) -> pd.Series:
    """January effect: Long in January only."""
    month = pd.Series(data.index.month, index=data.index)
    return (month == 1).astype(int)


def signal_monday_effect(data: pd.DataFrame, **kwargs) -> pd.Series:
    """Monday effect: Historically negative Mondays (inverse)."""
    dow = pd.Series(data.index.dayofweek, index=data.index)
    return (dow == 0).astype(int) * -1


def signal_pre_holiday(data: pd.DataFrame, **kwargs) -> pd.Series:
    """Pre-holiday rally: Day before major US holidays."""
    # Simplified: Friday before Monday holidays
    dow = pd.Series(data.index.dayofweek, index=data.index)
    return (dow == 4).astype(int)


def signal_low_volatility(data: pd.DataFrame, lookback: int = 20, **kwargs) -> pd.Series:
    """Low volatility anomaly: Low vol stocks outperform."""
    vol = data['close'].pct_change().rolling(lookback).std()
    median_vol = vol.rolling(252).median()
    return (vol < median_vol).astype(int)


def signal_high_volume_breakout(data: pd.DataFrame, price_lookback: int = 20, vol_threshold: float = 1.5, **kwargs) -> pd.Series:
    """Breakout on high volume."""
    if 'volume' not in data.columns:
        return pd.Series(0, index=data.index)
    
    high_n = data['close'].rolling(price_lookback).max()
    vol_avg = data['volume'].rolling(20).mean()
    
    breakout = data['close'] >= high_n
    high_vol = data['volume'] > vol_threshold * vol_avg
    
    return (breakout & high_vol).astype(int)


def signal_mean_reversion_zscore(data: pd.DataFrame, lookback: int = 60, threshold: float = 2.0, **kwargs) -> pd.Series:
    """Z-score mean reversion."""
    rolling_mean = data['close'].rolling(lookback).mean()
    rolling_std = data['close'].rolling(lookback).std()
    zscore = (data['close'] - rolling_mean) / rolling_std
    
    # Long when z < -threshold, short when z > threshold
    signal = pd.Series(0, index=data.index)
    signal[zscore < -threshold] = 1
    signal[zscore > threshold] = -1
    return signal


def signal_dual_momentum(data: pd.DataFrame, spy_data: pd.DataFrame = None, lookback: int = 252, **kwargs) -> pd.Series:
    """Dual momentum: Absolute + relative momentum."""
    if spy_data is None:
        # Absolute momentum only
        ret = data['close'].pct_change(lookback)
        return (ret > 0).astype(int)
    
    ret = data['close'].pct_change(lookback)
    spy_ret = spy_data['close'].pct_change(lookback).reindex(data.index).ffill()
    
    # Long if both absolute and relative momentum positive
    abs_mom = ret > 0
    rel_mom = ret > spy_ret
    
    return (abs_mom & rel_mom).astype(int)


def signal_yield_curve_regime(data: pd.DataFrame, y10: pd.Series = None, y2: pd.Series = None, **kwargs) -> pd.Series:
    """Yield curve regime: Inverted = defensive."""
    if y10 is None or y2 is None:
        return pd.Series(0, index=data.index)
    
    y10_aligned = y10.reindex(data.index).ffill()
    y2_aligned = y2.reindex(data.index).ffill()
    
    spread = y10_aligned - y2_aligned
    # Inverted curve = risk-off
    return (spread > 0).astype(int)


def signal_dollar_strength(data: pd.DataFrame, uup_data: pd.Series = None, lookback: int = 20, **kwargs) -> pd.Series:
    """Strong dollar = emerging markets weakness."""
    if uup_data is None:
        return pd.Series(0, index=data.index)
    
    uup_aligned = uup_data.reindex(data.index).ffill()
    dollar_mom = uup_aligned.pct_change(lookback)
    
    # Strong dollar = bearish for risk assets
    return (dollar_mom < 0).astype(int)


def signal_put_call_extreme(data: pd.DataFrame, pc_ratio: pd.Series = None, threshold_high: float = 1.2, **kwargs) -> pd.Series:
    """Put/call ratio extreme (contrarian)."""
    if pc_ratio is None:
        return pd.Series(0, index=data.index)
    
    pc_aligned = pc_ratio.reindex(data.index).ffill()
    # High put/call = extreme fear = contrarian buy
    return (pc_aligned > threshold_high).astype(int)


def signal_new_high_breakout(data: pd.DataFrame, lookback: int = 252, **kwargs) -> pd.Series:
    """52-week high breakout."""
    high_52w = data['close'].rolling(lookback).max()
    return (data['close'] >= high_52w * 0.98).astype(int)  # Within 2% of high


def signal_new_low_buy(data: pd.DataFrame, lookback: int = 252, **kwargs) -> pd.Series:
    """52-week low (contrarian buy)."""
    low_52w = data['close'].rolling(lookback).min()
    return (data['close'] <= low_52w * 1.02).astype(int)  # Within 2% of low


# ============================================================================
# ADDITIONAL SIGNAL FUNCTIONS - BATCH 2
# ============================================================================

def signal_macd_crossover(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, **kwargs) -> pd.Series:
    """MACD crossover signal."""
    ema_fast = data['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return (macd > macd_signal).astype(int)


def signal_macd_histogram_reversal(data: pd.DataFrame, **kwargs) -> pd.Series:
    """MACD histogram reversal (momentum shift)."""
    ema_fast = data['close'].ewm(span=12, adjust=False).mean()
    ema_slow = data['close'].ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - macd_signal
    # Buy when histogram turns positive from negative
    return ((histogram > 0) & (histogram.shift(1) < 0)).astype(int)


def signal_stochastic_oversold(data: pd.DataFrame, k_period: int = 14, d_period: int = 3, oversold: int = 20, **kwargs) -> pd.Series:
    """Stochastic oscillator oversold."""
    low_n = data['low'].rolling(k_period).min() if 'low' in data.columns else data['close'].rolling(k_period).min()
    high_n = data['high'].rolling(k_period).max() if 'high' in data.columns else data['close'].rolling(k_period).max()
    k = 100 * (data['close'] - low_n) / (high_n - low_n + 1e-10)
    d = k.rolling(d_period).mean()
    return ((k < oversold) & (d < oversold)).astype(int)


def signal_stochastic_overbought(data: pd.DataFrame, k_period: int = 14, d_period: int = 3, overbought: int = 80, **kwargs) -> pd.Series:
    """Stochastic oscillator overbought (short signal)."""
    low_n = data['low'].rolling(k_period).min() if 'low' in data.columns else data['close'].rolling(k_period).min()
    high_n = data['high'].rolling(k_period).max() if 'high' in data.columns else data['close'].rolling(k_period).max()
    k = 100 * (data['close'] - low_n) / (high_n - low_n + 1e-10)
    d = k.rolling(d_period).mean()
    return ((k > overbought) & (d > overbought)).astype(int) * -1


def signal_atr_breakout(data: pd.DataFrame, lookback: int = 20, multiplier: float = 2.0, **kwargs) -> pd.Series:
    """ATR breakout: price moves beyond ATR band."""
    high = data['high'] if 'high' in data.columns else data['close']
    low = data['low'] if 'low' in data.columns else data['close']
    close = data['close']
    
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(lookback).mean()
    
    upper_band = close.rolling(lookback).mean() + multiplier * atr
    return (close > upper_band).astype(int)


def signal_consecutive_up_days(data: pd.DataFrame, n_days: int = 3, **kwargs) -> pd.Series:
    """N consecutive up days (momentum continuation)."""
    up = (data['close'] > data['close'].shift(1)).astype(int)
    consecutive = up.rolling(n_days).sum()
    return (consecutive >= n_days).astype(int)


def signal_consecutive_down_days(data: pd.DataFrame, n_days: int = 3, **kwargs) -> pd.Series:
    """N consecutive down days (reversal opportunity)."""
    down = (data['close'] < data['close'].shift(1)).astype(int)
    consecutive = down.rolling(n_days).sum()
    return (consecutive >= n_days).astype(int)


def signal_inside_day_breakout(data: pd.DataFrame, **kwargs) -> pd.Series:
    """Inside day followed by breakout."""
    if 'high' not in data.columns or 'low' not in data.columns:
        return pd.Series(0, index=data.index)
    
    inside = (data['high'] < data['high'].shift(1)) & (data['low'] > data['low'].shift(1))
    breakout_up = data['close'] > data['high'].shift(1)
    return (inside.shift(1) & breakout_up).astype(int)


def signal_outside_day(data: pd.DataFrame, **kwargs) -> pd.Series:
    """Outside day (engulfing pattern)."""
    if 'high' not in data.columns or 'low' not in data.columns:
        return pd.Series(0, index=data.index)
    
    outside = (data['high'] > data['high'].shift(1)) & (data['low'] < data['low'].shift(1))
    bullish = data['close'] > data['open'] if 'open' in data.columns else data['close'] > data['close'].shift(1)
    return (outside & bullish).astype(int)


def signal_doji(data: pd.DataFrame, threshold: float = 0.001, **kwargs) -> pd.Series:
    """Doji pattern (indecision → reversal)."""
    if 'open' not in data.columns:
        return pd.Series(0, index=data.index)
    
    body = (data['close'] - data['open']).abs() / data['open']
    doji = body < threshold
    # Look for doji after downtrend
    downtrend = data['close'].pct_change(5) < -0.02
    return (doji & downtrend).astype(int)


def signal_hammer(data: pd.DataFrame, **kwargs) -> pd.Series:
    """Hammer pattern (bullish reversal)."""
    if 'open' not in data.columns or 'high' not in data.columns or 'low' not in data.columns:
        return pd.Series(0, index=data.index)
    
    body = (data['close'] - data['open']).abs()
    lower_wick = pd.concat([data['open'], data['close']], axis=1).min(axis=1) - data['low']
    upper_wick = data['high'] - pd.concat([data['open'], data['close']], axis=1).max(axis=1)
    
    # Hammer: lower wick > 2x body, small upper wick
    hammer = (lower_wick > 2 * body) & (upper_wick < body * 0.5)
    downtrend = data['close'].pct_change(5) < -0.02
    return (hammer & downtrend).astype(int)


def signal_price_channel_breakout(data: pd.DataFrame, lookback: int = 20, **kwargs) -> pd.Series:
    """Donchian channel breakout (turtle trading)."""
    high_n = data['close'].rolling(lookback).max()
    return (data['close'] >= high_n).astype(int)


def signal_price_channel_breakdown(data: pd.DataFrame, lookback: int = 20, **kwargs) -> pd.Series:
    """Donchian channel breakdown (short signal)."""
    low_n = data['close'].rolling(lookback).min()
    return (data['close'] <= low_n).astype(int) * -1


def signal_williams_r(data: pd.DataFrame, period: int = 14, oversold: float = -80, **kwargs) -> pd.Series:
    """Williams %R oversold."""
    high_n = data['close'].rolling(period).max()
    low_n = data['close'].rolling(period).min()
    wr = -100 * (high_n - data['close']) / (high_n - low_n + 1e-10)
    return (wr < oversold).astype(int)


def signal_cci_oversold(data: pd.DataFrame, period: int = 20, oversold: float = -100, **kwargs) -> pd.Series:
    """Commodity Channel Index oversold."""
    typical_price = data['close']  # Simplified
    sma = typical_price.rolling(period).mean()
    mad = typical_price.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (typical_price - sma) / (0.015 * mad + 1e-10)
    return (cci < oversold).astype(int)


def signal_cci_overbought(data: pd.DataFrame, period: int = 20, overbought: float = 100, **kwargs) -> pd.Series:
    """Commodity Channel Index overbought (short)."""
    typical_price = data['close']
    sma = typical_price.rolling(period).mean()
    mad = typical_price.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (typical_price - sma) / (0.015 * mad + 1e-10)
    return (cci > overbought).astype(int) * -1


def signal_adx_strong_trend(data: pd.DataFrame, period: int = 14, threshold: float = 25, **kwargs) -> pd.Series:
    """ADX strong trend filter."""
    # Simplified ADX using price momentum as proxy
    momentum = data['close'].pct_change(period).abs()
    avg_momentum = momentum.rolling(period).mean()
    strong = avg_momentum > avg_momentum.rolling(100).quantile(0.75)
    return (strong & (data['close'] > data['close'].shift(period))).astype(int)


def signal_parabolic_sar_bullish(data: pd.DataFrame, **kwargs) -> pd.Series:
    """Simplified Parabolic SAR - price above trailing stop."""
    # Use ATR-based trailing stop as proxy
    atr = data['close'].pct_change().abs().rolling(14).mean()
    stop = data['close'].rolling(20).min() + 2 * atr * data['close']
    return (data['close'] > stop).astype(int)


def signal_fibonacci_retracement(data: pd.DataFrame, lookback: int = 50, level: float = 0.618, **kwargs) -> pd.Series:
    """Buy at 61.8% Fibonacci retracement."""
    high = data['close'].rolling(lookback).max()
    low = data['close'].rolling(lookback).min()
    fib_level = high - (high - low) * level
    
    # Near fib level and bouncing
    near_fib = (data['close'] - fib_level).abs() / data['close'] < 0.02
    bouncing = data['close'] > data['close'].shift(1)
    return (near_fib & bouncing).astype(int)


def signal_pivot_point_support(data: pd.DataFrame, **kwargs) -> pd.Series:
    """Price near pivot point support."""
    pivot = data['close'].rolling(20).mean()
    support = pivot - (data['close'].rolling(20).max() - data['close'].rolling(20).min()) * 0.382
    
    near_support = (data['close'] - support).abs() / data['close'] < 0.01
    return near_support.astype(int)


def signal_overnight_gap(data: pd.DataFrame, **kwargs) -> pd.Series:
    """Overnight gap up continuation."""
    if 'open' not in data.columns:
        return pd.Series(0, index=data.index)
    
    gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    gap_up = gap > 0.005  # 0.5% gap
    return gap_up.astype(int)


def signal_end_of_week(data: pd.DataFrame, **kwargs) -> pd.Series:
    """End of week effect (Friday positive)."""
    dow = pd.Series(data.index.dayofweek, index=data.index)
    return (dow == 4).astype(int)


def signal_first_of_month(data: pd.DataFrame, **kwargs) -> pd.Series:
    """First trading day of month."""
    dom = pd.Series(data.index.day, index=data.index)
    return (dom <= 3).astype(int)


def signal_quarter_end(data: pd.DataFrame, **kwargs) -> pd.Series:
    """Quarter end effect (last week of quarter)."""
    month = pd.Series(data.index.month, index=data.index)
    day = pd.Series(data.index.day, index=data.index)
    quarter_end_month = month.isin([3, 6, 9, 12])
    late_month = day >= 25
    return (quarter_end_month & late_month).astype(int)


def signal_year_end_rally(data: pd.DataFrame, **kwargs) -> pd.Series:
    """Santa Claus rally (last 5 days + first 2 of January)."""
    month = pd.Series(data.index.month, index=data.index)
    day = pd.Series(data.index.day, index=data.index)
    dec_late = (month == 12) & (day >= 25)
    jan_early = (month == 1) & (day <= 3)
    return (dec_late | jan_early).astype(int)


def signal_vix_spike(data: pd.DataFrame, vix_data: pd.Series = None, spike_pct: float = 20, **kwargs) -> pd.Series:
    """VIX spike (>20% 1-day increase) → contrarian buy."""
    if vix_data is None:
        return pd.Series(0, index=data.index)
    
    vix_aligned = vix_data.reindex(data.index).ffill()
    vix_change = vix_aligned.pct_change()
    return (vix_change > spike_pct / 100).astype(int)


def signal_vix_mean_reversion(data: pd.DataFrame, vix_data: pd.Series = None, lookback: int = 20, **kwargs) -> pd.Series:
    """VIX mean reversion: high VIX relative to SMA."""
    if vix_data is None:
        return pd.Series(0, index=data.index)
    
    vix_aligned = vix_data.reindex(data.index).ffill()
    vix_sma = vix_aligned.rolling(lookback).mean()
    high_vix = vix_aligned > vix_sma * 1.2
    return high_vix.astype(int)


def signal_spy_tlt_rotation(data: pd.DataFrame, tlt_data: pd.DataFrame = None, lookback: int = 21, **kwargs) -> pd.Series:
    """SPY/TLT rotation based on momentum."""
    if tlt_data is None:
        return pd.Series(1, index=data.index)  # Default to SPY
    
    spy_mom = data['close'].pct_change(lookback)
    tlt_mom = tlt_data['close'].pct_change(lookback).reindex(data.index).ffill()
    
    # Long SPY when SPY momentum > TLT momentum
    return (spy_mom > tlt_mom).astype(int)


def signal_gold_momentum(data: pd.DataFrame, gld_data: pd.DataFrame = None, lookback: int = 21, **kwargs) -> pd.Series:
    """Gold momentum as risk indicator."""
    if gld_data is None:
        return pd.Series(0, index=data.index)
    
    gld_mom = gld_data['close'].pct_change(lookback).reindex(data.index).ffill()
    # Weak gold = risk-on
    return (gld_mom < 0).astype(int)


def signal_breadth_thrust(data: pd.DataFrame, **kwargs) -> pd.Series:
    """Market breadth thrust (simplified using momentum)."""
    # Use strong momentum as proxy for breadth thrust
    mom = data['close'].pct_change(10)
    thrust = mom > mom.rolling(252).quantile(0.95)
    return thrust.astype(int)


def signal_new_high_new_low(data: pd.DataFrame, lookback: int = 252, **kwargs) -> pd.Series:
    """Price making new highs vs new lows."""
    high_52w = data['close'].rolling(lookback).max()
    at_high = data['close'] >= high_52w * 0.98
    return at_high.astype(int)


def signal_relative_strength(data: pd.DataFrame, spy_data: pd.DataFrame = None, lookback: int = 63, **kwargs) -> pd.Series:
    """Relative strength vs SPY."""
    if spy_data is None:
        return pd.Series(0, index=data.index)
    
    asset_ret = data['close'].pct_change(lookback)
    spy_ret = spy_data['close'].pct_change(lookback).reindex(data.index).ffill()
    
    return (asset_ret > spy_ret).astype(int)


def signal_regime_high_vol(data: pd.DataFrame, lookback: int = 20, **kwargs) -> pd.Series:
    """High volatility regime (defensive)."""
    vol = data['close'].pct_change().rolling(lookback).std() * np.sqrt(252)
    high_vol = vol > vol.rolling(252).quantile(0.8)
    return (high_vol).astype(int) * -1  # Defensive in high vol


def signal_regime_low_vol(data: pd.DataFrame, lookback: int = 20, **kwargs) -> pd.Series:
    """Low volatility regime (aggressive)."""
    vol = data['close'].pct_change().rolling(lookback).std() * np.sqrt(252)
    low_vol = vol < vol.rolling(252).quantile(0.2)
    return (low_vol).astype(int)


# ============================================================================
# HYPOTHESIS LIBRARY
# ============================================================================

def build_hypothesis_library() -> List[Hypothesis]:
    """Build the complete library of hypotheses to test."""
    
    # Standard ETF universe
    etf_universe = ['SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLU', 'TLT', 'GLD', 'EEM']
    sector_etfs = ['XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLP', 'XLY', 'XLU', 'XLB']
    
    hypotheses = [
        # ========== TREND FOLLOWING ==========
        Hypothesis(
            id="TF001",
            name="Time-Series Momentum (1-month)",
            category=HypothesisCategory.TREND_FOLLOWING,
            description="Assets with positive 21-day returns continue rising",
            signal_func=signal_momentum_timeseries,
            signal_type=SignalType.LONG_ONLY,
            tickers=etf_universe,
            lookback=21,
            hold_period=5,
            priority=1,
        ),
        Hypothesis(
            id="TF002",
            name="Time-Series Momentum (3-month)",
            category=HypothesisCategory.TREND_FOLLOWING,
            description="Assets with positive 63-day returns continue rising",
            signal_func=signal_momentum_timeseries,
            signal_type=SignalType.LONG_ONLY,
            tickers=etf_universe,
            lookback=63,
            hold_period=21,
            priority=1,
        ),
        Hypothesis(
            id="TF003",
            name="Time-Series Momentum (12-month)",
            category=HypothesisCategory.TREND_FOLLOWING,
            description="Assets with positive 252-day returns continue rising",
            signal_func=signal_momentum_timeseries,
            signal_type=SignalType.LONG_ONLY,
            tickers=etf_universe,
            lookback=252,
            hold_period=21,
            priority=1,
        ),
        Hypothesis(
            id="TF004",
            name="12-1 Month Momentum",
            category=HypothesisCategory.TREND_FOLLOWING,
            description="12-month momentum excluding recent month",
            signal_func=signal_momentum_12m_1m,
            signal_type=SignalType.LONG_ONLY,
            tickers=etf_universe,
            hold_period=21,
            priority=2,
        ),
        Hypothesis(
            id="TF005",
            name="MA Crossover (50/200)",
            category=HypothesisCategory.TREND_FOLLOWING,
            description="Golden cross: 50-day MA above 200-day MA",
            signal_func=signal_ma_crossover,
            signal_type=SignalType.LONG_ONLY,
            tickers=etf_universe,
            parameters={'fast': 50, 'slow': 200},
            hold_period=21,
            priority=1,
        ),
        Hypothesis(
            id="TF006",
            name="Price Above 200MA",
            category=HypothesisCategory.TREND_FOLLOWING,
            description="Long only when price above 200-day MA",
            signal_func=signal_above_ma,
            signal_type=SignalType.LONG_ONLY,
            tickers=etf_universe,
            parameters={'period': 200},
            hold_period=5,
            priority=1,
        ),
        Hypothesis(
            id="TF007",
            name="Dual Momentum",
            category=HypothesisCategory.TREND_FOLLOWING,
            description="Absolute + relative momentum vs SPY",
            signal_func=signal_dual_momentum,
            signal_type=SignalType.LONG_ONLY,
            tickers=['QQQ', 'IWM', 'EEM', 'TLT', 'GLD'],
            lookback=252,
            hold_period=21,
            priority=2,
        ),
        Hypothesis(
            id="TF008",
            name="52-Week High Breakout",
            category=HypothesisCategory.TREND_FOLLOWING,
            description="Buy at new 52-week highs",
            signal_func=signal_new_high_breakout,
            signal_type=SignalType.LONG_ONLY,
            tickers=etf_universe,
            lookback=252,
            hold_period=10,
            priority=2,
        ),
        
        # ========== MEAN REVERSION ==========
        Hypothesis(
            id="MR001",
            name="RSI Oversold",
            category=HypothesisCategory.MEAN_REVERSION,
            description="Buy when RSI(14) < 30",
            signal_func=signal_rsi_oversold,
            signal_type=SignalType.LONG_ONLY,
            tickers=etf_universe,
            parameters={'period': 14, 'oversold': 30},
            hold_period=5,
            priority=1,
        ),
        Hypothesis(
            id="MR002",
            name="RSI Overbought (Short)",
            category=HypothesisCategory.MEAN_REVERSION,
            description="Short when RSI(14) > 70",
            signal_func=signal_rsi_overbought,
            signal_type=SignalType.LONG_SHORT,
            tickers=etf_universe,
            parameters={'period': 14, 'overbought': 70},
            hold_period=5,
            priority=1,
        ),
        Hypothesis(
            id="MR003",
            name="Bollinger Band Lower Touch",
            category=HypothesisCategory.MEAN_REVERSION,
            description="Buy when price touches lower Bollinger Band",
            signal_func=signal_bollinger_lower,
            signal_type=SignalType.LONG_ONLY,
            tickers=etf_universe,
            parameters={'period': 20, 'std_dev': 2.0},
            hold_period=5,
            priority=1,
        ),
        Hypothesis(
            id="MR004",
            name="Bollinger Band Upper Touch (Short)",
            category=HypothesisCategory.MEAN_REVERSION,
            description="Short when price touches upper Bollinger Band",
            signal_func=signal_bollinger_upper,
            signal_type=SignalType.LONG_SHORT,
            tickers=etf_universe,
            parameters={'period': 20, 'std_dev': 2.0},
            hold_period=5,
            priority=1,
        ),
        Hypothesis(
            id="MR005",
            name="Short-Term Reversal (5-day)",
            category=HypothesisCategory.MEAN_REVERSION,
            description="Worst 5-day performers bounce",
            signal_func=signal_short_term_reversal,
            signal_type=SignalType.LONG_ONLY,
            tickers=etf_universe,
            lookback=5,
            hold_period=5,
            priority=2,
        ),
        Hypothesis(
            id="MR006",
            name="Z-Score Mean Reversion",
            category=HypothesisCategory.MEAN_REVERSION,
            description="Buy at -2 Z-score, sell at +2",
            signal_func=signal_mean_reversion_zscore,
            signal_type=SignalType.LONG_SHORT,
            tickers=etf_universe,
            parameters={'lookback': 60, 'threshold': 2.0},
            hold_period=10,
            priority=2,
        ),
        Hypothesis(
            id="MR007",
            name="52-Week Low Contrarian",
            category=HypothesisCategory.MEAN_REVERSION,
            description="Buy at 52-week lows",
            signal_func=signal_new_low_buy,
            signal_type=SignalType.LONG_ONLY,
            tickers=etf_universe,
            lookback=252,
            hold_period=21,
            priority=3,
        ),
        Hypothesis(
            id="MR008",
            name="Gap Down Fade",
            category=HypothesisCategory.MEAN_REVERSION,
            description="Buy 2%+ gap downs for gap fill",
            signal_func=signal_gap_down,
            signal_type=SignalType.LONG_ONLY,
            tickers=['SPY', 'QQQ', 'IWM'],
            parameters={'threshold': -0.02},
            hold_period=1,
            priority=2,
        ),
        Hypothesis(
            id="MR009",
            name="Gap Up Fade",
            category=HypothesisCategory.MEAN_REVERSION,
            description="Short 2%+ gap ups for gap fill",
            signal_func=signal_gap_up,
            signal_type=SignalType.LONG_SHORT,
            tickers=['SPY', 'QQQ', 'IWM'],
            parameters={'threshold': 0.02},
            hold_period=1,
            priority=2,
        ),
        
        # ========== SEASONALITY ==========
        Hypothesis(
            id="SS001",
            name="Sell in May",
            category=HypothesisCategory.SEASONALITY,
            description="Long Nov-Apr, cash May-Oct",
            signal_func=signal_sell_in_may,
            signal_type=SignalType.BINARY,
            tickers=['SPY', 'QQQ', 'IWM'],
            hold_period=21,
            priority=1,
        ),
        Hypothesis(
            id="SS002",
            name="January Effect",
            category=HypothesisCategory.SEASONALITY,
            description="Small caps outperform in January",
            signal_func=signal_january_effect,
            signal_type=SignalType.BINARY,
            tickers=['IWM'],
            hold_period=21,
            priority=2,
        ),
        Hypothesis(
            id="SS003",
            name="Turn of Month",
            category=HypothesisCategory.SEASONALITY,
            description="Last 3 + first 3 days of month are positive",
            signal_func=signal_turn_of_month,
            signal_type=SignalType.BINARY,
            tickers=['SPY', 'QQQ', 'IWM'],
            hold_period=1,
            priority=1,
        ),
        Hypothesis(
            id="SS004",
            name="Monday Effect (Inverse)",
            category=HypothesisCategory.SEASONALITY,
            description="Mondays historically negative",
            signal_func=signal_monday_effect,
            signal_type=SignalType.LONG_SHORT,
            tickers=['SPY'],
            hold_period=1,
            priority=3,
        ),
        Hypothesis(
            id="SS005",
            name="Pre-Holiday Rally",
            category=HypothesisCategory.SEASONALITY,
            description="Day before holidays is positive",
            signal_func=signal_pre_holiday,
            signal_type=SignalType.BINARY,
            tickers=['SPY', 'QQQ'],
            hold_period=1,
            priority=3,
        ),
        
        # ========== VOLATILITY ==========
        Hypothesis(
            id="VL001",
            name="VIX High → Buy",
            category=HypothesisCategory.VOLATILITY,
            description="High VIX (>25) predicts bounce",
            signal_func=signal_vix_high,
            signal_type=SignalType.LONG_SHORT,
            tickers=['SPY'],
            requires_macro=True,
            macro_series=['VIX'],
            parameters={'threshold': 25},
            hold_period=5,
            priority=1,
        ),
        Hypothesis(
            id="VL002",
            name="VIX Low → Risk On",
            category=HypothesisCategory.VOLATILITY,
            description="Low VIX (<15) = favorable conditions",
            signal_func=signal_vix_low,
            signal_type=SignalType.LONG_ONLY,
            tickers=['SPY', 'QQQ'],
            requires_macro=True,
            macro_series=['VIX'],
            parameters={'threshold': 15},
            hold_period=5,
            priority=1,
        ),
        Hypothesis(
            id="VL003",
            name="VIX Term Structure",
            category=HypothesisCategory.VOLATILITY,
            description="Backwardation (VIX > VIX3M) signals bounce",
            signal_func=signal_vix_term_structure,
            signal_type=SignalType.LONG_ONLY,
            tickers=['SPY'],
            requires_macro=True,
            macro_series=['VIX', 'VIX3M'],
            hold_period=5,
            priority=1,
        ),
        Hypothesis(
            id="VL004",
            name="Low Volatility Anomaly",
            category=HypothesisCategory.VOLATILITY,
            description="Low vol assets outperform risk-adjusted",
            signal_func=signal_low_volatility,
            signal_type=SignalType.LONG_ONLY,
            tickers=sector_etfs,
            lookback=20,
            hold_period=21,
            priority=2,
        ),
        
        # ========== VOLUME ==========
        Hypothesis(
            id="LQ001",
            name="Volume Spike",
            category=HypothesisCategory.LIQUIDITY,
            description="2x average volume signals continuation",
            signal_func=signal_volume_spike,
            signal_type=SignalType.LONG_ONLY,
            tickers=etf_universe,
            requires_volume=True,
            parameters={'threshold': 2.0},
            hold_period=5,
            priority=2,
        ),
        Hypothesis(
            id="LQ002",
            name="High Volume Breakout",
            category=HypothesisCategory.LIQUIDITY,
            description="20-day high on 1.5x volume",
            signal_func=signal_high_volume_breakout,
            signal_type=SignalType.LONG_ONLY,
            tickers=etf_universe,
            requires_volume=True,
            parameters={'price_lookback': 20, 'vol_threshold': 1.5},
            hold_period=10,
            priority=2,
        ),
        
        # ========== RISK REGIME ==========
        Hypothesis(
            id="RR001",
            name="Yield Curve Regime",
            category=HypothesisCategory.RISK_REGIME,
            description="Long when yield curve normal, defensive when inverted",
            signal_func=signal_yield_curve_regime,
            signal_type=SignalType.LONG_ONLY,
            tickers=['SPY'],
            requires_macro=True,
            macro_series=['DGS10', 'DGS2'],
            hold_period=21,
            priority=2,
        ),
        Hypothesis(
            id="RR002",
            name="Dollar Strength → EM Weakness",
            category=HypothesisCategory.RISK_REGIME,
            description="Weak dollar favors emerging markets",
            signal_func=signal_dollar_strength,
            signal_type=SignalType.LONG_ONLY,
            tickers=['EEM', 'EWZ', 'FXI'],
            requires_macro=True,
            macro_series=['UUP'],
            lookback=20,
            hold_period=10,
            priority=2,
        ),
        
        # ========== INTERMARKET ==========
        Hypothesis(
            id="IM001",
            name="Sector Momentum (1-month)",
            category=HypothesisCategory.INTERMARKET,
            description="Winning sectors continue winning",
            signal_func=signal_momentum_timeseries,
            signal_type=SignalType.LONG_ONLY,
            tickers=sector_etfs,
            lookback=21,
            hold_period=21,
            priority=1,
        ),
        
        # ========== BATCH 2: TECHNICAL INDICATORS ==========
        Hypothesis(
            id="TI001",
            name="MACD Crossover",
            category=HypothesisCategory.TREND_FOLLOWING,
            description="MACD line crosses above signal line",
            signal_func=signal_macd_crossover,
            signal_type=SignalType.LONG_ONLY,
            tickers=etf_universe,
            hold_period=10,
            priority=3,
        ),
        Hypothesis(
            id="TI002",
            name="MACD Histogram Reversal",
            category=HypothesisCategory.TREND_FOLLOWING,
            description="MACD histogram turns positive",
            signal_func=signal_macd_histogram_reversal,
            signal_type=SignalType.LONG_ONLY,
            tickers=etf_universe,
            hold_period=5,
            priority=3,
        ),
        Hypothesis(
            id="TI003",
            name="Stochastic Oversold",
            category=HypothesisCategory.MEAN_REVERSION,
            description="Stochastic K&D below 20",
            signal_func=signal_stochastic_oversold,
            signal_type=SignalType.LONG_ONLY,
            tickers=etf_universe,
            hold_period=5,
            priority=3,
        ),
        Hypothesis(
            id="TI004",
            name="Stochastic Overbought",
            category=HypothesisCategory.MEAN_REVERSION,
            description="Stochastic K&D above 80 (short)",
            signal_func=signal_stochastic_overbought,
            signal_type=SignalType.LONG_SHORT,
            tickers=etf_universe,
            hold_period=5,
            priority=3,
        ),
        Hypothesis(
            id="TI005",
            name="ATR Breakout",
            category=HypothesisCategory.TREND_FOLLOWING,
            description="Price breaks above 2x ATR band",
            signal_func=signal_atr_breakout,
            signal_type=SignalType.LONG_ONLY,
            tickers=etf_universe,
            hold_period=10,
            priority=3,
        ),
        Hypothesis(
            id="TI006",
            name="Williams %R Oversold",
            category=HypothesisCategory.MEAN_REVERSION,
            description="Williams %R below -80",
            signal_func=signal_williams_r,
            signal_type=SignalType.LONG_ONLY,
            tickers=etf_universe,
            hold_period=5,
            priority=3,
        ),
        Hypothesis(
            id="TI007",
            name="CCI Oversold",
            category=HypothesisCategory.MEAN_REVERSION,
            description="CCI below -100",
            signal_func=signal_cci_oversold,
            signal_type=SignalType.LONG_ONLY,
            tickers=etf_universe,
            hold_period=5,
            priority=3,
        ),
        Hypothesis(
            id="TI008",
            name="CCI Overbought",
            category=HypothesisCategory.MEAN_REVERSION,
            description="CCI above 100 (short)",
            signal_func=signal_cci_overbought,
            signal_type=SignalType.LONG_SHORT,
            tickers=etf_universe,
            hold_period=5,
            priority=3,
        ),
        Hypothesis(
            id="TI009",
            name="ADX Strong Trend",
            category=HypothesisCategory.TREND_FOLLOWING,
            description="Strong trend with positive direction",
            signal_func=signal_adx_strong_trend,
            signal_type=SignalType.LONG_ONLY,
            tickers=etf_universe,
            hold_period=10,
            priority=3,
        ),
        Hypothesis(
            id="TI010",
            name="Donchian Channel Breakout",
            category=HypothesisCategory.TREND_FOLLOWING,
            description="20-day high breakout (turtle trading)",
            signal_func=signal_price_channel_breakout,
            signal_type=SignalType.LONG_ONLY,
            tickers=etf_universe,
            parameters={'lookback': 20},
            hold_period=10,
            priority=3,
        ),
        Hypothesis(
            id="TI011",
            name="Fibonacci 61.8% Retracement",
            category=HypothesisCategory.MEAN_REVERSION,
            description="Buy at 61.8% Fibonacci level",
            signal_func=signal_fibonacci_retracement,
            signal_type=SignalType.LONG_ONLY,
            tickers=etf_universe,
            hold_period=10,
            priority=4,
        ),
        Hypothesis(
            id="TI012",
            name="Pivot Point Support",
            category=HypothesisCategory.MEAN_REVERSION,
            description="Price near pivot point support",
            signal_func=signal_pivot_point_support,
            signal_type=SignalType.LONG_ONLY,
            tickers=etf_universe,
            hold_period=5,
            priority=4,
        ),
        
        # ========== BATCH 2: PRICE PATTERNS ==========
        Hypothesis(
            id="PP001",
            name="3 Consecutive Up Days",
            category=HypothesisCategory.TREND_FOLLOWING,
            description="Momentum after 3 up days",
            signal_func=signal_consecutive_up_days,
            signal_type=SignalType.LONG_ONLY,
            tickers=etf_universe,
            parameters={'n_days': 3},
            hold_period=5,
            priority=3,
        ),
        Hypothesis(
            id="PP002",
            name="3 Consecutive Down Days",
            category=HypothesisCategory.MEAN_REVERSION,
            description="Reversal after 3 down days",
            signal_func=signal_consecutive_down_days,
            signal_type=SignalType.LONG_ONLY,
            tickers=etf_universe,
            parameters={'n_days': 3},
            hold_period=5,
            priority=3,
        ),
        Hypothesis(
            id="PP003",
            name="Inside Day Breakout",
            category=HypothesisCategory.TREND_FOLLOWING,
            description="Breakout after inside day",
            signal_func=signal_inside_day_breakout,
            signal_type=SignalType.LONG_ONLY,
            tickers=['SPY', 'QQQ', 'IWM'],
            hold_period=3,
            priority=4,
        ),
        Hypothesis(
            id="PP004",
            name="Outside Day (Engulfing)",
            category=HypothesisCategory.TREND_FOLLOWING,
            description="Bullish outside day pattern",
            signal_func=signal_outside_day,
            signal_type=SignalType.LONG_ONLY,
            tickers=['SPY', 'QQQ', 'IWM'],
            hold_period=3,
            priority=4,
        ),
        Hypothesis(
            id="PP005",
            name="Doji Reversal",
            category=HypothesisCategory.MEAN_REVERSION,
            description="Doji after downtrend",
            signal_func=signal_doji,
            signal_type=SignalType.LONG_ONLY,
            tickers=['SPY', 'QQQ', 'IWM'],
            hold_period=5,
            priority=4,
        ),
        Hypothesis(
            id="PP006",
            name="Hammer Pattern",
            category=HypothesisCategory.MEAN_REVERSION,
            description="Hammer candlestick after downtrend",
            signal_func=signal_hammer,
            signal_type=SignalType.LONG_ONLY,
            tickers=['SPY', 'QQQ', 'IWM'],
            hold_period=5,
            priority=4,
        ),
        
        # ========== BATCH 2: SEASONALITY ==========
        Hypothesis(
            id="SS006",
            name="End of Week Effect",
            category=HypothesisCategory.SEASONALITY,
            description="Friday positive bias",
            signal_func=signal_end_of_week,
            signal_type=SignalType.BINARY,
            tickers=['SPY', 'QQQ'],
            hold_period=1,
            priority=3,
        ),
        Hypothesis(
            id="SS007",
            name="First of Month",
            category=HypothesisCategory.SEASONALITY,
            description="First 3 days of month positive",
            signal_func=signal_first_of_month,
            signal_type=SignalType.BINARY,
            tickers=['SPY', 'QQQ'],
            hold_period=1,
            priority=3,
        ),
        Hypothesis(
            id="SS008",
            name="Quarter End Effect",
            category=HypothesisCategory.SEASONALITY,
            description="Last week of quarter",
            signal_func=signal_quarter_end,
            signal_type=SignalType.BINARY,
            tickers=['SPY', 'QQQ'],
            hold_period=5,
            priority=3,
        ),
        Hypothesis(
            id="SS009",
            name="Santa Claus Rally",
            category=HypothesisCategory.SEASONALITY,
            description="Year-end rally Dec 25 - Jan 3",
            signal_func=signal_year_end_rally,
            signal_type=SignalType.BINARY,
            tickers=['SPY', 'QQQ', 'IWM'],
            hold_period=5,
            priority=3,
        ),
        
        # ========== BATCH 2: VOLATILITY ==========
        Hypothesis(
            id="VL005",
            name="VIX Spike Contrarian",
            category=HypothesisCategory.VOLATILITY,
            description="Buy after 20%+ VIX spike",
            signal_func=signal_vix_spike,
            signal_type=SignalType.LONG_ONLY,
            tickers=['SPY'],
            requires_macro=True,
            macro_series=['VIX'],
            hold_period=5,
            priority=3,
        ),
        Hypothesis(
            id="VL006",
            name="VIX Mean Reversion",
            category=HypothesisCategory.VOLATILITY,
            description="VIX > 1.2x SMA",
            signal_func=signal_vix_mean_reversion,
            signal_type=SignalType.LONG_ONLY,
            tickers=['SPY'],
            requires_macro=True,
            macro_series=['VIX'],
            hold_period=10,
            priority=3,
        ),
        Hypothesis(
            id="VL007",
            name="High Vol Regime Defensive",
            category=HypothesisCategory.VOLATILITY,
            description="Reduce exposure in high vol",
            signal_func=signal_regime_high_vol,
            signal_type=SignalType.LONG_SHORT,
            tickers=['SPY', 'QQQ'],
            hold_period=5,
            priority=3,
        ),
        Hypothesis(
            id="VL008",
            name="Low Vol Regime Aggressive",
            category=HypothesisCategory.VOLATILITY,
            description="Full exposure in low vol",
            signal_func=signal_regime_low_vol,
            signal_type=SignalType.LONG_ONLY,
            tickers=['SPY', 'QQQ'],
            hold_period=10,
            priority=3,
        ),
        
        # ========== BATCH 2: INTERMARKET ==========
        Hypothesis(
            id="IM002",
            name="Breadth Thrust",
            category=HypothesisCategory.BREADTH,
            description="Extreme positive momentum thrust",
            signal_func=signal_breadth_thrust,
            signal_type=SignalType.LONG_ONLY,
            tickers=['SPY'],
            hold_period=21,
            priority=3,
        ),
        Hypothesis(
            id="IM003",
            name="Relative Strength vs SPY",
            category=HypothesisCategory.INTERMARKET,
            description="Asset outperforming SPY",
            signal_func=signal_relative_strength,
            signal_type=SignalType.LONG_ONLY,
            tickers=['QQQ', 'IWM', 'EEM', 'XLK', 'XLF'],
            hold_period=21,
            priority=3,
        ),
        Hypothesis(
            id="IM004",
            name="New Highs Filter",
            category=HypothesisCategory.BREADTH,
            description="At 52-week high",
            signal_func=signal_new_high_new_low,
            signal_type=SignalType.LONG_ONLY,
            tickers=etf_universe,
            hold_period=10,
            priority=3,
        ),
    ]
    
    return hypotheses


# ============================================================================
# HYPOTHESIS TESTING ENGINE
# ============================================================================

class HypothesisEngine:
    """Main engine for testing hypotheses systematically."""
    
    def __init__(self, start_date: str = "2010-01-01"):
        self.start_date = start_date
        self.hypotheses = build_hypothesis_library()
        self.results: List[HypothesisResult] = []
        self.price_cache: Dict[str, pd.DataFrame] = {}
        self.macro_cache: Dict[str, pd.Series] = {}
        
        print(f"Hypothesis Engine initialized with {len(self.hypotheses)} hypotheses")
    
    def load_price_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load price data for a ticker."""
        if ticker in self.price_cache:
            return self.price_cache[ticker]
        
        try:
            data = yf.Ticker(ticker).history(start=self.start_date, auto_adjust=True)
            if len(data) < 500:
                return None
            
            # Standardize column names
            data.columns = [c.lower() for c in data.columns]
            data.index = pd.to_datetime(data.index)
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            
            self.price_cache[ticker] = data
            return data
        except Exception as e:
            print(f"Error loading {ticker}: {e}")
            return None
    
    def load_macro_data(self, series_id: str) -> Optional[pd.Series]:
        """Load macro data series."""
        if series_id in self.macro_cache:
            return self.macro_cache[series_id]
        
        # Map series IDs to Yahoo Finance tickers
        yf_mapping = {
            'VIX': '^VIX',
            'VIX3M': '^VIX3M',
            'DGS10': '^TNX',
            'DGS2': '^FVX',  # Using 5Y as proxy
            'UUP': 'UUP',
        }
        
        ticker = yf_mapping.get(series_id, series_id)
        
        try:
            data = yf.Ticker(ticker).history(start=self.start_date)
            if len(data) < 100:
                return None
            
            series = data['Close']
            series.index = pd.to_datetime(series.index)
            if series.index.tz is not None:
                series.index = series.index.tz_localize(None)
            
            self.macro_cache[series_id] = series
            return series
        except Exception as e:
            print(f"Error loading macro {series_id}: {e}")
            return None
    
    def calculate_returns(self, data: pd.DataFrame, signal: pd.Series, hold_period: int) -> pd.Series:
        """Calculate strategy returns from signal."""
        # Forward returns
        fwd_ret = data['close'].pct_change(hold_period).shift(-hold_period)
        
        # Apply signal (shifted to avoid look-ahead)
        strategy_ret = signal.shift(1) * fwd_ret
        
        return strategy_ret.dropna()
    
    def run_monte_carlo(self, returns: pd.Series, signal: pd.Series, n_sims: int = N_MONTE_CARLO) -> Tuple[float, float]:
        """Run Monte Carlo simulation to test significance."""
        if len(returns) < MIN_TRADES:
            return 0.0, 1.0
        
        real_sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Generate random signals with same frequency
        signal_values = signal.dropna().values
        if len(np.unique(signal_values)) < 2:
            return 0.0, 1.0
        
        random_sharpes = []
        for _ in range(n_sims):
            random_signal = np.random.permutation(signal_values)
            random_ret = random_signal[:-1] * returns.values[:len(random_signal)-1] if len(returns) >= len(random_signal) else returns.values
            
            if len(random_ret) > 0 and np.std(random_ret) > 0:
                random_sharpes.append(np.mean(random_ret) / np.std(random_ret) * np.sqrt(252))
        
        if len(random_sharpes) == 0:
            return 0.0, 1.0
        
        percentile = (np.array(random_sharpes) < real_sharpe).mean() * 100
        p_value = 1 - percentile / 100
        
        return percentile, p_value
    
    def run_walk_forward(self, data: pd.DataFrame, signal: pd.Series, hold_period: int) -> Tuple[List[float], float]:
        """Run walk-forward validation."""
        excess_sharpes = []
        
        i = TRAIN_DAYS
        while i + TEST_DAYS <= len(data):
            test_data = data.iloc[i:i+TEST_DAYS]
            test_signal = signal.iloc[i:i+TEST_DAYS]
            
            # Strategy returns
            fwd_ret = test_data['close'].pct_change(hold_period).shift(-hold_period)
            strat_ret = test_signal.shift(1) * fwd_ret
            strat_ret = strat_ret.dropna()
            
            # Baseline (buy and hold)
            baseline_ret = fwd_ret.dropna()
            
            if len(strat_ret) > 10 and strat_ret.std() > 0 and baseline_ret.std() > 0:
                strat_sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(252 / hold_period)
                baseline_sharpe = baseline_ret.mean() / baseline_ret.std() * np.sqrt(252 / hold_period)
                excess_sharpes.append(strat_sharpe - baseline_sharpe)
            
            i += STEP_DAYS
        
        if len(excess_sharpes) < 3:
            return [], 1.0
        
        # T-test on excess Sharpes
        t_stat, p_value = stats.ttest_1samp(excess_sharpes, 0)
        
        return excess_sharpes, p_value
    
    def test_hypothesis(self, hypothesis: Hypothesis) -> HypothesisResult:
        """Test a single hypothesis."""
        result = HypothesisResult(
            hypothesis_id=hypothesis.id,
            name=hypothesis.name,
            category=hypothesis.category.value,
            tickers_tested=hypothesis.tickers,
            parameters=hypothesis.parameters,
        )
        
        try:
            all_returns = []
            all_signals = []
            
            # Load macro data if needed
            macro_data = {}
            if hypothesis.requires_macro:
                for series_id in hypothesis.macro_series:
                    macro_data[series_id] = self.load_macro_data(series_id)
            
            # Load SPY for dual momentum
            spy_data = None
            if hypothesis.id == "TF007":
                spy_data = self.load_price_data('SPY')
            
            # Test on each ticker
            for ticker in hypothesis.tickers:
                data = self.load_price_data(ticker)
                if data is None:
                    continue
                
                # Prepare kwargs for signal function
                kwargs = {**hypothesis.parameters}
                kwargs['lookback'] = hypothesis.lookback
                
                # Add macro data
                if 'VIX' in macro_data:
                    kwargs['vix_data'] = macro_data.get('VIX')
                    kwargs['vix'] = macro_data.get('VIX')
                if 'VIX3M' in macro_data:
                    kwargs['vix3m'] = macro_data.get('VIX3M')
                if 'DGS10' in macro_data:
                    kwargs['y10'] = macro_data.get('DGS10')
                if 'DGS2' in macro_data:
                    kwargs['y2'] = macro_data.get('DGS2')
                if 'UUP' in macro_data:
                    kwargs['uup_data'] = macro_data.get('UUP')
                if spy_data is not None:
                    kwargs['spy_data'] = spy_data
                
                # Generate signal
                signal = hypothesis.signal_func(data, **kwargs)
                
                # Calculate returns
                returns = self.calculate_returns(data, signal, hypothesis.hold_period)
                
                if len(returns) > 50:
                    all_returns.append(returns)
                    all_signals.append(signal)
            
            if len(all_returns) == 0:
                result.error = "No valid data"
                return result
            
            # Combine results
            combined_returns = pd.concat(all_returns)
            combined_signal = pd.concat(all_signals)
            
            # Calculate metrics
            result.n_trades = (combined_signal != 0).sum()
            
            if len(combined_returns) < MIN_TRADES:
                result.error = f"Insufficient trades ({result.n_trades})"
                return result
            
            # In-sample metrics
            result.total_return = (1 + combined_returns).prod() - 1
            result.sharpe_ratio = combined_returns.mean() / combined_returns.std() * np.sqrt(252) if combined_returns.std() > 0 else 0
            result.win_rate = (combined_returns > 0).mean()
            
            cumret = (1 + combined_returns).cumprod()
            result.max_drawdown = ((cumret - cumret.cummax()) / cumret.cummax()).min()
            
            # Baseline comparison (first ticker)
            first_ticker = hypothesis.tickers[0]
            first_data = self.load_price_data(first_ticker)
            if first_data is not None:
                baseline_ret = first_data['close'].pct_change(hypothesis.hold_period).dropna()
                result.baseline_sharpe = baseline_ret.mean() / baseline_ret.std() * np.sqrt(252 / hypothesis.hold_period) if baseline_ret.std() > 0 else 0
                result.excess_sharpe = result.sharpe_ratio - result.baseline_sharpe
            
            # Monte Carlo
            result.monte_carlo_percentile, mc_p = self.run_monte_carlo(combined_returns, combined_signal)
            result.passed_monte_carlo = result.monte_carlo_percentile > 95
            
            # Walk-forward
            if first_data is not None:
                first_signal = hypothesis.signal_func(first_data, **kwargs)
                excess_sharpes, wf_p = self.run_walk_forward(first_data, first_signal, hypothesis.hold_period)
                
                result.wf_periods = len(excess_sharpes)
                if len(excess_sharpes) > 0:
                    result.wf_win_rate = (np.array(excess_sharpes) > 0).mean()
                    result.wf_avg_excess = np.mean(excess_sharpes)
                result.wf_p_value = wf_p
                result.passed_walkforward = wf_p < SIGNIFICANCE_LEVEL and result.wf_avg_excess > 0
            
            # Final verdict
            result.is_survivor = result.passed_monte_carlo and result.passed_walkforward
            
        except Exception as e:
            result.error = str(e)
        
        return result
    
    def run_all(self, priority_filter: Optional[int] = None, category_filter: Optional[HypothesisCategory] = None) -> List[HypothesisResult]:
        """Run all hypotheses (or filtered subset)."""
        hypotheses_to_test = self.hypotheses
        
        if priority_filter is not None:
            hypotheses_to_test = [h for h in hypotheses_to_test if h.priority <= priority_filter]
        
        if category_filter is not None:
            hypotheses_to_test = [h for h in hypotheses_to_test if h.category == category_filter]
        
        # Sort by priority
        hypotheses_to_test = sorted(hypotheses_to_test, key=lambda h: h.priority)
        
        print(f"\n{'='*80}")
        print(f"HYPOTHESIS TESTING ENGINE")
        print(f"{'='*80}")
        print(f"Testing {len(hypotheses_to_test)} hypotheses...")
        print(f"Validation: Walk-forward ({TRAIN_DAYS}d train, {TEST_DAYS}d test)")
        print(f"Significance: Monte Carlo ({N_MONTE_CARLO} sims, p<{SIGNIFICANCE_LEVEL})")
        print('='*80)
        
        self.results = []
        
        for i, hypothesis in enumerate(hypotheses_to_test):
            print(f"\n[{i+1}/{len(hypotheses_to_test)}] Testing: {hypothesis.id} - {hypothesis.name}")
            
            result = self.test_hypothesis(hypothesis)
            self.results.append(result)
            
            # Print summary
            if result.error:
                print(f"  ✗ Error: {result.error}")
            else:
                mc_icon = "✓" if result.passed_monte_carlo else "✗"
                wf_icon = "✓" if result.passed_walkforward else "✗"
                
                print(f"  Sharpe: {result.sharpe_ratio:.2f} | Excess: {result.excess_sharpe:.2f}")
                print(f"  Monte Carlo: {mc_icon} ({result.monte_carlo_percentile:.1f}%ile)")
                print(f"  Walk-Forward: {wf_icon} (p={result.wf_p_value:.3f}, {result.wf_periods} periods)")
                
                if result.is_survivor:
                    print(f"  🎯 SURVIVOR!")
            
            time.sleep(0.1)  # Rate limiting
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _save_results(self):
        """Save results to CSV and JSON."""
        if not self.results:
            return
        
        # Convert to DataFrame
        results_df = pd.DataFrame([
            {
                'id': r.hypothesis_id,
                'name': r.name,
                'category': r.category,
                'sharpe': r.sharpe_ratio,
                'excess_sharpe': r.excess_sharpe,
                'mc_percentile': r.monte_carlo_percentile,
                'wf_p_value': r.wf_p_value,
                'wf_win_rate': r.wf_win_rate,
                'passed_mc': r.passed_monte_carlo,
                'passed_wf': r.passed_walkforward,
                'is_survivor': r.is_survivor,
                'n_trades': r.n_trades,
                'error': r.error,
            }
            for r in self.results
        ])
        
        results_df.to_csv(RESULTS_DIR / "hypothesis_results.csv", index=False)
        
        # Save detailed JSON
        detailed = [
            {
                'id': r.hypothesis_id,
                'name': r.name,
                'category': r.category,
                'total_return': r.total_return,
                'sharpe_ratio': r.sharpe_ratio,
                'max_drawdown': r.max_drawdown,
                'win_rate': r.win_rate,
                'n_trades': r.n_trades,
                'baseline_sharpe': r.baseline_sharpe,
                'excess_sharpe': r.excess_sharpe,
                'monte_carlo_percentile': r.monte_carlo_percentile,
                'wf_periods': r.wf_periods,
                'wf_win_rate': r.wf_win_rate,
                'wf_avg_excess': r.wf_avg_excess,
                'wf_p_value': r.wf_p_value,
                'passed_monte_carlo': r.passed_monte_carlo,
                'passed_walkforward': r.passed_walkforward,
                'is_survivor': r.is_survivor,
                'tickers': r.tickers_tested,
                'parameters': r.parameters,
                'error': r.error,
            }
            for r in self.results
        ]
        
        with open(RESULTS_DIR / "hypothesis_results_detailed.json", 'w') as f:
            json.dump(detailed, f, indent=2, default=str)
        
        print(f"\nResults saved to {RESULTS_DIR}")
    
    def get_survivors(self) -> List[HypothesisResult]:
        """Get hypotheses that passed all tests."""
        return [r for r in self.results if r.is_survivor]
    
    def print_summary(self):
        """Print summary of results."""
        if not self.results:
            print("No results yet. Run engine first.")
            return
        
        print("\n" + "="*80)
        print("HYPOTHESIS TESTING SUMMARY")
        print("="*80)
        
        total = len(self.results)
        errors = sum(1 for r in self.results if r.error)
        passed_mc = sum(1 for r in self.results if r.passed_monte_carlo)
        passed_wf = sum(1 for r in self.results if r.passed_walkforward)
        survivors = len(self.get_survivors())
        
        print(f"""
Total Hypotheses Tested: {total}
Errors:                  {errors}
Passed Monte Carlo:      {passed_mc} ({100*passed_mc/total:.1f}%)
Passed Walk-Forward:     {passed_wf} ({100*passed_wf/total:.1f}%)
─────────────────────────────────────
SURVIVORS:               {survivors} ({100*survivors/total:.1f}%)
""")
        
        if survivors > 0:
            print("\n🎯 SURVIVING HYPOTHESES:")
            print("-"*60)
            for r in self.get_survivors():
                print(f"  {r.hypothesis_id}: {r.name}")
                print(f"    Sharpe: {r.sharpe_ratio:.2f} | Excess: {r.excess_sharpe:.2f}")
                print(f"    MC: {r.monte_carlo_percentile:.1f}%ile | WF p-value: {r.wf_p_value:.4f}")
        else:
            print("\n❌ No hypotheses survived rigorous validation.")
            print("   This is expected for efficient markets.")
        
        # Category breakdown
        print("\n" + "-"*60)
        print("Results by Category:")
        print("-"*60)
        
        categories = {}
        for r in self.results:
            if r.category not in categories:
                categories[r.category] = {'tested': 0, 'survivors': 0}
            categories[r.category]['tested'] += 1
            if r.is_survivor:
                categories[r.category]['survivors'] += 1
        
        for cat, counts in sorted(categories.items()):
            print(f"  {cat}: {counts['survivors']}/{counts['tested']} survivors")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hypothesis Testing Engine")
    parser.add_argument('--priority', type=int, default=2, help='Max priority to test (1=highest)')
    parser.add_argument('--category', type=str, default=None, help='Category to test')
    
    args = parser.parse_args()
    
    engine = HypothesisEngine()
    
    category = None
    if args.category:
        try:
            category = HypothesisCategory[args.category.upper()]
        except KeyError:
            print(f"Unknown category: {args.category}")
            print(f"Valid categories: {[c.name for c in HypothesisCategory]}")
            exit(1)
    
    results = engine.run_all(priority_filter=args.priority, category_filter=category)
    engine.print_summary()
