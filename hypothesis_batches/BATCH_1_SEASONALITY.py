#!/usr/bin/env python3
"""
HYPOTHESIS BATCH 1: SEASONALITY (H42-H55)
==========================================
Quick wins with clear calendar patterns.
Est. time: ~8 minutes for 5 tests
API calls: 1 (SPY only)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


# ============================================================================
# SIGNAL FUNCTIONS - SEASONALITY
# ============================================================================

def signal_monday_effect(data: pd.DataFrame, **kwargs) -> pd.Series:
    """H42: Monday Effect - Historically negative Mondays."""
    dow = pd.Series(data.index.dayofweek, index=data.index)
    return (dow != 0).astype(int)  # Avoid Monday = 1, else 0


def signal_friday_effect(data: pd.DataFrame, **kwargs) -> pd.Series:
    """H43: Friday Effect - Long into weekend."""
    dow = pd.Series(data.index.dayofweek, index=data.index)
    return (dow == 4).astype(int)


def signal_turn_of_month(data: pd.DataFrame, **kwargs) -> pd.Series:
    """H44: Turn of Month - Last 1 + first 4 trading days."""
    dom = pd.Series(data.index.day, index=data.index)
    
    # Get end of month day for each date
    eom = data.index.to_series().apply(lambda x: (x + pd.offsets.MonthEnd(0)).day)
    
    # Last trading day of month (within 1 day of EOM) or first 4 days
    is_eom = (eom - dom) <= 1
    is_bom = dom <= 4
    return (is_eom | is_bom).astype(int)


def signal_opex_week(data: pd.DataFrame, **kwargs) -> pd.Series:
    """H45: Monthly OpEx Week - 3rd Friday ± 2 days."""
    # Find 3rd Friday of each month
    dates = pd.Series(data.index, index=data.index)
    
    def is_opex_window(dt):
        # Find 3rd Friday
        first_day = dt.replace(day=1)
        first_friday = first_day + pd.Timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + pd.Timedelta(days=14)
        
        # Window: 3rd Friday ± 2 days
        return abs((dt - third_friday).days) <= 2
    
    return dates.apply(is_opex_window).astype(int)


def signal_first_half_month(data: pd.DataFrame, **kwargs) -> pd.Series:
    """H46: First Half vs Second Half of Month."""
    dom = pd.Series(data.index.day, index=data.index)
    return (dom <= 10).astype(int)


def signal_january_effect(data: pd.DataFrame, **kwargs) -> pd.Series:
    """H47: January Effect - Long small caps in January."""
    month = pd.Series(data.index.month, index=data.index)
    return (month == 1).astype(int)


def signal_sell_in_may(data: pd.DataFrame, **kwargs) -> pd.Series:
    """H48: Sell in May - Nov-Apr long, May-Oct cash."""
    month = pd.Series(data.index.month, index=data.index)
    return ((month >= 11) | (month <= 4)).astype(int)


def signal_september_effect(data: pd.DataFrame, **kwargs) -> pd.Series:
    """H49: September Effect - Reduce in September."""
    month = pd.Series(data.index.month, index=data.index)
    return (month != 9).astype(int)


def signal_december_rally(data: pd.DataFrame, **kwargs) -> pd.Series:
    """H50: December Rally - Long December."""
    month = pd.Series(data.index.month, index=data.index)
    return (month == 12).astype(int)


def signal_santa_claus_rally(data: pd.DataFrame, **kwargs) -> pd.Series:
    """H51: Santa Claus Rally - Last 5 Dec + first 2 Jan."""
    month = pd.Series(data.index.month, index=data.index)
    day = pd.Series(data.index.day, index=data.index)
    
    dec_late = (month == 12) & (day >= 25)
    jan_early = (month == 1) & (day <= 3)
    return (dec_late | jan_early).astype(int)


def signal_pre_holiday(data: pd.DataFrame, **kwargs) -> pd.Series:
    """H52: Pre-Holiday Effect - Day before US holidays."""
    # Major US market holidays (simplified - just check for gaps in trading)
    # A "pre-holiday" is when the next trading day is 2+ calendar days away
    dates = pd.Series(data.index, index=data.index)
    next_date = dates.shift(-1)
    gap_days = (next_date - dates).dt.days
    
    # Long weekends (3+ day gap) indicate holiday
    return (gap_days >= 3).astype(int).shift(1).fillna(0)


def signal_quarter_end(data: pd.DataFrame, **kwargs) -> pd.Series:
    """H53: Quarter-End Effect - Last week of quarter."""
    month = pd.Series(data.index.month, index=data.index)
    day = pd.Series(data.index.day, index=data.index)
    
    quarter_end_month = month.isin([3, 6, 9, 12])
    late_month = day >= 24
    return (quarter_end_month & late_month).astype(int)


# Note: H54/H55 FOMC Drift requires external FOMC dates - skip for now
# Can be added with manual FOMC calendar


# ============================================================================
# HYPOTHESIS DEFINITIONS
# ============================================================================

BATCH_1_HYPOTHESES = [
    {
        'id': 'H42',
        'name': 'Monday Effect',
        'category': 'Seasonality',
        'description': 'Mondays historically negative - avoid or short',
        'signal_func': signal_monday_effect,
        'tickers': ['SPY'],
        'hold_period': 1,
        'priority': 3,
    },
    {
        'id': 'H43',
        'name': 'Friday Effect',
        'category': 'Seasonality',
        'description': 'Long into weekend',
        'signal_func': signal_friday_effect,
        'tickers': ['SPY'],
        'hold_period': 1,
        'priority': 3,
    },
    {
        'id': 'H44',
        'name': 'Turn of Month',
        'category': 'Seasonality',
        'description': 'Last 1 + first 4 trading days positive',
        'signal_func': signal_turn_of_month,
        'tickers': ['SPY', 'QQQ', 'IWM'],
        'hold_period': 1,
        'priority': 1,  # Well-documented
    },
    {
        'id': 'H45',
        'name': 'Monthly OpEx Week',
        'category': 'Seasonality',
        'description': '3rd Friday ± 2 days volatility',
        'signal_func': signal_opex_week,
        'tickers': ['SPY'],
        'hold_period': 1,
        'priority': 3,
    },
    {
        'id': 'H46',
        'name': 'First Half of Month',
        'category': 'Seasonality',
        'description': 'Trading days 1-10 vs 11-end',
        'signal_func': signal_first_half_month,
        'tickers': ['SPY'],
        'hold_period': 1,
        'priority': 3,
    },
    {
        'id': 'H47',
        'name': 'January Effect',
        'category': 'Seasonality',
        'description': 'Small caps outperform in January',
        'signal_func': signal_january_effect,
        'tickers': ['IWM'],
        'hold_period': 21,
        'priority': 2,
    },
    {
        'id': 'H48',
        'name': 'Sell in May',
        'category': 'Seasonality',
        'description': 'Nov-Apr long, May-Oct reduced',
        'signal_func': signal_sell_in_may,
        'tickers': ['SPY', 'QQQ', 'IWM'],
        'hold_period': 21,
        'priority': 1,  # Famous, well-tested
    },
    {
        'id': 'H49',
        'name': 'September Effect',
        'category': 'Seasonality',
        'description': 'September historically weak',
        'signal_func': signal_september_effect,
        'tickers': ['SPY'],
        'hold_period': 21,
        'priority': 2,
    },
    {
        'id': 'H50',
        'name': 'December Rally',
        'category': 'Seasonality',
        'description': 'December historically positive',
        'signal_func': signal_december_rally,
        'tickers': ['SPY'],
        'hold_period': 21,
        'priority': 2,
    },
    {
        'id': 'H51',
        'name': 'Santa Claus Rally',
        'category': 'Seasonality',
        'description': 'Last 5 Dec + first 2 Jan',
        'signal_func': signal_santa_claus_rally,
        'tickers': ['SPY', 'QQQ'],
        'hold_period': 5,
        'priority': 2,
    },
    {
        'id': 'H52',
        'name': 'Pre-Holiday Effect',
        'category': 'Seasonality',
        'description': 'Day before market holidays positive',
        'signal_func': signal_pre_holiday,
        'tickers': ['SPY'],
        'hold_period': 1,
        'priority': 3,
    },
    {
        'id': 'H53',
        'name': 'Quarter-End Effect',
        'category': 'Seasonality',
        'description': 'Window dressing last week of quarter',
        'signal_func': signal_quarter_end,
        'tickers': ['SPY'],
        'hold_period': 5,
        'priority': 3,
    },
]


def get_batch_1_hypotheses():
    """Return all Batch 1 hypotheses."""
    return BATCH_1_HYPOTHESES


if __name__ == "__main__":
    print(f"Batch 1: Seasonality - {len(BATCH_1_HYPOTHESES)} hypotheses")
    for h in BATCH_1_HYPOTHESES:
        print(f"  {h['id']}: {h['name']} (Priority: {h['priority']})")
