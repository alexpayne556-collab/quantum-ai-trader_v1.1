#!/usr/bin/env python3
"""
HYPOTHESIS BATCH LOADER & EXECUTOR
===================================
Loads all hypothesis batches and provides unified execution.

Usage:
    from hypothesis_batches import load_all_batches, get_all_hypotheses
    
    all_hypotheses = get_all_hypotheses()
    print(f"Total: {len(all_hypotheses)} hypotheses")
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

# Add batch directory to path
BATCH_DIR = Path(__file__).parent
sys.path.insert(0, str(BATCH_DIR))

# ============================================================================
# IMPORT ALL BATCHES
# ============================================================================

try:
    from BATCH_1_SEASONALITY import get_batch_1_hypotheses, BATCH_1_HYPOTHESES
except ImportError:
    BATCH_1_HYPOTHESES = []

try:
    from BATCH_2_VOLATILITY import get_batch_2_hypotheses, BATCH_2_HYPOTHESES
except ImportError:
    BATCH_2_HYPOTHESES = []

try:
    from BATCH_3_MOMENTUM import get_batch_3_hypotheses, BATCH_3_HYPOTHESES
except ImportError:
    BATCH_3_HYPOTHESES = []

try:
    from BATCH_4_MEAN_REVERSION import get_batch_4_hypotheses, BATCH_4_HYPOTHESES
except ImportError:
    BATCH_4_HYPOTHESES = []

try:
    from BATCH_5_CROSS_ASSET import get_batch_5_hypotheses, BATCH_5_HYPOTHESES
except ImportError:
    BATCH_5_HYPOTHESES = []

try:
    from BATCH_6_MACRO import get_batch_6_hypotheses, BATCH_6_HYPOTHESES
except ImportError:
    BATCH_6_HYPOTHESES = []

try:
    from BATCH_7_TECHNICAL import get_batch_7_hypotheses, BATCH_7_HYPOTHESES
except ImportError:
    BATCH_7_HYPOTHESES = []

try:
    from BATCH_8_SENTIMENT import get_batch_8_hypotheses, BATCH_8_HYPOTHESES
except ImportError:
    BATCH_8_HYPOTHESES = []

try:
    from BATCH_9_CREATIVE import get_batch_9_hypotheses, BATCH_9_HYPOTHESES
except ImportError:
    BATCH_9_HYPOTHESES = []


# ============================================================================
# BATCH REGISTRY
# ============================================================================

BATCH_REGISTRY = {
    1: {
        'name': 'Seasonality',
        'hypotheses': BATCH_1_HYPOTHESES,
        'description': 'Calendar effects (Monday, January, Sell in May, etc.)',
        'est_time_min': 8,
        'api_calls': 1,
        'ids': 'H42-H53',
    },
    2: {
        'name': 'Volatility',
        'hypotheses': BATCH_2_HYPOTHESES,
        'description': 'VIX-based signals (term structure, VRP, regime)',
        'est_time_min': 12,
        'api_calls': 3,
        'ids': 'H20-H41',
    },
    3: {
        'name': 'Momentum',
        'hypotheses': BATCH_3_HYPOTHESES,
        'description': 'Asset class and cross-sectional momentum',
        'est_time_min': 10,
        'api_calls': 4,
        'ids': 'H11-H15, H70-H79',
    },
    4: {
        'name': 'Mean Reversion',
        'hypotheses': BATCH_4_HYPOTHESES,
        'description': 'RSI, Bollinger, Z-score mean reversion',
        'est_time_min': 10,
        'api_calls': 1,
        'ids': 'H16-H27',
    },
    5: {
        'name': 'Cross-Asset & Credit',
        'hypotheses': BATCH_5_HYPOTHESES,
        'description': 'Stock-bond corr, HYG/LQD, copper/gold, FX',
        'est_time_min': 12,
        'api_calls': 4,
        'ids': 'H56-H73',
    },
    6: {
        'name': 'Yield Curve & Macro',
        'hypotheses': BATCH_6_HYPOTHESES,
        'description': 'PMI, unemployment, yield curve, inflation',
        'est_time_min': 15,
        'api_calls': 5,
        'ids': 'H65-H68, H100-H110',
    },
    7: {
        'name': 'Technical Patterns',
        'hypotheses': BATCH_7_HYPOTHESES,
        'description': '52W high/low, Donchian, volume, OBV',
        'est_time_min': 10,
        'api_calls': 0,
        'ids': 'H88-H99',
    },
    8: {
        'name': 'Sentiment Proxies',
        'hypotheses': BATCH_8_HYPOTHESES,
        'description': 'ARKK, XBI, high beta vs low vol',
        'est_time_min': 10,
        'api_calls': 3,
        'ids': 'H119-H127',
    },
    9: {
        'name': 'Creative/Novel (NEW!)',
        'hypotheses': BATCH_9_HYPOTHESES,
        'description': 'VIX turbulence, divergences, FOMC, regime HMM',
        'est_time_min': 18,
        'api_calls': 2,
        'ids': 'H128-H145',
    },
}


# ============================================================================
# LOADER FUNCTIONS
# ============================================================================

def get_all_hypotheses() -> List[Dict]:
    """Return all hypotheses from all batches."""
    all_hyps = []
    for batch_num, batch_info in BATCH_REGISTRY.items():
        for hyp in batch_info['hypotheses']:
            hyp_copy = hyp.copy()
            hyp_copy['batch'] = batch_num
            all_hyps.append(hyp_copy)
    return all_hyps


def get_batch_hypotheses(batch_num: int) -> List[Dict]:
    """Return hypotheses for a specific batch."""
    if batch_num not in BATCH_REGISTRY:
        raise ValueError(f"Batch {batch_num} not found")
    return BATCH_REGISTRY[batch_num]['hypotheses']


def count_hypotheses() -> Dict[str, int]:
    """Count hypotheses per batch."""
    counts = {}
    total = 0
    for batch_num, batch_info in BATCH_REGISTRY.items():
        count = len(batch_info['hypotheses'])
        counts[f"Batch {batch_num}: {batch_info['name']}"] = count
        total += count
    counts['TOTAL'] = total
    return counts


def list_batches() -> None:
    """Print summary of all batches."""
    print("\n" + "="*70)
    print("HYPOTHESIS BATCH REGISTRY - 116 HYPOTHESES READY!")
    print("="*70)
    
    total_hyps = 0
    total_time = 0
    
    for batch_num, info in BATCH_REGISTRY.items():
        count = len(info['hypotheses'])
        status = "✓" if count > 0 else "○"
        print(f"{status} Batch {batch_num:2d}: {info['name']:<25} | "
              f"{count:3d} hypotheses | ~{info['est_time_min']:2d} min | {info['ids']}")
        total_hyps += count
        if count > 0:
            total_time += info['est_time_min']
    
    print("-"*70)
    print(f"TOTAL: {total_hyps} hypotheses | ~{total_time} minutes")
    print("="*70)


def get_priority_hypotheses(priority: int = 1) -> List[Dict]:
    """Get hypotheses by priority level."""
    all_hyps = get_all_hypotheses()
    return [h for h in all_hyps if h.get('priority', 3) <= priority]


def get_execution_plan() -> Dict:
    """Get execution plan with estimated times."""
    plan = {
        'batches': [],
        'total_hypotheses': 0,
        'total_time_minutes': 0,
        'total_api_calls': 0,
    }
    
    for batch_num, info in BATCH_REGISTRY.items():
        if len(info['hypotheses']) > 0:
            batch_plan = {
                'batch': batch_num,
                'name': info['name'],
                'count': len(info['hypotheses']),
                'est_minutes': info['est_time_min'],
                'api_calls': info['api_calls'],
            }
            plan['batches'].append(batch_plan)
            plan['total_hypotheses'] += len(info['hypotheses'])
            plan['total_time_minutes'] += info['est_time_min']
            plan['total_api_calls'] += info['api_calls']
    
    return plan


def get_tickers_needed() -> Dict[str, set]:
    """Get all unique tickers needed across batches."""
    tickers_by_batch = {}
    all_tickers = set()
    
    for batch_num, info in BATCH_REGISTRY.items():
        batch_tickers = set()
        for hyp in info['hypotheses']:
            batch_tickers.update(hyp.get('tickers', []))
        tickers_by_batch[f"Batch {batch_num}"] = batch_tickers
        all_tickers.update(batch_tickers)
    
    tickers_by_batch['ALL'] = all_tickers
    return tickers_by_batch


def get_fred_series_needed() -> set:
    """Get all FRED series IDs needed."""
    fred_ids = set()
    for batch_num, info in BATCH_REGISTRY.items():
        for hyp in info['hypotheses']:
            fred_ids.update(hyp.get('requires_fred', []))
    return fred_ids


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Hypothesis Batch Manager')
    parser.add_argument('--list', action='store_true', help='List all batches')
    parser.add_argument('--count', action='store_true', help='Count hypotheses')
    parser.add_argument('--tickers', action='store_true', help='List tickers needed')
    parser.add_argument('--fred', action='store_true', help='List FRED series needed')
    parser.add_argument('--plan', action='store_true', help='Show execution plan')
    parser.add_argument('--priority', type=int, help='Filter by priority')
    args = parser.parse_args()
    
    if args.list or not any(vars(args).values()):
        list_batches()
    
    if args.count:
        counts = count_hypotheses()
        for k, v in counts.items():
            print(f"{k}: {v}")
    
    if args.tickers:
        tickers = get_tickers_needed()
        print("\n📊 TICKERS NEEDED:")
        for batch, ticker_set in tickers.items():
            if ticker_set:
                print(f"  {batch}: {', '.join(sorted(ticker_set))}")
    
    if args.fred:
        fred_ids = get_fred_series_needed()
        print(f"\n📈 FRED SERIES: {', '.join(sorted(fred_ids))}")
    
    if args.plan:
        plan = get_execution_plan()
        print("\n📋 EXECUTION PLAN:")
        print(f"  Total Hypotheses: {plan['total_hypotheses']}")
        print(f"  Total Time: ~{plan['total_time_minutes']} minutes")
        print(f"  Total API Calls: {plan['total_api_calls']}")
    
    if args.priority:
        priority_hyps = get_priority_hypotheses(args.priority)
        print(f"\n⭐ PRIORITY {args.priority} HYPOTHESES ({len(priority_hyps)}):")
        for h in priority_hyps:
            print(f"  {h['id']}: {h['name']}")
