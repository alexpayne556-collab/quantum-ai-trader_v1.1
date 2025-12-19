#!/usr/bin/env python3
"""
GPU TESTS - Run this on Shadow PC
Command: python RUN_GPU_TESTS.py
"""

import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm
import os

# Find database
DB_PATHS = ['data/market_data.db', 'market_data.db', '../data/market_data.db']
DB_PATH = None
for p in DB_PATHS:
    if os.path.exists(p):
        DB_PATH = p
        break

if not DB_PATH:
    print("ERROR: market_data.db not found!")
    print("Run: git pull")
    exit(1)

print(f"Using database: {DB_PATH}")

# Check table exists
conn = sqlite3.connect(DB_PATH)
tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
print(f"Tables found: {tables}")

if 'ohlcv' not in tables:
    print("ERROR: 'ohlcv' table not found!")
    print(f"Available tables: {tables}")
    exit(1)

# Load data
print("Loading data...")
df = pd.read_sql("SELECT * FROM ohlcv", conn)
conn.close()
print(f"Loaded {len(df):,} rows, {df['ticker'].nunique():,} tickers")

# Try GPU
try:
    import cupy as cp
    GPU = True
    print("GPU: ENABLED (CuPy)")
except:
    GPU = False
    print("GPU: Disabled (using NumPy)")

# Try Numba
try:
    from numba import jit, prange
    NUMBA = True
    print("Numba: ENABLED")
except:
    NUMBA = False

def calculate_t_stat(returns):
    """Harvey-Liu-Zhu t-statistic"""
    if len(returns) < 30:
        return 0, 0, 0
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    if std == 0:
        return 0, 0, 0
    t = mean / (std / np.sqrt(len(returns)))
    return mean, len(returns), t

results = []

# ============ CALENDAR EFFECTS ============
print("\n" + "="*50)
print("TESTING: Calendar Effects")
print("="*50)

df['date'] = pd.to_datetime(df['date'])
df['returns'] = df.groupby('ticker')['close'].pct_change()
df['dow'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['dom'] = df['date'].dt.day

# Day of week
for day in tqdm(range(5), desc="Day of Week"):
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    rets = df[df['dow'] == day]['returns'].dropna()
    mean, n, t = calculate_t_stat(rets)
    results.append({
        'category': 'CALENDAR',
        'strategy': f'{day_names[day]}_Effect',
        'avg_return': mean,
        'n_samples': n,
        't_stat': t,
        'significant': abs(t) > 3.0
    })

# Month of year
for month in tqdm(range(1, 13), desc="Month of Year"):
    month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    rets = df[df['month'] == month]['returns'].dropna()
    mean, n, t = calculate_t_stat(rets)
    results.append({
        'category': 'CALENDAR',
        'strategy': f'{month_names[month-1]}_Effect',
        'avg_return': mean,
        'n_samples': n,
        't_stat': t,
        'significant': abs(t) > 3.0
    })

# Turn of month (first 3 days, last 3 days)
first_3 = df[df['dom'] <= 3]['returns'].dropna()
mean, n, t = calculate_t_stat(first_3)
results.append({
    'category': 'CALENDAR',
    'strategy': 'First3Days_Month',
    'avg_return': mean,
    'n_samples': n,
    't_stat': t,
    'significant': abs(t) > 3.0
})

last_3 = df[df['dom'] >= 28]['returns'].dropna()
mean, n, t = calculate_t_stat(last_3)
results.append({
    'category': 'CALENDAR',
    'strategy': 'Last3Days_Month',
    'avg_return': mean,
    'n_samples': n,
    't_stat': t,
    'significant': abs(t) > 3.0
})

# ============ VOLATILITY REGIMES ============
print("\n" + "="*50)
print("TESTING: Volatility Regimes")
print("="*50)

for lookback in tqdm([5, 10, 20, 60], desc="Vol Lookbacks"):
    df[f'vol_{lookback}'] = df.groupby('ticker')['returns'].transform(
        lambda x: x.rolling(lookback).std()
    )
    
    for threshold in [0.25, 0.5, 0.75]:
        vol_thresh = df[f'vol_{lookback}'].quantile(threshold)
        
        # Low vol regime
        low_vol = df[df[f'vol_{lookback}'] < vol_thresh]['returns'].dropna()
        mean, n, t = calculate_t_stat(low_vol)
        results.append({
            'category': 'VOLATILITY',
            'strategy': f'LowVol{lookback}_Q{int(threshold*100)}',
            'avg_return': mean,
            'n_samples': n,
            't_stat': t,
            'significant': abs(t) > 3.0
        })
        
        # High vol regime
        high_vol = df[df[f'vol_{lookback}'] >= vol_thresh]['returns'].dropna()
        mean, n, t = calculate_t_stat(high_vol)
        results.append({
            'category': 'VOLATILITY',
            'strategy': f'HighVol{lookback}_Q{int(threshold*100)}',
            'avg_return': mean,
            'n_samples': n,
            't_stat': t,
            'significant': abs(t) > 3.0
        })

# ============ MOMENTUM ============
print("\n" + "="*50)
print("TESTING: Momentum Strategies")
print("="*50)

for lookback in tqdm([5, 10, 20, 60, 120, 252], desc="Momentum Lookbacks"):
    df[f'mom_{lookback}'] = df.groupby('ticker')['close'].transform(
        lambda x: x.pct_change(lookback)
    )
    
    for hold in [1, 5, 10, 20]:
        df[f'fwd_{hold}'] = df.groupby('ticker')['close'].transform(
            lambda x: x.shift(-hold) / x - 1
        )
        
        for pct in [0.1, 0.2, 0.3]:
            thresh = df[f'mom_{lookback}'].quantile(1 - pct)
            winners = df[df[f'mom_{lookback}'] >= thresh][f'fwd_{hold}'].dropna()
            mean, n, t = calculate_t_stat(winners)
            results.append({
                'category': 'MOMENTUM',
                'strategy': f'Mom{lookback}_Top{int(pct*100)}_H{hold}',
                'avg_return': mean,
                'n_samples': n,
                't_stat': t,
                'significant': abs(t) > 3.0
            })
            
            thresh_low = df[f'mom_{lookback}'].quantile(pct)
            losers = df[df[f'mom_{lookback}'] <= thresh_low][f'fwd_{hold}'].dropna()
            mean, n, t = calculate_t_stat(losers)
            results.append({
                'category': 'MOMENTUM',
                'strategy': f'MomRev{lookback}_Bot{int(pct*100)}_H{hold}',
                'avg_return': mean,
                'n_samples': n,
                't_stat': t,
                'significant': abs(t) > 3.0
            })

# ============ VOLUME PATTERNS ============
print("\n" + "="*50)
print("TESTING: Volume Patterns")
print("="*50)

for lookback in tqdm([5, 10, 20], desc="Volume Lookbacks"):
    df[f'vol_ratio_{lookback}'] = df.groupby('ticker')['volume'].transform(
        lambda x: x / x.rolling(lookback).mean()
    )
    
    for thresh in [1.5, 2.0, 3.0]:
        high_vol = df[df[f'vol_ratio_{lookback}'] >= thresh]
        
        for hold in [1, 3, 5]:
            df[f'fwd_{hold}'] = df.groupby('ticker')['close'].transform(
                lambda x: x.shift(-hold) / x - 1
            )
            rets = high_vol[f'fwd_{hold}'].dropna()
            mean, n, t = calculate_t_stat(rets)
            results.append({
                'category': 'VOLUME',
                'strategy': f'HighVol{lookback}_{thresh}x_H{hold}',
                'avg_return': mean,
                'n_samples': n,
                't_stat': t,
                'significant': abs(t) > 3.0
            })

# ============ GAP PATTERNS ============
print("\n" + "="*50)
print("TESTING: Gap Patterns")
print("="*50)

df['gap'] = df.groupby('ticker').apply(
    lambda x: x['open'] / x['close'].shift(1) - 1
).reset_index(level=0, drop=True)

for gap_thresh in tqdm([0.02, 0.03, 0.05], desc="Gap Thresholds"):
    for hold in [1, 3, 5]:
        df[f'fwd_{hold}'] = df.groupby('ticker')['close'].transform(
            lambda x: x.shift(-hold) / x - 1
        )
        
        # Gap up
        gap_up = df[df['gap'] >= gap_thresh][f'fwd_{hold}'].dropna()
        mean, n, t = calculate_t_stat(gap_up)
        results.append({
            'category': 'GAP',
            'strategy': f'GapUp{int(gap_thresh*100)}pct_H{hold}',
            'avg_return': mean,
            'n_samples': n,
            't_stat': t,
            'significant': abs(t) > 3.0
        })
        
        # Gap down
        gap_dn = df[df['gap'] <= -gap_thresh][f'fwd_{hold}'].dropna()
        mean, n, t = calculate_t_stat(gap_dn)
        results.append({
            'category': 'GAP',
            'strategy': f'GapDown{int(gap_thresh*100)}pct_H{hold}',
            'avg_return': mean,
            'n_samples': n,
            't_stat': t,
            'significant': abs(t) > 3.0
        })

# ============ RESULTS ============
print("\n" + "="*50)
print("RESULTS SUMMARY")
print("="*50)

results_df = pd.DataFrame(results)
sig = results_df[results_df['significant']]

print(f"\nTotal strategies tested: {len(results_df)}")
print(f"Significant (|t| > 3.0): {len(sig)} ({100*len(sig)/len(results_df):.1f}%)")

# Save
output_file = 'data/GPU_TEST_RESULTS.csv' if os.path.exists('data') else 'GPU_TEST_RESULTS.csv'
results_df.to_csv(output_file, index=False)
print(f"\nSaved to: {output_file}")

# Top 20
print("\n" + "="*50)
print("TOP 20 STRATEGIES")
print("="*50)
top20 = results_df.nlargest(20, 't_stat')
for _, row in top20.iterrows():
    print(f"{row['category']:12} | {row['strategy']:30} | t={row['t_stat']:.2f} | ret={row['avg_return']*100:.2f}%")

print("\nDONE!")
