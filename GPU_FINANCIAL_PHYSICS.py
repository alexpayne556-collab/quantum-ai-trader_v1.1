#!/usr/bin/env python3
"""
FINANCIAL PHYSICS - Deep Law Discovery
Maps the fundamental forces of markets like physicists map the universe.

Run on Shadow PC: python GPU_FINANCIAL_PHYSICS.py
"""

import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

# Find database
DB_PATHS = ['data/market_data.db', 'market_data.db']
DB_PATH = next((p for p in DB_PATHS if os.path.exists(p)), None)
if not DB_PATH:
    print("ERROR: Run git pull first")
    exit(1)

print(f"Database: {DB_PATH}")
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM ohlcv", conn)
conn.close()
print(f"Loaded {len(df):,} rows, {df['ticker'].nunique():,} tickers")

# Prepare data
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['ticker', 'date'])
df['returns'] = df.groupby('ticker')['close'].pct_change()

# Pre-compute forward returns
print("Computing forward returns...")
for h in [1, 2, 3, 5, 10, 15, 20, 40, 60]:
    df[f'fwd_{h}'] = df.groupby('ticker')['close'].transform(lambda x: x.shift(-h) / x - 1)

results = []

def calc_t(rets):
    """Harvey-Liu-Zhu t-statistic"""
    if len(rets) < 30: return 0, 0, 0
    m = np.mean(rets)
    s = np.std(rets, ddof=1)
    if s == 0: return 0, 0, 0
    return m, len(rets), m / (s / np.sqrt(len(rets)))

# ============================================================
# GRAVITATIONAL FORCES - Mean reversion strength at distances
# ============================================================
print("\n" + "="*60)
print("GRAVITATIONAL FORCES - Mean Reversion at Different Distances")
print("="*60)

for lookback in tqdm([5, 10, 20, 50, 100, 200], desc="Gravity"):
    df[f'dist_ma_{lookback}'] = (df['close'] - df.groupby('ticker')['close'].transform(
        lambda x: x.rolling(lookback).mean())) / df['close']
    
    # Test reversion strength at different distances
    for dist in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        for hold in [5, 10, 20]:
            # Below MA (should revert up)
            below = df[df[f'dist_ma_{lookback}'] < -dist][f'fwd_{hold}'].dropna()
            m, n, t = calc_t(below)
            results.append({
                'category': 'GRAVITY',
                'strategy': f'MeanRevert_MA{lookback}_Dist{int(dist*100)}_H{hold}',
                'avg_return': m, 'n_samples': n, 't_stat': t,
                'physics': f'Gravitational pull to MA{lookback} from {dist*100}% below'
            })

# ============================================================
# ORBITAL MECHANICS - Cyclical patterns
# ============================================================
print("\n" + "="*60)
print("ORBITAL MECHANICS - Cyclical Time Patterns")
print("="*60)

df['dow'] = df['date'].dt.dayofweek
df['dom'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['week'] = df['date'].dt.isocalendar().week
df['quarter'] = df['date'].dt.quarter

# Week of month
df['wom'] = ((df['dom'] - 1) // 7) + 1

for hold in tqdm([1, 3, 5], desc="Orbits"):
    # First week vs last week of month
    first_week = df[df['wom'] == 1][f'fwd_{hold}'].dropna()
    m, n, t = calc_t(first_week)
    results.append({'category': 'ORBITAL', 'strategy': f'FirstWeekMonth_H{hold}', 
                    'avg_return': m, 'n_samples': n, 't_stat': t,
                    'physics': 'Monthly cycle beginning'})
    
    last_week = df[df['wom'] >= 4][f'fwd_{hold}'].dropna()
    m, n, t = calc_t(last_week)
    results.append({'category': 'ORBITAL', 'strategy': f'LastWeekMonth_H{hold}',
                    'avg_return': m, 'n_samples': n, 't_stat': t,
                    'physics': 'Monthly cycle ending'})
    
    # Quarter effects
    for q in [1, 2, 3, 4]:
        qrets = df[df['quarter'] == q][f'fwd_{hold}'].dropna()
        m, n, t = calc_t(qrets)
        results.append({'category': 'ORBITAL', 'strategy': f'Q{q}_H{hold}',
                        'avg_return': m, 'n_samples': n, 't_stat': t,
                        'physics': f'Q{q} orbital position'})

# ============================================================
# DARK MATTER - Hidden volume patterns
# ============================================================
print("\n" + "="*60)
print("DARK MATTER - Hidden Volume Forces")
print("="*60)

df['vol_ma_20'] = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(20).mean())
df['vol_ratio'] = df['volume'] / df['vol_ma_20']

# On-Balance Volume concept
df['obv_sign'] = np.sign(df['returns'])
df['obv_vol'] = df['volume'] * df['obv_sign']
df['obv_cum'] = df.groupby('ticker')['obv_vol'].cumsum()
df['obv_ma'] = df.groupby('ticker')['obv_cum'].transform(lambda x: x.rolling(20).mean())
df['obv_diverge'] = df['obv_cum'] - df['obv_ma']

for hold in tqdm([1, 5, 10], desc="Dark Matter"):
    # Volume leading price (accumulation)
    for vol_mult in [1.5, 2.0, 3.0]:
        # High volume + small price move = accumulation
        accum = df[(df['vol_ratio'] > vol_mult) & (abs(df['returns']) < 0.01)][f'fwd_{hold}'].dropna()
        m, n, t = calc_t(accum)
        results.append({'category': 'DARK_MATTER', 'strategy': f'Accumulation_{vol_mult}x_H{hold}',
                        'avg_return': m, 'n_samples': n, 't_stat': t,
                        'physics': 'Hidden buying pressure'})
        
        # Low volume + big price move = weak move
        weak = df[(df['vol_ratio'] < 0.5) & (abs(df['returns']) > 0.02)][f'fwd_{hold}'].dropna()
        m, n, t = calc_t(weak)
        results.append({'category': 'DARK_MATTER', 'strategy': f'WeakMove_H{hold}',
                        'avg_return': m, 'n_samples': n, 't_stat': t,
                        'physics': 'Unsupported price move'})

# ============================================================
# QUANTUM EFFECTS - Small signals that compound
# ============================================================
print("\n" + "="*60)
print("QUANTUM EFFECTS - Small Persistent Patterns")
print("="*60)

# Micro-momentum (very small consistent gains)
for lookback in tqdm([3, 5, 7, 10], desc="Quantum"):
    df[f'pos_days_{lookback}'] = df.groupby('ticker')['returns'].transform(
        lambda x: (x > 0).rolling(lookback).sum())
    
    for thresh in [lookback-1, lookback]:  # Almost all or all positive
        for hold in [1, 3, 5]:
            consist_up = df[df[f'pos_days_{lookback}'] >= thresh][f'fwd_{hold}'].dropna()
            m, n, t = calc_t(consist_up)
            results.append({'category': 'QUANTUM', 'strategy': f'ConsistentUp{lookback}_{thresh}_H{hold}',
                            'avg_return': m, 'n_samples': n, 't_stat': t,
                            'physics': 'Quantum persistence - small consistent moves'})
            
            consist_dn = df[df[f'pos_days_{lookback}'] <= lookback - thresh][f'fwd_{hold}'].dropna()
            m, n, t = calc_t(consist_dn)
            results.append({'category': 'QUANTUM', 'strategy': f'ConsistentDn{lookback}_{thresh}_H{hold}',
                            'avg_return': m, 'n_samples': n, 't_stat': t,
                            'physics': 'Quantum persistence - consistent down'})

# ============================================================
# ENTROPY - Volatility compression/expansion cycles
# ============================================================
print("\n" + "="*60)
print("ENTROPY - Volatility State Changes")
print("="*60)

for vol_lb in tqdm([5, 10, 20], desc="Entropy"):
    df[f'vol_{vol_lb}'] = df.groupby('ticker')['returns'].transform(lambda x: x.rolling(vol_lb).std())
    df[f'vol_{vol_lb}_ma'] = df.groupby('ticker')[f'vol_{vol_lb}'].transform(lambda x: x.rolling(20).mean())
    df[f'vol_ratio_{vol_lb}'] = df[f'vol_{vol_lb}'] / df[f'vol_{vol_lb}_ma']
    
    # Volatility compression (low entropy - about to expand)
    for thresh in [0.5, 0.7]:
        for hold in [5, 10, 20]:
            compressed = df[df[f'vol_ratio_{vol_lb}'] < thresh][f'fwd_{hold}'].dropna()
            m, n, t = calc_t(compressed)
            results.append({'category': 'ENTROPY', 'strategy': f'VolCompress{vol_lb}_{int(thresh*100)}_H{hold}',
                            'avg_return': m, 'n_samples': n, 't_stat': t,
                            'physics': 'Low entropy state - expansion imminent'})
            
            # Volatility expansion (high entropy)
            expanded = df[df[f'vol_ratio_{vol_lb}'] > (1/thresh)][f'fwd_{hold}'].dropna()
            m, n, t = calc_t(expanded)
            results.append({'category': 'ENTROPY', 'strategy': f'VolExpand{vol_lb}_{int(100/thresh)}_H{hold}',
                            'avg_return': m, 'n_samples': n, 't_stat': t,
                            'physics': 'High entropy state - contraction expected'})

# ============================================================
# THERMODYNAMICS - Price "temperature" and energy
# ============================================================
print("\n" + "="*60)
print("THERMODYNAMICS - Market Temperature")
print("="*60)

# Range as temperature indicator
df['range_pct'] = (df['high'] - df['low']) / df['close']
df['range_ma'] = df.groupby('ticker')['range_pct'].transform(lambda x: x.rolling(20).mean())
df['temp'] = df['range_pct'] / df['range_ma']

for hold in tqdm([1, 5, 10], desc="Thermo"):
    # Hot market (high range)
    for temp_thresh in [1.5, 2.0, 2.5]:
        hot = df[df['temp'] > temp_thresh][f'fwd_{hold}'].dropna()
        m, n, t = calc_t(hot)
        results.append({'category': 'THERMO', 'strategy': f'HotMarket_{temp_thresh}x_H{hold}',
                        'avg_return': m, 'n_samples': n, 't_stat': t,
                        'physics': 'High temperature - energy dissipation expected'})
    
    # Cold market (low range)
    for temp_thresh in [0.3, 0.5]:
        cold = df[df['temp'] < temp_thresh][f'fwd_{hold}'].dropna()
        m, n, t = calc_t(cold)
        results.append({'category': 'THERMO', 'strategy': f'ColdMarket_{temp_thresh}x_H{hold}',
                        'avg_return': m, 'n_samples': n, 't_stat': t,
                        'physics': 'Low temperature - energy building'})

# ============================================================
# WAVE MECHANICS - Price oscillations
# ============================================================
print("\n" + "="*60)
print("WAVE MECHANICS - Oscillation Patterns")
print("="*60)

# Stochastic as wave position
for k_period in tqdm([5, 14, 21], desc="Waves"):
    df[f'lowest_{k_period}'] = df.groupby('ticker')['low'].transform(lambda x: x.rolling(k_period).min())
    df[f'highest_{k_period}'] = df.groupby('ticker')['high'].transform(lambda x: x.rolling(k_period).max())
    df[f'stoch_{k_period}'] = (df['close'] - df[f'lowest_{k_period}']) / (df[f'highest_{k_period}'] - df[f'lowest_{k_period}']) * 100
    
    for hold in [1, 5, 10]:
        # Wave trough (oversold)
        for thresh in [10, 20]:
            trough = df[df[f'stoch_{k_period}'] < thresh][f'fwd_{hold}'].dropna()
            m, n, t = calc_t(trough)
            results.append({'category': 'WAVE', 'strategy': f'Stoch{k_period}_Trough{thresh}_H{hold}',
                            'avg_return': m, 'n_samples': n, 't_stat': t,
                            'physics': 'Wave at minimum - reversal expected'})
        
        # Wave peak (overbought)
        for thresh in [80, 90]:
            peak = df[df[f'stoch_{k_period}'] > thresh][f'fwd_{hold}'].dropna()
            m, n, t = calc_t(peak)
            results.append({'category': 'WAVE', 'strategy': f'Stoch{k_period}_Peak{thresh}_H{hold}',
                            'avg_return': m, 'n_samples': n, 't_stat': t,
                            'physics': 'Wave at maximum - reversal expected'})

# ============================================================
# RELATIVITY - Returns relative to market
# ============================================================
print("\n" + "="*60)
print("RELATIVITY - Performance vs Market")
print("="*60)

market_ret = df.groupby('date')['returns'].mean()
df['mkt_ret'] = df['date'].map(market_ret)
df['alpha'] = df['returns'] - df['mkt_ret']

for lookback in tqdm([5, 10, 20, 60], desc="Relativity"):
    df[f'cum_alpha_{lookback}'] = df.groupby('ticker')['alpha'].transform(lambda x: x.rolling(lookback).sum())
    
    for pct in [0.1, 0.2]:
        thresh_high = df[f'cum_alpha_{lookback}'].quantile(1 - pct)
        thresh_low = df[f'cum_alpha_{lookback}'].quantile(pct)
        
        for hold in [5, 10, 20]:
            # Strong alpha (outperformers)
            strong = df[df[f'cum_alpha_{lookback}'] > thresh_high][f'fwd_{hold}'].dropna()
            m, n, t = calc_t(strong)
            results.append({'category': 'RELATIVITY', 'strategy': f'StrongAlpha{lookback}_Top{int(pct*100)}_H{hold}',
                            'avg_return': m, 'n_samples': n, 't_stat': t,
                            'physics': 'Relativistic outperformance persistence'})
            
            # Weak alpha (underperformers - mean reversion?)
            weak = df[df[f'cum_alpha_{lookback}'] < thresh_low][f'fwd_{hold}'].dropna()
            m, n, t = calc_t(weak)
            results.append({'category': 'RELATIVITY', 'strategy': f'WeakAlpha{lookback}_Bot{int(pct*100)}_H{hold}',
                            'avg_return': m, 'n_samples': n, 't_stat': t,
                            'physics': 'Relativistic underperformance reversal'})

# ============================================================
# FIELD THEORY - Cross-sectional momentum
# ============================================================
print("\n" + "="*60)
print("FIELD THEORY - Cross-Sectional Rankings")
print("="*60)

for lookback in tqdm([20, 60, 120, 252], desc="Field Theory"):
    df[f'mom_{lookback}'] = df.groupby('ticker')['close'].transform(lambda x: x.pct_change(lookback))
    
    # Rank within each day
    df[f'rank_{lookback}'] = df.groupby('date')[f'mom_{lookback}'].rank(pct=True)
    
    for hold in [5, 10, 20]:
        # Top decile
        top = df[df[f'rank_{lookback}'] > 0.9][f'fwd_{hold}'].dropna()
        m, n, t = calc_t(top)
        results.append({'category': 'FIELD', 'strategy': f'TopDecile{lookback}_H{hold}',
                        'avg_return': m, 'n_samples': n, 't_stat': t,
                        'physics': 'Field strength - top performers'})
        
        # Bottom decile
        bot = df[df[f'rank_{lookback}'] < 0.1][f'fwd_{hold}'].dropna()
        m, n, t = calc_t(bot)
        results.append({'category': 'FIELD', 'strategy': f'BotDecile{lookback}_H{hold}',
                        'avg_return': m, 'n_samples': n, 't_stat': t,
                        'physics': 'Field strength - bottom performers'})

# ============================================================
# SAVE RESULTS
# ============================================================
print("\n" + "="*60)
print("RESULTS")
print("="*60)

results_df = pd.DataFrame(results)
results_df['significant'] = results_df['t_stat'].abs() > 3.0

output = 'data/FINANCIAL_PHYSICS_LAWS.csv' if os.path.exists('data') else 'FINANCIAL_PHYSICS_LAWS.csv'
results_df.to_csv(output, index=False)

sig = results_df[results_df['significant']]
print(f"\nTotal strategies: {len(results_df)}")
print(f"Significant (|t|>3): {len(sig)} ({100*len(sig)/len(results_df):.1f}%)")

print("\n" + "="*60)
print("TOP 30 FINANCIAL LAWS DISCOVERED")
print("="*60)
for _, r in results_df.nlargest(30, 't_stat').iterrows():
    print(f"{r['category']:12} | {r['strategy']:40} | t={r['t_stat']:6.2f}")

# Group by category
print("\n" + "="*60)
print("LAWS BY CATEGORY")
print("="*60)
for cat in results_df['category'].unique():
    cat_df = results_df[results_df['category'] == cat]
    cat_sig = cat_df[cat_df['significant']]
    best = cat_df.nlargest(1, 't_stat').iloc[0] if len(cat_df) > 0 else None
    print(f"{cat:15} | {len(cat_sig):3}/{len(cat_df):3} significant | Best: {best['strategy'] if best is not None else 'N/A'} (t={best['t_stat']:.1f})")

print(f"\nSaved to: {output}")
print("\nDONE!")
