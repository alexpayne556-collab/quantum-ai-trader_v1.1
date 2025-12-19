#!/usr/bin/env python3
"""
DEEPER LAYERS - Beyond the Surface
===================================

Layer 1: Single factors (done)
Layer 2: INTERACTIONS between factors
Layer 3: Conditional effects (when does X work?)
Layer 4: Regime-dependent behavior
Layer 5: Temporal dynamics (does it decay?)

Like finding quarks inside protons...
"""

import sqlite3
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("DEEPER LAYERS - Peeling Back the Universe")
print("Every layer reveals another beneath...")
print("="*70)

# Load
conn = sqlite3.connect('data/market_data.db')
df = pd.read_sql("SELECT * FROM ohlcv", conn)
conn.close()

print(f"\nUniverse: {len(df):,} observations, {df['ticker'].nunique():,} stocks")

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
df['returns'] = df.groupby('ticker')['close'].pct_change()

# Forward returns
for h in [1, 2, 3, 5, 10, 20, 40, 60]:
    df[f'fwd_{h}'] = df.groupby('ticker')['close'].transform(lambda x: x.shift(-h) / x - 1)

results = []

def calc_t(rets, name=""):
    rets = rets.dropna()
    if len(rets) < 100:
        return None
    m = np.mean(rets)
    s = np.std(rets, ddof=1)
    if s == 0:
        return None
    t = m / (s / np.sqrt(len(rets)))
    return {'strategy': name, 'mean_return': m, 'n': len(rets), 't_stat': t, 'significant': abs(t) > 3.0}

# Pre-compute key features
print("\nComputing features...")
df['vol_20'] = df.groupby('ticker')['returns'].transform(lambda x: x.rolling(20).std())
df['mom_20'] = df.groupby('ticker')['close'].transform(lambda x: x.pct_change(20))
df['mom_60'] = df.groupby('ticker')['close'].transform(lambda x: x.pct_change(60))
df['vol_ratio'] = df['volume'] / df.groupby('ticker')['volume'].transform(lambda x: x.rolling(20).mean())
df['month'] = df['date'].dt.month
df['dow'] = df['date'].dt.dayofweek

# Z-score
ma20 = df.groupby('ticker')['close'].transform(lambda x: x.rolling(20).mean())
std20 = df.groupby('ticker')['close'].transform(lambda x: x.rolling(20).std())
df['zscore'] = (df['close'] - ma20) / std20

# RSI
df['rsi'] = df.groupby('ticker')['returns'].transform(
    lambda x: 100 - 100 / (1 + x.clip(lower=0).rolling(14).mean() / (-x.clip(upper=0)).rolling(14).mean().replace(0, 0.0001))
)

# ================================================================
# LAYER 2: TWO-FACTOR INTERACTIONS
# Does momentum + volatility together work better than alone?
# ================================================================
print("\n" + "="*70)
print("LAYER 2: TWO-FACTOR INTERACTIONS")
print("Finding particles that only exist when combined...")
print("="*70)

# Momentum x Volatility
print("\n--- Momentum × Volatility ---")
for mom_q in tqdm(['top', 'bottom'], desc="Mom×Vol"):
    mom_thresh_high = df['mom_20'].quantile(0.80)
    mom_thresh_low = df['mom_20'].quantile(0.20)
    vol_thresh_high = df['vol_20'].quantile(0.75)
    vol_thresh_low = df['vol_20'].quantile(0.25)
    
    for vol_q in ['high', 'low']:
        for hold in [1, 5, 10, 20]:
            if mom_q == 'top' and vol_q == 'high':
                subset = df[(df['mom_20'] >= mom_thresh_high) & (df['vol_20'] >= vol_thresh_high)]
            elif mom_q == 'top' and vol_q == 'low':
                subset = df[(df['mom_20'] >= mom_thresh_high) & (df['vol_20'] <= vol_thresh_low)]
            elif mom_q == 'bottom' and vol_q == 'high':
                subset = df[(df['mom_20'] <= mom_thresh_low) & (df['vol_20'] >= vol_thresh_high)]
            else:
                subset = df[(df['mom_20'] <= mom_thresh_low) & (df['vol_20'] <= vol_thresh_low)]
            
            r = calc_t(subset[f'fwd_{hold}'], f'Mom{mom_q}_Vol{vol_q}_H{hold}')
            if r:
                r['layer'] = 'INTERACTION'
                r['factors'] = 'MOM×VOL'
                results.append(r)

# RSI x Volume
print("\n--- RSI × Volume ---")
for rsi_cond in tqdm(['oversold', 'overbought'], desc="RSI×Vol"):
    for vol_cond in ['spike', 'quiet']:
        for hold in [1, 5, 10, 20]:
            if rsi_cond == 'oversold' and vol_cond == 'spike':
                subset = df[(df['rsi'] < 30) & (df['vol_ratio'] > 2.0)]
            elif rsi_cond == 'oversold' and vol_cond == 'quiet':
                subset = df[(df['rsi'] < 30) & (df['vol_ratio'] < 0.5)]
            elif rsi_cond == 'overbought' and vol_cond == 'spike':
                subset = df[(df['rsi'] > 70) & (df['vol_ratio'] > 2.0)]
            else:
                subset = df[(df['rsi'] > 70) & (df['vol_ratio'] < 0.5)]
            
            r = calc_t(subset[f'fwd_{hold}'], f'RSI{rsi_cond}_Vol{vol_cond}_H{hold}')
            if r:
                r['layer'] = 'INTERACTION'
                r['factors'] = 'RSI×VOL'
                results.append(r)

# Z-Score x Momentum (Mean Reversion in Trend)
print("\n--- Z-Score × Trend ---")
for zscore_cond in tqdm(['oversold', 'overbought'], desc="Z×Trend"):
    for trend_cond in ['uptrend', 'downtrend']:
        for hold in [1, 5, 10, 20]:
            if zscore_cond == 'oversold' and trend_cond == 'uptrend':
                subset = df[(df['zscore'] < -2) & (df['mom_60'] > 0)]
            elif zscore_cond == 'oversold' and trend_cond == 'downtrend':
                subset = df[(df['zscore'] < -2) & (df['mom_60'] < 0)]
            elif zscore_cond == 'overbought' and trend_cond == 'uptrend':
                subset = df[(df['zscore'] > 2) & (df['mom_60'] > 0)]
            else:
                subset = df[(df['zscore'] > 2) & (df['mom_60'] < 0)]
            
            r = calc_t(subset[f'fwd_{hold}'], f'Z{zscore_cond}_Trend{trend_cond}_H{hold}')
            if r:
                r['layer'] = 'INTERACTION'
                r['factors'] = 'Z×TREND'
                results.append(r)

# ================================================================
# LAYER 3: THREE-FACTOR COMBINATIONS
# The "strong nuclear force" of markets
# ================================================================
print("\n" + "="*70)
print("LAYER 3: THREE-FACTOR COMBINATIONS")
print("Like finding the strong nuclear force...")
print("="*70)

for hold in tqdm([1, 5, 10, 20], desc="3-Factor"):
    # Oversold + Uptrend + High Volume (capitulation bottom?)
    cap_bottom = df[(df['zscore'] < -2) & (df['mom_60'] > 0) & (df['vol_ratio'] > 2)]
    r = calc_t(cap_bottom[f'fwd_{hold}'], f'CapitulationBottom_H{hold}')
    if r:
        r['layer'] = 'THREE_FACTOR'
        r['factors'] = 'Z+TREND+VOL'
        results.append(r)
    
    # Overbought + Downtrend + High Volume (blow-off top?)
    blow_off = df[(df['zscore'] > 2) & (df['mom_60'] < 0) & (df['vol_ratio'] > 2)]
    r = calc_t(blow_off[f'fwd_{hold}'], f'BlowOffTop_H{hold}')
    if r:
        r['layer'] = 'THREE_FACTOR'
        r['factors'] = 'Z+TREND+VOL'
        results.append(r)
    
    # Strong momentum + Low vol + Volume spike (breakout)
    breakout = df[(df['mom_20'] > df['mom_20'].quantile(0.9)) & 
                  (df['vol_20'] < df['vol_20'].quantile(0.3)) & 
                  (df['vol_ratio'] > 2)]
    r = calc_t(breakout[f'fwd_{hold}'], f'BreakoutSetup_H{hold}')
    if r:
        r['layer'] = 'THREE_FACTOR'
        r['factors'] = 'MOM+LOWVOL+SPIKE'
        results.append(r)
    
    # Weak momentum + High vol + Volume spike (panic selling?)
    panic = df[(df['mom_20'] < df['mom_20'].quantile(0.1)) & 
               (df['vol_20'] > df['vol_20'].quantile(0.7)) & 
               (df['vol_ratio'] > 2)]
    r = calc_t(panic[f'fwd_{hold}'], f'PanicSelling_H{hold}')
    if r:
        r['layer'] = 'THREE_FACTOR'
        r['factors'] = 'WEAKMOM+HIGHVOL+SPIKE'
        results.append(r)

# ================================================================
# LAYER 4: CONDITIONAL EFFECTS (When does X work?)
# ================================================================
print("\n" + "="*70)
print("LAYER 4: CONDITIONAL EFFECTS")
print("When does mean reversion work? When doesn't it?")
print("="*70)

# Does mean reversion work better in certain months?
print("\n--- Mean Reversion by Month ---")
for month in tqdm(range(1, 13), desc="MR by Month"):
    month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    for hold in [5, 10, 20]:
        # Oversold in this month
        subset = df[(df['zscore'] < -2) & (df['month'] == month)]
        r = calc_t(subset[f'fwd_{hold}'], f'MR_Oversold_{month_names[month-1]}_H{hold}')
        if r:
            r['layer'] = 'CONDITIONAL'
            r['condition'] = f'MONTH={month_names[month-1]}'
            results.append(r)

# Does momentum work better on certain days?
print("\n--- Momentum by Day of Week ---")
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
for dow in tqdm(range(5), desc="Mom by Day"):
    for hold in [1, 5, 10]:
        # Top momentum on this day
        subset = df[(df['mom_20'] > df['mom_20'].quantile(0.8)) & (df['dow'] == dow)]
        r = calc_t(subset[f'fwd_{hold}'], f'TopMom_{days[dow]}_H{hold}')
        if r:
            r['layer'] = 'CONDITIONAL'
            r['condition'] = f'DOW={days[dow]}'
            results.append(r)

# Does volatility regime change everything?
print("\n--- Strategies in High vs Low Vol Regime ---")
vol_regime_high = df['vol_20'] > df['vol_20'].quantile(0.75)
vol_regime_low = df['vol_20'] < df['vol_20'].quantile(0.25)

for hold in tqdm([5, 10, 20], desc="Vol Regime"):
    # Mean reversion in HIGH vol regime
    subset = df[(df['zscore'] < -2) & vol_regime_high]
    r = calc_t(subset[f'fwd_{hold}'], f'MR_HighVolRegime_H{hold}')
    if r:
        r['layer'] = 'CONDITIONAL'
        r['condition'] = 'HIGH_VOL_REGIME'
        results.append(r)
    
    # Mean reversion in LOW vol regime
    subset = df[(df['zscore'] < -2) & vol_regime_low]
    r = calc_t(subset[f'fwd_{hold}'], f'MR_LowVolRegime_H{hold}')
    if r:
        r['layer'] = 'CONDITIONAL'
        r['condition'] = 'LOW_VOL_REGIME'
        results.append(r)
    
    # Momentum in HIGH vol regime
    subset = df[(df['mom_20'] > df['mom_20'].quantile(0.8)) & vol_regime_high]
    r = calc_t(subset[f'fwd_{hold}'], f'Mom_HighVolRegime_H{hold}')
    if r:
        r['layer'] = 'CONDITIONAL'
        r['condition'] = 'HIGH_VOL_REGIME'
        results.append(r)
    
    # Momentum in LOW vol regime
    subset = df[(df['mom_20'] > df['mom_20'].quantile(0.8)) & vol_regime_low]
    r = calc_t(subset[f'fwd_{hold}'], f'Mom_LowVolRegime_H{hold}')
    if r:
        r['layer'] = 'CONDITIONAL'
        r['condition'] = 'LOW_VOL_REGIME'
        results.append(r)

# ================================================================
# LAYER 5: CONSECUTIVE PATTERNS (Sequences)
# ================================================================
print("\n" + "="*70)
print("LAYER 5: SEQUENTIAL PATTERNS")
print("What happens after 2 up days? 3 down days? 5 up days?")
print("="*70)

# Consecutive up/down days
df['up_day'] = (df['returns'] > 0).astype(int)
df['dn_day'] = (df['returns'] < 0).astype(int)

for consec in tqdm([2, 3, 4, 5], desc="Consecutive"):
    df[f'consec_up_{consec}'] = df.groupby('ticker')['up_day'].transform(lambda x: x.rolling(consec).sum() == consec)
    df[f'consec_dn_{consec}'] = df.groupby('ticker')['dn_day'].transform(lambda x: x.rolling(consec).sum() == consec)
    
    for hold in [1, 3, 5, 10]:
        # After N consecutive up days
        subset = df[df[f'consec_up_{consec}']]
        r = calc_t(subset[f'fwd_{hold}'], f'After{consec}UpDays_H{hold}')
        if r:
            r['layer'] = 'SEQUENTIAL'
            r['pattern'] = f'{consec}_UP_DAYS'
            results.append(r)
        
        # After N consecutive down days
        subset = df[df[f'consec_dn_{consec}']]
        r = calc_t(subset[f'fwd_{hold}'], f'After{consec}DnDays_H{hold}')
        if r:
            r['layer'] = 'SEQUENTIAL'
            r['pattern'] = f'{consec}_DN_DAYS'
            results.append(r)

# ================================================================
# LAYER 6: EXTREME COMBINATIONS
# ================================================================
print("\n" + "="*70)
print("LAYER 6: EXTREME EVENTS")
print("What happens at the edges of the distribution?")
print("="*70)

for hold in tqdm([1, 5, 10, 20], desc="Extremes"):
    # Extreme oversold + Extreme volume
    extreme_os_vol = df[(df['zscore'] < -3) & (df['vol_ratio'] > 3)]
    r = calc_t(extreme_os_vol[f'fwd_{hold}'], f'ExtremeOversold_ExtremeVol_H{hold}')
    if r:
        r['layer'] = 'EXTREME'
        results.append(r)
    
    # Extreme momentum + Extreme volume
    extreme_mom_vol = df[(df['mom_20'] > df['mom_20'].quantile(0.99)) & (df['vol_ratio'] > 3)]
    r = calc_t(extreme_mom_vol[f'fwd_{hold}'], f'ExtremeMom_ExtremeVol_H{hold}')
    if r:
        r['layer'] = 'EXTREME'
        results.append(r)
    
    # RSI at absolute extremes
    rsi_extreme_low = df[df['rsi'] < 10]
    r = calc_t(rsi_extreme_low[f'fwd_{hold}'], f'RSI_Under10_H{hold}')
    if r:
        r['layer'] = 'EXTREME'
        results.append(r)
    
    rsi_extreme_high = df[df['rsi'] > 90]
    r = calc_t(rsi_extreme_high[f'fwd_{hold}'], f'RSI_Over90_H{hold}')
    if r:
        r['layer'] = 'EXTREME'
        results.append(r)

# ================================================================
# SAVE DEEPER LAYERS
# ================================================================
print("\n" + "="*70)
print("SAVING DEEPER LAYER DISCOVERIES")
print("="*70)

results_df = pd.DataFrame(results)
results_df.to_csv('data/DEEPER_LAYERS.csv', index=False)

sig = results_df[results_df['significant']]

print(f"\nTotal patterns tested: {len(results_df):,}")
print(f"Significant: {len(sig):,} ({100*len(sig)/max(1,len(results_df)):.1f}%)")

# By layer
print(f"\n{'='*70}")
print("DISCOVERIES BY LAYER")
print(f"{'='*70}")
if 'layer' in results_df.columns:
    for layer in results_df['layer'].unique():
        layer_df = results_df[results_df['layer'] == layer]
        layer_sig = layer_df[layer_df['significant']]
        print(f"{layer:20} | {len(layer_sig):3}/{len(layer_df):3} significant")

# Top discoveries
print(f"\n{'='*70}")
print("TOP 30 DEEPER LAYER DISCOVERIES")
print(f"{'='*70}")
for i, (_, r) in enumerate(results_df.nlargest(30, 't_stat').iterrows(), 1):
    layer = r.get('layer', 'N/A')
    print(f"{i:2}. [{layer:12}] {r['strategy']:40} | t={r['t_stat']:7.2f}")

# Bottom discoveries (what DOESN'T work)
print(f"\n{'='*70}")
print("BOTTOM 20 - WHAT DOESN'T WORK (Negative t-stats)")
print(f"{'='*70}")
for i, (_, r) in enumerate(results_df.nsmallest(20, 't_stat').iterrows(), 1):
    layer = r.get('layer', 'N/A')
    print(f"{i:2}. [{layer:12}] {r['strategy']:40} | t={r['t_stat']:7.2f}")

print(f"\n{'='*70}")
print("DEEPER LAYERS MAPPED")
print("Every layer reveals another beneath...")
print(f"{'='*70}")
