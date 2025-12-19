#!/usr/bin/env python3
"""
SHADOW PC TEST #2 - MACD, Williams %R, and Indicator Combinations
Run on Shadow PC: python SHADOW_MACD_WILLIAMS.py
"""

import sqlite3
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("SHADOW PC TEST #2 - MACD, Williams %R, CCI, ADX")
print("="*70)

# Find database
DB_PATHS = ['data/market_data.db', 'market_data.db', '../data/market_data.db']
DB_PATH = None
for p in DB_PATHS:
    if os.path.exists(p):
        DB_PATH = p
        break

if not DB_PATH:
    print("ERROR: market_data.db not found. Run: git pull")
    input("Press Enter to exit...")
    exit(1)

print(f"\nDatabase: {DB_PATH}")
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM ohlcv", conn)
conn.close()
print(f"Loaded {len(df):,} rows, {df['ticker'].nunique():,} tickers")

# Prepare data
print("\nPreparing data...")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
df['returns'] = df.groupby('ticker')['close'].pct_change()

# Pre-compute forward returns
print("Pre-computing forward returns...")
for h in tqdm([1, 3, 5, 10, 15, 20], desc="Forward Returns"):
    df[f'fwd_{h}'] = df.groupby('ticker')['close'].transform(lambda x: x.shift(-h) / x - 1)

# Calendar
df['month'] = df['date'].dt.month
df['dow'] = df['date'].dt.dayofweek

results = []

def calc_t(rets, name=''):
    rets = rets.dropna()
    if len(rets) < 100:
        return None
    m = np.mean(rets)
    s = np.std(rets, ddof=1)
    if s == 0 or np.isnan(s):
        return None
    t = m / (s / np.sqrt(len(rets)))
    return {'category': name.split('_')[0], 'strategy': name, 'mean': m, 'n': len(rets), 't': t}

# ============================================================
# SECTION 1: MACD STRATEGIES
# ============================================================
print("\n" + "="*70)
print("SECTION 1: MACD STRATEGIES")
print("="*70)

for fast, slow, signal in tqdm([(8, 17, 9), (12, 26, 9), (5, 35, 5), (19, 39, 9)], desc="MACD"):
    ema_fast = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=fast).mean())
    ema_slow = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=slow).mean())
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df.groupby('ticker')['macd'].transform(lambda x: x.ewm(span=signal).mean())
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # MACD crosses
    df['macd_cross_up'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    df['macd_cross_dn'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
    
    for hold in [1, 3, 5, 10, 15, 20]:
        # Bullish cross
        r = calc_t(df[df['macd_cross_up']][f'fwd_{hold}'], f'MACD_{fast}_{slow}_CrossUp_H{hold}')
        if r: results.append(r)
        
        # Bearish cross
        r = calc_t(df[df['macd_cross_dn']][f'fwd_{hold}'], f'MACD_{fast}_{slow}_CrossDn_H{hold}')
        if r: results.append(r)
        
        # MACD positive
        r = calc_t(df[df['macd'] > 0][f'fwd_{hold}'], f'MACD_{fast}_{slow}_Pos_H{hold}')
        if r: results.append(r)
        
        # MACD negative
        r = calc_t(df[df['macd'] < 0][f'fwd_{hold}'], f'MACD_{fast}_{slow}_Neg_H{hold}')
        if r: results.append(r)
        
        # Histogram increasing
        hist_inc = df['macd_hist'] > df['macd_hist'].shift(1)
        r = calc_t(df[hist_inc][f'fwd_{hold}'], f'MACD_{fast}_{slow}_HistUp_H{hold}')
        if r: results.append(r)
        
        # Histogram decreasing
        hist_dec = df['macd_hist'] < df['macd_hist'].shift(1)
        r = calc_t(df[hist_dec][f'fwd_{hold}'], f'MACD_{fast}_{slow}_HistDn_H{hold}')
        if r: results.append(r)

# ============================================================
# SECTION 2: WILLIAMS %R
# ============================================================
print("\n" + "="*70)
print("SECTION 2: WILLIAMS %R")
print("="*70)

for period in tqdm([5, 10, 14, 21, 28], desc="Williams %R"):
    highest = df.groupby('ticker')['high'].transform(lambda x: x.rolling(period).max())
    lowest = df.groupby('ticker')['low'].transform(lambda x: x.rolling(period).min())
    df[f'willr_{period}'] = -100 * (highest - df['close']) / (highest - lowest)
    
    for oversold in [-90, -80, -70]:
        for overbought in [-30, -20, -10]:
            for hold in [1, 3, 5, 10, 15]:
                # Oversold
                r = calc_t(df[df[f'willr_{period}'] < oversold][f'fwd_{hold}'], 
                          f'WillR{period}_OS{abs(oversold)}_H{hold}')
                if r: results.append(r)
                
                # Overbought
                r = calc_t(df[df[f'willr_{period}'] > overbought][f'fwd_{hold}'], 
                          f'WillR{period}_OB{abs(overbought)}_H{hold}')
                if r: results.append(r)

# ============================================================
# SECTION 3: CCI (Commodity Channel Index)
# ============================================================
print("\n" + "="*70)
print("SECTION 3: CCI")
print("="*70)

for period in tqdm([10, 14, 20, 30], desc="CCI"):
    df['_tp'] = (df['high'] + df['low'] + df['close']) / 3
    sma = df.groupby('ticker')['_tp'].transform(lambda x: x.rolling(period).mean())
    mad = df.groupby('ticker')['_tp'].transform(lambda x: x.rolling(period).apply(lambda y: np.abs(y - y.mean()).mean()))
    df[f'cci_{period}'] = (df['_tp'] - sma) / (0.015 * mad)
    
    for thresh in [100, 150, 200]:
        for hold in [1, 3, 5, 10, 15]:
            # Oversold (below -thresh)
            r = calc_t(df[df[f'cci_{period}'] < -thresh][f'fwd_{hold}'], 
                      f'CCI{period}_Below{thresh}_H{hold}')
            if r: results.append(r)
            
            # Overbought (above thresh)
            r = calc_t(df[df[f'cci_{period}'] > thresh][f'fwd_{hold}'], 
                      f'CCI{period}_Above{thresh}_H{hold}')
            if r: results.append(r)

# ============================================================
# SECTION 4: ADX (Average Directional Index)
# ============================================================
print("\n" + "="*70)
print("SECTION 4: ADX")
print("="*70)

for period in tqdm([7, 14, 21], desc="ADX"):
    prev_close = df.groupby('ticker')['close'].shift(1)
    prev_high = df.groupby('ticker')['high'].shift(1)
    prev_low = df.groupby('ticker')['low'].shift(1)
    
    # True Range
    tr = np.maximum(df['high'] - df['low'], 
                   np.maximum(abs(df['high'] - prev_close), abs(df['low'] - prev_close)))
    
    # +DM and -DM
    plus_dm = np.where((df['high'] - prev_high) > (prev_low - df['low']), 
                       np.maximum(df['high'] - prev_high, 0), 0)
    minus_dm = np.where((prev_low - df['low']) > (df['high'] - prev_high), 
                        np.maximum(prev_low - df['low'], 0), 0)
    
    df['_tr'] = tr
    df['_plus_dm'] = plus_dm
    df['_minus_dm'] = minus_dm
    
    # Smoothed averages
    atr = df.groupby('ticker')['_tr'].transform(lambda x: x.rolling(period).mean())
    plus_di = 100 * df.groupby('ticker')['_plus_dm'].transform(lambda x: x.rolling(period).mean()) / atr
    minus_di = 100 * df.groupby('ticker')['_minus_dm'].transform(lambda x: x.rolling(period).mean()) / atr
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df['_dx'] = dx  # Store in dataframe to avoid Series.name bug
    df[f'adx_{period}'] = df.groupby('ticker')['_dx'].transform(lambda x: x.rolling(period).mean())
    df[f'plus_di_{period}'] = plus_di
    df[f'minus_di_{period}'] = minus_di
    
    for adx_thresh in [20, 25, 30, 40]:
        for hold in [1, 5, 10, 20]:
            # Strong trend (high ADX)
            r = calc_t(df[df[f'adx_{period}'] > adx_thresh][f'fwd_{hold}'], 
                      f'ADX{period}_Strong{adx_thresh}_H{hold}')
            if r: results.append(r)
            
            # Weak trend (low ADX)
            r = calc_t(df[df[f'adx_{period}'] < adx_thresh][f'fwd_{hold}'], 
                      f'ADX{period}_Weak{adx_thresh}_H{hold}')
            if r: results.append(r)
            
            # Bullish (+DI > -DI)
            r = calc_t(df[df[f'plus_di_{period}'] > df[f'minus_di_{period}']][f'fwd_{hold}'], 
                      f'ADX{period}_Bullish_H{hold}')
            if r: results.append(r)

# ============================================================
# SECTION 5: INDICATOR COMBINATIONS
# ============================================================
print("\n" + "="*70)
print("SECTION 5: INDICATOR COMBINATIONS")
print("="*70)

# Compute base indicators for combos
df['rsi_14'] = df.groupby('ticker')['returns'].transform(
    lambda x: 100 - 100 / (1 + x.clip(lower=0).rolling(14).mean() / 
                          (-x.clip(upper=0)).rolling(14).mean().replace(0, 0.0001)))
df['vol_20'] = df.groupby('ticker')['returns'].transform(lambda x: x.rolling(20).std())
df['mom_20'] = df.groupby('ticker')['close'].transform(lambda x: x.pct_change(20))

for hold in tqdm([5, 10, 20], desc="Combos"):
    # RSI oversold + MACD bullish cross
    combo = (df['rsi_14'] < 30) & df['macd_cross_up']
    r = calc_t(df[combo][f'fwd_{hold}'], f'COMBO_RSI30_MACDup_H{hold}')
    if r: results.append(r)
    
    # RSI oversold + Williams %R oversold
    combo = (df['rsi_14'] < 30) & (df['willr_14'] < -80)
    r = calc_t(df[combo][f'fwd_{hold}'], f'COMBO_RSI30_WillR80_H{hold}')
    if r: results.append(r)
    
    # Strong trend + Pullback
    combo = (df['adx_14'] > 25) & (df['rsi_14'] < 40)
    r = calc_t(df[combo][f'fwd_{hold}'], f'COMBO_StrongTrend_Pullback_H{hold}')
    if r: results.append(r)
    
    # Low vol + MACD positive
    low_vol = df['vol_20'] < df['vol_20'].quantile(0.3)
    combo = low_vol & (df['macd'] > 0)
    r = calc_t(df[combo][f'fwd_{hold}'], f'COMBO_LowVol_MACDpos_H{hold}')
    if r: results.append(r)
    
    # CCI oversold + RSI oversold
    combo = (df['cci_14'] < -100) & (df['rsi_14'] < 30)
    r = calc_t(df[combo][f'fwd_{hold}'], f'COMBO_CCI100_RSI30_H{hold}')
    if r: results.append(r)
    
    # MACD positive + ADX strong + Best months
    best_month = df['month'].isin([4, 6, 8, 9])
    combo = (df['macd'] > 0) & (df['adx_14'] > 25) & best_month
    r = calc_t(df[combo][f'fwd_{hold}'], f'COMBO_MACD_ADX_BestMo_H{hold}')
    if r: results.append(r)
    
    # Triple indicator alignment
    combo = (df['rsi_14'] < 35) & (df['willr_14'] < -70) & (df['cci_14'] < -80)
    r = calc_t(df[combo][f'fwd_{hold}'], f'COMBO_Triple_Oversold_H{hold}')
    if r: results.append(r)

# ============================================================
# SECTION 6: OBV (On Balance Volume)
# ============================================================
print("\n" + "="*70)
print("SECTION 6: OBV")
print("="*70)

df['obv_change'] = np.where(df['close'] > df['close'].shift(1), df['volume'],
                   np.where(df['close'] < df['close'].shift(1), -df['volume'], 0))
df['obv'] = df.groupby('ticker')['obv_change'].cumsum()
df['obv_ma'] = df.groupby('ticker')['obv'].transform(lambda x: x.rolling(20).mean())
df['obv_trend'] = df['obv'] > df['obv_ma']

for hold in tqdm([1, 5, 10, 20], desc="OBV"):
    # OBV above MA (bullish)
    r = calc_t(df[df['obv_trend']][f'fwd_{hold}'], f'OBV_AboveMA_H{hold}')
    if r: results.append(r)
    
    # OBV below MA (bearish)
    r = calc_t(df[~df['obv_trend']][f'fwd_{hold}'], f'OBV_BelowMA_H{hold}')
    if r: results.append(r)
    
    # OBV divergence: Price up + OBV down
    price_up = df['mom_20'] > 0.05
    obv_down = df['obv'] < df['obv'].shift(20)
    r = calc_t(df[price_up & obv_down][f'fwd_{hold}'], f'OBV_BearishDiv_H{hold}')
    if r: results.append(r)
    
    # OBV divergence: Price down + OBV up
    price_down = df['mom_20'] < -0.05
    obv_up = df['obv'] > df['obv'].shift(20)
    r = calc_t(df[price_down & obv_up][f'fwd_{hold}'], f'OBV_BullishDiv_H{hold}')
    if r: results.append(r)

# ============================================================
# SAVE RESULTS
# ============================================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

results_df = pd.DataFrame(results)
results_df['significant'] = results_df['t'].abs() > 3.0

output_dir = 'data' if os.path.exists('data') else '.'
output_file = f'{output_dir}/MACD_WILLIAMS_RESULTS.csv'
results_df.to_csv(output_file, index=False)

sig = results_df[results_df['significant']]

print(f"\n{'='*70}")
print(f"TEST COMPLETE")
print(f"{'='*70}")
print(f"\nTotal strategies tested: {len(results_df):,}")
print(f"Significant (|t| > 3.0): {len(sig):,} ({100*len(sig)/len(results_df):.1f}%)")
print(f"\nResults saved to: {output_file}")

# Top 50
print(f"\n{'='*70}")
print("TOP 50 STRATEGIES DISCOVERED")
print(f"{'='*70}")
for i, (_, r) in enumerate(results_df.nlargest(50, 't').iterrows(), 1):
    print(f"{i:2}. {r['category']:15} | {r['strategy']:45} | t={r['t']:7.2f} | ret={r['mean']*100:6.2f}%")

# Top 20 SHORT signals
print(f"\n{'='*70}")
print("TOP 20 SHORT SIGNALS (Negative t-stat)")
print(f"{'='*70}")
for i, (_, r) in enumerate(results_df.nsmallest(20, 't').iterrows(), 1):
    print(f"{i:2}. {r['category']:15} | {r['strategy']:45} | t={r['t']:7.2f} | ret={r['mean']*100:6.2f}%")

print(f"\n{'='*70}")
print("DONE! Results saved to:", output_file)
print(f"{'='*70}")

# Removed input() for automated running

