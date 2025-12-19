#!/usr/bin/env python3
"""
MEGA TEST - 15+ minute comprehensive strategy discovery
Run on Shadow PC: python SHADOW_MEGA_TEST.py
"""

import sqlite3
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("SHADOW PC MEGA TEST - Financial Universe Mapping")
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

# Check tables
tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
print(f"Tables: {tables}")
if 'ohlcv' not in tables:
    print("ERROR: 'ohlcv' table not found")
    input("Press Enter to exit...")
    exit(1)

# Load data
print("\nLoading data...")
df = pd.read_sql("SELECT * FROM ohlcv", conn)
conn.close()
print(f"Loaded {len(df):,} rows, {df['ticker'].nunique():,} tickers")

# Prepare data
print("\nPreparing data...")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
df['returns'] = df.groupby('ticker')['close'].pct_change()

# Pre-compute ALL forward returns upfront
print("Pre-computing forward returns...")
hold_periods = [1, 2, 3, 5, 10, 15, 20, 40, 60]
for h in tqdm(hold_periods, desc="Forward Returns"):
    df[f'fwd_{h}'] = df.groupby('ticker')['close'].transform(lambda x: x.shift(-h) / x - 1)

# Calendar features
print("Computing calendar features...")
df['dow'] = df['date'].dt.dayofweek
df['dom'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter
df['wom'] = ((df['dom'] - 1) // 7) + 1
df['year'] = df['date'].dt.year

results = []

def calc_t(rets):
    """Harvey-Liu-Zhu t-statistic"""
    rets = rets.dropna()
    if len(rets) < 30:
        return 0, 0, 0
    m = np.mean(rets)
    s = np.std(rets, ddof=1)
    if s == 0 or np.isnan(s):
        return 0, 0, 0
    t = m / (s / np.sqrt(len(rets)))
    return m, len(rets), t

# ============================================================
# SECTION 1: RSI STRATEGIES (Comprehensive)
# ============================================================
print("\n" + "="*70)
print("SECTION 1: RSI STRATEGIES")
print("="*70)

for period in tqdm([3, 5, 7, 10, 14, 21, 30], desc="RSI Periods"):
    delta = df.groupby('ticker')['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = df.groupby('ticker')[gain.name].transform(lambda x: gain.loc[x.index].rolling(period).mean())
    avg_loss = df.groupby('ticker')[loss.name].transform(lambda x: loss.loc[x.index].rolling(period).mean())
    # Simpler RSI calculation
    df[f'rsi_{period}'] = df.groupby('ticker')['returns'].transform(
        lambda x: 100 - 100 / (1 + x.clip(lower=0).rolling(period).mean() / (-x.clip(upper=0)).rolling(period).mean().replace(0, 0.0001))
    )
    
    for oversold in [20, 25, 30, 35]:
        for overbought in [65, 70, 75, 80]:
            for hold in [1, 3, 5, 10, 15]:
                # Oversold buy
                buy = df[df[f'rsi_{period}'] < oversold][f'fwd_{hold}']
                m, n, t = calc_t(buy)
                if n > 100:
                    results.append({'category': 'RSI', 'strategy': f'RSI{period}_OV{oversold}_H{hold}', 
                                   'avg_return': m, 'n_samples': n, 't_stat': t})
                
                # Overbought sell
                sell = df[df[f'rsi_{period}'] > overbought][f'fwd_{hold}']
                m, n, t = calc_t(sell)
                if n > 100:
                    results.append({'category': 'RSI', 'strategy': f'RSI{period}_OB{overbought}_H{hold}',
                                   'avg_return': m, 'n_samples': n, 't_stat': t})

# ============================================================
# SECTION 2: MEAN REVERSION (Z-Score Based)
# ============================================================
print("\n" + "="*70)
print("SECTION 2: MEAN REVERSION STRATEGIES")
print("="*70)

for lookback in tqdm([10, 20, 30, 50, 60, 90, 120], desc="Z-Score Lookbacks"):
    ma = df.groupby('ticker')['close'].transform(lambda x: x.rolling(lookback).mean())
    std = df.groupby('ticker')['close'].transform(lambda x: x.rolling(lookback).std())
    df[f'zscore_{lookback}'] = (df['close'] - ma) / std.replace(0, np.nan)
    
    for thresh in [1.0, 1.5, 2.0, 2.5, 3.0]:
        for hold in [1, 2, 3, 5, 10, 15, 20]:
            # Buy when oversold (negative z-score)
            buy = df[df[f'zscore_{lookback}'] < -thresh][f'fwd_{hold}']
            m, n, t = calc_t(buy)
            if n > 100:
                results.append({'category': 'MEAN_REVERSION', 'strategy': f'ZScore{lookback}_BUY{thresh}_H{hold}',
                               'avg_return': m, 'n_samples': n, 't_stat': t})
            
            # Sell when overbought
            sell = df[df[f'zscore_{lookback}'] > thresh][f'fwd_{hold}']
            m, n, t = calc_t(sell)
            if n > 100:
                results.append({'category': 'MEAN_REVERSION', 'strategy': f'ZScore{lookback}_SELL{thresh}_H{hold}',
                               'avg_return': m, 'n_samples': n, 't_stat': t})

# ============================================================
# SECTION 3: MOMENTUM STRATEGIES
# ============================================================
print("\n" + "="*70)
print("SECTION 3: MOMENTUM STRATEGIES")
print("="*70)

for lookback in tqdm([5, 10, 20, 50, 100, 200, 252], desc="Momentum Lookbacks"):
    df[f'mom_{lookback}'] = df.groupby('ticker')['close'].transform(lambda x: x.pct_change(lookback))
    
    for pct in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        for hold in [1, 5, 10, 20, 40]:
            # Top momentum
            thresh = df[f'mom_{lookback}'].quantile(1 - pct)
            top = df[df[f'mom_{lookback}'] >= thresh][f'fwd_{hold}']
            m, n, t = calc_t(top)
            if n > 100:
                results.append({'category': 'MOMENTUM', 'strategy': f'Mom{lookback}_Top{int(pct*100)}_H{hold}',
                               'avg_return': m, 'n_samples': n, 't_stat': t})
            
            # Bottom momentum (reversal)
            thresh_low = df[f'mom_{lookback}'].quantile(pct)
            bot = df[df[f'mom_{lookback}'] <= thresh_low][f'fwd_{hold}']
            m, n, t = calc_t(bot)
            if n > 100:
                results.append({'category': 'MOMENTUM', 'strategy': f'MomRev{lookback}_Bot{int(pct*100)}_H{hold}',
                               'avg_return': m, 'n_samples': n, 't_stat': t})

# ============================================================
# SECTION 4: VOLATILITY STRATEGIES
# ============================================================
print("\n" + "="*70)
print("SECTION 4: VOLATILITY STRATEGIES")
print("="*70)

for vol_lb in tqdm([5, 10, 20, 30, 60], desc="Volatility Lookbacks"):
    df[f'vol_{vol_lb}'] = df.groupby('ticker')['returns'].transform(lambda x: x.rolling(vol_lb).std())
    
    for pct in [0.10, 0.25, 0.50]:
        thresh_low = df[f'vol_{vol_lb}'].quantile(pct)
        thresh_high = df[f'vol_{vol_lb}'].quantile(1 - pct)
        
        for hold in [1, 5, 10, 20]:
            # Low volatility
            low_vol = df[df[f'vol_{vol_lb}'] <= thresh_low][f'fwd_{hold}']
            m, n, t = calc_t(low_vol)
            if n > 100:
                results.append({'category': 'VOLATILITY', 'strategy': f'LowVol{vol_lb}_Q{int(pct*100)}_H{hold}',
                               'avg_return': m, 'n_samples': n, 't_stat': t})
            
            # High volatility
            high_vol = df[df[f'vol_{vol_lb}'] >= thresh_high][f'fwd_{hold}']
            m, n, t = calc_t(high_vol)
            if n > 100:
                results.append({'category': 'VOLATILITY', 'strategy': f'HighVol{vol_lb}_Q{int(100-int(pct*100))}_H{hold}',
                               'avg_return': m, 'n_samples': n, 't_stat': t})

# ============================================================
# SECTION 5: MOVING AVERAGE STRATEGIES
# ============================================================
print("\n" + "="*70)
print("SECTION 5: MOVING AVERAGE STRATEGIES")
print("="*70)

for fast in tqdm([5, 10, 20], desc="MA Fast"):
    df[f'ma_{fast}'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(fast).mean())
    
    for slow in [20, 50, 100, 200]:
        if slow <= fast:
            continue
        df[f'ma_{slow}'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(slow).mean())
        
        # Cross signals
        df['ma_diff'] = df[f'ma_{fast}'] - df[f'ma_{slow}']
        df['ma_cross_up'] = (df['ma_diff'] > 0) & (df['ma_diff'].shift(1) <= 0)
        df['ma_cross_dn'] = (df['ma_diff'] < 0) & (df['ma_diff'].shift(1) >= 0)
        
        for hold in [1, 5, 10, 20]:
            # Golden cross
            golden = df[df['ma_cross_up']][f'fwd_{hold}']
            m, n, t = calc_t(golden)
            if n > 100:
                results.append({'category': 'MA_CROSS', 'strategy': f'Golden_{fast}_{slow}_H{hold}',
                               'avg_return': m, 'n_samples': n, 't_stat': t})
            
            # Death cross
            death = df[df['ma_cross_dn']][f'fwd_{hold}']
            m, n, t = calc_t(death)
            if n > 100:
                results.append({'category': 'MA_CROSS', 'strategy': f'Death_{fast}_{slow}_H{hold}',
                               'avg_return': m, 'n_samples': n, 't_stat': t})
            
            # Above MA
            above = df[df['close'] > df[f'ma_{slow}']][f'fwd_{hold}']
            m, n, t = calc_t(above)
            if n > 100:
                results.append({'category': 'MA_TREND', 'strategy': f'AboveMA{slow}_H{hold}',
                               'avg_return': m, 'n_samples': n, 't_stat': t})

# ============================================================
# SECTION 6: BOLLINGER BAND STRATEGIES
# ============================================================
print("\n" + "="*70)
print("SECTION 6: BOLLINGER BAND STRATEGIES")
print("="*70)

for period in tqdm([10, 20, 30, 50], desc="Bollinger Periods"):
    ma = df.groupby('ticker')['close'].transform(lambda x: x.rolling(period).mean())
    std = df.groupby('ticker')['close'].transform(lambda x: x.rolling(period).std())
    
    for num_std in [1.5, 2.0, 2.5, 3.0]:
        upper = ma + num_std * std
        lower = ma - num_std * std
        df[f'bb_pos_{period}_{num_std}'] = (df['close'] - lower) / (upper - lower)
        
        for hold in [1, 3, 5, 10, 15]:
            # Below lower band
            below = df[df['close'] < lower][f'fwd_{hold}']
            m, n, t = calc_t(below)
            if n > 100:
                results.append({'category': 'BOLLINGER', 'strategy': f'BB{period}_{num_std}std_BelowLow_H{hold}',
                               'avg_return': m, 'n_samples': n, 't_stat': t})
            
            # Above upper band
            above = df[df['close'] > upper][f'fwd_{hold}']
            m, n, t = calc_t(above)
            if n > 100:
                results.append({'category': 'BOLLINGER', 'strategy': f'BB{period}_{num_std}std_AboveUp_H{hold}',
                               'avg_return': m, 'n_samples': n, 't_stat': t})

# ============================================================
# SECTION 7: VOLUME STRATEGIES
# ============================================================
print("\n" + "="*70)
print("SECTION 7: VOLUME STRATEGIES")
print("="*70)

for vol_lb in tqdm([5, 10, 20], desc="Volume Lookbacks"):
    df[f'vol_ma_{vol_lb}'] = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(vol_lb).mean())
    df[f'vol_ratio_{vol_lb}'] = df['volume'] / df[f'vol_ma_{vol_lb}']
    
    for mult in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        for hold in [1, 3, 5, 10]:
            # Volume spike
            spike = df[df[f'vol_ratio_{vol_lb}'] >= mult][f'fwd_{hold}']
            m, n, t = calc_t(spike)
            if n > 100:
                results.append({'category': 'VOLUME', 'strategy': f'VolSpike{vol_lb}_{mult}x_H{hold}',
                               'avg_return': m, 'n_samples': n, 't_stat': t})
            
            # Volume spike + up day
            spike_up = df[(df[f'vol_ratio_{vol_lb}'] >= mult) & (df['returns'] > 0)][f'fwd_{hold}']
            m, n, t = calc_t(spike_up)
            if n > 100:
                results.append({'category': 'VOLUME', 'strategy': f'VolSpikeUp{vol_lb}_{mult}x_H{hold}',
                               'avg_return': m, 'n_samples': n, 't_stat': t})
            
            # Volume spike + down day
            spike_dn = df[(df[f'vol_ratio_{vol_lb}'] >= mult) & (df['returns'] < 0)][f'fwd_{hold}']
            m, n, t = calc_t(spike_dn)
            if n > 100:
                results.append({'category': 'VOLUME', 'strategy': f'VolSpikeDn{vol_lb}_{mult}x_H{hold}',
                               'avg_return': m, 'n_samples': n, 't_stat': t})

# ============================================================
# SECTION 8: PRICE BREAKOUTS
# ============================================================
print("\n" + "="*70)
print("SECTION 8: PRICE BREAKOUT STRATEGIES")
print("="*70)

for lookback in tqdm([10, 20, 50, 100, 200], desc="Breakout Lookbacks"):
    df[f'high_{lookback}'] = df.groupby('ticker')['high'].transform(lambda x: x.rolling(lookback).max())
    df[f'low_{lookback}'] = df.groupby('ticker')['low'].transform(lambda x: x.rolling(lookback).min())
    
    for hold in [1, 3, 5, 10, 20]:
        # Breakout above high
        breakout_up = df[df['close'] >= df[f'high_{lookback}'].shift(1)][f'fwd_{hold}']
        m, n, t = calc_t(breakout_up)
        if n > 100:
            results.append({'category': 'BREAKOUT', 'strategy': f'BreakHigh{lookback}_H{hold}',
                           'avg_return': m, 'n_samples': n, 't_stat': t})
        
        # Breakdown below low
        breakout_dn = df[df['close'] <= df[f'low_{lookback}'].shift(1)][f'fwd_{hold}']
        m, n, t = calc_t(breakout_dn)
        if n > 100:
            results.append({'category': 'BREAKOUT', 'strategy': f'BreakLow{lookback}_H{hold}',
                           'avg_return': m, 'n_samples': n, 't_stat': t})

# ============================================================
# SECTION 9: GAP STRATEGIES
# ============================================================
print("\n" + "="*70)
print("SECTION 9: GAP STRATEGIES")
print("="*70)

df['prev_close'] = df.groupby('ticker')['close'].shift(1)
df['gap'] = (df['open'] - df['prev_close']) / df['prev_close']

for gap_thresh in tqdm([0.01, 0.02, 0.03, 0.05, 0.07, 0.10], desc="Gap Thresholds"):
    for hold in [1, 3, 5, 10]:
        # Gap up
        gap_up = df[df['gap'] >= gap_thresh][f'fwd_{hold}']
        m, n, t = calc_t(gap_up)
        if n > 100:
            results.append({'category': 'GAP', 'strategy': f'GapUp{int(gap_thresh*100)}pct_H{hold}',
                           'avg_return': m, 'n_samples': n, 't_stat': t})
        
        # Gap down
        gap_dn = df[df['gap'] <= -gap_thresh][f'fwd_{hold}']
        m, n, t = calc_t(gap_dn)
        if n > 100:
            results.append({'category': 'GAP', 'strategy': f'GapDn{int(gap_thresh*100)}pct_H{hold}',
                           'avg_return': m, 'n_samples': n, 't_stat': t})

# ============================================================
# SECTION 10: CALENDAR EFFECTS
# ============================================================
print("\n" + "="*70)
print("SECTION 10: CALENDAR EFFECTS")
print("="*70)

for hold in tqdm([1, 3, 5, 10], desc="Calendar"):
    # Day of week
    for day in range(5):
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        rets = df[df['dow'] == day][f'fwd_{hold}']
        m, n, t = calc_t(rets)
        if n > 100:
            results.append({'category': 'CALENDAR', 'strategy': f'{day_names[day]}_H{hold}',
                           'avg_return': m, 'n_samples': n, 't_stat': t})
    
    # Month of year
    for month in range(1, 13):
        month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        rets = df[df['month'] == month][f'fwd_{hold}']
        m, n, t = calc_t(rets)
        if n > 100:
            results.append({'category': 'CALENDAR', 'strategy': f'{month_names[month-1]}_H{hold}',
                           'avg_return': m, 'n_samples': n, 't_stat': t})
    
    # Week of month
    for wom in [1, 2, 3, 4]:
        rets = df[df['wom'] == wom][f'fwd_{hold}']
        m, n, t = calc_t(rets)
        if n > 100:
            results.append({'category': 'CALENDAR', 'strategy': f'Week{wom}OfMonth_H{hold}',
                           'avg_return': m, 'n_samples': n, 't_stat': t})

# ============================================================
# SECTION 11: EXTREME MOVES
# ============================================================
print("\n" + "="*70)
print("SECTION 11: EXTREME MOVE STRATEGIES")
print("="*70)

for thresh in tqdm([0.03, 0.05, 0.07, 0.10, 0.15, 0.20], desc="Extreme Moves"):
    for hold in [1, 3, 5, 10, 20]:
        # Big up day
        big_up = df[df['returns'] >= thresh][f'fwd_{hold}']
        m, n, t = calc_t(big_up)
        if n > 100:
            results.append({'category': 'EXTREME', 'strategy': f'BigUp{int(thresh*100)}pct_H{hold}',
                           'avg_return': m, 'n_samples': n, 't_stat': t})
        
        # Big down day
        big_dn = df[df['returns'] <= -thresh][f'fwd_{hold}']
        m, n, t = calc_t(big_dn)
        if n > 100:
            results.append({'category': 'EXTREME', 'strategy': f'BigDn{int(thresh*100)}pct_H{hold}',
                           'avg_return': m, 'n_samples': n, 't_stat': t})

# ============================================================
# SECTION 12: CANDLESTICK PATTERNS
# ============================================================
print("\n" + "="*70)
print("SECTION 12: CANDLESTICK PATTERNS")
print("="*70)

df['body'] = (df['close'] - df['open']) / df['open']
df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']
df['range'] = (df['high'] - df['low']) / df['close']

for hold in tqdm([1, 3, 5, 10], desc="Candlesticks"):
    # Hammer (long lower wick, small body)
    hammer = df[(df['lower_wick'] > 0.02) & (abs(df['body']) < 0.01)][f'fwd_{hold}']
    m, n, t = calc_t(hammer)
    if n > 100:
        results.append({'category': 'CANDLE', 'strategy': f'Hammer_H{hold}',
                       'avg_return': m, 'n_samples': n, 't_stat': t})
    
    # Shooting star
    shooting = df[(df['upper_wick'] > 0.02) & (abs(df['body']) < 0.01)][f'fwd_{hold}']
    m, n, t = calc_t(shooting)
    if n > 100:
        results.append({'category': 'CANDLE', 'strategy': f'ShootingStar_H{hold}',
                       'avg_return': m, 'n_samples': n, 't_stat': t})
    
    # Big bullish
    big_bull = df[df['body'] > 0.03][f'fwd_{hold}']
    m, n, t = calc_t(big_bull)
    if n > 100:
        results.append({'category': 'CANDLE', 'strategy': f'BigBull_H{hold}',
                       'avg_return': m, 'n_samples': n, 't_stat': t})
    
    # Big bearish
    big_bear = df[df['body'] < -0.03][f'fwd_{hold}']
    m, n, t = calc_t(big_bear)
    if n > 100:
        results.append({'category': 'CANDLE', 'strategy': f'BigBear_H{hold}',
                       'avg_return': m, 'n_samples': n, 't_stat': t})
    
    # Doji
    doji = df[abs(df['body']) < 0.002][f'fwd_{hold}']
    m, n, t = calc_t(doji)
    if n > 100:
        results.append({'category': 'CANDLE', 'strategy': f'Doji_H{hold}',
                       'avg_return': m, 'n_samples': n, 't_stat': t})

# ============================================================
# SECTION 13: PULLBACK STRATEGIES (Multi-Timeframe)
# ============================================================
print("\n" + "="*70)
print("SECTION 13: PULLBACK STRATEGIES")
print("="*70)

for short_lb in tqdm([3, 5, 10], desc="Pullbacks"):
    df[f'short_mom_{short_lb}'] = df.groupby('ticker')['close'].transform(lambda x: x.pct_change(short_lb))
    
    for long_lb in [20, 50, 100, 200]:
        df[f'long_mom_{long_lb}'] = df.groupby('ticker')['close'].transform(lambda x: x.pct_change(long_lb))
        
        for hold in [1, 5, 10, 20]:
            # Pullback in uptrend (short neg, long pos)
            pullback = df[(df[f'short_mom_{short_lb}'] < 0) & (df[f'long_mom_{long_lb}'] > 0)][f'fwd_{hold}']
            m, n, t = calc_t(pullback)
            if n > 100:
                results.append({'category': 'PULLBACK', 'strategy': f'Pullback_{short_lb}_{long_lb}_H{hold}',
                               'avg_return': m, 'n_samples': n, 't_stat': t})
            
            # Rally in downtrend (short pos, long neg)
            rally = df[(df[f'short_mom_{short_lb}'] > 0) & (df[f'long_mom_{long_lb}'] < 0)][f'fwd_{hold}']
            m, n, t = calc_t(rally)
            if n > 100:
                results.append({'category': 'PULLBACK', 'strategy': f'Rally_{short_lb}_{long_lb}_H{hold}',
                               'avg_return': m, 'n_samples': n, 't_stat': t})

# ============================================================
# SECTION 14: ATR STRATEGIES
# ============================================================
print("\n" + "="*70)
print("SECTION 14: ATR STRATEGIES")
print("="*70)

for atr_period in tqdm([5, 10, 14, 20], desc="ATR"):
    prev_close = df.groupby('ticker')['close'].shift(1)
    df['_tr'] = np.maximum(df['high'] - df['low'],
                   np.maximum(abs(df['high'] - prev_close),
                             abs(df['low'] - prev_close)))
    df[f'atr_{atr_period}'] = df.groupby('ticker')['_tr'].transform(lambda x: x.rolling(atr_period).mean())
    df[f'atr_pct_{atr_period}'] = df[f'atr_{atr_period}'] / df['close']
    
    for pct in [0.1, 0.25]:
        thresh_high = df[f'atr_pct_{atr_period}'].quantile(1 - pct)
        thresh_low = df[f'atr_pct_{atr_period}'].quantile(pct)
        
        for hold in [1, 5, 10, 20]:
            # High ATR
            high_atr = df[df[f'atr_pct_{atr_period}'] >= thresh_high][f'fwd_{hold}']
            m, n, t = calc_t(high_atr)
            if n > 100:
                results.append({'category': 'ATR', 'strategy': f'HighATR{atr_period}_Top{int(pct*100)}_H{hold}',
                               'avg_return': m, 'n_samples': n, 't_stat': t})
            
            # Low ATR
            low_atr = df[df[f'atr_pct_{atr_period}'] <= thresh_low][f'fwd_{hold}']
            m, n, t = calc_t(low_atr)
            if n > 100:
                results.append({'category': 'ATR', 'strategy': f'LowATR{atr_period}_Bot{int(pct*100)}_H{hold}',
                               'avg_return': m, 'n_samples': n, 't_stat': t})

# ============================================================
# SECTION 15: STOCHASTIC STRATEGIES
# ============================================================
print("\n" + "="*70)
print("SECTION 15: STOCHASTIC STRATEGIES")
print("="*70)

for k_period in tqdm([5, 9, 14, 21], desc="Stochastic"):
    lowest = df.groupby('ticker')['low'].transform(lambda x: x.rolling(k_period).min())
    highest = df.groupby('ticker')['high'].transform(lambda x: x.rolling(k_period).max())
    df[f'stoch_{k_period}'] = (df['close'] - lowest) / (highest - lowest) * 100
    
    for oversold in [10, 20, 30]:
        for overbought in [70, 80, 90]:
            for hold in [1, 3, 5, 10]:
                # Oversold
                os_rets = df[df[f'stoch_{k_period}'] < oversold][f'fwd_{hold}']
                m, n, t = calc_t(os_rets)
                if n > 100:
                    results.append({'category': 'STOCH', 'strategy': f'Stoch{k_period}_OS{oversold}_H{hold}',
                                   'avg_return': m, 'n_samples': n, 't_stat': t})
                
                # Overbought
                ob_rets = df[df[f'stoch_{k_period}'] > overbought][f'fwd_{hold}']
                m, n, t = calc_t(ob_rets)
                if n > 100:
                    results.append({'category': 'STOCH', 'strategy': f'Stoch{k_period}_OB{overbought}_H{hold}',
                                   'avg_return': m, 'n_samples': n, 't_stat': t})

# ============================================================
# SAVE RESULTS
# ============================================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

results_df = pd.DataFrame(results)
results_df['significant'] = results_df['t_stat'].abs() > 3.0

# Save to data folder if exists, otherwise current dir
output_dir = 'data' if os.path.exists('data') else '.'
output_file = f'{output_dir}/MEGA_TEST_RESULTS.csv'
results_df.to_csv(output_file, index=False)

sig = results_df[results_df['significant']]

print(f"\n{'='*70}")
print(f"MEGA TEST COMPLETE")
print(f"{'='*70}")
print(f"\nTotal strategies tested: {len(results_df):,}")
print(f"Significant (|t| > 3.0): {len(sig):,} ({100*len(sig)/len(results_df):.1f}%)")
print(f"\nResults saved to: {output_file}")

# Top 50
print(f"\n{'='*70}")
print("TOP 50 STRATEGIES DISCOVERED")
print(f"{'='*70}")
for i, (_, r) in enumerate(results_df.nlargest(50, 't_stat').iterrows(), 1):
    print(f"{i:2}. {r['category']:15} | {r['strategy']:40} | t={r['t_stat']:7.2f} | ret={r['avg_return']*100:6.2f}%")

# By category
print(f"\n{'='*70}")
print("SUMMARY BY CATEGORY")
print(f"{'='*70}")
for cat in sorted(results_df['category'].unique()):
    cat_df = results_df[results_df['category'] == cat]
    cat_sig = cat_df[cat_df['significant']]
    best = cat_df.nlargest(1, 't_stat').iloc[0] if len(cat_df) > 0 else None
    print(f"{cat:15} | {len(cat_sig):4}/{len(cat_df):4} significant | Best t={best['t_stat']:.1f}")

print(f"\n{'='*70}")
print("DONE! Results saved to:", output_file)
print(f"{'='*70}")

input("\nPress Enter to exit...")
