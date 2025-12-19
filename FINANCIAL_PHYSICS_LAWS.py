#!/usr/bin/env python3
"""
FINANCIAL PHYSICS - MAPPING THE UNIVERSE
=========================================

Like Copernicus proving the Earth revolves around the Sun,
we will TEST every assumption the financial world holds dear.

We have what ancient astronomers didn't have:
- DATA (millions of data points)
- BACKTESTING (we can "move time forward")
- STATISTICS (t-stats to prove/disprove)

CONVENTIONAL WISDOM TO TEST:
1. "Buy low, sell high" - Is this actually true?
2. "The trend is your friend" - Or is mean reversion king?
3. "Volume confirms price" - Does it really?
4. "Markets are efficient" - Or are there exploitable patterns?
5. "Past performance doesn't predict future" - Let's see...

Let's find out what ACTUALLY moves the planets (prices).
"""

import sqlite3
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("FINANCIAL PHYSICS - MAPPING THE UNIVERSE")
print("Questioning Everything. Proving Nothing Without Data.")
print("="*70)

# Load data
DB_PATH = 'data/market_data.db'
if not os.path.exists(DB_PATH):
    print("ERROR: Database not found")
    exit(1)

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM ohlcv", conn)
conn.close()

print(f"\nOur Universe: {len(df):,} observations across {df['ticker'].nunique():,} stocks")
print("Like astronomers with millions of star observations...")

# Prepare
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
df['returns'] = df.groupby('ticker')['close'].pct_change()

# Forward returns
print("\nPre-computing forward returns (our time machine)...")
for h in [1, 2, 3, 5, 10, 20, 40, 60]:
    df[f'fwd_{h}'] = df.groupby('ticker')['close'].transform(lambda x: x.shift(-h) / x - 1)

results = []

def calc_t(rets, name=""):
    """Calculate t-statistic - our proof mechanism"""
    rets = rets.dropna()
    if len(rets) < 100:
        return None
    m = np.mean(rets)
    s = np.std(rets, ddof=1)
    if s == 0:
        return None
    t = m / (s / np.sqrt(len(rets)))
    return {'strategy': name, 'mean_return': m, 'n': len(rets), 't_stat': t, 'significant': abs(t) > 3.0}

# ================================================================
# FUNDAMENTAL LAW 1: GRAVITY (Mean Reversion)
# Does what goes up, come down? Does what falls, rise again?
# ================================================================
print("\n" + "="*70)
print("LAW 1: GRAVITY - Does what goes up come down?")
print("Testing: Mean reversion vs Momentum")
print("="*70)

gravity_results = []

# Test: After extreme moves, what happens?
for move_size in tqdm([0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30], desc="Testing Gravity"):
    for hold in [1, 2, 3, 5, 10, 20, 40]:
        # After big UP move - does gravity pull it down?
        big_up = df[df['returns'] >= move_size][f'fwd_{hold}']
        r = calc_t(big_up, f'After_{int(move_size*100)}pct_UP_H{hold}')
        if r:
            r['hypothesis'] = 'GRAVITY_DOWN' if r['mean_return'] < 0 else 'MOMENTUM_UP'
            gravity_results.append(r)
        
        # After big DOWN move - does gravity pull it up?
        big_dn = df[df['returns'] <= -move_size][f'fwd_{hold}']
        r = calc_t(big_dn, f'After_{int(move_size*100)}pct_DN_H{hold}')
        if r:
            r['hypothesis'] = 'GRAVITY_UP' if r['mean_return'] > 0 else 'MOMENTUM_DOWN'
            gravity_results.append(r)

gravity_df = pd.DataFrame(gravity_results)
if len(gravity_df) > 0:
    sig = gravity_df[gravity_df['significant']]
    gravity_up = len(sig[sig['hypothesis'] == 'GRAVITY_UP'])
    gravity_dn = len(sig[sig['hypothesis'] == 'GRAVITY_DOWN'])
    mom_up = len(sig[sig['hypothesis'] == 'MOMENTUM_UP'])
    mom_dn = len(sig[sig['hypothesis'] == 'MOMENTUM_DOWN'])
    
    print(f"\nGRAVITY VERDICT:")
    print(f"  Gravity (mean reversion) wins: {gravity_up + gravity_dn} cases")
    print(f"  Momentum wins: {mom_up + mom_dn} cases")
    if gravity_up + gravity_dn > mom_up + mom_dn:
        print("  >>> MEAN REVERSION IS THE DOMINANT FORCE <<<")
    else:
        print("  >>> MOMENTUM IS THE DOMINANT FORCE <<<")
    results.extend(gravity_results)

# ================================================================
# FUNDAMENTAL LAW 2: INERTIA (Trend Persistence)
# Does a body in motion stay in motion?
# ================================================================
print("\n" + "="*70)
print("LAW 2: INERTIA - Does a body in motion stay in motion?")
print("Testing: Do trends persist or reverse?")
print("="*70)

inertia_results = []

for lookback in tqdm([5, 10, 20, 50, 100, 200], desc="Testing Inertia"):
    df[f'trend_{lookback}'] = df.groupby('ticker')['close'].transform(lambda x: x.pct_change(lookback))
    
    # Strong uptrend - does it continue?
    for thresh in [0.10, 0.20, 0.30, 0.50]:
        for hold in [1, 5, 10, 20, 40]:
            # Uptrend continuation
            uptrend = df[df[f'trend_{lookback}'] >= thresh][f'fwd_{hold}']
            r = calc_t(uptrend, f'Uptrend{lookback}_{int(thresh*100)}pct_H{hold}')
            if r:
                r['hypothesis'] = 'INERTIA' if r['mean_return'] > 0 else 'REVERSAL'
                r['law'] = 'INERTIA'
                inertia_results.append(r)
            
            # Downtrend continuation
            downtrend = df[df[f'trend_{lookback}'] <= -thresh][f'fwd_{hold}']
            r = calc_t(downtrend, f'Downtrend{lookback}_{int(thresh*100)}pct_H{hold}')
            if r:
                r['hypothesis'] = 'INERTIA' if r['mean_return'] < 0 else 'REVERSAL'
                r['law'] = 'INERTIA'
                inertia_results.append(r)

inertia_df = pd.DataFrame(inertia_results)
if len(inertia_df) > 0:
    sig = inertia_df[inertia_df['significant']]
    inertia_wins = len(sig[sig['hypothesis'] == 'INERTIA'])
    reversal_wins = len(sig[sig['hypothesis'] == 'REVERSAL'])
    print(f"\nINERTIA VERDICT:")
    print(f"  Inertia (trend continues) wins: {inertia_wins} cases")
    print(f"  Reversal wins: {reversal_wins} cases")
    if inertia_wins > reversal_wins:
        print("  >>> TRENDS TEND TO PERSIST <<<")
    else:
        print("  >>> TRENDS TEND TO REVERSE <<<")
    results.extend(inertia_results)

# ================================================================
# FUNDAMENTAL LAW 3: VOLUME = MASS (Does volume move price?)
# ================================================================
print("\n" + "="*70)
print("LAW 3: MASS - Does volume (mass) affect price movement?")
print("Testing: Does volume predict future returns?")
print("="*70)

mass_results = []

for vol_lb in tqdm([5, 10, 20], desc="Testing Mass"):
    df[f'vol_ma_{vol_lb}'] = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(vol_lb).mean())
    df[f'vol_ratio_{vol_lb}'] = df['volume'] / df[f'vol_ma_{vol_lb}']
    
    # Does HIGH volume predict anything?
    for mult in [1.5, 2.0, 3.0, 5.0]:
        for hold in [1, 3, 5, 10, 20]:
            # High volume alone
            high_vol = df[df[f'vol_ratio_{vol_lb}'] >= mult][f'fwd_{hold}']
            r = calc_t(high_vol, f'HighVol{vol_lb}_{mult}x_H{hold}')
            if r:
                r['law'] = 'MASS'
                r['hypothesis'] = 'VOLUME_PREDICTS' if abs(r['t_stat']) > 3 else 'VOLUME_NEUTRAL'
                mass_results.append(r)
            
            # High volume + UP day
            hv_up = df[(df[f'vol_ratio_{vol_lb}'] >= mult) & (df['returns'] > 0)][f'fwd_{hold}']
            r = calc_t(hv_up, f'HighVolUp{vol_lb}_{mult}x_H{hold}')
            if r:
                r['law'] = 'MASS'
                r['hypothesis'] = 'VOL_CONFIRMS_UP' if r['mean_return'] > 0 else 'VOL_REVERSAL'
                mass_results.append(r)
            
            # High volume + DOWN day
            hv_dn = df[(df[f'vol_ratio_{vol_lb}'] >= mult) & (df['returns'] < 0)][f'fwd_{hold}']
            r = calc_t(hv_dn, f'HighVolDn{vol_lb}_{mult}x_H{hold}')
            if r:
                r['law'] = 'MASS'
                r['hypothesis'] = 'VOL_CONFIRMS_DN' if r['mean_return'] < 0 else 'VOL_REVERSAL'
                mass_results.append(r)

    # Does LOW volume predict anything?
    for pct in [0.25, 0.10]:
        thresh = df[f'vol_ratio_{vol_lb}'].quantile(pct)
        for hold in [1, 5, 10, 20]:
            low_vol = df[df[f'vol_ratio_{vol_lb}'] <= thresh][f'fwd_{hold}']
            r = calc_t(low_vol, f'LowVol{vol_lb}_Q{int(pct*100)}_H{hold}')
            if r:
                r['law'] = 'MASS'
                mass_results.append(r)

mass_df = pd.DataFrame(mass_results)
if len(mass_df) > 0:
    sig = mass_df[mass_df['significant']]
    print(f"\nMASS (VOLUME) VERDICT:")
    print(f"  Total significant volume patterns: {len(sig)}")
    confirms = len([r for r in mass_results if r.get('hypothesis', '').startswith('VOL_CONFIRMS')])
    reversals = len([r for r in mass_results if r.get('hypothesis') == 'VOL_REVERSAL'])
    print(f"  Volume confirms direction: {confirms} cases")
    print(f"  Volume predicts reversal: {reversals} cases")
    results.extend(mass_results)

# ================================================================
# FUNDAMENTAL LAW 4: TIME (Does day/month/year matter?)
# Is there order in the calendar, or is it random?
# ================================================================
print("\n" + "="*70)
print("LAW 4: TIME - Is the calendar random or ordered?")
print("Testing: Calendar anomalies")
print("="*70)

df['dow'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['dom'] = df['date'].dt.day

time_results = []

# Day of week
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
for hold in tqdm([1, 5, 10], desc="Testing Time"):
    for d, name in enumerate(days):
        rets = df[df['dow'] == d][f'fwd_{hold}']
        r = calc_t(rets, f'{name}_H{hold}')
        if r:
            r['law'] = 'TIME'
            time_results.append(r)

# Month of year
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
for hold in [1, 5, 10, 20]:
    for m, name in enumerate(months, 1):
        rets = df[df['month'] == m][f'fwd_{hold}']
        r = calc_t(rets, f'{name}_H{hold}')
        if r:
            r['law'] = 'TIME'
            time_results.append(r)

# Turn of month effect
for hold in [1, 3, 5]:
    # Last 3 days of month
    last_days = df[df['dom'] >= 28][f'fwd_{hold}']
    r = calc_t(last_days, f'MonthEnd_H{hold}')
    if r:
        r['law'] = 'TIME'
        time_results.append(r)
    
    # First 3 days of month
    first_days = df[df['dom'] <= 3][f'fwd_{hold}']
    r = calc_t(first_days, f'MonthStart_H{hold}')
    if r:
        r['law'] = 'TIME'
        time_results.append(r)

time_df = pd.DataFrame(time_results)
if len(time_df) > 0:
    sig = time_df[time_df['significant']]
    print(f"\nTIME VERDICT:")
    print(f"  Significant calendar patterns: {len(sig)}")
    if len(sig) > 0:
        print("  >>> TIME IS NOT RANDOM <<<")
        print("  Best calendar patterns:")
        for _, r in sig.nlargest(5, 't_stat').iterrows():
            print(f"    {r['strategy']}: t={r['t_stat']:.2f}")
    results.extend(time_results)

# ================================================================
# FUNDAMENTAL LAW 5: VOLATILITY (Does chaos predict?)
# ================================================================
print("\n" + "="*70)
print("LAW 5: CHAOS - Does volatility predict future returns?")
print("Testing: Volatility forecasting power")
print("="*70)

chaos_results = []

for vol_lb in tqdm([5, 10, 20, 30, 60], desc="Testing Chaos"):
    df[f'volatility_{vol_lb}'] = df.groupby('ticker')['returns'].transform(lambda x: x.rolling(vol_lb).std())
    
    # Quintiles of volatility
    for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
        thresh = df[f'volatility_{vol_lb}'].quantile(q)
        
        for hold in [1, 5, 10, 20]:
            # Below threshold
            if q <= 0.5:
                low_vol = df[df[f'volatility_{vol_lb}'] <= thresh][f'fwd_{hold}']
                r = calc_t(low_vol, f'LowVol{vol_lb}_Q{int(q*100)}_H{hold}')
                if r:
                    r['law'] = 'CHAOS'
                    chaos_results.append(r)
            else:
                high_vol = df[df[f'volatility_{vol_lb}'] >= thresh][f'fwd_{hold}']
                r = calc_t(high_vol, f'HighVol{vol_lb}_Q{int(q*100)}_H{hold}')
                if r:
                    r['law'] = 'CHAOS'
                    chaos_results.append(r)

chaos_df = pd.DataFrame(chaos_results)
if len(chaos_df) > 0:
    sig = chaos_df[chaos_df['significant']]
    print(f"\nCHAOS (VOLATILITY) VERDICT:")
    print(f"  Significant volatility patterns: {len(sig)}")
    results.extend(chaos_results)

# ================================================================
# FUNDAMENTAL LAW 6: MARKET REGIME (Bull vs Bear)
# Does the overall market state change individual stock behavior?
# ================================================================
print("\n" + "="*70)
print("LAW 6: REGIME - Do market conditions change the rules?")
print("Testing: Bull market vs Bear market behavior")
print("="*70)

regime_results = []

# Calculate market regime using rolling returns
for regime_lb in tqdm([20, 50, 100, 200], desc="Testing Regime"):
    df[f'regime_{regime_lb}'] = df.groupby('ticker')['close'].transform(lambda x: x.pct_change(regime_lb))
    
    for hold in [1, 5, 10, 20]:
        # Bull market (positive regime)
        bull = df[df[f'regime_{regime_lb}'] > 0.10][f'fwd_{hold}']
        r = calc_t(bull, f'BullMarket{regime_lb}_H{hold}')
        if r:
            r['law'] = 'REGIME'
            r['regime'] = 'BULL'
            regime_results.append(r)
        
        # Bear market (negative regime)
        bear = df[df[f'regime_{regime_lb}'] < -0.10][f'fwd_{hold}']
        r = calc_t(bear, f'BearMarket{regime_lb}_H{hold}')
        if r:
            r['law'] = 'REGIME'
            r['regime'] = 'BEAR'
            regime_results.append(r)
        
        # Strong bull
        strong_bull = df[df[f'regime_{regime_lb}'] > 0.25][f'fwd_{hold}']
        r = calc_t(strong_bull, f'StrongBull{regime_lb}_H{hold}')
        if r:
            r['law'] = 'REGIME'
            r['regime'] = 'STRONG_BULL'
            regime_results.append(r)
        
        # Strong bear
        strong_bear = df[df[f'regime_{regime_lb}'] < -0.25][f'fwd_{hold}']
        r = calc_t(strong_bear, f'StrongBear{regime_lb}_H{hold}')
        if r:
            r['law'] = 'REGIME'
            r['regime'] = 'STRONG_BEAR'
            regime_results.append(r)

regime_df = pd.DataFrame(regime_results)
if len(regime_df) > 0:
    sig = regime_df[regime_df['significant']]
    bull_sig = len(sig[sig.get('regime', '') == 'BULL']) if 'regime' in sig.columns else 0
    bear_sig = len(sig[sig.get('regime', '') == 'BEAR']) if 'regime' in sig.columns else 0
    print(f"\nREGIME VERDICT:")
    print(f"  Significant bull market patterns: {bull_sig}")
    print(f"  Significant bear market patterns: {bear_sig}")
    results.extend(regime_results)

# ================================================================
# FUNDAMENTAL LAW 7: PRICE LEVELS (Support/Resistance)
# Do certain price levels have meaning?
# ================================================================
print("\n" + "="*70)
print("LAW 7: PRICE LEVELS - Do support/resistance levels exist?")
print("Testing: 52-week high/low, round numbers")
print("="*70)

level_results = []

for lb in tqdm([50, 100, 200, 252], desc="Testing Price Levels"):
    df[f'high_{lb}'] = df.groupby('ticker')['high'].transform(lambda x: x.rolling(lb).max())
    df[f'low_{lb}'] = df.groupby('ticker')['low'].transform(lambda x: x.rolling(lb).min())
    df[f'range_{lb}'] = df[f'high_{lb}'] - df[f'low_{lb}']
    df[f'pct_of_range_{lb}'] = (df['close'] - df[f'low_{lb}']) / df[f'range_{lb}']
    
    for hold in [1, 5, 10, 20]:
        # Near 52-week high
        near_high = df[df[f'pct_of_range_{lb}'] > 0.95][f'fwd_{hold}']
        r = calc_t(near_high, f'Near{lb}High_H{hold}')
        if r:
            r['law'] = 'PRICE_LEVEL'
            level_results.append(r)
        
        # Near 52-week low
        near_low = df[df[f'pct_of_range_{lb}'] < 0.05][f'fwd_{hold}']
        r = calc_t(near_low, f'Near{lb}Low_H{hold}')
        if r:
            r['law'] = 'PRICE_LEVEL'
            level_results.append(r)
        
        # Breakout above high
        breakout = df[df['close'] >= df[f'high_{lb}'].shift(1)][f'fwd_{hold}']
        r = calc_t(breakout, f'Breakout{lb}_H{hold}')
        if r:
            r['law'] = 'PRICE_LEVEL'
            level_results.append(r)

level_df = pd.DataFrame(level_results)
if len(level_df) > 0:
    sig = level_df[level_df['significant']]
    print(f"\nPRICE LEVEL VERDICT:")
    print(f"  Significant price level patterns: {len(sig)}")
    results.extend(level_results)

# ================================================================
# FUNDAMENTAL LAW 8: THE EFFICIENT MARKET HYPOTHESIS TEST
# Is the market efficient or are there exploitable patterns?
# ================================================================
print("\n" + "="*70)
print("LAW 8: EFFICIENCY - Is the market efficient?")
print("Testing: Random walk vs predictable patterns")
print("="*70)

# Count all significant results
all_results_df = pd.DataFrame(results)
sig = all_results_df[all_results_df['significant']]

print(f"\nEFFICIENT MARKET HYPOTHESIS TEST:")
print(f"  Total patterns tested: {len(all_results_df):,}")
print(f"  Statistically significant (|t| > 3.0): {len(sig):,} ({100*len(sig)/max(1,len(all_results_df)):.1f}%)")
print(f"  Expected by chance (5%): ~{int(len(all_results_df)*0.05):,}")

if len(sig) > len(all_results_df) * 0.10:
    print("\n  >>> MARKET IS NOT EFFICIENT <<<")
    print("  We found FAR more patterns than random chance would predict.")
    print("  The 'efficient market hypothesis' is WRONG.")
else:
    print("\n  >>> MARKET MAY BE EFFICIENT <<<")
    print("  Pattern count is close to random expectation.")

# ================================================================
# SAVE RESULTS - THE LAWS OF FINANCIAL PHYSICS
# ================================================================
print("\n" + "="*70)
print("SAVING THE LAWS OF FINANCIAL PHYSICS")
print("="*70)

all_results_df = pd.DataFrame(results)
all_results_df.to_csv('data/FINANCIAL_PHYSICS_LAWS.csv', index=False)

print(f"\nTotal laws tested: {len(all_results_df):,}")
print(f"Significant discoveries: {len(sig):,}")

# Summary by law
print(f"\n{'='*70}")
print("SUMMARY BY FUNDAMENTAL LAW")
print(f"{'='*70}")
if 'law' in all_results_df.columns:
    for law in all_results_df['law'].unique():
        law_df = all_results_df[all_results_df['law'] == law]
        law_sig = law_df[law_df['significant']]
        if len(law_df) > 0:
            best = law_df.nlargest(1, 't_stat').iloc[0]
            print(f"{law:20} | {len(law_sig):3}/{len(law_df):3} significant | Best: {best['strategy']} (t={best['t_stat']:.2f})")

# Top 30 discoveries
print(f"\n{'='*70}")
print("TOP 30 DISCOVERIES - THE FUNDAMENTAL LAWS")
print(f"{'='*70}")
for i, (_, r) in enumerate(all_results_df.nlargest(30, 't_stat').iterrows(), 1):
    law = r.get('law', 'N/A')
    print(f"{i:2}. [{law:12}] {r['strategy']:40} | t={r['t_stat']:7.2f} | ret={r['mean_return']*100:5.2f}%")

print(f"\n{'='*70}")
print("THE FINANCIAL UNIVERSE HAS BEEN MAPPED")
print(f"{'='*70}")
print(f"\nResults saved to: data/FINANCIAL_PHYSICS_LAWS.csv")
print("\nLike Copernicus, we have proven what actually moves the markets.")
print("The data doesn't lie. The conventional wisdom often does.")
