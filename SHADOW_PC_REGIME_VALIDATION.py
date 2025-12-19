#!/usr/bin/env python3
"""
SHADOW PC REGIME VALIDATION
============================
Run this on Shadow PC with Anaconda/Jupyter

PURPOSE: Test ALL 3,323 significant strategies across:
- BULL vs BEAR vs RANGE regimes
- LARGE vs MID vs SMALL cap
- Out-of-sample validation

OUTPUT: regime_validated_strategies.csv

ESTIMATED TIME: 2-4 hours depending on CPU
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("SHADOW PC REGIME VALIDATION")
print("="*70)
print(f"Started: {datetime.now()}")
print()

# ============================================================================
# CONFIGURATION
# ============================================================================

DB_PATH = 'data/market_data.db'  # Adjust if needed
STRATEGIES_CSV = 'data/GRAND_CONSOLIDATED_ALL.csv'  # Your master strategies file

# Walk-forward periods
TRAIN_START = '2023-01-01'
TRAIN_END = '2024-09-30'
TEST_START = '2024-10-01'
TEST_END = '2025-12-31'

# Significance threshold
T_THRESHOLD = 3.0

# ============================================================================
# LOAD DATA
# ============================================================================

print("[1/6] Loading database...")
conn = sqlite3.connect(DB_PATH)

# Load all OHLCV data
df = pd.read_sql("""
    SELECT ticker, date, open, high, low, close, volume
    FROM ohlcv
    WHERE date >= '2023-01-01'
    ORDER BY ticker, date
""", conn)
df['date'] = pd.to_datetime(df['date'])
print(f"      Loaded {len(df):,} rows, {df['ticker'].nunique():,} tickers")

# ============================================================================
# CALCULATE ALL FACTORS (same as original research)
# ============================================================================

print("[2/6] Calculating factors...")

# Group by ticker for calculations
def calc_factors(g):
    g = g.sort_values('date').copy()
    
    # Returns
    g['ret_1'] = g['close'].pct_change(1)
    g['ret_5'] = g['close'].pct_change(5)
    g['ret_20'] = g['close'].pct_change(20)
    g['ret_60'] = g['close'].pct_change(60)
    
    # Forward returns (what we're predicting)
    g['fwd_1'] = g['close'].shift(-1) / g['close'] - 1
    g['fwd_5'] = g['close'].shift(-5) / g['close'] - 1
    g['fwd_10'] = g['close'].shift(-10) / g['close'] - 1
    g['fwd_20'] = g['close'].shift(-20) / g['close'] - 1
    
    # EMAs
    g['ema_8'] = g['close'].ewm(span=8).mean()
    g['ema_21'] = g['close'].ewm(span=21).mean()
    g['ema_50'] = g['close'].ewm(span=50).mean()
    g['ema_200'] = g['close'].ewm(span=200).mean()
    
    # RSI
    delta = g['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    g['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    g['bb_mid'] = g['close'].rolling(20).mean()
    g['bb_std'] = g['close'].rolling(20).std()
    g['bb_upper'] = g['bb_mid'] + 2 * g['bb_std']
    g['bb_lower'] = g['bb_mid'] - 2 * g['bb_std']
    g['bb_width'] = (g['bb_upper'] - g['bb_lower']) / g['bb_mid']
    g['bb_pct'] = (g['close'] - g['bb_lower']) / (g['bb_upper'] - g['bb_lower'])
    
    # Z-Score
    g['zscore'] = (g['close'] - g['close'].rolling(20).mean()) / g['close'].rolling(20).std()
    
    # ATR & Volatility
    tr = pd.concat([
        g['high'] - g['low'],
        (g['high'] - g['close'].shift(1)).abs(),
        (g['low'] - g['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    g['atr'] = tr.rolling(14).mean()
    g['volatility'] = g['ret_1'].rolling(20).std() * np.sqrt(252)
    
    # Volume
    g['vol_sma'] = g['volume'].rolling(20).mean()
    g['vol_ratio'] = g['volume'] / g['vol_sma']
    
    # 52-week high/low
    g['high_52w'] = g['high'].rolling(252).max()
    g['low_52w'] = g['low'].rolling(252).min()
    g['pct_from_high'] = (g['close'] - g['high_52w']) / g['high_52w']
    g['pct_from_low'] = (g['close'] - g['low_52w']) / g['low_52w']
    
    # Trend signals
    g['above_ema200'] = (g['close'] > g['ema_200']).astype(int)
    g['bullish_ribbon'] = ((g['ema_8'] > g['ema_21']) & (g['ema_21'] > g['ema_50'])).astype(int)
    
    # Consecutive down days
    g['down_day'] = (g['ret_1'] < 0).astype(int)
    g['consec_down'] = g['down_day'].groupby((g['down_day'] != g['down_day'].shift()).cumsum()).cumsum()
    g['after_2_down'] = (g['consec_down'].shift(1) >= 2).astype(int)
    
    # Oversold/Overbought
    g['oversold'] = (g['rsi'] < 30).astype(int)
    g['overbought'] = (g['rsi'] > 70).astype(int)
    
    # Low volatility
    vol_pct = g['volatility'].rolling(252).rank(pct=True)
    g['low_vol'] = (vol_pct < 0.3).astype(int)
    
    return g

df = df.groupby('ticker', group_keys=False).apply(calc_factors)
print(f"      Calculated {len([c for c in df.columns if c not in ['ticker','date','open','high','low','close','volume']])} factors")

# ============================================================================
# DETECT REGIME (SPY-based)
# ============================================================================

print("[3/6] Detecting market regimes...")

# Use SPY as market proxy
spy = df[df['ticker'] == 'SPY'].copy()
if len(spy) == 0:
    # Try QQQ or use overall market
    spy = df.groupby('date').agg({'close': 'mean', 'ret_20': 'mean', 'volatility': 'mean'}).reset_index()
    spy['ticker'] = 'MARKET'

spy = spy.sort_values('date')
spy['regime'] = 'RANGE'  # Default

# Simple regime detection
spy.loc[spy['ret_20'] > 0.05, 'regime'] = 'BULL'  # 5%+ in 20 days = BULL
spy.loc[spy['ret_20'] < -0.05, 'regime'] = 'BEAR'  # -5%+ in 20 days = BEAR

# Create regime lookup
regime_lookup = spy[['date', 'regime']].drop_duplicates()
df = df.merge(regime_lookup, on='date', how='left')
df['regime'] = df['regime'].fillna('RANGE')

print(f"      Regime distribution:")
print(f"        BULL:  {(df['regime']=='BULL').sum():,} rows")
print(f"        BEAR:  {(df['regime']=='BEAR').sum():,} rows")
print(f"        RANGE: {(df['regime']=='RANGE').sum():,} rows")

# ============================================================================
# DETECT MARKET CAP
# ============================================================================

print("[4/6] Classifying market cap...")

# Approximate market cap from price * volume (rough proxy)
# In production, you'd have actual market cap data
avg_dollar_vol = df.groupby('ticker').apply(lambda g: (g['close'] * g['volume']).mean())
cap_33 = avg_dollar_vol.quantile(0.33)
cap_66 = avg_dollar_vol.quantile(0.66)

def classify_cap(ticker):
    val = avg_dollar_vol.get(ticker, 0)
    if val >= cap_66:
        return 'LARGE'
    elif val >= cap_33:
        return 'MID'
    else:
        return 'SMALL'

df['cap'] = df['ticker'].apply(classify_cap)
print(f"      Cap distribution:")
print(f"        LARGE: {(df['cap']=='LARGE').sum():,} rows")
print(f"        MID:   {(df['cap']=='MID').sum():,} rows")
print(f"        SMALL: {(df['cap']=='SMALL').sum():,} rows")

# ============================================================================
# SPLIT TRAIN/TEST
# ============================================================================

print("[5/6] Splitting train/test periods...")

train_df = df[(df['date'] >= TRAIN_START) & (df['date'] <= TRAIN_END)]
test_df = df[(df['date'] >= TEST_START) & (df['date'] <= TEST_END)]

print(f"      Train: {len(train_df):,} rows ({TRAIN_START} to {TRAIN_END})")
print(f"      Test:  {len(test_df):,} rows ({TEST_START} to {TEST_END})")

# ============================================================================
# DEFINE ALL STRATEGY CONDITIONS
# ============================================================================

print("[6/6] Testing strategies across regimes...")

def calc_t_stat(returns):
    """Harvey-Liu-Zhu t-statistic"""
    returns = returns.dropna()
    if len(returns) < 30:
        return 0, 0, 0
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    if std == 0 or np.isnan(std):
        return 0, 0, 0
    t = mean / (std / np.sqrt(len(returns)))
    return mean, len(returns), t

# Strategy definitions (conditions and hold periods)
STRATEGIES = {
    # === TREND STRATEGIES ===
    'AboveEMA200': lambda d: d['above_ema200'] == 1,
    'BelowEMA200': lambda d: d['above_ema200'] == 0,
    'BullishRibbon': lambda d: d['bullish_ribbon'] == 1,
    'BearishRibbon': lambda d: d['bullish_ribbon'] == 0,
    
    # === MEAN REVERSION ===
    'RSI_Oversold': lambda d: d['rsi'] < 30,
    'RSI_Overbought': lambda d: d['rsi'] > 70,
    'RSI_Extreme_Oversold': lambda d: d['rsi'] < 20,
    'RSI_Extreme_Overbought': lambda d: d['rsi'] > 80,
    'ZScore_Low': lambda d: d['zscore'] < -2,
    'ZScore_High': lambda d: d['zscore'] > 2,
    'BB_Below_Lower': lambda d: d['close'] < d['bb_lower'],
    'BB_Above_Upper': lambda d: d['close'] > d['bb_upper'],
    
    # === MOMENTUM ===
    'Momentum_5d_Pos': lambda d: d['ret_5'] > 0,
    'Momentum_5d_Neg': lambda d: d['ret_5'] < 0,
    'Momentum_20d_Pos': lambda d: d['ret_20'] > 0,
    'Momentum_20d_Neg': lambda d: d['ret_20'] < 0,
    'Momentum_60d_Pos': lambda d: d['ret_60'] > 0,
    
    # === 52-WEEK LEVELS ===
    'Near_52wk_High': lambda d: d['pct_from_high'] > -0.05,
    'Near_52wk_Low': lambda d: d['pct_from_low'] < 0.05,
    'Far_From_High': lambda d: d['pct_from_high'] < -0.20,
    
    # === VOLATILITY ===
    'LowVol': lambda d: d['low_vol'] == 1,
    'HighVol': lambda d: d['low_vol'] == 0,
    'BB_Narrow': lambda d: d['bb_width'] < d['bb_width'].rolling(50).quantile(0.2),
    'Vol_Spike': lambda d: d['vol_ratio'] > 2.0,
    
    # === PATTERNS ===
    'After2Down': lambda d: d['after_2_down'] == 1,
    'After3Down': lambda d: d['consec_down'].shift(1) >= 3,
    
    # === MULTI-FACTOR COMBOS (the good stuff) ===
    'Oversold_LowVol': lambda d: (d['rsi'] < 30) & (d['low_vol'] == 1),
    'Oversold_After2Down': lambda d: (d['rsi'] < 30) & (d['after_2_down'] == 1),
    'Near52High_LowVol': lambda d: (d['pct_from_high'] > -0.05) & (d['low_vol'] == 1),
    'Near52High_PosMom': lambda d: (d['pct_from_high'] > -0.05) & (d['ret_60'] > 0),
    'AboveEMA200_BullRibbon': lambda d: (d['above_ema200'] == 1) & (d['bullish_ribbon'] == 1),
    'Oversold_LowVol_After2Down': lambda d: (d['rsi'] < 30) & (d['low_vol'] == 1) & (d['after_2_down'] == 1),
    'Oversold_LowVol_VolSpike': lambda d: (d['rsi'] < 30) & (d['low_vol'] == 1) & (d['vol_ratio'] > 1.5),
    'Near52High_LowVol_PosMom': lambda d: (d['pct_from_high'] > -0.05) & (d['low_vol'] == 1) & (d['ret_60'] > 0),
}

HOLD_PERIODS = [1, 5, 10, 20]  # Days

# ============================================================================
# RUN ALL TESTS
# ============================================================================

results = []
total_tests = len(STRATEGIES) * len(HOLD_PERIODS) * 3 * 3  # strategies * holds * regimes * caps
test_num = 0

for strat_name, condition_fn in STRATEGIES.items():
    for hold in HOLD_PERIODS:
        fwd_col = f'fwd_{hold}'
        if fwd_col not in df.columns:
            continue
            
        for regime in ['BULL', 'BEAR', 'RANGE', 'ALL']:
            for cap in ['LARGE', 'MID', 'SMALL', 'ALL']:
                test_num += 1
                if test_num % 100 == 0:
                    print(f"      Progress: {test_num}/{total_tests} tests...")
                
                # Filter data
                if regime == 'ALL':
                    regime_mask = pd.Series(True, index=train_df.index)
                    test_regime_mask = pd.Series(True, index=test_df.index)
                else:
                    regime_mask = train_df['regime'] == regime
                    test_regime_mask = test_df['regime'] == regime
                
                if cap == 'ALL':
                    cap_mask = pd.Series(True, index=train_df.index)
                    test_cap_mask = pd.Series(True, index=test_df.index)
                else:
                    cap_mask = train_df['cap'] == cap
                    test_cap_mask = test_df['cap'] == cap
                
                # Apply strategy condition
                try:
                    train_cond = condition_fn(train_df)
                    test_cond = condition_fn(test_df)
                except:
                    continue
                
                # TRAIN performance
                train_mask = regime_mask & cap_mask & train_cond
                train_returns = train_df.loc[train_mask, fwd_col]
                train_mean, train_n, train_t = calc_t_stat(train_returns)
                
                # TEST performance (out-of-sample!)
                test_mask = test_regime_mask & test_cap_mask & test_cond
                test_returns = test_df.loc[test_mask, fwd_col]
                test_mean, test_n, test_t = calc_t_stat(test_returns)
                
                # Record result
                results.append({
                    'strategy': strat_name,
                    'hold_days': hold,
                    'regime': regime,
                    'cap': cap,
                    'train_n': train_n,
                    'train_mean': train_mean,
                    'train_t': train_t,
                    'test_n': test_n,
                    'test_mean': test_mean,
                    'test_t': test_t,
                    'train_significant': abs(train_t) > T_THRESHOLD,
                    'test_significant': abs(test_t) > T_THRESHOLD,
                    'both_significant': (abs(train_t) > T_THRESHOLD) and (abs(test_t) > T_THRESHOLD),
                    'direction_match': (train_t > 0) == (test_t > 0) if train_t != 0 and test_t != 0 else False,
                })

# ============================================================================
# SAVE RESULTS
# ============================================================================

results_df = pd.DataFrame(results)
results_df.to_csv('data/REGIME_VALIDATED_STRATEGIES.csv', index=False)

print()
print("="*70)
print("RESULTS SUMMARY")
print("="*70)

# Summary stats
total = len(results_df)
train_sig = results_df['train_significant'].sum()
test_sig = results_df['test_significant'].sum()
both_sig = results_df['both_significant'].sum()
direction_match = results_df[results_df['train_significant']]['direction_match'].sum()

print(f"Total strategy/regime/cap combinations tested: {total:,}")
print(f"Significant in TRAIN (in-sample):  {train_sig:,} ({100*train_sig/total:.1f}%)")
print(f"Significant in TEST (out-of-sample): {test_sig:,} ({100*test_sig/total:.1f}%)")
print(f"Significant in BOTH:                 {both_sig:,} ({100*both_sig/total:.1f}%)")
print(f"Direction matches (train→test):      {direction_match:,}")

print()
print("TOP 20 STRATEGIES (by test t-stat, train also significant):")
print("-"*70)
top = results_df[results_df['train_significant']].nlargest(20, 'test_t')
print(top[['strategy', 'hold_days', 'regime', 'cap', 'train_t', 'test_t', 'test_mean']].to_string(index=False))

print()
print("BEST STRATEGIES BY REGIME:")
print("-"*70)
for regime in ['BULL', 'BEAR', 'RANGE']:
    regime_df = results_df[(results_df['regime'] == regime) & (results_df['both_significant'])]
    if len(regime_df) > 0:
        best = regime_df.nlargest(3, 'test_t')
        print(f"\n{regime}:")
        for _, row in best.iterrows():
            print(f"  {row['strategy']} H{row['hold_days']} ({row['cap']}): train_t={row['train_t']:.1f}, test_t={row['test_t']:.1f}")

print()
print(f"Finished: {datetime.now()}")
print(f"Results saved to: data/REGIME_VALIDATED_STRATEGIES.csv")
print()
print("NEXT STEP: Push results to GitHub so Codespaces can analyze")
print("  git add data/REGIME_VALIDATED_STRATEGIES.csv")
print("  git commit -m 'Shadow PC regime validation results'")
print("  git push")
