#!/usr/bin/env python3
"""
CORRECTED VALIDATION FRAMEWORK
==============================
Fixes the fatal flaws identified in DeepSeek review:

1. LOOK-AHEAD BIAS FIX: Regime uses LAGGED data (t-1)
2. MULTIPLE TESTING: Benjamini-Hochberg FDR correction
3. WALK-FORWARD: Rolling windows, not single split
4. TRANSACTION COSTS: 0.1% one-way (0.2% round-trip)
5. MINIMUM SAMPLE SIZE: n >= 100 (not 30)
6. WINSORIZATION: Cap extreme returns at ±20%

NOTE: Survivorship bias requires historical constituents data
      which we don't have. This is acknowledged as a limitation.

Author: Research Team
Date: Dec 2025
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("CORRECTED VALIDATION FRAMEWORK")
print("Fixing: Look-ahead, Multiple Testing, Single Split")
print("="*70)
print(f"Started: {datetime.now()}")
print()

# ============================================================================
# CONFIGURATION
# ============================================================================

DB_PATH = 'data/market_data.db'

# Walk-Forward Parameters (Rolling Windows)
TRAIN_MONTHS = 12       # 12 months training
TEST_MONTHS = 3         # 3 months testing
STEP_MONTHS = 3         # Roll forward 3 months

# Statistical Thresholds (MORE CONSERVATIVE)
MIN_SAMPLE_SIZE = 100   # Was 30, now 100
T_THRESHOLD = 3.5       # Was 3.0, now 3.5 (per DeepSeek)
FDR_ALPHA = 0.05        # False Discovery Rate threshold

# Transaction Costs
COST_ONE_WAY = 0.001    # 0.1% per trade (10 bps)
COST_ROUND_TRIP = 0.002 # 0.2% round trip

# Return Winsorization (cap extreme outliers)
RETURN_CAP = 0.20       # Cap at ±20% to prevent penny stock distortion

# ============================================================================
# LOAD DATA
# ============================================================================

print("[1/8] Loading database...")
conn = sqlite3.connect(DB_PATH)

df = pd.read_sql("""
    SELECT ticker, date, open, high, low, close, volume
    FROM ohlcv
    WHERE date >= '2022-01-01'  -- Extra year for lookback
    ORDER BY ticker, date
""", conn)
df['date'] = pd.to_datetime(df['date'], format='mixed')
print(f"      Loaded {len(df):,} rows, {df['ticker'].nunique():,} tickers")

# ============================================================================
# CALCULATE FACTORS (No Look-Ahead!)
# ============================================================================

print("[2/8] Calculating factors (no look-ahead)...")

def calc_factors_safe(g):
    """Calculate factors using ONLY past data - no future leakage"""
    g = g.sort_values('date').copy()
    
    # Returns (backward-looking only)
    g['ret_1'] = g['close'].pct_change(1)
    g['ret_5'] = g['close'].pct_change(5)
    g['ret_20'] = g['close'].pct_change(20)
    g['ret_60'] = g['close'].pct_change(60)
    
    # Forward returns (what we're predicting) - WINSORIZED
    for days in [1, 5, 10, 20]:
        fwd = g['close'].shift(-days) / g['close'] - 1
        # Winsorize to prevent outlier distortion
        g[f'fwd_{days}'] = fwd.clip(-RETURN_CAP, RETURN_CAP)
    
    # EMAs (backward-looking)
    g['ema_8'] = g['close'].ewm(span=8).mean()
    g['ema_21'] = g['close'].ewm(span=21).mean()
    g['ema_50'] = g['close'].ewm(span=50).mean()
    g['ema_200'] = g['close'].ewm(span=200).mean()
    
    # RSI (backward-looking)
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
    
    # Z-Score
    g['zscore'] = (g['close'] - g['bb_mid']) / g['bb_std']
    
    # Volatility
    g['volatility'] = g['ret_1'].rolling(20).std() * np.sqrt(252)
    
    # Volume ratio
    g['vol_sma'] = g['volume'].rolling(20).mean()
    g['vol_ratio'] = g['volume'] / g['vol_sma']
    
    # 52-week high/low
    g['high_52w'] = g['high'].rolling(252).max()
    g['low_52w'] = g['low'].rolling(252).min()
    g['pct_from_high'] = (g['close'] - g['high_52w']) / g['high_52w']
    
    # Trend signals
    g['above_ema200'] = (g['close'] > g['ema_200']).astype(int)
    g['bullish_ribbon'] = ((g['ema_8'] > g['ema_21']) & (g['ema_21'] > g['ema_50'])).astype(int)
    
    # Consecutive down days
    g['down_day'] = (g['ret_1'] < 0).astype(int)
    g['consec_down'] = g['down_day'].groupby((g['down_day'] != g['down_day'].shift()).cumsum()).cumsum()
    g['after_2_down'] = (g['consec_down'].shift(1) >= 2).astype(int)
    
    # Low volatility percentile
    vol_pct = g['volatility'].rolling(252, min_periods=60).rank(pct=True)
    g['low_vol'] = (vol_pct < 0.3).astype(int)
    
    return g

df = df.groupby('ticker', group_keys=False).apply(calc_factors_safe)
print(f"      Calculated factors with winsorization at ±{RETURN_CAP*100:.0f}%")

# ============================================================================
# FIX #1: LAGGED REGIME DETECTION
# ============================================================================

print("[3/8] Detecting regimes with LAGGED data (t-1)...")

# Get SPY data
spy = df[df['ticker'] == 'SPY'][['date', 'ret_20']].copy()
spy = spy.sort_values('date').drop_duplicates('date')

# CRITICAL FIX: Use YESTERDAY's 20-day return for today's regime
# This is the return we KNOW at market open
spy['regime_signal'] = spy['ret_20'].shift(1)  # <-- THE FIX

spy['regime'] = 'RANGE'
spy.loc[spy['regime_signal'] > 0.05, 'regime'] = 'BULL'
spy.loc[spy['regime_signal'] < -0.05, 'regime'] = 'BEAR'

# Merge regime to main dataframe
regime_lookup = spy[['date', 'regime']].copy()
df = df.merge(regime_lookup, on='date', how='left')
df['regime'] = df['regime'].fillna('RANGE')

print(f"      Regime distribution (LAGGED - no look-ahead):")
regime_counts = df.groupby('regime')['ticker'].count()
for r in ['BULL', 'BEAR', 'RANGE']:
    if r in regime_counts:
        print(f"        {r}: {regime_counts[r]:,} rows")

# ============================================================================
# MARKET CAP CLASSIFICATION
# ============================================================================

print("[4/8] Classifying market cap...")

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

# ============================================================================
# DEFINE STRATEGIES (Reduced Set - Only Distinct Concepts)
# ============================================================================

print("[5/8] Defining strategy conditions...")

# Fewer, more distinct strategies (avoid correlated tests)
STRATEGIES = {
    # Trend
    'AboveEMA200': lambda d: d['above_ema200'] == 1,
    'BullishRibbon': lambda d: d['bullish_ribbon'] == 1,
    
    # Mean Reversion
    'RSI_Oversold': lambda d: d['rsi'] < 30,
    'ZScore_Low': lambda d: d['zscore'] < -2,
    
    # Momentum
    'Momentum_20d_Pos': lambda d: d['ret_20'] > 0,
    'Momentum_60d_Pos': lambda d: d['ret_60'] > 0,
    
    # 52-Week
    'Near_52wk_High': lambda d: d['pct_from_high'] > -0.05,
    
    # Volatility
    'LowVol': lambda d: d['low_vol'] == 1,
    'Vol_Spike': lambda d: d['vol_ratio'] > 2.0,
    
    # Multi-Factor (only the distinct ones)
    'Oversold_After2Down': lambda d: (d['rsi'] < 30) & (d['after_2_down'] == 1),
    'Near52High_LowVol': lambda d: (d['pct_from_high'] > -0.05) & (d['low_vol'] == 1),
}

HOLD_PERIODS = [5, 10, 20]  # Skip 1-day (too noisy, too expensive)

print(f"      {len(STRATEGIES)} strategies × {len(HOLD_PERIODS)} holds = {len(STRATEGIES)*len(HOLD_PERIODS)} base tests")

# ============================================================================
# FIX #2: WALK-FORWARD VALIDATION (Rolling Windows)
# ============================================================================

print("[6/8] Running walk-forward validation...")

def calc_t_stat(returns):
    """Calculate t-statistic with stricter requirements"""
    returns = returns.dropna()
    if len(returns) < MIN_SAMPLE_SIZE:
        return 0, 0, 0
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    if std == 0 or np.isnan(std):
        return 0, 0, 0
    t = mean / (std / np.sqrt(len(returns)))
    return mean, len(returns), t

def apply_transaction_cost(mean_return, hold_days):
    """Subtract transaction costs from mean return"""
    # Assume one round-trip per trade
    # Annualized turnover depends on hold period
    trades_per_year = 252 / hold_days
    annual_cost = COST_ROUND_TRIP * trades_per_year
    # Convert to per-trade cost for the hold period
    cost_per_trade = COST_ROUND_TRIP
    return mean_return - cost_per_trade

# Generate walk-forward windows
df['month'] = df['date'].dt.to_period('M')
all_months = sorted(df['month'].unique())

# Start from month where we have enough history
start_idx = TRAIN_MONTHS + 12  # Extra 12 months for indicator lookback
windows = []

idx = start_idx
while idx + TEST_MONTHS <= len(all_months):
    train_start = all_months[idx - TRAIN_MONTHS]
    train_end = all_months[idx - 1]
    test_start = all_months[idx]
    test_end = all_months[min(idx + TEST_MONTHS - 1, len(all_months) - 1)]
    
    windows.append({
        'train_start': train_start,
        'train_end': train_end,
        'test_start': test_start,
        'test_end': test_end
    })
    idx += STEP_MONTHS

print(f"      Generated {len(windows)} walk-forward windows")

# Run validation across all windows
all_results = []

for win_idx, window in enumerate(windows):
    train_mask = (df['month'] >= window['train_start']) & (df['month'] <= window['train_end'])
    test_mask = (df['month'] >= window['test_start']) & (df['month'] <= window['test_end'])
    
    train_df = df[train_mask]
    test_df = df[test_mask]
    
    if len(train_df) < 1000 or len(test_df) < 100:
        continue
    
    for strat_name, condition_fn in STRATEGIES.items():
        for hold in HOLD_PERIODS:
            fwd_col = f'fwd_{hold}'
            
            for regime in ['BULL', 'BEAR', 'RANGE', 'ALL']:
                # Filter by regime
                if regime == 'ALL':
                    train_regime = train_df
                    test_regime = test_df
                else:
                    train_regime = train_df[train_df['regime'] == regime]
                    test_regime = test_df[test_df['regime'] == regime]
                
                if len(train_regime) < MIN_SAMPLE_SIZE or len(test_regime) < 30:
                    continue
                
                try:
                    train_cond = condition_fn(train_regime)
                    test_cond = condition_fn(test_regime)
                except:
                    continue
                
                train_returns = train_regime.loc[train_cond, fwd_col]
                test_returns = test_regime.loc[test_cond, fwd_col]
                
                train_mean, train_n, train_t = calc_t_stat(train_returns)
                test_mean, test_n, test_t = calc_t_stat(test_returns)
                
                # Apply transaction costs
                train_mean_net = apply_transaction_cost(train_mean, hold)
                test_mean_net = apply_transaction_cost(test_mean, hold)
                
                all_results.append({
                    'window': win_idx,
                    'strategy': strat_name,
                    'hold_days': hold,
                    'regime': regime,
                    'train_n': train_n,
                    'train_mean_gross': train_mean,
                    'train_mean_net': train_mean_net,
                    'train_t': train_t,
                    'test_n': test_n,
                    'test_mean_gross': test_mean,
                    'test_mean_net': test_mean_net,
                    'test_t': test_t,
                })

results_df = pd.DataFrame(all_results)
print(f"      Collected {len(results_df):,} window-strategy results")

# ============================================================================
# FIX #3: MULTIPLE TESTING CORRECTION (Benjamini-Hochberg)
# ============================================================================

print("[7/8] Applying Benjamini-Hochberg FDR correction...")

def benjamini_hochberg(p_values, alpha=FDR_ALPHA):
    """
    Apply Benjamini-Hochberg procedure to control False Discovery Rate.
    Returns boolean array of which tests are significant.
    """
    n = len(p_values)
    if n == 0:
        return np.array([])
    
    # Sort p-values and track original indices
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    
    # Calculate BH threshold for each rank
    ranks = np.arange(1, n + 1)
    thresholds = (ranks / n) * alpha
    
    # Find largest k where p[k] <= threshold[k]
    below_threshold = sorted_p <= thresholds
    
    if not below_threshold.any():
        return np.zeros(n, dtype=bool)
    
    # All tests up to and including the largest k are significant
    max_k = np.max(np.where(below_threshold)[0])
    significant = np.zeros(n, dtype=bool)
    significant[sorted_idx[:max_k + 1]] = True
    
    return significant

# Aggregate results by strategy (average across windows)
agg_results = results_df.groupby(['strategy', 'hold_days', 'regime']).agg({
    'train_n': 'mean',
    'train_mean_net': 'mean',
    'train_t': 'mean',
    'test_n': 'mean',
    'test_mean_net': 'mean',
    'test_t': 'mean',
    'window': 'count'  # Number of windows
}).reset_index()
agg_results.rename(columns={'window': 'n_windows'}, inplace=True)

# Filter: Must appear in at least 3 windows
agg_results = agg_results[agg_results['n_windows'] >= 3]

# Convert t-stats to p-values (two-tailed)
# Using approximation for large samples
agg_results['train_p'] = 2 * (1 - stats.norm.cdf(np.abs(agg_results['train_t'])))
agg_results['test_p'] = 2 * (1 - stats.norm.cdf(np.abs(agg_results['test_t'])))

# Apply BH correction separately to train and test
train_significant = benjamini_hochberg(agg_results['train_p'].values)
test_significant = benjamini_hochberg(agg_results['test_p'].values)

agg_results['train_significant_bh'] = train_significant
agg_results['test_significant_bh'] = test_significant
agg_results['both_significant_bh'] = train_significant & test_significant

# Also check direction consistency
agg_results['same_direction'] = (agg_results['train_t'] > 0) == (agg_results['test_t'] > 0)

# Final filter: significant in both AND same direction AND positive net return
agg_results['validated'] = (
    agg_results['both_significant_bh'] & 
    agg_results['same_direction'] & 
    (agg_results['test_mean_net'] > 0) &
    (agg_results['test_t'] > T_THRESHOLD)  # Also require t > 3.5
)

print(f"      After BH correction: {agg_results['validated'].sum()} validated strategies")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("[8/8] Saving results...")

# Save all results
agg_results.to_csv('data/CORRECTED_VALIDATION_RESULTS.csv', index=False)

# Save only validated
validated = agg_results[agg_results['validated']].sort_values('test_t', ascending=False)
validated.to_csv('data/VALIDATED_EDGES_FINAL.csv', index=False)

# ============================================================================
# SUMMARY REPORT
# ============================================================================

print()
print("="*70)
print("CORRECTED VALIDATION RESULTS")
print("="*70)

print(f"""
METHODOLOGY FIXES APPLIED:
✅ Look-ahead bias: Regime uses t-1 lagged data
✅ Multiple testing: Benjamini-Hochberg FDR correction (α={FDR_ALPHA})
✅ Walk-forward: {len(windows)} rolling windows ({TRAIN_MONTHS}mo train, {TEST_MONTHS}mo test)
✅ Transaction costs: {COST_ROUND_TRIP*100:.1f}% round-trip deducted
✅ Winsorization: Returns capped at ±{RETURN_CAP*100:.0f}%
✅ Min sample size: n ≥ {MIN_SAMPLE_SIZE}
✅ Higher threshold: t > {T_THRESHOLD}

⚠️  KNOWN LIMITATION: Survivorship bias not addressed (need historical constituents)
""")

print(f"\nTOTAL TESTS: {len(agg_results):,}")
print(f"SIGNIFICANT (BH-corrected): {agg_results['both_significant_bh'].sum():,}")
print(f"VALIDATED (all criteria): {agg_results['validated'].sum():,}")

if len(validated) > 0:
    print(f"\n{'='*70}")
    print("VALIDATED EDGES (Survive All Corrections)")
    print("="*70)
    print(validated[['strategy', 'hold_days', 'regime', 'train_t', 'test_t', 
                     'test_mean_net', 'n_windows']].to_string(index=False))
else:
    print(f"\n⚠️  NO STRATEGIES SURVIVED THE CORRECTED VALIDATION")
    print("    This is actually a good sign - it means the corrections are working.")
    print("    The previous 'edges' were likely artifacts of bias.")

print(f"\nResults saved to:")
print(f"  - data/CORRECTED_VALIDATION_RESULTS.csv (all)")
print(f"  - data/VALIDATED_EDGES_FINAL.csv (validated only)")
print(f"\nFinished: {datetime.now()}")
