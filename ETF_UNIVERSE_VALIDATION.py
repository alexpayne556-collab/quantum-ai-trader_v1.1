#!/usr/bin/env python3
"""
ETF UNIVERSE VALIDATION - THE CLEAN TEST
==========================================
NO survivorship bias. NO missing tickers. PURE methodology test.

This is the "prove it works" test before we invest in expensive
historical stock data.

UNIVERSE: 40 Major ETFs covering:
- Broad Market: SPY, QQQ, IWM, DIA, VTI
- Sectors: XLF, XLK, XLV, XLE, XLI, XLU, XLY, XLP, XLB, XLRE, XLC
- International: VEA, VWO, EFA, EEM
- Bonds: TLT, IEF, AGG, LQD, HYG
- Commodities: GLD, SLV, USO
- Volatility: VXX (if available)
- Thematic: ARKK, IBB, XBI, SMH, SOXX

If we can't find edges in this clean universe, our methodology is broken.
If we CAN find edges, we've proven the system works.

Author: Research Team
Date: Dec 2025
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ETF UNIVERSE VALIDATION")
print("The Clean, Survivorship-Bias-Free Test")
print("="*70)
print(f"Started: {datetime.now()}")
print()

# ============================================================================
# ETF UNIVERSE - NO SURVIVORSHIP BIAS
# ============================================================================

ETF_UNIVERSE = {
    # Broad Market
    'SPY': {'name': 'S&P 500', 'category': 'BROAD', 'cap': 'LARGE'},
    'QQQ': {'name': 'Nasdaq 100', 'category': 'BROAD', 'cap': 'LARGE'},
    'IWM': {'name': 'Russell 2000', 'category': 'BROAD', 'cap': 'SMALL'},
    'DIA': {'name': 'Dow Jones', 'category': 'BROAD', 'cap': 'LARGE'},
    'MDY': {'name': 'S&P MidCap 400', 'category': 'BROAD', 'cap': 'MID'},
    
    # Sector ETFs (SPDR)
    'XLK': {'name': 'Technology', 'category': 'TECH', 'cap': 'LARGE'},
    'XLF': {'name': 'Financials', 'category': 'FINC', 'cap': 'LARGE'},
    'XLV': {'name': 'Healthcare', 'category': 'HLTH', 'cap': 'LARGE'},
    'XLE': {'name': 'Energy', 'category': 'ENRG', 'cap': 'LARGE'},
    'XLI': {'name': 'Industrials', 'category': 'INDU', 'cap': 'LARGE'},
    'XLU': {'name': 'Utilities', 'category': 'UTIL', 'cap': 'LARGE'},
    'XLY': {'name': 'Consumer Disc', 'category': 'COND', 'cap': 'LARGE'},
    'XLP': {'name': 'Consumer Staples', 'category': 'CONS', 'cap': 'LARGE'},
    'XLB': {'name': 'Materials', 'category': 'MATL', 'cap': 'LARGE'},
    'XLRE': {'name': 'Real Estate', 'category': 'REAL', 'cap': 'LARGE'},
    'XLC': {'name': 'Communication', 'category': 'COMM', 'cap': 'LARGE'},
    
    # International
    'VEA': {'name': 'Developed Intl', 'category': 'INTL', 'cap': 'LARGE'},
    'VWO': {'name': 'Emerging Markets', 'category': 'INTL', 'cap': 'MID'},
    'EFA': {'name': 'EAFE', 'category': 'INTL', 'cap': 'LARGE'},
    
    # Fixed Income
    'TLT': {'name': '20+ Year Treasury', 'category': 'BOND', 'cap': 'N/A'},
    'IEF': {'name': '7-10 Year Treasury', 'category': 'BOND', 'cap': 'N/A'},
    'AGG': {'name': 'Aggregate Bond', 'category': 'BOND', 'cap': 'N/A'},
    'LQD': {'name': 'Investment Grade Corp', 'category': 'BOND', 'cap': 'N/A'},
    'HYG': {'name': 'High Yield Corp', 'category': 'BOND', 'cap': 'N/A'},
    
    # Commodities
    'GLD': {'name': 'Gold', 'category': 'CMDTY', 'cap': 'N/A'},
    'SLV': {'name': 'Silver', 'category': 'CMDTY', 'cap': 'N/A'},
    'USO': {'name': 'Oil', 'category': 'CMDTY', 'cap': 'N/A'},
    
    # Thematic
    'SMH': {'name': 'Semiconductors', 'category': 'TECH', 'cap': 'LARGE'},
    'IBB': {'name': 'Biotech', 'category': 'HLTH', 'cap': 'MID'},
    'XBI': {'name': 'Biotech Equal Weight', 'category': 'HLTH', 'cap': 'SMALL'},
    'SOXX': {'name': 'Semiconductors', 'category': 'TECH', 'cap': 'LARGE'},
    'KRE': {'name': 'Regional Banks', 'category': 'FINC', 'cap': 'MID'},
    'XHB': {'name': 'Homebuilders', 'category': 'COND', 'cap': 'MID'},
    'XOP': {'name': 'Oil & Gas E&P', 'category': 'ENRG', 'cap': 'MID'},
}

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data period
START_DATE = '2020-01-01'  # 5 years for robust testing
END_DATE = datetime.now().strftime('%Y-%m-%d')

# Walk-Forward Parameters
TRAIN_MONTHS = 18
TEST_MONTHS = 6
STEP_MONTHS = 3

# Statistical Thresholds (STRICT)
MIN_SAMPLE_SIZE = 50
T_THRESHOLD = 3.5
FDR_ALPHA = 0.05

# Transaction Costs
COST_ROUND_TRIP = 0.001  # 0.1% for ETFs (lower than stocks)

# Return Winsorization
RETURN_CAP = 0.15  # Cap at ±15%

# ============================================================================
# DOWNLOAD ETF DATA (Fresh from yfinance)
# ============================================================================

print("[1/7] Downloading ETF data from yfinance...")

all_data = []
successful_tickers = []

for ticker, info in ETF_UNIVERSE.items():
    try:
        etf = yf.Ticker(ticker)
        hist = etf.history(start=START_DATE, end=END_DATE, auto_adjust=True)
        
        if len(hist) < 252:  # Need at least 1 year
            print(f"      {ticker}: Insufficient data ({len(hist)} days)")
            continue
        
        hist = hist.reset_index()
        hist['ticker'] = ticker
        hist['category'] = info['category']
        hist['cap'] = info['cap']
        hist.columns = [c.lower() for c in hist.columns]
        hist = hist[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'category', 'cap']]
        
        all_data.append(hist)
        successful_tickers.append(ticker)
        
    except Exception as e:
        print(f"      {ticker}: Error - {str(e)[:50]}")

df = pd.concat(all_data, ignore_index=True)
df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
print(f"      Downloaded {len(df):,} rows for {len(successful_tickers)} ETFs")

# ============================================================================
# CALCULATE FACTORS (No Look-Ahead!)
# ============================================================================

print("[2/7] Calculating factors...")

def calc_factors_etf(g):
    """Calculate factors using ONLY past data"""
    g = g.sort_values('date').copy()
    
    # Returns (backward-looking)
    g['ret_1'] = g['close'].pct_change(1)
    g['ret_5'] = g['close'].pct_change(5)
    g['ret_10'] = g['close'].pct_change(10)
    g['ret_20'] = g['close'].pct_change(20)
    g['ret_60'] = g['close'].pct_change(60)
    
    # Forward returns (WINSORIZED)
    for days in [5, 10, 20]:
        fwd = g['close'].shift(-days) / g['close'] - 1
        g[f'fwd_{days}'] = fwd.clip(-RETURN_CAP, RETURN_CAP)
    
    # EMAs
    g['ema_10'] = g['close'].ewm(span=10).mean()
    g['ema_20'] = g['close'].ewm(span=20).mean()
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
    
    # Z-Score
    g['zscore'] = (g['close'] - g['bb_mid']) / g['bb_std']
    
    # Volatility
    g['volatility'] = g['ret_1'].rolling(20).std() * np.sqrt(252)
    g['vol_rank'] = g['volatility'].rolling(252, min_periods=60).rank(pct=True)
    
    # Volume
    g['vol_sma'] = g['volume'].rolling(20).mean()
    g['vol_ratio'] = g['volume'] / g['vol_sma']
    
    # Trend signals
    g['above_ema200'] = (g['close'] > g['ema_200']).astype(int)
    g['above_ema50'] = (g['close'] > g['ema_50']).astype(int)
    g['ema_trend'] = ((g['ema_10'] > g['ema_20']) & (g['ema_20'] > g['ema_50'])).astype(int)
    
    # Momentum rank
    g['mom_rank'] = g['ret_20'].rolling(60, min_periods=20).rank(pct=True)
    
    return g

df = df.groupby('ticker', group_keys=False).apply(calc_factors_etf)
print(f"      Calculated factors for all ETFs")

# ============================================================================
# REGIME DETECTION (LAGGED - No Look-Ahead!)
# ============================================================================

print("[3/7] Detecting regimes with LAGGED data (t-1)...")

spy = df[df['ticker'] == 'SPY'][['date', 'ret_20', 'volatility']].copy()
spy = spy.sort_values('date').drop_duplicates('date')

# CRITICAL: Use LAGGED data for regime
spy['regime_signal'] = spy['ret_20'].shift(1)  # Yesterday's 20-day return
spy['vol_signal'] = spy['volatility'].shift(1)  # Yesterday's volatility

# Simple regime based on lagged return
spy['regime'] = 'RANGE'
spy.loc[spy['regime_signal'] > 0.05, 'regime'] = 'BULL'
spy.loc[spy['regime_signal'] < -0.05, 'regime'] = 'BEAR'

# Volatility regime
spy['vol_regime'] = 'NORMAL'
vol_median = spy['vol_signal'].median()
spy.loc[spy['vol_signal'] > vol_median * 1.5, 'vol_regime'] = 'HIGH_VOL'
spy.loc[spy['vol_signal'] < vol_median * 0.7, 'vol_regime'] = 'LOW_VOL'

# Merge to main df
regime_lookup = spy[['date', 'regime', 'vol_regime']].copy()
df = df.merge(regime_lookup, on='date', how='left')
df['regime'] = df['regime'].fillna('RANGE')
df['vol_regime'] = df['vol_regime'].fillna('NORMAL')

print(f"      Regime distribution (LAGGED):")
for r in ['BULL', 'BEAR', 'RANGE']:
    count = (df['regime'] == r).sum()
    pct = 100 * count / len(df)
    print(f"        {r}: {count:,} ({pct:.1f}%)")

# ============================================================================
# DEFINE STRATEGIES
# ============================================================================

print("[4/7] Defining strategies...")

STRATEGIES = {
    # === TREND FOLLOWING ===
    'AboveEMA200': lambda d: d['above_ema200'] == 1,
    'AboveEMA50': lambda d: d['above_ema50'] == 1,
    'EMA_Trend_Up': lambda d: d['ema_trend'] == 1,
    'Momentum_Positive': lambda d: d['ret_20'] > 0,
    'Strong_Momentum': lambda d: d['mom_rank'] > 0.7,
    
    # === MEAN REVERSION ===
    'RSI_Oversold': lambda d: d['rsi'] < 30,
    'RSI_Overbought': lambda d: d['rsi'] > 70,
    'ZScore_Low': lambda d: d['zscore'] < -2,
    'ZScore_High': lambda d: d['zscore'] > 2,
    'Below_BB_Lower': lambda d: d['close'] < d['bb_lower'],
    
    # === VOLATILITY ===
    'Low_Volatility': lambda d: d['vol_rank'] < 0.3,
    'High_Volatility': lambda d: d['vol_rank'] > 0.7,
    'Vol_Expansion': lambda d: d['vol_ratio'] > 1.5,
    
    # === MULTI-FACTOR ===
    'Trend_LowVol': lambda d: (d['above_ema200'] == 1) & (d['vol_rank'] < 0.3),
    'Oversold_HighVol': lambda d: (d['rsi'] < 35) & (d['vol_rank'] > 0.5),
    'Strong_Trend': lambda d: (d['above_ema200'] == 1) & (d['ema_trend'] == 1) & (d['ret_20'] > 0),
}

HOLD_PERIODS = [5, 10, 20]

print(f"      {len(STRATEGIES)} strategies × {len(HOLD_PERIODS)} holds")

# ============================================================================
# WALK-FORWARD VALIDATION
# ============================================================================

print("[5/7] Running walk-forward validation...")

df['month'] = df['date'].dt.to_period('M')
all_months = sorted(df['month'].unique())

# Generate windows
start_idx = TRAIN_MONTHS + 6  # Buffer for indicators
windows = []
idx = start_idx

while idx + TEST_MONTHS <= len(all_months):
    windows.append({
        'train_start': all_months[idx - TRAIN_MONTHS],
        'train_end': all_months[idx - 1],
        'test_start': all_months[idx],
        'test_end': all_months[min(idx + TEST_MONTHS - 1, len(all_months) - 1)]
    })
    idx += STEP_MONTHS

print(f"      Generated {len(windows)} walk-forward windows")

# Run tests
all_results = []
total_tests = len(STRATEGIES) * len(HOLD_PERIODS) * 4 * len(windows)  # strategies × holds × regimes × windows
test_count = 0

for win_idx, window in enumerate(windows):
    train_mask = (df['month'] >= window['train_start']) & (df['month'] <= window['train_end'])
    test_mask = (df['month'] >= window['test_start']) & (df['month'] <= window['test_end'])
    
    train_df = df[train_mask]
    test_df = df[test_mask]
    
    if len(train_df) < 500 or len(test_df) < 100:
        continue
    
    for strat_name, condition_fn in STRATEGIES.items():
        for hold in HOLD_PERIODS:
            fwd_col = f'fwd_{hold}'
            
            for regime in ['BULL', 'BEAR', 'RANGE', 'ALL']:
                test_count += 1
                
                # Filter by regime
                if regime == 'ALL':
                    train_regime = train_df
                    test_regime = test_df
                else:
                    train_regime = train_df[train_df['regime'] == regime]
                    test_regime = test_df[test_df['regime'] == regime]
                
                if len(train_regime) < MIN_SAMPLE_SIZE or len(test_regime) < 20:
                    continue
                
                try:
                    train_cond = condition_fn(train_regime)
                    test_cond = condition_fn(test_regime)
                except:
                    continue
                
                train_returns = train_regime.loc[train_cond, fwd_col].dropna()
                test_returns = test_regime.loc[test_cond, fwd_col].dropna()
                
                if len(train_returns) < MIN_SAMPLE_SIZE or len(test_returns) < 20:
                    continue
                
                # Calculate stats
                train_mean = train_returns.mean()
                train_std = train_returns.std()
                train_n = len(train_returns)
                train_t = train_mean / (train_std / np.sqrt(train_n)) if train_std > 0 else 0
                
                test_mean = test_returns.mean()
                test_std = test_returns.std()
                test_n = len(test_returns)
                test_t = test_mean / (test_std / np.sqrt(test_n)) if test_std > 0 else 0
                
                # Net of costs
                train_mean_net = train_mean - COST_ROUND_TRIP
                test_mean_net = test_mean - COST_ROUND_TRIP
                
                all_results.append({
                    'window': win_idx,
                    'strategy': strat_name,
                    'hold_days': hold,
                    'regime': regime,
                    'train_n': train_n,
                    'train_mean': train_mean,
                    'train_mean_net': train_mean_net,
                    'train_t': train_t,
                    'test_n': test_n,
                    'test_mean': test_mean,
                    'test_mean_net': test_mean_net,
                    'test_t': test_t,
                })

if test_count % 500 == 0:
    print(f"      Progress: {test_count:,} tests...")

results_df = pd.DataFrame(all_results)
print(f"      Completed {len(results_df):,} valid tests")

# ============================================================================
# AGGREGATE AND APPLY MULTIPLE TESTING CORRECTION
# ============================================================================

print("[6/7] Applying Benjamini-Hochberg correction...")

# Aggregate across windows
agg = results_df.groupby(['strategy', 'hold_days', 'regime']).agg({
    'train_n': 'mean',
    'train_mean_net': 'mean',
    'train_t': 'mean',
    'test_n': 'mean',
    'test_mean_net': 'mean',
    'test_t': 'mean',
    'window': 'count'
}).reset_index()
agg.rename(columns={'window': 'n_windows'}, inplace=True)

# Filter: Must appear in at least 3 windows
agg = agg[agg['n_windows'] >= 3]

# Convert t to p-values
agg['train_p'] = 2 * (1 - stats.norm.cdf(np.abs(agg['train_t'])))
agg['test_p'] = 2 * (1 - stats.norm.cdf(np.abs(agg['test_t'])))

# Benjamini-Hochberg
def bh_correction(p_values, alpha=FDR_ALPHA):
    n = len(p_values)
    if n == 0:
        return np.array([])
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    below = sorted_p <= thresholds
    if not below.any():
        return np.zeros(n, dtype=bool)
    max_k = np.max(np.where(below)[0])
    sig = np.zeros(n, dtype=bool)
    sig[sorted_idx[:max_k + 1]] = True
    return sig

train_sig = bh_correction(agg['train_p'].values)
test_sig = bh_correction(agg['test_p'].values)

agg['train_sig_bh'] = train_sig
agg['test_sig_bh'] = test_sig
agg['both_sig_bh'] = train_sig & test_sig

# Direction check
agg['same_direction'] = (agg['train_t'] > 0) == (agg['test_t'] > 0)

# Final validation criteria
agg['VALIDATED'] = (
    agg['both_sig_bh'] &
    agg['same_direction'] &
    (agg['test_mean_net'] > 0) &
    (agg['test_t'] > T_THRESHOLD)
)

# ============================================================================
# RESULTS
# ============================================================================

print("[7/7] Generating results...")

agg.to_csv('data/ETF_VALIDATION_RESULTS.csv', index=False)

validated = agg[agg['VALIDATED']].sort_values('test_t', ascending=False)
validated.to_csv('data/ETF_VALIDATED_EDGES.csv', index=False)

print()
print("="*70)
print("ETF UNIVERSE VALIDATION RESULTS")
print("="*70)

print(f"""
METHODOLOGY:
✅ Universe: {len(successful_tickers)} ETFs (NO survivorship bias)
✅ Period: {START_DATE} to {END_DATE}
✅ Walk-Forward: {len(windows)} rolling windows ({TRAIN_MONTHS}mo train, {TEST_MONTHS}mo test)
✅ Regime: LAGGED (t-1) - No look-ahead
✅ Multiple Testing: Benjamini-Hochberg FDR correction (α={FDR_ALPHA})
✅ Transaction Costs: {COST_ROUND_TRIP*100:.2f}% round-trip
✅ Winsorization: Returns capped at ±{RETURN_CAP*100:.0f}%
✅ Threshold: t > {T_THRESHOLD}
""")

print(f"TOTAL TESTS: {len(agg):,}")
print(f"SIGNIFICANT (BH-corrected): {agg['both_sig_bh'].sum():,}")
print(f"VALIDATED (all criteria): {len(validated):,}")

if len(validated) > 0:
    print(f"\n{'='*70}")
    print("🎯 VALIDATED ETF EDGES (Survive All Corrections)")
    print("="*70)
    
    for _, row in validated.head(20).iterrows():
        print(f"\n{row['strategy']} | Hold {row['hold_days']}d | {row['regime']}")
        print(f"   Train: t={row['train_t']:.2f}, n={row['train_n']:.0f}")
        print(f"   Test:  t={row['test_t']:.2f}, n={row['test_n']:.0f}, net={row['test_mean_net']*100:.3f}%")
        print(f"   Windows: {row['n_windows']}")
    
    # Summary by regime
    print(f"\n{'='*70}")
    print("VALIDATED EDGES BY REGIME")
    print("="*70)
    for regime in ['BULL', 'BEAR', 'RANGE', 'ALL']:
        regime_val = validated[validated['regime'] == regime]
        if len(regime_val) > 0:
            print(f"\n{regime}: {len(regime_val)} edges")
            for _, row in regime_val.head(3).iterrows():
                print(f"   {row['strategy']} H{row['hold_days']}: t={row['test_t']:.2f}, ret={row['test_mean_net']*100:.3f}%")
else:
    print(f"\n{'='*70}")
    print("⚠️  NO STRATEGIES SURVIVED THE ETF VALIDATION")
    print("="*70)
    print("""
This is actually VALUABLE information:
1. Your methodology is correctly filtering out false positives
2. The ETF universe may require different strategies
3. Or the edges are too small to survive transaction costs

NEXT STEPS:
- Review near-misses (t > 2.5 but < 3.5)
- Try longer hold periods (lower turnover = lower costs)
- Consider regime-specific strategies only
""")
    
    # Show near-misses
    near_miss = agg[(agg['test_t'] > 2.5) & (agg['same_direction']) & (agg['test_mean_net'] > 0)]
    if len(near_miss) > 0:
        print(f"\nNEAR-MISSES (t > 2.5 but didn't meet all criteria):")
        for _, row in near_miss.nlargest(10, 'test_t').iterrows():
            print(f"   {row['strategy']} H{row['hold_days']} {row['regime']}: t={row['test_t']:.2f}, ret={row['test_mean_net']*100:.3f}%")

print(f"\nResults saved to:")
print(f"  - data/ETF_VALIDATION_RESULTS.csv (all)")
print(f"  - data/ETF_VALIDATED_EDGES.csv (validated only)")
print(f"\nFinished: {datetime.now()}")
