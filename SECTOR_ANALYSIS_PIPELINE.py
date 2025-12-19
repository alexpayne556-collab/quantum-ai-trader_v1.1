#!/usr/bin/env python3
"""
SECTOR-SPECIFIC PATTERN DISCOVERY
==================================

PURPOSE: Find which strategies work for which sectors
- Technology vs Energy vs Healthcare vs Financials
- Sector rotation patterns
- Risk-on vs risk-off behavior

OUTPUT: sector_specific_strategies.csv
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("SECTOR-SPECIFIC PATTERN DISCOVERY")
print("="*70)
print(f"Started: {datetime.now()}")

# ============================================================================
# SECTOR DEFINITIONS
# ============================================================================

SECTORS = {
    # Technology
    'TECH': ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AVGO', 'ORCL', 'AMD', 'CRM', 'ADBE',
             'CSCO', 'ACN', 'INTC', 'IBM', 'NOW', 'QCOM', 'TXN', 'AMAT', 'MU', 'INTU'],
    
    # Healthcare
    'HLTH': ['UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'PFE', 'TMO', 'ABT', 'DHR', 'BMY',
             'AMGN', 'GILD', 'CVS', 'CI', 'MDT', 'ISRG', 'REGN', 'VRTX', 'ZTS', 'SYK'],
    
    # Financial
    'FINC': ['BRK.B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'MS', 'GS', 'SPGI', 'BLK',
             'C', 'AXP', 'SCHW', 'CB', 'MMC', 'PGR', 'AON', 'USB', 'TFC', 'PNC'],
    
    # Energy
    'ENRG': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PSX', 'MPC', 'VLO', 'OXY', 'HES',
             'KMI', 'WMB', 'HAL', 'BKR', 'FANG', 'DVN', 'MRO', 'APA', 'CTRA', 'EQT'],
    
    # Consumer Discretionary
    'DISC': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX', 'BKNG',
             'CMG', 'MAR', 'GM', 'F', 'ABNB', 'YUM', 'DHI', 'LEN', 'ROST', 'EBAY'],
    
    # Consumer Staples
    'STPL': ['PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MO', 'CL', 'MDLZ', 'KMB',
             'GIS', 'KHC', 'TSN', 'SYY', 'HSY', 'K', 'CPB', 'CAG', 'CHD', 'CLX'],
    
    # Industrials
    'INDU': ['CAT', 'UNP', 'RTX', 'HON', 'UPS', 'BA', 'LMT', 'GE', 'DE', 'MMM',
             'EMR', 'ETN', 'ITW', 'CSX', 'NSC', 'FDX', 'WM', 'PH', 'PCAR', 'CMI'],
    
    # Materials
    'MATL': ['LIN', 'APD', 'ECL', 'SHW', 'FCX', 'NEM', 'DD', 'NUE', 'DOW', 'PPG',
             'VMC', 'MLM', 'IFF', 'ALB', 'CTVA', 'CE', 'MOS', 'FMC', 'EMN', 'CF'],
    
    # Utilities
    'UTIL': ['NEE', 'DUK', 'SO', 'D', 'EXC', 'AEP', 'SRE', 'XEL', 'ES', 'ED',
             'PEG', 'EIX', 'WEC', 'AWK', 'DTE', 'PPL', 'AEE', 'CMS', 'EVRG', 'CNP'],
    
    # Real Estate
    'REAL': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'WELL', 'SPG', 'DLR', 'O', 'VICI',
             'AVB', 'EQR', 'SBAC', 'VTR', 'ARE', 'MAA', 'INVH', 'ESS', 'KIM', 'UDR'],
    
    # Communication
    'COMM': ['GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA', 'T', 'VZ', 'TMUS', 'EA', 'TTWO',
             'CHTR', 'PARA', 'WBD', 'OMC', 'IPG', 'FOXA', 'NWSA', 'LUMN', 'MTCH', 'LYV']
}

# ============================================================================
# LOAD DATA
# ============================================================================

print("[1/5] Loading database...")
conn = sqlite3.connect('data/market_data.db')

# Load OHLCV for sector tickers
all_sector_tickers = []
for sector_tickers in SECTORS.values():
    all_sector_tickers.extend(sector_tickers)

placeholders = ','.join(['?'] * len(all_sector_tickers))
query = f"""
    SELECT ticker, date, open, high, low, close, volume
    FROM ohlcv
    WHERE ticker IN ({placeholders})
    AND date >= '2023-01-01'
    ORDER BY ticker, date
"""

df = pd.read_sql(query, conn, params=all_sector_tickers)
df['date'] = pd.to_datetime(df['date'])
print(f"      Loaded {len(df):,} rows for {df['ticker'].nunique():,} tickers")

# Add sector classification
ticker_to_sector = {}
for sector, tickers in SECTORS.items():
    for ticker in tickers:
        ticker_to_sector[ticker] = sector

df['sector'] = df['ticker'].map(ticker_to_sector)
df = df.dropna(subset=['sector'])

print("      Sector distribution:")
for sector in sorted(df['sector'].unique()):
    count = (df['sector'] == sector).sum()
    print(f"        {sector}: {count:,} rows")

# ============================================================================
# CALCULATE FACTORS
# ============================================================================

print("[2/5] Calculating factors...")

def calc_factors(g):
    g = g.sort_values('date').copy()
    
    # Returns
    g['ret_1'] = g['close'].pct_change(1)
    g['ret_5'] = g['close'].pct_change(5)
    g['ret_20'] = g['close'].pct_change(20)
    
    # Forward returns
    g['fwd_5'] = g['close'].shift(-5) / g['close'] - 1
    g['fwd_20'] = g['close'].shift(-20) / g['close'] - 1
    
    # EMAs
    g['ema_50'] = g['close'].ewm(span=50).mean()
    g['ema_200'] = g['close'].ewm(span=200).mean()
    
    # RSI
    delta = g['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    g['rsi'] = 100 - (100 / (1 + rs))
    
    # Volatility
    g['volatility'] = g['ret_1'].rolling(20).std() * np.sqrt(252)
    vol_pct = g['volatility'].rolling(252).rank(pct=True)
    g['low_vol'] = (vol_pct < 0.3).astype(int)
    
    # 52-week
    g['high_52w'] = g['high'].rolling(252).max()
    g['pct_from_high'] = (g['close'] - g['high_52w']) / g['high_52w']
    
    # Signals
    g['above_ema200'] = (g['close'] > g['ema_200']).astype(int)
    g['oversold'] = (g['rsi'] < 30).astype(int)
    
    return g

df = df.groupby('ticker', group_keys=False).apply(calc_factors)
print(f"      Calculated factors")

# ============================================================================
# DEFINE STRATEGIES
# ============================================================================

print("[3/5] Testing sector-specific strategies...")

def calc_t_stat(returns):
    returns = returns.dropna()
    if len(returns) < 30:
        return 0, 0, 0
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    if std == 0 or np.isnan(std):
        return 0, 0, 0
    t = mean / (std / np.sqrt(len(returns)))
    return mean, len(returns), t

# Test these strategies per sector
STRATEGIES = {
    'AboveEMA200': lambda d: d['above_ema200'] == 1,
    'RSI_Oversold': lambda d: d['rsi'] < 30,
    'Near52High': lambda d: d['pct_from_high'] > -0.05,
    'LowVol': lambda d: d['low_vol'] == 1,
    'Momentum_Pos': lambda d: d['ret_20'] > 0,
    'Oversold_LowVol': lambda d: (d['rsi'] < 30) & (d['low_vol'] == 1),
    'Near52High_LowVol': lambda d: (d['pct_from_high'] > -0.05) & (d['low_vol'] == 1),
}

results = []
for sector in sorted(df['sector'].unique()):
    sector_df = df[df['sector'] == sector]
    
    for strat_name, condition_fn in STRATEGIES.items():
        for hold_col in ['fwd_5', 'fwd_20']:
            try:
                condition = condition_fn(sector_df)
                returns = sector_df.loc[condition, hold_col]
                mean, n, t = calc_t_stat(returns)
                
                results.append({
                    'sector': sector,
                    'strategy': strat_name,
                    'hold': hold_col,
                    'n': n,
                    'mean': mean,
                    't_stat': t,
                    'significant': abs(t) > 3.0
                })
            except:
                continue

results_df = pd.DataFrame(results)

# ============================================================================
# ANALYZE SECTOR DIFFERENCES
# ============================================================================

print("[4/5] Analyzing sector-specific patterns...")

# Find strategies that work well in specific sectors
sector_specialists = []
for strat in results_df['strategy'].unique():
    strat_df = results_df[results_df['strategy'] == strat]
    
    # Which sector has highest t-stat?
    best_sector = strat_df.nlargest(1, 't_stat').iloc[0]
    worst_sector = strat_df.nsmallest(1, 't_stat').iloc[0]
    
    spread = best_sector['t_stat'] - worst_sector['t_stat']
    
    if spread > 5:  # Big difference across sectors
        sector_specialists.append({
            'strategy': strat,
            'best_sector': best_sector['sector'],
            'best_t': best_sector['t_stat'],
            'worst_sector': worst_sector['sector'],
            'worst_t': worst_sector['t_stat'],
            'spread': spread
        })

specialist_df = pd.DataFrame(sector_specialists)

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("[5/5] Saving results...")

results_df.to_csv('data/SECTOR_SPECIFIC_STRATEGIES.csv', index=False)
if len(specialist_df) > 0:
    specialist_df.to_csv('data/SECTOR_SPECIALISTS.csv', index=False)

# ============================================================================
# SUMMARY
# ============================================================================

print()
print("="*70)
print("SECTOR ANALYSIS SUMMARY")
print("="*70)

print(f"\nTotal tests: {len(results_df):,}")
print(f"Significant: {results_df['significant'].sum():,} ({100*results_df['significant'].sum()/len(results_df):.1f}%)")

print("\n" + "="*70)
print("BEST STRATEGY BY SECTOR:")
print("="*70)
for sector in sorted(results_df['sector'].unique()):
    sector_results = results_df[results_df['sector'] == sector]
    if len(sector_results) > 0:
        best = sector_results.nlargest(1, 't_stat').iloc[0]
        print(f"\n{sector}:")
        print(f"  Best: {best['strategy']} ({best['hold']})")
        print(f"  t-stat: {best['t_stat']:.1f}")
        print(f"  Avg return: {best['mean']*100:.2f}%")

if len(specialist_df) > 0:
    print("\n" + "="*70)
    print("SECTOR SPECIALISTS (strategies with big differences):")
    print("="*70)
    for _, row in specialist_df.iterrows():
        print(f"\n{row['strategy']}:")
        print(f"  Best in {row['best_sector']}: t={row['best_t']:.1f}")
        print(f"  Worst in {row['worst_sector']}: t={row['worst_t']:.1f}")
        print(f"  Spread: {row['spread']:.1f}")

print()
print(f"Finished: {datetime.now()}")
print("\nResults saved to:")
print("  - data/SECTOR_SPECIFIC_STRATEGIES.csv")
print("  - data/SECTOR_SPECIALISTS.csv")
