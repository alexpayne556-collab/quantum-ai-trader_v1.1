"""
CPU Speed Test - Check if GPU acceleration is needed
Tests: Correlation matrix, PCA, basic operations
"""

import sqlite3
import pandas as pd
import numpy as np
import time
from sklearn.decomposition import PCA

print("=" * 70)
print("CPU SPEED TEST - Benchmarking Operations")
print("=" * 70)
print()

# Load data
print("📂 Loading data from database...")
start = time.time()
conn = sqlite3.connect('data/market_data.db')

# Get all data into pandas
df = pd.read_sql("""
    SELECT ticker, date, close 
    FROM ohlcv 
    ORDER BY ticker, date
""", conn)
conn.close()

print(f"✅ Loaded {len(df):,} rows in {time.time()-start:.2f}s")
print(f"   Tickers: {df['ticker'].nunique()}")
print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
print()

# Pivot to wide format (tickers as columns)
print("🔄 Pivoting to wide format (ticker × date matrix)...")
start = time.time()
returns_wide = df.pivot(index='date', columns='ticker', values='close')
print(f"✅ Pivoted in {time.time()-start:.2f}s")
print(f"   Shape: {returns_wide.shape} (dates × tickers)")
print()

# Calculate returns
print("📈 Computing daily returns...")
start = time.time()
returns = returns_wide.pct_change().dropna()
print(f"✅ Returns computed in {time.time()-start:.2f}s")
print()

# Remove tickers with too many NaNs (keep only tickers with >80% data)
min_data = int(0.8 * len(returns))
valid_tickers = returns.columns[returns.count() >= min_data]
returns_clean = returns[valid_tickers].fillna(0)
print(f"📊 After cleaning: {returns_clean.shape[1]} tickers with >80% data")
print()

# TEST 1: Correlation Matrix
print("=" * 70)
print("TEST 1: CORRELATION MATRIX")
print("=" * 70)
print(f"Computing {returns_clean.shape[1]} × {returns_clean.shape[1]} correlation matrix...")
start = time.time()
corr_matrix = returns_clean.corr()
elapsed = time.time() - start
print(f"✅ Correlation matrix computed in {elapsed:.2f}s")
print(f"   Matrix size: {corr_matrix.shape}")
print(f"   Memory: {corr_matrix.memory_usage().sum() / 1024**2:.2f} MB")

if elapsed > 30:
    print("⚠️  SLOW - GPU acceleration recommended!")
elif elapsed > 10:
    print("⚡ Moderate - GPU would help but not critical")
else:
    print("🚀 FAST - CPU is fine for now!")
print()

# TEST 2: Rolling Correlation (subset)
print("=" * 70)
print("TEST 2: ROLLING CORRELATION (60-day window, subset)")
print("=" * 70)
subset_size = min(100, returns_clean.shape[1])  # Use 100 tickers max
print(f"Testing on {subset_size} tickers...")
subset = returns_clean.iloc[:, :subset_size]
start = time.time()
rolling_corr = subset.rolling(window=60).corr()
elapsed = time.time() - start
print(f"✅ Rolling correlation computed in {elapsed:.2f}s")

if elapsed > 60:
    print("⚠️  VERY SLOW - GPU strongly recommended!")
elif elapsed > 20:
    print("⚡ Moderate - GPU would help")
else:
    print("🚀 ACCEPTABLE - CPU manageable")
print()

# TEST 3: PCA
print("=" * 70)
print("TEST 3: PCA (Principal Component Analysis)")
print("=" * 70)
print(f"Computing PCA on {returns_clean.shape} matrix...")
start = time.time()
pca = PCA(n_components=10)
components = pca.fit_transform(returns_clean.fillna(0))
elapsed = time.time() - start
print(f"✅ PCA computed in {elapsed:.2f}s")
print(f"   Explained variance (first 3 components): {pca.explained_variance_ratio_[:3]}")

if elapsed > 20:
    print("⚠️  SLOW - GPU would help")
elif elapsed > 5:
    print("⚡ Moderate speed")
else:
    print("🚀 FAST - CPU is fine")
print()

# SUMMARY
print("=" * 70)
print("SUMMARY & RECOMMENDATION")
print("=" * 70)
print(f"Dataset: {returns_clean.shape[1]} tickers × {returns_clean.shape[0]} days")
print(f"Total data points: {returns_clean.size:,}")
print()

# Decision logic
total_time = time.time()
if elapsed > 30:
    print("🎯 RECOMMENDATION: Set up GPU acceleration")
    print("   - Correlation matrix took >30s")
    print("   - Expected GPU speedup: 50-100x")
    print("   - Full pipeline would take hours on CPU vs minutes on GPU")
elif elapsed > 10:
    print("🎯 RECOMMENDATION: CPU is workable, GPU optional")
    print("   - Current speed is acceptable for initial testing")
    print("   - Consider GPU for large parameter sweeps later")
else:
    print("🎯 RECOMMENDATION: CPU is fine for now!")
    print("   - Operations are fast enough")
    print("   - Only set up GPU if running 100+ iterations")

print()
print("✅ Speed test complete!")
print("=" * 70)
