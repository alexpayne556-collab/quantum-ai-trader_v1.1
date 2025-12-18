"""
DATA QUALITY AUDIT - Find Every Anomaly
========================================
Like physicists checking instruments before measurements,
we check data quality before discovering "laws" (edges).

Checks:
1. Survivorship bias indicators
2. Look-ahead bias potential  
3. Corporate actions (splits, extreme moves)
4. Data gaps and coverage issues
5. Price sanity (negatives, zeros, outliers)
6. Transaction cost feasibility
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Database connection
DB_PATH = 'data/market_data.db'

def print_section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")

def check_survivorship_bias():
    """Identify tickers that disappeared (potential delisting/bankruptcy)"""
    print_section("1. SURVIVORSHIP BIAS CHECK")
    
    conn = sqlite3.connect(DB_PATH)
    
    # Find tickers by their last trading date
    query = """
    SELECT ticker, MAX(date) as last_date, COUNT(*) as total_bars
    FROM ohlcv
    GROUP BY ticker
    """
    df = pd.read_sql(query, conn)
    df['last_date'] = pd.to_datetime(df['last_date'])
    
    # Current date (or most recent date in database)
    max_date = df['last_date'].max()
    cutoff_date = max_date - timedelta(days=90)  # Stopped trading >90 days ago
    
    # Potential delistings
    dead_tickers = df[df['last_date'] < cutoff_date].copy()
    dead_tickers['days_since_last'] = (max_date - dead_tickers['last_date']).dt.days
    
    print(f"📊 Total tickers in database: {len(df)}")
    print(f"⚠️  Tickers that stopped trading >90 days ago: {len(dead_tickers)}")
    print(f"📈 Currently active tickers: {len(df) - len(dead_tickers)}")
    
    if len(dead_tickers) > 0:
        print(f"\n🚨 SURVIVORSHIP BIAS DETECTED!")
        print(f"   Missing {len(dead_tickers)} potentially delisted stocks")
        print(f"   This inflates backtest returns by excluding losers\n")
        
        # Show worst cases (earliest delistings)
        print("Sample of earliest delistings:")
        print(dead_tickers.nsmallest(10, 'last_date')[['ticker', 'last_date', 'days_since_last', 'total_bars']])
        
        # Save to file
        dead_tickers.to_csv('data/potentially_delisted.csv', index=False)
        print(f"\n💾 Saved full list to data/potentially_delisted.csv")
    
    conn.close()
    return dead_tickers

def check_corporate_actions():
    """Find stock splits, reverse splits, and extreme moves"""
    print_section("2. CORPORATE ACTIONS & EXTREME MOVES")
    
    conn = sqlite3.connect(DB_PATH)
    
    # Get data for all tickers
    query = """
    SELECT ticker, date, close, volume
    FROM ohlcv
    ORDER BY ticker, date
    """
    df = pd.read_sql(query, conn)
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate daily returns
    df['return'] = df.groupby('ticker')['close'].pct_change()
    
    # Identify extreme single-day moves (potential data errors or corporate actions)
    extreme_moves = df[abs(df['return']) > 0.5].copy()  # >50% single-day move
    extreme_moves = extreme_moves.dropna()
    
    print(f"🔍 Found {len(extreme_moves)} extreme single-day moves (>50%)")
    
    if len(extreme_moves) > 0:
        print("\n⚠️  Potential stock splits, reverse splits, or data errors:")
        print(extreme_moves.nlargest(20, 'return')[['ticker', 'date', 'close', 'return', 'volume']])
        
        # Flag tickers with multiple extreme moves (likely data issues)
        extreme_counts = extreme_moves['ticker'].value_counts()
        problematic = extreme_counts[extreme_counts > 2]
        
        if len(problematic) > 0:
            print(f"\n🚨 SUSPICIOUS: {len(problematic)} tickers with >2 extreme moves (likely data quality issues)")
            print(problematic.head(10))
            
        extreme_moves.to_csv('data/extreme_moves.csv', index=False)
        print(f"\n💾 Saved to data/extreme_moves.csv")
    
    conn.close()
    return extreme_moves

def check_data_gaps():
    """Identify missing trading days and irregular coverage"""
    print_section("3. DATA GAPS & COVERAGE ANOMALIES")
    
    conn = sqlite3.connect(DB_PATH)
    
    # Get coverage stats per ticker
    query = """
    SELECT ticker, 
           COUNT(*) as total_bars,
           MIN(date) as first_date,
           MAX(date) as last_date
    FROM ohlcv
    GROUP BY ticker
    """
    df = pd.read_sql(query, conn)
    df['first_date'] = pd.to_datetime(df['first_date'])
    df['last_date'] = pd.to_datetime(df['last_date'])
    df['date_span_days'] = (df['last_date'] - df['first_date']).dt.days
    
    # Expected bars: ~252 trading days per year
    df['expected_bars'] = df['date_span_days'] / 365 * 252
    df['coverage_pct'] = (df['total_bars'] / df['expected_bars'] * 100).clip(upper=100)
    
    print(f"📊 Coverage statistics:")
    print(f"   Mean coverage: {df['coverage_pct'].mean():.1f}%")
    print(f"   Median coverage: {df['coverage_pct'].median():.1f}%")
    print(f"   Min coverage: {df['coverage_pct'].min():.1f}%")
    
    # Identify poor coverage tickers
    poor_coverage = df[df['coverage_pct'] < 70].copy()
    print(f"\n⚠️  Tickers with <70% coverage: {len(poor_coverage)}")
    
    if len(poor_coverage) > 0:
        print(poor_coverage.nsmallest(10, 'coverage_pct')[['ticker', 'total_bars', 'coverage_pct', 'first_date', 'last_date']])
        poor_coverage.to_csv('data/poor_coverage.csv', index=False)
        print(f"\n💾 Saved to data/poor_coverage.csv")
    
    # Check for recent IPOs (< 6 months of data)
    recent_ipos = df[df['total_bars'] < 126]  # ~6 months = 126 trading days
    print(f"\n📅 Recent IPOs (<6 months data): {len(recent_ipos)}")
    if len(recent_ipos) > 0:
        print(recent_ipos.head(10)[['ticker', 'total_bars', 'first_date']])
        recent_ipos.to_csv('data/recent_ipos.csv', index=False)
        print(f"💾 Saved to data/recent_ipos.csv")
    
    conn.close()
    return df

def check_price_sanity():
    """Check for impossible prices (negative, zero, extreme)"""
    print_section("4. PRICE SANITY CHECKS")
    
    conn = sqlite3.connect(DB_PATH)
    
    # Check for problematic prices
    query = """
    SELECT ticker, date, open, high, low, close, volume
    FROM ohlcv
    WHERE close <= 0 OR high <= 0 OR low <= 0 OR open <= 0
       OR high < low
       OR close > high OR close < low
       OR open > high OR open < low
    """
    bad_prices = pd.read_sql(query, conn)
    
    print(f"🔍 Checking for impossible prices...")
    print(f"   Negative/zero prices: {len(bad_prices)}")
    
    if len(bad_prices) > 0:
        print("\n🚨 DATA ERRORS FOUND:")
        print(bad_prices.head(20))
        bad_prices.to_csv('data/price_errors.csv', index=False)
        print(f"\n💾 Saved to data/price_errors.csv")
    else:
        print("✅ No obvious price errors detected")
    
    # Check for zero volume (halted/suspended trading)
    query_volume = """
    SELECT ticker, COUNT(*) as zero_volume_days
    FROM ohlcv
    WHERE volume = 0
    GROUP BY ticker
    HAVING COUNT(*) > 5
    """
    zero_volume = pd.read_sql(query_volume, conn)
    
    print(f"\n📊 Tickers with >5 zero-volume days: {len(zero_volume)}")
    if len(zero_volume) > 0:
        print(zero_volume.nlargest(10, 'zero_volume_days'))
        zero_volume.to_csv('data/zero_volume.csv', index=False)
        print(f"💾 Saved to data/zero_volume.csv")
    
    conn.close()
    return bad_prices

def estimate_transaction_costs():
    """Estimate trading costs by liquidity tier"""
    print_section("5. TRANSACTION COST ESTIMATES")
    
    conn = sqlite3.connect(DB_PATH)
    
    # Get recent average price and volume for each ticker
    query = """
    SELECT ticker,
           AVG(close) as avg_price,
           AVG(volume) as avg_volume,
           AVG(close * volume) as avg_dollar_volume
    FROM ohlcv
    WHERE date >= (SELECT MAX(date) FROM ohlcv WHERE date < '2025-12-01')  -- Last month
    GROUP BY ticker
    """
    df = pd.read_sql(query, conn)
    
    # Classify by liquidity tier
    def get_cost_tier(row):
        price = row['avg_price']
        dollar_vol = row['avg_dollar_volume']
        
        # Penny stocks (<$5): high cost
        if price < 5:
            return 'penny', 0.03  # 3% round-trip
        # Low liquidity (<$1M daily volume): high cost
        elif dollar_vol < 1_000_000:
            return 'illiquid', 0.02  # 2% round-trip
        # Small cap ($1M-$10M daily volume)
        elif dollar_vol < 10_000_000:
            return 'small', 0.008  # 0.8% round-trip
        # Mid cap ($10M-$100M daily volume)
        elif dollar_vol < 100_000_000:
            return 'mid', 0.003  # 0.3% round-trip
        # Large cap (>$100M daily volume)
        else:
            return 'large', 0.001  # 0.1% round-trip
    
    df[['tier', 'round_trip_cost']] = df.apply(get_cost_tier, axis=1, result_type='expand')
    
    print("📊 Liquidity tier distribution:")
    print(df['tier'].value_counts().sort_index())
    
    print(f"\n💰 Transaction cost estimates (round-trip):")
    for tier in ['penny', 'illiquid', 'small', 'mid', 'large']:
        count = (df['tier'] == tier).sum()
        if count > 0:
            cost = df[df['tier'] == tier]['round_trip_cost'].iloc[0]
            print(f"   {tier.capitalize():10s}: {count:5d} tickers @ {cost*100:.1f}% cost")
    
    # High-cost tickers (>1% round trip)
    high_cost = df[df['round_trip_cost'] >= 0.01].copy()
    print(f"\n⚠️  High transaction cost tickers (>1%): {len(high_cost)}")
    print(f"   These will kill most edges in backtesting!")
    
    df.to_csv('data/transaction_costs.csv', index=False)
    print(f"\n💾 Saved to data/transaction_costs.csv")
    
    conn.close()
    return df

def check_look_ahead_bias():
    """Check for potential look-ahead bias in data"""
    print_section("6. LOOK-AHEAD BIAS CHECK")
    
    conn = sqlite3.connect(DB_PATH)
    
    # Check if we have future data (shouldn't happen with yfinance)
    query = """
    SELECT ticker, MAX(date) as latest_date
    FROM ohlcv
    GROUP BY ticker
    HAVING MAX(date) > date('now')
    """
    future_data = pd.read_sql(query, conn)
    
    if len(future_data) > 0:
        print(f"🚨 FUTURE DATA DETECTED: {len(future_data)} tickers have dates in the future!")
        print(future_data)
    else:
        print("✅ No future dates detected (good)")
    
    # Check for adj_close vs close discrepancies (adjustment happened retroactively)
    query = """
    SELECT ticker, date, close, adj_close,
           ABS(close - adj_close) / close as diff_pct
    FROM ohlcv
    WHERE ABS(close - adj_close) / close > 0.01
    LIMIT 100
    """
    adjustments = pd.read_sql(query, conn)
    
    print(f"\n📊 Price adjustments (close vs adj_close >1% diff): {len(adjustments)}")
    if len(adjustments) > 0:
        print("   This is normal for dividends/splits, but verify it's applied correctly:")
        print(adjustments.head(10))
    
    conn.close()
    return future_data

def main():
    """Run all data quality checks"""
    print("\n" + "="*80)
    print("  DATA QUALITY AUDIT - Finding All Anomalies")
    print("="*80)
    print(f"  Like physicists checking instruments before experiments,")
    print(f"  we verify data quality before discovering 'laws' (edges).")
    print("="*80)
    
    # Run all checks
    dead_tickers = check_survivorship_bias()
    extreme_moves = check_corporate_actions()
    coverage = check_data_gaps()
    price_errors = check_price_sanity()
    costs = estimate_transaction_costs()
    future_data = check_look_ahead_bias()
    
    # Summary
    print_section("AUDIT SUMMARY")
    
    issues = []
    if len(dead_tickers) > 0:
        issues.append(f"⚠️  Survivorship bias: {len(dead_tickers)} delisted tickers")
    if len(extreme_moves) > 0:
        issues.append(f"⚠️  {len(extreme_moves)} extreme moves (splits/errors)")
    if len(price_errors) > 0:
        issues.append(f"🚨 {len(price_errors)} price data errors")
    if len(future_data) > 0:
        issues.append(f"🚨 {len(future_data)} tickers with future dates")
    
    if len(issues) == 0:
        print("✅ No critical data quality issues found!")
    else:
        print("🔍 Issues identified:")
        for issue in issues:
            print(f"   {issue}")
    
    print(f"\n📁 All audit results saved to data/ folder")
    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. Review exported CSV files for each anomaly type")
    print(f"   2. Decide: exclude problematic tickers OR build corrections")
    print(f"   3. Re-run backtests with survivorship bias correction")
    print(f"   4. Model transaction costs in all strategy tests")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
