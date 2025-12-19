#!/usr/bin/env python3
"""
SHADOW PC DATABASE DOWNLOAD
============================
Downloads market data to populate market_data.db

Run this: python SHADOW_PC_DATABASE_DOWNLOAD.py
"""

import yfinance as yf
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os

print("=" * 70)
print("SHADOW PC DATABASE DOWNLOAD")
print("=" * 70)

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Connect to database
DB_PATH = 'data/market_data.db'
print(f"Database: {DB_PATH}")
conn = sqlite3.connect(DB_PATH)

# Create table if it doesn't exist
conn.execute("""
    CREATE TABLE IF NOT EXISTS ohlcv (
        ticker TEXT,
        date TEXT,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume INTEGER,
        PRIMARY KEY (ticker, date)
    )
""")
conn.commit()

# Tickers to download (start with major ones)
tickers = [
    # Major indexes
    'SPY', 'QQQ', 'IWM', 'DIA',
    
    # Mega caps
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA',
    
    # Large caps
    'JPM', 'V', 'MA', 'WMT', 'JNJ', 'PG', 'UNH', 'HD', 'BAC',
    
    # Your current positions
    'MU', 'ASTS', 'LUNR'
]

# Download 2 years of data
start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
print(f"Downloading from {start_date} to today...")
print()

success = 0
failed = []

for i, ticker in enumerate(tickers, 1):
    try:
        print(f"[{i}/{len(tickers)}] Fetching {ticker}...", end=' ')
        
        # Download data
        df = yf.download(ticker, start=start_date, progress=False)
        
        if len(df) == 0:
            print("❌ No data")
            failed.append(ticker)
            continue
        
        # Prepare for database
        df = df.reset_index()
        df['ticker'] = ticker
        
        # Rename columns to match database schema
        df.columns = [col.lower() for col in df.columns]
        df['date'] = df['date'].astype(str)
        
        # Insert into database
        df.to_sql('ohlcv', conn, if_exists='append', index=False)
        
        print(f"✅ {len(df)} bars")
        success += 1
        
    except Exception as e:
        print(f"❌ Error: {e}")
        failed.append(ticker)

conn.close()

print()
print("=" * 70)
print("DOWNLOAD COMPLETE")
print("=" * 70)
print(f"✅ Success: {success}/{len(tickers)} tickers")
if failed:
    print(f"❌ Failed: {', '.join(failed)}")

# Verify database
conn = sqlite3.connect(DB_PATH)
cursor = conn.execute("SELECT COUNT(DISTINCT ticker) as n_tickers, COUNT(*) as n_bars FROM ohlcv")
n_tickers, n_bars = cursor.fetchone()
conn.close()

print()
print(f"📊 Database now contains:")
print(f"   Tickers: {n_tickers}")
print(f"   Total bars: {n_bars:,}")
print()
print("✅ Ready to run: python SHADOW_PC_REGIME_VALIDATION.py")
