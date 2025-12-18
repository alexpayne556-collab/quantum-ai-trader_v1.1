"""
VERBOSE DATA DOWNLOAD - Shows progress in real-time
This version flushes output immediately so you can see what's happening
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import time
import sys
import os

# Force output to flush immediately
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

def print_flush(msg):
    """Print and flush immediately"""
    print(msg, flush=True)

print_flush("=" * 70)
print_flush("VERBOSE DATA DOWNLOAD - Real-time progress")
print_flush("=" * 70)
print_flush("")

# Load universe
print_flush("📂 Loading universe...")
universe_df = pd.read_csv('data/complete_us_universe.csv')
tickers = universe_df['ticker'].unique().tolist()

print_flush(f"✅ Loaded {len(tickers)} tickers")
print_flush("")

# Exchange breakdown
exchange_counts = universe_df['exchange'].value_counts().to_dict()
print_flush("Exchanges:")
for exchange, count in exchange_counts.items():
    print_flush(f"  {exchange}: {count}")
print_flush("")

# Date range
end_date = datetime.now()
start_date = end_date - timedelta(days=730)  # 2 years

print_flush(f"📅 Date range: {start_date.date()} to {end_date.date()}")
print_flush(f"📊 Expected: ~{len(tickers)} × 504 days = ~{len(tickers) * 504:,} bars")
print_flush("")

# Database setup
db_path = 'data/market_data.db'
print_flush(f"💾 Database: {db_path}")

# Check if database exists
if os.path.exists(db_path):
    db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
    print_flush(f"   Existing database: {db_size_mb:.2f} MB")
    
    # Check how many tickers already downloaded
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT ticker) FROM ohlcv")
        existing_count = cursor.fetchone()[0]
        conn.close()
        print_flush(f"   Existing tickers: {existing_count}")
        
        if existing_count > 100:
            print_flush("")
            response = input("⚠️  Database has data. Continue/Resume? (y/n): ")
            if response.lower() != 'y':
                print_flush("❌ Download cancelled")
                sys.exit(0)
    except:
        pass

print_flush("")
print_flush("=" * 70)
print_flush("STARTING DOWNLOAD")
print_flush("=" * 70)
print_flush("")

# Import yfinance (after banner for cleaner output)
import yfinance as yf

# Connect to database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS ohlcv (
        ticker TEXT,
        date TEXT,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume INTEGER,
        adj_close REAL,
        PRIMARY KEY (ticker, date)
    )
''')
conn.commit()

# Download stats
total_tickers = len(tickers)
completed = 0
failed = 0
skipped = 0
start_time = time.time()

print_flush("Progress format: [ticker] status - details")
print_flush("")

# Process each ticker
for i, ticker in enumerate(tickers, 1):
    try:
        # Check if already exists
        cursor.execute("SELECT COUNT(*) FROM ohlcv WHERE ticker = ?", (ticker,))
        count = cursor.fetchone()[0]
        
        if count > 400:  # Already have data
            skipped += 1
            if skipped % 50 == 1:  # Show occasional skip message
                print_flush(f"[{i}/{total_tickers}] {ticker} - SKIP (already downloaded)")
            continue
        
        # Download
        print_flush(f"[{i}/{total_tickers}] {ticker} - Downloading...", end='')
        
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            show_errors=False
        )
        
        if df.empty or len(df) < 100:
            print_flush(f" ❌ FAIL (insufficient data: {len(df)} days)")
            failed += 1
            continue
        
        # Prepare data
        df = df.reset_index()
        df['ticker'] = ticker
        df.columns = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'ticker']
        
        # Insert
        df.to_sql('ohlcv', conn, if_exists='append', index=False)
        
        completed += 1
        print_flush(f" ✅ OK ({len(df)} days)")
        
        # Progress summary every 10 tickers
        if i % 10 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            remaining = (total_tickers - i) / rate if rate > 0 else 0
            remaining_hrs = remaining / 3600
            
            db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
            
            print_flush("")
            print_flush(f"📊 Progress: {i}/{total_tickers} ({i/total_tickers*100:.1f}%)")
            print_flush(f"   ✅ Success: {completed} | ❌ Failed: {failed} | ⏭️  Skipped: {skipped}")
            print_flush(f"   💾 Database: {db_size_mb:.2f} MB")
            print_flush(f"   ⏱️  Rate: {rate:.1f} tickers/sec")
            print_flush(f"   🎯 ETA: {remaining_hrs:.1f} hours")
            print_flush("")
        
        # Commit every 50 tickers
        if i % 50 == 0:
            conn.commit()
            print_flush(f"💾 Committed batch (database saved)")
            print_flush("")
        
        # Rate limit (be nice to yfinance)
        time.sleep(0.1)
        
    except KeyboardInterrupt:
        print_flush("")
        print_flush("⚠️  Download interrupted by user")
        break
    except Exception as e:
        print_flush(f" ❌ ERROR: {str(e)}")
        failed += 1

# Final commit
conn.commit()
conn.close()

# Final stats
total_time = time.time() - start_time
db_size_mb = os.path.getsize(db_path) / (1024 * 1024)

print_flush("")
print_flush("=" * 70)
print_flush("DOWNLOAD COMPLETE")
print_flush("=" * 70)
print_flush("")
print_flush(f"✅ Successfully downloaded: {completed}")
print_flush(f"❌ Failed: {failed}")
print_flush(f"⏭️  Skipped (already had): {skipped}")
print_flush(f"⏱️  Total time: {total_time/3600:.1f} hours")
print_flush(f"💾 Final database size: {db_size_mb:.2f} MB")
print_flush("")
print_flush("Ready for analysis!")
