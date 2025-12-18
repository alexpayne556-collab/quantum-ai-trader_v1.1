#!/usr/bin/env python3
"""
Quick status checker for the industrial data pipeline
Run this anytime to see progress
"""

import sqlite3
import pandas as pd
from datetime import datetime
import os

def check_status():
    print("=" * 70)
    print("INDUSTRIAL DATA PIPELINE STATUS")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Check if process is running
    pid_file = 'research_lab/pipeline.pid'
    if os.path.exists(pid_file):
        with open(pid_file) as f:
            pid = f.read().strip()
        
        # Check if PID is running
        try:
            os.kill(int(pid), 0)
            print(f"✓ Pipeline RUNNING (PID: {pid})")
        except:
            print(f"✗ Pipeline STOPPED (PID {pid} not found)")
    else:
        print("? Pipeline status unknown (no PID file)")
    
    # Database stats
    conn = sqlite3.connect('data/market_data.db')
    
    total_bars = pd.read_sql_query("SELECT COUNT(*) as count FROM daily_bars", conn).iloc[0]['count']
    total_tickers = pd.read_sql_query("SELECT COUNT(DISTINCT ticker) FROM daily_bars", conn).iloc[0]['COUNT(DISTINCT ticker)']
    
    # Data sources
    sources = pd.read_sql_query("""
        SELECT data_source, COUNT(DISTINCT ticker) as tickers
        FROM daily_bars
        WHERE data_source IS NOT NULL
        GROUP BY data_source
    """, conn)
    
    # Download status
    status_counts = pd.read_sql_query("""
        SELECT status, COUNT(*) as count
        FROM download_status
        GROUP BY status
    """, conn)
    
    # Quality check stats
    quality_stats = pd.read_sql_query("""
        SELECT 
            passed_check,
            COUNT(*) as count
        FROM data_quality
        GROUP BY passed_check
    """, conn)
    
    # Recent activity
    recent = pd.read_sql_query("""
        SELECT ticker, status, last_attempt
        FROM download_status
        ORDER BY last_attempt DESC
        LIMIT 10
    """, conn)
    
    conn.close()
    
    # Print stats
    print(f"\n{'Database Stats:':<30}")
    print(f"{'  Total bars:':<30} {total_bars:,}")
    print(f"{'  Unique tickers:':<30} {total_tickers:,}")
    print(f"{'  Avg bars/ticker:':<30} {total_bars/total_tickers if total_tickers > 0 else 0:.0f}")
    print(f"{'  Progress:':<30} {total_tickers:,} / 10,986 ({total_tickers/10986*100:.1f}%)")
    
    if not sources.empty:
        print(f"\n{'Data Sources:':<30}")
        for _, row in sources.iterrows():
            print(f"{'  ' + row['data_source'] + ':':<30} {row['tickers']:,} tickers")
    
    if not status_counts.empty:
        print(f"\n{'Download Status:':<30}")
        for _, row in status_counts.iterrows():
            print(f"{'  ' + row['status'] + ':':<30} {row['count']:,}")
    
    if not quality_stats.empty:
        print(f"\n{'Quality Checks:':<30}")
        for _, row in quality_stats.iterrows():
            status = "Passed" if row['passed_check'] == 1 else "Failed"
            print(f"{'  ' + status + ':':<30} {row['count']:,}")
    
    if not recent.empty and recent['last_attempt'].notna().any():
        print(f"\n{'Recent Activity:':<30}")
        print(recent.to_string(index=False))
    
    # Estimate completion
    if total_tickers > 196:  # We started with 175, had 196 after 45 seconds
        tickers_per_minute = (total_tickers - 175) / ((datetime.now() - datetime(2025, 12, 18, 2, 35, 26)).total_seconds() / 60)
        remaining = 10986 - total_tickers
        eta_minutes = remaining / tickers_per_minute if tickers_per_minute > 0 else 0
        eta_hours = eta_minutes / 60
        
        print(f"\n{'Estimates:':<30}")
        print(f"{'  Rate:':<30} {tickers_per_minute:.1f} tickers/minute")
        print(f"{'  Remaining:':<30} {remaining:,} tickers")
        print(f"{'  ETA:':<30} {eta_hours:.1f} hours ({eta_minutes:.0f} minutes)")
    
    print("=" * 70)
    print("\nMonitor live progress:")
    print("  tail -f research_lab/data_download.log")
    print("\nCheck detailed logs:")
    print("  tail -f research_lab/download_progress.log")
    print("=" * 70)

if __name__ == "__main__":
    check_status()
