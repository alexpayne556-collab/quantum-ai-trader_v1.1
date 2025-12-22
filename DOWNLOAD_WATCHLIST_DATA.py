"""
Download 2 years of data for all watchlist tickers
Run this first on Shadow PC
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime
from WATCHLIST_2026 import WATCHLIST

DATA_DIR = Path("data/watchlist_2026")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_ticker(ticker, period="2y"):
    """Download OHLCV data for a ticker"""
    print(f"  Downloading {ticker}...", end=" ")
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if len(df) > 0:
            df.to_csv(DATA_DIR / f"{ticker}.csv")
            print(f"OK ({len(df)} days)")
            return True
        else:
            print("NO DATA")
            return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    print("="*60)
    print("DOWNLOADING WATCHLIST DATA")
    print(f"Tickers: {len(WATCHLIST)}")
    print(f"Output: {DATA_DIR}")
    print("="*60)
    
    success = 0
    failed = []
    
    for ticker in WATCHLIST:
        if download_ticker(ticker):
            success += 1
        else:
            failed.append(ticker)
    
    print("="*60)
    print(f"SUCCESS: {success}/{len(WATCHLIST)}")
    if failed:
        print(f"FAILED: {failed}")
    print("="*60)
    
    # Save summary
    summary = {
        "date": datetime.now().isoformat(),
        "tickers": WATCHLIST,
        "success": success,
        "failed": failed
    }
    pd.Series(summary).to_json(DATA_DIR / "download_summary.json")

if __name__ == "__main__":
    main()
