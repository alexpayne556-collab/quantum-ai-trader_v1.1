"""
DOWNLOAD WATCHLIST DATA - Fixed for proper CSV format
"""
import yfinance as yf
import pandas as pd
from pathlib import Path
from WATCHLIST_2026 import WATCHLIST

DATA_DIR = Path("data/watchlist_2026")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_ticker(ticker: str, period: str = "2y"):
    """Download and save single ticker with CLEAN format"""
    print(f"Downloading {ticker}...")
    
    try:
        df = yf.download(ticker, period=period, progress=False)
        
        if df.empty:
            print(f"  WARNING: No data for {ticker}")
            return False
        
        # FIX: Flatten multi-index columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Ensure proper column names
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Ensure numeric types
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').astype(int)
        
        # Drop any rows with NaN
        df = df.dropna()
        
        # Save with clean format
        path = DATA_DIR / f"{ticker}.csv"
        df.to_csv(path, date_format='%Y-%m-%d')
        
        print(f"  {ticker}: {len(df)} days, ${df['Close'].iloc[-1]:.2f}")
        return True
        
    except Exception as e:
        print(f"  ERROR {ticker}: {e}")
        return False

def main():
    print("="*60)
    print("DOWNLOADING WATCHLIST DATA (CLEAN FORMAT)")
    print("="*60)
    
    success = 0
    for ticker in WATCHLIST:
        if download_ticker(ticker):
            success += 1
    
    print(f"\nDownloaded {success}/{len(WATCHLIST)} tickers")
    print(f"Data saved to: {DATA_DIR.absolute()}")

if __name__ == "__main__":
    main()
