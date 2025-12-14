"""
TICKER SCANNER
==============
Scans a watchlist of 50+ tickers using the Ultimate Predictor.
Ranks opportunities by confidence and signal strength.

Usage:
    python ticker_scanner.py --limit 10
    python ticker_scanner.py --watchlist watchlist.txt
    python ticker_scanner.py --watchlist custom.json --limit 25
"""

import os
import json
import argparse
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import sys
from typing import List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultimate_predictor import UltimatePredictor

# Default Watchlist (Liquid Large Caps & ETFs)
WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "AMD", "NFLX", "INTC",
    "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "USO", "TLT",
    "JPM", "BAC", "GS", "MS", "V", "MA",
    "JNJ", "PFE", "UNH", "LLY",
    "XOM", "CVX", "COP",
    "KO", "PEP", "MCD", "SBUX",
    "DIS", "CMCSA",
    "BA", "CAT", "DE",
    "WMT", "TGT", "COST",
    "CRM", "ADBE", "ORCL", "CSCO"
]


def load_watchlist(path: str | None) -> List[str]:
    """Load tickers from a watchlist file.

    Supports:
    - .txt: one ticker per line, comments starting with '#'
    - .json: ["AAPL", "MSFT", ...]
    If path is None, uses `watchlist.txt` if present, else the default WATCHLIST.
    """
    if path is None:
        default_path = os.path.join(os.getcwd(), "watchlist.txt")
        if os.path.exists(default_path):
            path = default_path
        else:
            return WATCHLIST

    try:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".json":
            with open(path, "r") as f:
                data = json.load(f)
                tickers = [str(t).strip().upper() for t in data if str(t).strip()]
        else:
            # Treat as txt by default
            tickers = []
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    tickers.append(line.upper())

        # De-duplicate while preserving order
        seen = set()
        deduped = []
        for t in tickers:
            if t not in seen:
                seen.add(t)
                deduped.append(t)
        return deduped if deduped else WATCHLIST
    except Exception as e:
        print(f"⚠️ Failed to load watchlist from '{path}': {e}. Using default list.")
        return WATCHLIST

class TickerScanner:
    def __init__(self):
        self.predictor = UltimatePredictor()
        self.results = []
        
    def scan(self, tickers: list = WATCHLIST, limit: int = None):
        """Scan list of tickers"""
        if limit:
            tickers = tickers[:limit]
            
        print(f"🚀 Starting scan for {len(tickers)} tickers...")
        
        for ticker in tqdm(tickers):
            try:
                # Predict (will train if needed)
                result = self.predictor.predict(ticker)
                
                # Add ticker to result
                result['ticker'] = ticker
                self.results.append(result)
                
            except Exception as e:
                print(f"❌ Error scanning {ticker}: {str(e)}")
        
        self._save_results()
        self._print_summary()
        
    def _save_results(self):
        """Save results to JSON"""
        output_file = "scan_results.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✅ Results saved to {output_file}")
        
    def _print_summary(self):
        """Print ranked summary table"""
        if not self.results:
            print("No results to display.")
            return
            
        # Convert to DataFrame for sorting
        df = pd.DataFrame(self.results)
        
        # Filter for BUY signals
        buys = df[df['signal'] == 'BUY'].sort_values('confidence', ascending=False)
        sells = df[df['signal'] == 'SELL'].sort_values('confidence', ascending=False)
        
        print("\n" + "="*60)
        print("🏆 TOP BUY OPPORTUNITIES")
        print("="*60)
        if not buys.empty:
            print(f"{'TICKER':<8} {'CONFIDENCE':<12} {'SIZE':<10} {'REGIME':<15} {'PATTERN'}")
            print("-" * 60)
            for _, row in buys.head(10).iterrows():
                print(f"{row['ticker']:<8} {row['confidence']:.1%}       {row['position_size']:.1%}       {row['regime']:<15} {row['pattern']}")
        else:
            print("No BUY signals found.")
            
        print("\n" + "="*60)
        print("📉 TOP SELL OPPORTUNITIES")
        print("="*60)
        if not sells.empty:
            print(f"{'TICKER':<8} {'CONFIDENCE':<12} {'SIZE':<10} {'REGIME':<15} {'PATTERN'}")
            print("-" * 60)
            for _, row in sells.head(10).iterrows():
                print(f"{row['ticker']:<8} {row['confidence']:.1%}       {row['position_size']:.1%}       {row['regime']:<15} {row['pattern']}")
        else:
            print("No SELL signals found.")

def main():
    parser = argparse.ArgumentParser(description="Ticker Scanner")
    parser.add_argument("--limit", type=int, help="Limit number of tickers to scan")
    parser.add_argument(
        "--watchlist",
        type=str,
        default=None,
        help="Path to watchlist file (.txt one-per-line or .json list). Defaults to watchlist.txt if present."
    )
    args = parser.parse_args()

    tickers = load_watchlist(args.watchlist)
    scanner = TickerScanner()
    scanner.scan(tickers=tickers, limit=args.limit)

if __name__ == "__main__":
    main()
