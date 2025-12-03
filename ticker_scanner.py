"""
TICKER SCANNER
==============
Scans a watchlist of 50+ tickers using the Ultimate Predictor.
Ranks opportunities by confidence and signal strength.

Usage:
    python ticker_scanner.py --limit 10
"""

import os
import json
import argparse
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import sys

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
    args = parser.parse_args()
    
    scanner = TickerScanner()
    scanner.scan(limit=args.limit)

if __name__ == "__main__":
    main()
