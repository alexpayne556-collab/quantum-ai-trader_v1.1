"""
ULTIMATE PREDICTOR
==================
The production-ready wrapper for the Golden Architecture.
Handles data fetching, model persistence, and live predictions.

Usage:
    python ultimate_predictor.py --ticker AAPL --action predict
    python ultimate_predictor.py --ticker AAPL --action train
"""

import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import json
import sys
from typing import Tuple, Dict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from golden_architecture import GoldenArchitecture

class UltimatePredictor:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.arch = GoldenArchitecture(verbose=True)
        
    def _get_model_path(self, ticker: str) -> str:
        return os.path.join(self.model_dir, f"{ticker}_golden_v1.pkl")
    
    def fetch_data(self, ticker: str, days: int = 730) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch data for ticker and market benchmark"""
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        print(f"Fetching data for {ticker} from {start_date}...")
        df = yf.download(ticker, start=start_date, progress=False)
        market_df = yf.download("SPY", start=start_date, progress=False)
        
        if len(df) < 100:
            raise ValueError(f"Insufficient data for {ticker}. Got {len(df)} rows.")
            
        return df, market_df
    
    def train(self, ticker: str, force: bool = False):
        """Train model for a specific ticker"""
        model_path = self._get_model_path(ticker)
        
        if os.path.exists(model_path) and not force:
            print(f"Model for {ticker} already exists. Use --force to retrain.")
            return
        
        df, market_df = self.fetch_data(ticker)
        
        print(f"\nTraining Golden Architecture for {ticker}...")
        self.arch.build(
            df=df,
            market_df=market_df,
            label_method='triple_barrier',
            use_visual=True,  # Enable all engines
            use_logic=True,
            use_regime=True
        )
        
        self.arch.save(model_path)
        print(f"✅ Model saved to {model_path}")
        
    def predict(self, ticker: str) -> Dict:
        """Load model and make prediction"""
        model_path = self._get_model_path(ticker)
        
        if not os.path.exists(model_path):
            print(f"Model for {ticker} not found. Training now...")
            self.train(ticker)
        else:
            self.arch.load(model_path)
            
        df, market_df = self.fetch_data(ticker)
        
        print(f"\nGenerating prediction for {ticker}...")
        prediction = self.arch.predict(df, market_df)
        
        return prediction

def main():
    parser = argparse.ArgumentParser(description="Ultimate Predictor CLI")
    parser.add_argument("--ticker", type=str, required=True, help="Ticker symbol (e.g., AAPL)")
    parser.add_argument("--action", type=str, choices=["train", "predict"], default="predict")
    parser.add_argument("--force", action="store_true", help="Force retraining")
    
    args = parser.parse_args()
    
    predictor = UltimatePredictor()
    
    if args.action == "train":
        predictor.train(args.ticker, force=args.force)
    elif args.action == "predict":
        result = predictor.predict(args.ticker)
        
        print("\n" + "="*50)
        print(f"🔮 PREDICTION FOR {args.ticker}")
        print("="*50)
        print(f"Signal:        {result['signal']}")
        print(f"Confidence:    {result['confidence']:.1%}")
        print(f"Position Size: {result['position_size']:.1%}")
        print(f"Regime:        {result['regime']}")
        print(f"Pattern:       {result['pattern']}")
        print("\nProbabilities:")
        for k, v in result['probabilities'].items():
            print(f"  {k}: {v:.1%}")
        print("\nAnalysis:")
        for note in result['explanation']:
            print(f"  • {note}")
        print("="*50)

if __name__ == "__main__":
    main()
