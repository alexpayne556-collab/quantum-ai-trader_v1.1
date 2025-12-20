#!/usr/bin/env python3
"""
=============================================================================
WATCHLIST TRAINING FOUNDATION
=============================================================================

This is the FOUNDATION - waiting for YOUR watchlist decisions.

NOT production ready. This is the infrastructure that will:
1. Accept your watchlist when you define it
2. Train validated signals on YOUR chosen stocks
3. Build asset-specific profiles
4. Store results for later analysis

NO TRADING. Just learning.

Usage:
    1. Define your watchlist in watchlist_config.json
    2. Run: python WATCHLIST_TRAINING_FOUNDATION.py
    3. Review results in watchlist_training_results/
    4. Iterate and refine

=============================================================================
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import warnings

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - EDIT THIS WHEN READY
# =============================================================================

CONFIG_FILE = Path('./watchlist_config.json')
RESULTS_DIR = Path('./watchlist_training_results/')
RESULTS_DIR.mkdir(exist_ok=True)

# Default empty config template
DEFAULT_CONFIG = {
    "watchlist": {
        "symbols": [],  # <-- YOU FILL THIS IN
        "notes": "Add your chosen symbols here when ready"
    },
    "training": {
        "start_date": "2020-01-01",
        "end_date": None,  # None = today
        "min_history_days": 252,  # Require 1 year minimum
    },
    "signals_to_test": [
        "H16_weekly_reversal",
        "H19_bollinger_mr",
        "H20_vix_mr",
        "H128_vix_turbulence",
        "RSI_oversold",
        "gap_reversal",
        "volume_spike",
        "momentum_score"
    ],
    "status": "NOT_READY",  # Change to "READY" when watchlist is defined
    "last_updated": None
}


def create_config_template():
    """Create the config file template if it doesn't exist."""
    if not CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        print(f"✅ Created config template: {CONFIG_FILE}")
        print(f"   Edit this file to add your watchlist symbols")
    return DEFAULT_CONFIG


def load_config() -> Dict:
    """Load configuration from file."""
    if not CONFIG_FILE.exists():
        return create_config_template()
    
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)


def save_config(config: Dict):
    """Save configuration to file."""
    config['last_updated'] = datetime.now().isoformat()
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


# =============================================================================
# SIGNAL LIBRARY (From your validated hypotheses)
# =============================================================================

def calc_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


class SignalLibrary:
    """
    Your validated signals, ready to test on any stock.
    All signals use proper lag (shift(1)) to avoid look-ahead bias.
    """
    
    @staticmethod
    def h16_weekly_reversal(data: pd.DataFrame, threshold: float = -0.03) -> pd.Series:
        """H16: Buy when 5-day return < -3%."""
        weekly_ret = data['close'].pct_change(5)
        return (weekly_ret < threshold).shift(1).fillna(0).astype(int)
    
    @staticmethod
    def h19_bollinger_mr(data: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.Series:
        """H19: Buy when price below lower Bollinger Band."""
        ma = data['close'].rolling(period).mean()
        std = data['close'].rolling(period).std()
        lower = ma - num_std * std
        return (data['close'] < lower).shift(1).fillna(0).astype(int)
    
    @staticmethod
    def rsi_oversold(data: pd.DataFrame, threshold: int = 30) -> pd.Series:
        """RSI oversold signal."""
        rsi = calc_rsi(data['close'], 14)
        return (rsi < threshold).shift(1).fillna(0).astype(int)
    
    @staticmethod
    def volume_spike(data: pd.DataFrame, threshold: float = 2.0) -> pd.Series:
        """Detect volume spikes (>2x 20-day average)."""
        avg_vol = data['volume'].rolling(20).mean()
        spike = data['volume'] > (avg_vol * threshold)
        return spike.shift(1).fillna(0).astype(int)
    
    @staticmethod
    def gap_reversal(data: pd.DataFrame, threshold: float = -0.02) -> pd.Series:
        """Detect gap downs that might reverse."""
        gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
        return (gap < threshold).shift(1).fillna(0).astype(int)
    
    @staticmethod
    def momentum_score(data: pd.DataFrame, periods: list = None) -> pd.Series:
        """Multi-timeframe momentum (returns 0-1 score)."""
        if periods is None:
            periods = [5, 21, 63]
        scores = []
        for p in periods:
            ret = data['close'].pct_change(p)
            score = (ret > 0).astype(int)
            scores.append(score)
        return pd.concat(scores, axis=1).mean(axis=1)


# =============================================================================
# DATA FETCHER
# =============================================================================

def fetch_data(symbol: str, start_date: str = '2020-01-01') -> Optional[pd.DataFrame]:
    """
    Fetch historical data for a symbol.
    Uses yfinance for free data.
    """
    try:
        import yfinance as yf
        
        data = yf.download(symbol, start=start_date, progress=False)
        
        if data.empty:
            print(f"   ⚠️ No data for {symbol}")
            return None
        
        # Standardize columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.columns = [c.lower().replace(' ', '_') for c in data.columns]
        
        return data
    
    except Exception as e:
        print(f"   ❌ Error fetching {symbol}: {e}")
        return None


# =============================================================================
# SIGNAL TESTER
# =============================================================================

class SignalTester:
    """
    Tests signals on individual stocks.
    Calculates edge metrics without any trading.
    """
    
    def __init__(self):
        self.signals = SignalLibrary()
    
    def test_signal(self, data: pd.DataFrame, signal_func, 
                    signal_name: str, hold_period: int = 21) -> Dict:
        """
        Test a signal on historical data.
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Generate signal
            signal = signal_func(data)
            
            # Calculate forward returns
            fwd_ret = data['close'].pct_change(hold_period).shift(-hold_period)
            
            # Split by signal
            signal_rets = fwd_ret[signal == 1].dropna()
            other_rets = fwd_ret[signal == 0].dropna()
            
            if len(signal_rets) < 10:
                return {
                    'valid': False,
                    'reason': f'Only {len(signal_rets)} signal days (need 10+)',
                    'signal_name': signal_name
                }
            
            # Calculate metrics
            spread = (signal_rets.mean() - other_rets.mean()) * (252/hold_period)
            
            signal_sharpe = 0
            if signal_rets.std() > 0:
                signal_sharpe = signal_rets.mean() / signal_rets.std() * np.sqrt(252/hold_period)
            
            win_rate = (signal_rets > 0).mean()
            
            return {
                'valid': True,
                'signal_name': signal_name,
                'n_signals': len(signal_rets),
                'signal_frequency': len(signal_rets) / len(data),
                'spread': spread,  # Annualized outperformance
                'sharpe': signal_sharpe,
                'win_rate': win_rate,
                'avg_return': signal_rets.mean() * (252/hold_period),
                'avg_return_other': other_rets.mean() * (252/hold_period),
            }
        
        except Exception as e:
            return {
                'valid': False,
                'reason': str(e),
                'signal_name': signal_name
            }
    
    def test_all_signals(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Test all signals on a stock."""
        results = {
            'symbol': symbol,
            'data_start': data.index[0].strftime('%Y-%m-%d'),
            'data_end': data.index[-1].strftime('%Y-%m-%d'),
            'n_days': len(data),
            'signals': {}
        }
        
        # Define signal tests
        signal_tests = [
            ('H16_Weekly_Reversal', lambda d: self.signals.h16_weekly_reversal(d)),
            ('H19_Bollinger_MR', lambda d: self.signals.h19_bollinger_mr(d)),
            ('RSI_Oversold', lambda d: self.signals.rsi_oversold(d)),
            ('Volume_Spike', lambda d: self.signals.volume_spike(d)),
            ('Gap_Reversal', lambda d: self.signals.gap_reversal(d)),
            ('Momentum_Bullish', lambda d: (self.signals.momentum_score(d) > 0.66).astype(int)),
        ]
        
        for signal_name, signal_func in signal_tests:
            result = self.test_signal(data, signal_func, signal_name)
            results['signals'][signal_name] = result
        
        return results


# =============================================================================
# TRAINING PIPELINE
# =============================================================================

class WatchlistTrainer:
    """
    Trains signals on your watchlist.
    Stores results for analysis - NO TRADING.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.tester = SignalTester()
        self.results = {}
    
    def check_ready(self) -> bool:
        """Check if watchlist is ready for training."""
        symbols = self.config.get('watchlist', {}).get('symbols', [])
        status = self.config.get('status', 'NOT_READY')
        
        if not symbols:
            print("❌ No symbols in watchlist")
            print(f"   Edit {CONFIG_FILE} to add your symbols")
            return False
        
        if status != 'READY':
            print(f"❌ Status is '{status}' - change to 'READY' when watchlist is finalized")
            return False
        
        return True
    
    def train(self) -> Dict:
        """Run training on all watchlist symbols."""
        if not self.check_ready():
            return {}
        
        symbols = self.config['watchlist']['symbols']
        start_date = self.config['training']['start_date']
        
        print("\n" + "="*70)
        print("WATCHLIST TRAINING PIPELINE")
        print("="*70)
        print(f"Symbols: {len(symbols)}")
        print(f"Start Date: {start_date}")
        print(f"Signals: {len(self.tester.signals.__class__.__dict__) - 2}")  # Rough count
        print("="*70)
        
        all_results = {}
        
        for i, symbol in enumerate(symbols, 1):
            print(f"\n[{i}/{len(symbols)}] Training on {symbol}...")
            
            # Fetch data
            data = fetch_data(symbol, start_date)
            if data is None or len(data) < self.config['training']['min_history_days']:
                print(f"   ⚠️ Insufficient data for {symbol}")
                continue
            
            # Test all signals
            results = self.tester.test_all_signals(data, symbol)
            all_results[symbol] = results
            
            # Print quick summary
            valid_signals = [s for s in results['signals'].values() if s.get('valid')]
            if valid_signals:
                best = max(valid_signals, key=lambda x: x.get('spread', 0))
                print(f"   ✅ Best signal: {best['signal_name']} ({best['spread']*100:+.1f}% spread)")
        
        self.results = all_results
        self.save_results()
        self.print_summary()
        
        return all_results
    
    def save_results(self):
        """Save training results to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save full results as JSON
        results_file = RESULTS_DIR / f'training_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save summary as CSV
        summary_rows = []
        for symbol, data in self.results.items():
            for signal_name, metrics in data.get('signals', {}).items():
                if metrics.get('valid'):
                    summary_rows.append({
                        'symbol': symbol,
                        'signal': signal_name,
                        'spread': metrics['spread'],
                        'sharpe': metrics['sharpe'],
                        'win_rate': metrics['win_rate'],
                        'n_signals': metrics['n_signals'],
                        'frequency': metrics['signal_frequency'],
                    })
        
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_file = RESULTS_DIR / f'training_summary_{timestamp}.csv'
            summary_df.to_csv(summary_file, index=False)
            print(f"\n✅ Results saved to {RESULTS_DIR}/")
    
    def print_summary(self):
        """Print training summary."""
        if not self.results:
            return
        
        print("\n" + "="*70)
        print("TRAINING SUMMARY")
        print("="*70)
        
        # Find best signals per asset
        best_by_asset = []
        for symbol, data in self.results.items():
            valid_signals = [
                (name, metrics) 
                for name, metrics in data.get('signals', {}).items()
                if metrics.get('valid') and metrics.get('spread', 0) > 0
            ]
            
            if valid_signals:
                best = max(valid_signals, key=lambda x: x[1]['spread'])
                best_by_asset.append({
                    'symbol': symbol,
                    'best_signal': best[0],
                    'spread': best[1]['spread'],
                    'sharpe': best[1]['sharpe'],
                    'win_rate': best[1]['win_rate'],
                })
        
        if best_by_asset:
            # Sort by spread
            best_by_asset.sort(key=lambda x: -x['spread'])
            
            print("\n📊 Best Signal for Each Asset:")
            print("-"*70)
            print(f"{'Symbol':<10} {'Best Signal':<25} {'Spread':>10} {'Sharpe':>8} {'Win Rate':>10}")
            print("-"*70)
            
            for row in best_by_asset[:20]:  # Top 20
                print(f"{row['symbol']:<10} {row['best_signal']:<25} "
                      f"{row['spread']*100:>+9.1f}% {row['sharpe']:>8.2f} "
                      f"{row['win_rate']*100:>9.1f}%")
            
            # Best signals overall
            print("\n🏆 Top 5 Asset-Signal Combinations:")
            for i, row in enumerate(best_by_asset[:5], 1):
                print(f"   {i}. {row['symbol']} + {row['best_signal']}: "
                      f"{row['spread']*100:+.1f}% spread, {row['sharpe']:.2f} Sharpe")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("WATCHLIST TRAINING FOUNDATION")
    print("="*70)
    print("This is NOT production code. This trains signals on YOUR watchlist.")
    print("="*70)
    
    # Load or create config
    config = load_config()
    
    # Check status
    symbols = config.get('watchlist', {}).get('symbols', [])
    status = config.get('status', 'NOT_READY')
    
    if not symbols:
        print(f"\n⏳ WAITING FOR YOUR WATCHLIST")
        print(f"   1. Edit {CONFIG_FILE}")
        print(f"   2. Add your symbols to 'symbols' array")
        print(f"   3. Change 'status' to 'READY'")
        print(f"   4. Run this script again")
        print(f"\n   Example symbols array:")
        print(f'   "symbols": ["AAPL", "MSFT", "GOOGL", "NVDA", "AMD"]')
        return
    
    if status != 'READY':
        print(f"\n📋 Watchlist defined: {symbols}")
        print(f"⚠️  Status is '{status}'")
        print(f"   When ready, change 'status' to 'READY' in {CONFIG_FILE}")
        return
    
    # Ready to train
    print(f"\n✅ Watchlist ready: {symbols}")
    trainer = WatchlistTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
