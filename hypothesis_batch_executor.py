#!/usr/bin/env python3
"""
HYPOTHESIS BATCH EXECUTOR
==========================
Rate-limit safe hypothesis testing with incremental saves.

Usage:
    python hypothesis_batch_executor.py --batch 1      # Run Batch 1 only
    python hypothesis_batch_executor.py --all          # Run all batches
    python hypothesis_batch_executor.py --quick        # Run priority 1 only
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import warnings
import time
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# Import batch loader
from hypothesis_batches import (
    get_all_hypotheses, 
    get_batch_hypotheses, 
    get_priority_hypotheses,
    BATCH_REGISTRY,
    list_batches
)


class HypothesisBatchExecutor:
    """
    Executes hypothesis testing batches with built-in rate limiting.
    Saves results incrementally to prevent loss on connection failure.
    """
    
    def __init__(self, data_cache_path='./hypothesis_data/', start_date='2010-01-01'):
        self.data_cache = {}
        self.results = []
        self.cache_path = Path(data_cache_path)
        self.cache_path.mkdir(exist_ok=True)
        self.start_date = start_date
        self.rate_limit_pause = 0.5  # seconds between API calls
        
        self.ticker_universe = self._init_ticker_universe()
        
    def _init_ticker_universe(self):
        """Initialize the ticker universe needed for all batches"""
        return {
            'core': ['SPY', 'QQQ', 'IWM'],
            'sectors': ['XLK', 'XLV', 'XLI', 'XLY', 'XLE', 'XLF', 'XLU', 'XLB', 'XLRE', 'XLP', 'XLC'],
            'intl': ['EFA', 'EEM', 'VEA', 'VWO'],
            'bonds': ['TLT', 'IEF', 'SHY', 'TIP'],
            'credit': ['HYG', 'LQD', 'JNK', 'EMB'],
            'commodities': ['GLD', 'DBC', 'USO', 'COPX'],
            'fx': ['UUP', 'FXY'],
            'factors': ['MTUM', 'VLUE', 'QUAL', 'SPLV', 'SPHB', 'ARKK', 'XBI'],
            'vix': ['^VIX', '^VIX3M', '^VIX9D', '^SKEW'],
        }
    
    def download_batch_data(self, batch_name: str, ticker_list: list) -> tuple:
        """
        Download data for a batch in one efficient call.
        Returns dict of DataFrames, caches results.
        """
        print(f"\n📥 Downloading data for {batch_name}...")
        print(f"   Tickers: {', '.join(ticker_list[:5])}{'...' if len(ticker_list) > 5 else ''}")
        
        # Check cache first
        cache_file = self.cache_path / f"{batch_name}_data.pkl"
        if cache_file.exists():
            print(f"   ⚡ Loading from cache: {cache_file}")
            data = pd.read_pickle(cache_file)
            return data, []
        
        data = {}
        failed = []
        
        # Batch download
        try:
            time.sleep(self.rate_limit_pause)  # Rate limit
            df = yf.download(ticker_list, start=self.start_date, progress=False)
            
            # Handle single vs multiple tickers
            if len(ticker_list) == 1:
                data[ticker_list[0]] = df
            else:
                for ticker in ticker_list:
                    try:
                        if isinstance(df.columns, pd.MultiIndex):
                            if ticker in df.columns.get_level_values(1):
                                ticker_df = df.xs(ticker, axis=1, level=1)
                                data[ticker] = ticker_df
                            else:
                                failed.append(ticker)
                        else:
                            data[ticker_list[0]] = df
                    except Exception as e:
                        failed.append(ticker)
            
            # Cache the data
            pd.to_pickle(data, cache_file)
            self.data_cache[batch_name] = data
            print(f"   ✓ Downloaded {len(data)} tickers, {len(failed)} failed")
            return data, failed
            
        except Exception as e:
            print(f"   ✗ Download failed: {e}")
            return {}, ticker_list
    
    def test_hypothesis(self, hypothesis: dict, data: dict) -> dict:
        """Test a single hypothesis and return results."""
        result = {
            'hypothesis_id': hypothesis['id'],
            'name': hypothesis['name'],
            'category': hypothesis['category'],
            'batch': hypothesis.get('batch', 0),
            'spread': 0,
            'p_value': 1.0,
            'sharpe': 0,
            'win_rate': 0,
            'n_signals': 0,
            'pass': False,
            'note': '',
            'error': None,
        }
        
        try:
            # Get primary ticker data
            primary_ticker = hypothesis['tickers'][0]
            if primary_ticker not in data:
                result['error'] = f"Missing data for {primary_ticker}"
                return result
            
            ticker_data = data[primary_ticker]
            
            # Standardize column names
            if 'Adj Close' in ticker_data.columns:
                df = pd.DataFrame({
                    'close': ticker_data['Adj Close'],
                    'open': ticker_data['Open'],
                    'high': ticker_data['High'],
                    'low': ticker_data['Low'],
                    'volume': ticker_data.get('Volume', 0),
                })
            else:
                df = ticker_data.copy()
                df.columns = df.columns.str.lower()
            
            # Generate signal
            signal_func = hypothesis['signal_func']
            
            # Prepare kwargs for signal function
            kwargs = {}
            
            # Add secondary data if needed
            for ticker in hypothesis['tickers'][1:]:
                if ticker in data:
                    secondary_data = data[ticker]
                    if 'Adj Close' in secondary_data.columns:
                        kwargs[f'{ticker.lower()}_data'] = pd.DataFrame({
                            'close': secondary_data['Adj Close'],
                            'open': secondary_data['Open'],
                            'high': secondary_data['High'],
                            'low': secondary_data['Low'],
                            'volume': secondary_data.get('Volume', 0),
                        })
                    else:
                        kwargs[f'{ticker.lower()}_data'] = secondary_data
            
            # Add VIX if needed
            if '^VIX' in data:
                kwargs['vix_data'] = data['^VIX']['Adj Close'] if 'Adj Close' in data['^VIX'].columns else data['^VIX']['close']
            
            # Generate signal
            signal = signal_func(df, **kwargs)
            
            # Calculate returns
            hold_period = hypothesis.get('hold_period', 21)
            forward_returns = df['close'].pct_change(hold_period).shift(-hold_period)
            
            # Align signal and returns
            valid_idx = signal.notna() & forward_returns.notna()
            
            if valid_idx.sum() < 50:
                result['error'] = f"Insufficient signals: {valid_idx.sum()}"
                return result
            
            # Calculate strategy returns
            long_returns = forward_returns[valid_idx & (signal == 1)]
            other_returns = forward_returns[valid_idx & (signal == 0)]
            
            result['n_signals'] = len(long_returns)
            
            if len(long_returns) > 20 and len(other_returns) > 20:
                # Spread (annualized)
                ann_factor = 252 / hold_period
                long_avg = long_returns.mean() * ann_factor
                other_avg = other_returns.mean() * ann_factor
                result['spread'] = long_avg - other_avg
                
                # T-test
                t_stat, p_value = stats.ttest_ind(long_returns, other_returns)
                result['p_value'] = p_value
                
                # Sharpe ratio
                if long_returns.std() > 0:
                    result['sharpe'] = (long_returns.mean() / long_returns.std()) * np.sqrt(ann_factor)
                
                # Win rate
                result['win_rate'] = (long_returns > 0).mean()
                
                # Pass criteria
                result['pass'] = (p_value < 0.1) and (result['spread'] > 0)
                result['note'] = f"Long: {long_avg:.2%}, Other: {other_avg:.2%}"
            else:
                result['error'] = f"Insufficient data: long={len(long_returns)}, other={len(other_returns)}"
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def run_batch(self, batch_num: int, save_results: bool = True) -> pd.DataFrame:
        """Run all hypotheses in a specific batch."""
        if batch_num not in BATCH_REGISTRY:
            print(f"❌ Batch {batch_num} not found!")
            return pd.DataFrame()
        
        batch_info = BATCH_REGISTRY[batch_num]
        hypotheses = batch_info['hypotheses']
        
        if not hypotheses:
            print(f"❌ Batch {batch_num} has no hypotheses!")
            return pd.DataFrame()
        
        print("\n" + "="*60)
        print(f"BATCH {batch_num}: {batch_info['name']} ({len(hypotheses)} hypotheses)")
        print("="*60)
        
        # Collect all tickers needed
        all_tickers = set()
        for hyp in hypotheses:
            all_tickers.update(hyp.get('tickers', []))
        
        # Add VIX if any hypothesis needs it
        if any(hyp.get('requires_macro', []) for hyp in hypotheses):
            all_tickers.add('^VIX')
        
        # Download data
        data, failed = self.download_batch_data(f'batch_{batch_num}', list(all_tickers))
        
        # Test each hypothesis
        results = []
        for i, hyp in enumerate(hypotheses, 1):
            print(f"\n[{hyp['id']}] {hyp['name']} ({i}/{len(hypotheses)})...")
            result = self.test_hypothesis(hyp, data)
            results.append(result)
            
            status = "✓ PASS" if result['pass'] else "✗ FAIL" if result['error'] is None else "⚠ ERROR"
            print(f"  {status} | Spread: {result['spread']:.2%} | p={result['p_value']:.4f}")
            if result['error']:
                print(f"  Error: {result['error']}")
        
        # Save results
        results_df = pd.DataFrame(results)
        
        if save_results:
            filename = self.cache_path / f'batch_{batch_num}_results.csv'
            results_df.to_csv(filename, index=False)
            print(f"\n💾 Saved to {filename}")
        
        self.results.extend(results)
        return results_df
    
    def run_all_batches(self, save_results: bool = True) -> pd.DataFrame:
        """Run all available batches."""
        all_results = []
        
        for batch_num in sorted(BATCH_REGISTRY.keys()):
            if len(BATCH_REGISTRY[batch_num]['hypotheses']) > 0:
                try:
                    batch_results = self.run_batch(batch_num, save_results=save_results)
                    all_results.append(batch_results)
                    print(f"\n✅ Batch {batch_num} complete: {len(batch_results)} hypotheses")
                except Exception as e:
                    print(f"\n❌ Batch {batch_num} failed: {e}")
                    continue
        
        if all_results:
            summary_df = pd.concat(all_results, ignore_index=True)
            self._print_summary(summary_df)
            
            if save_results:
                summary_file = self.cache_path / 'all_results_summary.csv'
                summary_df.to_csv(summary_file, index=False)
                print(f"\n💾 Full summary saved to {summary_file}")
            
            return summary_df
        
        return pd.DataFrame()
    
    def run_priority(self, priority_level: int = 1) -> pd.DataFrame:
        """Run only high-priority hypotheses."""
        hypotheses = get_priority_hypotheses(priority_level)
        
        print(f"\n🎯 Running {len(hypotheses)} Priority-{priority_level} hypotheses...")
        
        # Collect all tickers
        all_tickers = set()
        for hyp in hypotheses:
            all_tickers.update(hyp.get('tickers', []))
        all_tickers.add('^VIX')
        
        # Download data
        data, _ = self.download_batch_data('priority', list(all_tickers))
        
        # Test each
        results = []
        for hyp in hypotheses:
            print(f"\n[{hyp['id']}] {hyp['name']}...")
            result = self.test_hypothesis(hyp, data)
            results.append(result)
        
        results_df = pd.DataFrame(results)
        self._print_summary(results_df)
        
        return results_df
    
    def _print_summary(self, results_df: pd.DataFrame):
        """Print summary statistics."""
        print("\n" + "="*60)
        print("TESTING SUMMARY")
        print("="*60)
        
        total = len(results_df)
        passed = results_df['pass'].sum()
        errors = results_df['error'].notna().sum()
        
        print(f"Total hypotheses tested: {total}")
        print(f"Passed (p < 0.1 & spread > 0): {passed} ({100*passed/total:.1f}%)")
        print(f"Errors: {errors}")
        
        # Top winners
        winners = results_df[results_df['pass']].nlargest(10, 'spread')
        if len(winners) > 0:
            print(f"\n🏆 TOP WINNERS:")
            print(winners[['hypothesis_id', 'name', 'spread', 'sharpe', 'p_value']].to_string(index=False))
        
        # By category
        print(f"\n📊 BY CATEGORY:")
        category_stats = results_df.groupby('category').agg({
            'pass': 'sum',
            'spread': 'mean',
            'hypothesis_id': 'count'
        }).rename(columns={'hypothesis_id': 'total'})
        category_stats['pass_rate'] = category_stats['pass'] / category_stats['total'] * 100
        print(category_stats.to_string())


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Hypothesis Batch Executor')
    parser.add_argument('--batch', type=int, help='Run specific batch (1-9)')
    parser.add_argument('--all', action='store_true', help='Run all batches')
    parser.add_argument('--quick', action='store_true', help='Run priority-1 only')
    parser.add_argument('--list', action='store_true', help='List available batches')
    parser.add_argument('--start', default='2010-01-01', help='Start date for data')
    args = parser.parse_args()
    
    if args.list:
        list_batches()
    elif args.batch:
        executor = HypothesisBatchExecutor(start_date=args.start)
        executor.run_batch(args.batch)
    elif args.all:
        executor = HypothesisBatchExecutor(start_date=args.start)
        executor.run_all_batches()
    elif args.quick:
        executor = HypothesisBatchExecutor(start_date=args.start)
        executor.run_priority(1)
    else:
        # Default: show batches
        list_batches()
        print("\n💡 Usage:")
        print("  python hypothesis_batch_executor.py --batch 1   # Run Batch 1")
        print("  python hypothesis_batch_executor.py --all       # Run all batches")
        print("  python hypothesis_batch_executor.py --quick     # Priority-1 only")
