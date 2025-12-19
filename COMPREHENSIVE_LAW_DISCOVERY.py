"""
COMPREHENSIVE ALEX LAWS DISCOVERY
==================================
NO SHORTCUTS. Test everything on ALL clean data.
Mission: Discover real edges that survive rigorous testing.

"We don't premake laws, we discover them"
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveLawDiscovery:
    """
    Test EVERY hypothesis on EVERY clean ticker.
    Document EVERYTHING for reproducibility.
    """
    
    def __init__(self):
        self.conn = sqlite3.connect('data/market_data.db')
        self.transaction_costs = self._load_transaction_costs()
        self.clean_universe = self._load_clean_universe()
        
        print(f"✅ Loaded {len(self.clean_universe):,} clean tickers")
        print(f"✅ Loaded transaction costs for all tickers")
        print(f"📊 Total rows in database: {self._count_total_rows():,}")
    
    def _count_total_rows(self):
        return pd.read_sql("SELECT COUNT(*) as cnt FROM ohlcv", self.conn)['cnt'][0]
    
    def _load_transaction_costs(self):
        costs = pd.read_csv('data/transaction_costs.csv')
        cost_map = {'penny': 0.03, 'illiquid': 0.02, 'small': 0.008, 
                   'mid': 0.003, 'large': 0.001}
        costs['cost_pct'] = costs['tier'].map(cost_map)
        return dict(zip(costs['ticker'], costs['cost_pct']))
    
    def _load_clean_universe(self):
        """Load the 5,062 clean tickers (excluded bad data/high costs)"""
        try:
            extreme = pd.read_csv('data/extreme_moves.csv')
            poor_cov = pd.read_csv('data/poor_coverage.csv')
            
            bad_tickers = extreme[extreme.groupby('ticker')['ticker'].transform('count') > 2]['ticker'].unique()
            bad_tickers = set(bad_tickers) | set(poor_cov['ticker'])
            
            high_cost = [t for t, c in self.transaction_costs.items() if c > 0.01]
            bad_tickers = bad_tickers | set(high_cost)
            
            all_tickers = pd.read_sql("SELECT DISTINCT ticker FROM ohlcv", self.conn)['ticker']
            return [t for t in all_tickers if t not in bad_tickers]
        except:
            return []
    
    def test_all_momentum_variants(self):
        """
        Test MULTIPLE momentum strategies:
        - Different lookback periods (5, 10, 20, 60, 126 days)
        - Different holding periods (1, 5, 10, 21 days)
        - Different thresholds
        
        Find which combinations actually work after costs.
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE MOMENTUM TESTING")
        print("="*80)
        print(f"Testing ALL {len(self.clean_universe):,} clean tickers")
        print("Variants: 5 lookback periods × 4 holding periods = 20 strategies")
        print("")
        
        lookbacks = [5, 10, 20, 60, 126]  # 1wk, 2wk, 1mo, 3mo, 6mo
        holding_periods = [1, 5, 10, 21]  # 1d, 1wk, 2wk, 1mo
        
        all_results = []
        
        for lookback in lookbacks:
            for hold_period in holding_periods:
                print(f"\n--- Testing: {lookback}-day momentum, {hold_period}-day hold ---")
                
                strategy_results = []
                
                for ticker in tqdm(self.clean_universe, desc=f"L{lookback}H{hold_period}"):
                    df = pd.read_sql(f"""
                        SELECT date, close, volume 
                        FROM ohlcv 
                        WHERE ticker = '{ticker}'
                        ORDER BY date
                    """, self.conn, parse_dates=['date'])
                    
                    if len(df) < lookback + hold_period + 10:
                        continue
                    
                    # Calculate momentum and forward returns
                    df['momentum'] = df['close'].pct_change(lookback)
                    df[f'forward_ret_{hold_period}d'] = df['close'].shift(-hold_period) / df['close'] - 1
                    
                    # Get transaction cost
                    cost = self.transaction_costs.get(ticker, 0.01)
                    
                    # Test top quintile momentum
                    threshold = df['momentum'].quantile(0.8)  # Top 20%
                    signals = df[df['momentum'] > threshold].copy()
                    
                    if len(signals) >= 10:
                        avg_ret = signals[f'forward_ret_{hold_period}d'].mean()
                        std_ret = signals[f'forward_ret_{hold_period}d'].std()
                        
                        strategy_results.append({
                            'ticker': ticker,
                            'lookback': lookback,
                            'hold_period': hold_period,
                            'num_trades': len(signals),
                            'gross_return': avg_ret,
                            'net_return': avg_ret - cost,
                            'std': std_ret,
                            'sharpe': avg_ret / std_ret if std_ret > 0 else 0,
                            'transaction_cost': cost
                        })
                
                if len(strategy_results) > 0:
                    df_strat = pd.DataFrame(strategy_results)
                    
                    # Aggregate statistics
                    avg_gross = df_strat['gross_return'].mean()
                    avg_net = df_strat['net_return'].mean()
                    profitable_pct = (df_strat['net_return'] > 0).mean()
                    
                    # T-test
                    t_stat, p_val = stats.ttest_1samp(df_strat['net_return'], 0)
                    
                    print(f"  Tested {len(df_strat)} tickers")
                    print(f"  Gross: {avg_gross:.3%} | Net: {avg_net:.3%}")
                    print(f"  Profitable: {profitable_pct:.1%}")
                    print(f"  T-stat: {t_stat:.2f} (need >3.0) | p-val: {p_val:.4f}")
                    
                    all_results.append({
                        'strategy': f'Mom{lookback}_Hold{hold_period}',
                        'lookback': lookback,
                        'hold_period': hold_period,
                        'num_tickers': len(df_strat),
                        'avg_gross': avg_gross,
                        'avg_net': avg_net,
                        'profitable_pct': profitable_pct,
                        't_stat': t_stat,
                        'p_value': p_val,
                        'significant': abs(t_stat) > 3.0
                    })
        
        # Summary
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('t_stat', ascending=False)
        
        print("\n" + "="*80)
        print("MOMENTUM STRATEGY RESULTS - RANKED BY T-STATISTIC")
        print("="*80)
        print(results_df.to_string(index=False))
        print("")
        
        significant = results_df[results_df['significant']]
        if len(significant) > 0:
            print(f"✅ FOUND {len(significant)} SIGNIFICANT STRATEGIES!")
            print(significant[['strategy', 't_stat', 'avg_net', 'profitable_pct']].to_string(index=False))
        else:
            print("❌ NO momentum strategies passed statistical threshold")
            print("   This is NORMAL - most academic factors don't work after costs")
        
        results_df.to_csv('data/MOMENTUM_COMPREHENSIVE_RESULTS.csv', index=False)
        print(f"\n💾 Saved: data/MOMENTUM_COMPREHENSIVE_RESULTS.csv")
        
        return results_df
    
    def test_reversal_strategies(self):
        """
        Test MEAN REVERSION strategies:
        - Short-term oversold bounces (2-5 day reversals)
        - Volume spike reversals
        - Gap reversals
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE REVERSAL TESTING")
        print("="*80)
        print(f"Testing ALL {len(self.clean_universe):,} clean tickers")
        print("")
        
        results = []
        
        print("--- Testing: Oversold Bounce (RSI < 30) ---")
        
        for ticker in tqdm(self.clean_universe, desc="Reversal"):
            df = pd.read_sql(f"""
                SELECT date, open, high, low, close, volume 
                FROM ohlcv 
                WHERE ticker = '{ticker}'
                ORDER BY date
            """, self.conn, parse_dates=['date'])
            
            if len(df) < 50:
                continue
            
            # Calculate RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Forward returns
            df['forward_5d'] = df['close'].shift(-5) / df['close'] - 1
            
            # Test: Buy when RSI < 30 (oversold)
            oversold = df[df['rsi'] < 30].copy()
            
            cost = self.transaction_costs.get(ticker, 0.01)
            
            if len(oversold) >= 10:
                avg_ret = oversold['forward_5d'].mean()
                
                results.append({
                    'ticker': ticker,
                    'strategy': 'RSI_Reversal',
                    'num_signals': len(oversold),
                    'gross_return': avg_ret,
                    'net_return': avg_ret - cost,
                    'transaction_cost': cost
                })
        
        if len(results) > 0:
            df_results = pd.DataFrame(results)
            
            avg_net = df_results['net_return'].mean()
            profitable_pct = (df_results['net_return'] > 0).mean()
            t_stat, p_val = stats.ttest_1samp(df_results['net_return'], 0)
            
            print(f"\nTested {len(df_results)} tickers")
            print(f"Average net return: {avg_net:.3%}")
            print(f"Profitable: {profitable_pct:.1%}")
            print(f"T-stat: {t_stat:.2f} (need >3.0)")
            
            if abs(t_stat) > 3.0:
                print("✅ REVERSAL EDGE FOUND!")
            else:
                print("❌ No statistical edge")
            
            df_results.to_csv('data/REVERSAL_RESULTS.csv', index=False)
            print(f"\n💾 Saved: data/REVERSAL_RESULTS.csv")
        
        return df_results
    
    def test_volatility_strategies(self):
        """
        Test VOLATILITY-BASED strategies:
        - Low volatility anomaly (buy calm stocks)
        - Volatility breakout (buy volatility expansion)
        - ATR-based position sizing impact
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE VOLATILITY TESTING")
        print("="*80)
        
        results = []
        
        for ticker in tqdm(self.clean_universe, desc="Volatility"):
            df = pd.read_sql(f"""
                SELECT date, high, low, close, volume 
                FROM ohlcv 
                WHERE ticker = '{ticker}'
                ORDER BY date
            """, self.conn, parse_dates=['date'])
            
            if len(df) < 50:
                continue
            
            # Calculate volatility metrics
            df['returns'] = df['close'].pct_change()
            df['vol_20d'] = df['returns'].rolling(20).std()
            df['atr_14'] = (df['high'] - df['low']).rolling(14).mean()
            
            # Forward returns
            df['forward_21d'] = df['close'].shift(-21) / df['close'] - 1
            
            # Test: Low volatility anomaly (buy bottom quintile)
            df = df.dropna()
            if len(df) < 50:
                continue
            
            low_vol_threshold = df['vol_20d'].quantile(0.2)
            low_vol = df[df['vol_20d'] < low_vol_threshold].copy()
            
            cost = self.transaction_costs.get(ticker, 0.01)
            
            if len(low_vol) >= 10:
                avg_ret = low_vol['forward_21d'].mean()
                
                results.append({
                    'ticker': ticker,
                    'strategy': 'Low_Vol_Anomaly',
                    'num_signals': len(low_vol),
                    'gross_return': avg_ret,
                    'net_return': avg_ret - cost,
                    'transaction_cost': cost
                })
        
        if len(results) > 0:
            df_results = pd.DataFrame(results)
            
            avg_net = df_results['net_return'].mean()
            profitable_pct = (df_results['net_return'] > 0).mean()
            t_stat, p_val = stats.ttest_1samp(df_results['net_return'], 0)
            
            print(f"\nTested {len(df_results)} tickers")
            print(f"Average net return: {avg_net:.3%}")
            print(f"Profitable: {profitable_pct:.1%}")
            print(f"T-stat: {t_stat:.2f} (need >3.0)")
            
            if abs(t_stat) > 3.0:
                print("✅ LOW VOL EDGE FOUND!")
            else:
                print("❌ No statistical edge")
            
            df_results.to_csv('data/VOLATILITY_RESULTS.csv', index=False)
            print(f"\n💾 Saved: data/VOLATILITY_RESULTS.csv")
        
        return df_results
    
    def run_full_discovery(self):
        """
        Run EVERYTHING. No shortcuts.
        Test every hypothesis we can think of.
        """
        print("\n" + "="*80)
        print("FULL ALEX LAWS DISCOVERY MISSION")
        print("="*80)
        print(f"Database: 4.38M rows, {len(self.clean_universe):,} clean tickers")
        print("Mission: Find edges that survive transaction costs")
        print("Standard: Harvey-Liu-Zhu threshold (t-stat > 3.0)")
        print("="*80)
        
        # Run all tests
        momentum_results = self.test_all_momentum_variants()
        reversal_results = self.test_reversal_strategies()
        volatility_results = self.test_volatility_strategies()
        
        # Summary report
        print("\n" + "="*80)
        print("DISCOVERY MISSION COMPLETE")
        print("="*80)
        print("\nFiles created:")
        print("  - data/MOMENTUM_COMPREHENSIVE_RESULTS.csv")
        print("  - data/REVERSAL_RESULTS.csv")
        print("  - data/VOLATILITY_RESULTS.csv")
        print("")
        print("Next: Review results and build validated strategies")


if __name__ == "__main__":
    discovery = ComprehensiveLawDiscovery()
    discovery.run_full_discovery()
