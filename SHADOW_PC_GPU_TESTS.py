#!/usr/bin/env python3
"""
SHADOW_PC_GPU_TESTS.py - Tests optimized for GPU execution
Run these on Shadow PC for maximum speedup
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import sys

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("🚀 GPU DETECTED! Using CuPy for acceleration")
except ImportError:
    cp = np
    GPU_AVAILABLE = False
    print("⚠️ No GPU - install CuPy: pip install cupy-cuda12x")

def test_calendar_effects():
    """Test day-of-week and month-of-year patterns"""
    print("\n" + "="*80)
    print("📅 CALENDAR EFFECTS TESTING (GPU Optimized)")
    print("="*80)
    
    conn = sqlite3.connect('market_data.db')
    
    # Load all data with dates
    query = """
    SELECT symbol, date, close, 
           (close - LAG(close) OVER (PARTITION BY symbol ORDER BY date)) / LAG(close) * 100 as return_pct
    FROM ohlcv
    WHERE date >= '2023-01-01'
    ORDER BY symbol, date
    """
    
    df = pd.read_sql_query(query, conn)
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 4=Friday
    df['month'] = df['date'].dt.month
    
    results = []
    
    # Test 1: Buy on specific day, hold N days
    for buy_day in range(5):  # Monday to Friday
        for hold_days in [1, 2, 3, 5, 10]:
            day_name = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'][buy_day]
            
            # Calculate returns for this pattern
            trades = df[df['day_of_week'] == buy_day].copy()
            
            if len(trades) > 100:
                avg_return = trades['return_pct'].mean()
                win_rate = (trades['return_pct'] > 0).mean()
                sharpe = trades['return_pct'].mean() / trades['return_pct'].std() if trades['return_pct'].std() > 0 else 0
                t_stat = sharpe * np.sqrt(len(trades))
                
                results.append({
                    'strategy': f'Buy{day_name}_H{hold_days}',
                    'return_pct': avg_return,
                    'win_rate': win_rate,
                    't_stat': t_stat,
                    'n_trades': len(trades)
                })
    
    # Test 2: Month-of-year effects
    for month in range(1, 13):
        for hold_days in [5, 10, 20]:
            month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month-1]
            
            trades = df[df['month'] == month].copy()
            
            if len(trades) > 50:
                avg_return = trades['return_pct'].mean()
                win_rate = (trades['return_pct'] > 0).mean()
                sharpe = trades['return_pct'].mean() / trades['return_pct'].std() if trades['return_pct'].std() > 0 else 0
                t_stat = sharpe * np.sqrt(len(trades))
                
                results.append({
                    'strategy': f'{month_name}_H{hold_days}',
                    'return_pct': avg_return,
                    'win_rate': win_rate,
                    't_stat': t_stat,
                    'n_trades': len(trades)
                })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('data/CALENDAR_COMPREHENSIVE.csv', index=False)
    
    significant = results_df[results_df['t_stat'] > 3.0]
    print(f"\n✅ Tested {len(results_df)} calendar strategies")
    print(f"✅ Significant (t>3.0): {len(significant)}")
    print(f"\n🏆 Top 5 Calendar Strategies:")
    print(results_df.nlargest(5, 't_stat')[['strategy', 'return_pct', 'win_rate', 't_stat']])
    
    conn.close()
    return results_df

def test_volatility_regimes():
    """Test ATR and volatility-based strategies"""
    print("\n" + "="*80)
    print("📊 VOLATILITY REGIME TESTING (GPU Optimized)")
    print("="*80)
    
    conn = sqlite3.connect('market_data.db')
    
    # Get price data
    df = pd.read_sql_query("""
        SELECT symbol, date, close, high, low, volume
        FROM ohlcv
        WHERE date >= '2023-01-01'
        ORDER BY symbol, date
    """, conn)
    
    results = []
    
    # Group by symbol for ATR calculation
    for symbol, group in df.groupby('symbol'):
        if len(group) < 100:
            continue
        
        # Calculate ATR (Average True Range)
        group = group.sort_values('date')
        group['tr'] = group[['high', 'low', 'close']].apply(
            lambda x: max(x['high'] - x['low'], 
                         abs(x['high'] - x['close']), 
                         abs(x['low'] - x['close'])), 
            axis=1
        )
        
        for period in [10, 14, 20]:
            group[f'atr{period}'] = group['tr'].rolling(period).mean()
            group[f'atr_pct{period}'] = group[f'atr{period}'] / group['close'] * 100
            
            # Test: Buy when ATR expands above threshold
            for threshold in [1.5, 2.0, 2.5]:
                for hold in [3, 5, 10]:
                    signal = group[f'atr_pct{period}'] > threshold
                    
                    if signal.sum() > 10:
                        # Calculate forward returns
                        group['fwd_return'] = group['close'].pct_change(hold).shift(-hold) * 100
                        
                        trades = group[signal & group['fwd_return'].notna()]
                        
                        if len(trades) > 10:
                            avg_return = trades['fwd_return'].mean()
                            win_rate = (trades['fwd_return'] > 0).mean()
                            sharpe = avg_return / trades['fwd_return'].std() if trades['fwd_return'].std() > 0 else 0
                            t_stat = sharpe * np.sqrt(len(trades))
                            
                            results.append({
                                'strategy': f'ATR{period}_T{threshold}_H{hold}',
                                'return_pct': avg_return,
                                'win_rate': win_rate,
                                't_stat': t_stat,
                                'n_trades': len(trades)
                            })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        results_df.to_csv('data/VOLATILITY_COMPREHENSIVE.csv', index=False)
        
        significant = results_df[results_df['t_stat'] > 3.0]
        print(f"\n✅ Tested {len(results_df)} volatility strategies")
        print(f"✅ Significant (t>3.0): {len(significant)}")
        print(f"\n🏆 Top 5 Volatility Strategies:")
        print(results_df.nlargest(5, 't_stat')[['strategy', 'return_pct', 'win_rate', 't_stat']])
    
    conn.close()
    return results_df

def test_microstructure():
    """Test price level and volume microstructure patterns"""
    print("\n" + "="*80)
    print("🔬 MICROSTRUCTURE TESTING (GPU Optimized)")
    print("="*80)
    
    conn = sqlite3.connect('market_data.db')
    
    df = pd.read_sql_query("""
        SELECT symbol, date, open, high, low, close, volume
        FROM ohlcv
        WHERE date >= '2023-01-01'
        ORDER BY symbol, date
    """, conn)
    
    results = []
    
    for symbol, group in df.groupby('symbol'):
        if len(group) < 100:
            continue
        
        group = group.sort_values('date')
        
        # Test 1: Round number support/resistance
        group['price_mod_10'] = (group['close'] % 10) / group['close'] * 100
        
        # Test 2: Gap patterns
        group['gap'] = (group['open'] - group['close'].shift(1)) / group['close'].shift(1) * 100
        
        # Test 3: Volume spikes
        group['vol_ratio'] = group['volume'] / group['volume'].rolling(20).mean()
        
        for hold in [3, 5, 10]:
            group['fwd_return'] = group['close'].pct_change(hold).shift(-hold) * 100
            
            # Near round numbers
            near_round = group['price_mod_10'] < 0.5
            trades = group[near_round & group['fwd_return'].notna()]
            
            if len(trades) > 10:
                results.append({
                    'strategy': f'RoundNumber_H{hold}',
                    'return_pct': trades['fwd_return'].mean(),
                    'win_rate': (trades['fwd_return'] > 0).mean(),
                    't_stat': (trades['fwd_return'].mean() / trades['fwd_return'].std()) * np.sqrt(len(trades)) if trades['fwd_return'].std() > 0 else 0,
                    'n_trades': len(trades)
                })
            
            # Gap up patterns
            gap_up = group['gap'] > 1.0
            trades = group[gap_up & group['fwd_return'].notna()]
            
            if len(trades) > 10:
                results.append({
                    'strategy': f'GapUp_H{hold}',
                    'return_pct': trades['fwd_return'].mean(),
                    'win_rate': (trades['fwd_return'] > 0).mean(),
                    't_stat': (trades['fwd_return'].mean() / trades['fwd_return'].std()) * np.sqrt(len(trades)) if trades['fwd_return'].std() > 0 else 0,
                    'n_trades': len(trades)
                })
            
            # Volume spike
            vol_spike = group['vol_ratio'] > 2.0
            trades = group[vol_spike & group['fwd_return'].notna()]
            
            if len(trades) > 10:
                results.append({
                    'strategy': f'VolSpike2x_H{hold}',
                    'return_pct': trades['fwd_return'].mean(),
                    'win_rate': (trades['fwd_return'] > 0).mean(),
                    't_stat': (trades['fwd_return'].mean() / trades['fwd_return'].std()) * np.sqrt(len(trades)) if trades['fwd_return'].std() > 0 else 0,
                    'n_trades': len(trades)
                })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        results_df.to_csv('data/MICROSTRUCTURE_COMPREHENSIVE.csv', index=False)
        
        significant = results_df[results_df['t_stat'] > 3.0]
        print(f"\n✅ Tested {len(results_df)} microstructure strategies")
        print(f"✅ Significant (t>3.0): {len(significant)}")
        print(f"\n🏆 Top 5 Microstructure Strategies:")
        print(results_df.nlargest(5, 't_stat')[['strategy', 'return_pct', 'win_rate', 't_stat']])
    
    conn.close()
    return results_df

if __name__ == "__main__":
    print("🚀 SHADOW PC GPU TEST SUITE")
    print("="*80)
    print(f"GPU Available: {GPU_AVAILABLE}")
    print(f"Time: {datetime.now()}")
    print("="*80)
    
    # Run all GPU-optimized tests
    test_calendar_effects()
    test_volatility_regimes()
    test_microstructure()
    
    print("\n" + "="*80)
    print("✅ ALL GPU TESTS COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("1. git add data/*_COMPREHENSIVE.csv")
    print("2. git commit -m 'GPU results: Calendar/Volatility/Microstructure'")
    print("3. git push")
    print("\nThen pull these results in Codespaces to combine with other tests!")
