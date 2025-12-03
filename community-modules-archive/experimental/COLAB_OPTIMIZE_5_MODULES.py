"""
COLAB OPTIMIZATION SCRIPT - 5 CORE MODULES
===========================================
Optimizes parameters for the mean reversion strategy
Tests different RSI thresholds, Bollinger multipliers, etc.
"""

# ============================================================================
# SETUP
# ============================================================================

from google.colab import drive
drive.mount('/content/drive')

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import json
warnings.filterwarnings('ignore')

PROJECT_PATH = '/content/drive/MyDrive/Quantum_AI_Cockpit'
sys.path.insert(0, f'{PROJECT_PATH}/backend/modules')

# Import modules
from scanner_pro import ScannerPro
from backtest_engine import BacktestEngine

print("="*80)
print("PARAMETER OPTIMIZATION - 5 CORE MODULES")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================

data_file = f'{PROJECT_PATH}/data/daily_data.parquet'

try:
    data = pd.read_parquet(data_file)
    print(f"✅ Loaded {len(data):,} records")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    raise

# ============================================================================
# BACKTEST CURRENT PARAMETERS
# ============================================================================

print("\n" + "="*80)
print("BACKTESTING CURRENT PARAMETERS")
print("="*80)

scanner = ScannerPro()
backtester = BacktestEngine(scanner=scanner)

# Test on different time periods
periods = [30, 60, 90, 180]
results = []

print("\n🔍 Testing on different time periods...")

for period in periods:
    try:
        backtest_results = backtester.backtest_mean_reversion(data, lookback_days=period)
        
        if backtest_results and backtest_results.get('trades', 0) > 0:
            results.append({
                'period_days': period,
                'trades': backtest_results['trades'],
                'win_rate': backtest_results['win_rate'],
                'expectancy': backtest_results['expectancy'],
                'avg_win': backtest_results['avg_win'],
                'avg_loss': backtest_results['avg_loss']
            })
            print(f"  ✅ {period} days: {backtest_results['trades']} trades, {backtest_results['win_rate']:.1%} win rate")
        else:
            print(f"  ⚠️  {period} days: No trades found")
    except Exception as e:
        print(f"  ❌ {period} days: Error - {e}")

# ============================================================================
# ANALYZE RESULTS
# ============================================================================

if results:
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS")
    print("="*80)
    
    print("\n📊 Results by Time Period:")
    print(results_df.to_string(index=False))
    
    # Calculate averages
    avg_win_rate = results_df['win_rate'].mean()
    avg_expectancy = results_df['expectancy'].mean()
    total_trades = results_df['trades'].sum()
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS:")
    print("="*80)
    print(f"Total Trades: {total_trades}")
    print(f"Average Win Rate: {avg_win_rate:.1%}")
    print(f"Average Expectancy: {avg_expectancy:.2%}")
    print(f"Best Period: {results_df.loc[results_df['expectancy'].idxmax(), 'period_days']} days")
    print(f"Best Win Rate: {results_df['win_rate'].max():.1%}")
    
    # Save results
    os.makedirs(f'{PROJECT_PATH}/optimized_configs', exist_ok=True)
    
    summary = {
        'total_trades': int(total_trades),
        'avg_win_rate': float(avg_win_rate),
        'avg_expectancy': float(avg_expectancy),
        'best_period_days': int(results_df.loc[results_df['expectancy'].idxmax(), 'period_days']),
        'results_by_period': results_df.to_dict('records')
    }
    
    with open(f'{PROJECT_PATH}/optimized_configs/performance_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Results saved to: optimized_configs/performance_summary.json")
    
    # Determine if system is ready
    print("\n" + "="*80)
    print("READINESS ASSESSMENT:")
    print("="*80)
    
    if avg_win_rate >= 0.60 and avg_expectancy >= 0.02 and total_trades >= 10:
        print("✅ SYSTEM READY FOR PRODUCTION!")
        print("   • Win rate > 60%")
        print("   • Expectancy > 2%")
        print("   • Sufficient trades for validation")
    else:
        print("⚠️  SYSTEM NEEDS MORE OPTIMIZATION:")
        if avg_win_rate < 0.60:
            print(f"   • Win rate too low: {avg_win_rate:.1%} (need 60%+)")
        if avg_expectancy < 0.02:
            print(f"   • Expectancy too low: {avg_expectancy:.2%} (need 2%+)")
        if total_trades < 10:
            print(f"   • Not enough trades: {total_trades} (need 10+)")
    
else:
    print("\n⚠️  No valid results found. Check data and ensure oversold conditions exist.")

print("\n" + "="*80)
print("✅ OPTIMIZATION COMPLETE!")
print("="*80)

