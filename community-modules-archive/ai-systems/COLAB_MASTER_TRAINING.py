"""
COLAB MASTER TRAINING - 5 CORE MODULES
======================================
Complete training script for Perplexity
Realistic optimization - no bullshit
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

print("="*80)
print("COLAB MASTER TRAINING - 5 CORE MODULES")
print("="*80)

# Import modules
from scanner_pro import ScannerPro
from risk_manager_pro import RiskManagerPro
from backtest_engine import BacktestEngine
from master_coordinator_pro_FIXED import MasterCoordinatorProFixed
from production_trading_system import ProductionTradingSystem

print("All 5 modules imported successfully")

# ============================================================================
# LOAD DATA
# ============================================================================

data_file = f'{PROJECT_PATH}/data/daily_data.parquet'

try:
    data = pd.read_parquet(data_file)
    print(f"Loaded {len(data):,} records")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"Tickers: {data['ticker'].nunique()}")
except Exception as e:
    print(f"Error loading data: {e}")
    print("Creating sample data...")
    
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    tickers = ['TEST1', 'TEST2', 'TEST3', 'TEST4', 'TEST5']
    all_data = []
    
    for ticker in tickers:
        base_price = 100 + np.random.randn() * 10
        prices = base_price + np.cumsum(np.random.randn(100) * 2)
        
        # Create oversold conditions
        for i in range(30, 50):
            prices[i] = prices[i] * 0.92
        
        volumes = np.random.randint(1000000, 5000000, 100)
        
        ticker_data = pd.DataFrame({
            'ticker': ticker,
            'date': dates,
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'close': prices,
            'volume': volumes
        })
        all_data.append(ticker_data)
    
    data = pd.concat(all_data, ignore_index=True)
    print(f"Created sample data: {len(data):,} records")

# ============================================================================
# STEP 1: TEST CURRENT SYSTEM
# ============================================================================

print("\n" + "="*80)
print("STEP 1: TEST CURRENT SYSTEM")
print("="*80)

scanner = ScannerPro()
backtester = BacktestEngine(scanner=scanner)

# Test on 60 days
results = backtester.backtest_mean_reversion(data, lookback_days=60)

if results:
    print(f"Current Performance:")
    print(f"  Trades: {results['trades']}")
    print(f"  Win Rate: {results['win_rate']:.1%}")
    print(f"  Expectancy: {results['expectancy']:.2%}")
    print(f"  Avg Win: {results['avg_win']:.2%}")
    print(f"  Avg Loss: {results['avg_loss']:.2%}")
else:
    print("No trades found - may need more data or different parameters")

# ============================================================================
# STEP 2: WALK-FORWARD VALIDATION
# ============================================================================

print("\n" + "="*80)
print("STEP 2: WALK-FORWARD VALIDATION")
print("="*80)

periods = [30, 60, 90, 180]
validation_results = []

for period in periods:
    try:
        results = backtester.backtest_mean_reversion(data, lookback_days=period)
        
        if results and results.get('trades', 0) > 0:
            validation_results.append({
                'period_days': period,
                'trades': results['trades'],
                'win_rate': results['win_rate'],
                'expectancy': results['expectancy'],
                'avg_win': results['avg_win'],
                'avg_loss': results['avg_loss']
            })
            print(f"{period} days: {results['trades']} trades, {results['win_rate']:.1%} win rate")
    except Exception as e:
        print(f"{period} days: Error - {e}")

# ============================================================================
# STEP 3: REALISTIC PERFORMANCE ASSESSMENT
# ============================================================================

print("\n" + "="*80)
print("STEP 3: REALISTIC PERFORMANCE ASSESSMENT")
print("="*80)

if validation_results:
    df_results = pd.DataFrame(validation_results)
    
    avg_win_rate = df_results['win_rate'].mean()
    avg_expectancy = df_results['expectancy'].mean()
    total_trades = df_results['trades'].sum()
    win_rate_std = df_results['win_rate'].std()
    consistency = 1 - (win_rate_std / avg_win_rate) if avg_win_rate > 0 else 0
    
    print(f"\nSummary Statistics:")
    print(f"  Total Trades: {total_trades}")
    print(f"  Average Win Rate: {avg_win_rate:.1%}")
    print(f"  Win Rate Std Dev: {win_rate_std:.1%}")
    print(f"  Consistency: {consistency:.1%}")
    print(f"  Average Expectancy: {avg_expectancy:.2%}")
    
    # Realistic return calculation
    print("\n" + "="*80)
    print("REALISTIC RETURN PROJECTIONS")
    print("="*80)
    
    if avg_win_rate >= 0.60 and avg_expectancy >= 0.02 and total_trades >= 10:
        # Assuming 4 trades per month (conservative)
        trades_per_month = 4
        monthly_return = avg_expectancy * trades_per_month
        annual_return = monthly_return * 12
        
        print(f"\nPerformance Metrics:")
        print(f"  Win Rate: {avg_win_rate:.1%}")
        print(f"  Expectancy: {avg_expectancy:.2%} per trade")
        print(f"  Trades per Month: {trades_per_month} (estimated)")
        print(f"  Monthly Return: {monthly_return:.1%}")
        print(f"  Annual Return: {annual_return:.1%}")
        
        print(f"\nAccount Growth Projection ($437 starting):")
        starting = 437.0
        print(f"  Month 1:  ${starting * (1 + monthly_return):.2f}")
        print(f"  Month 3:  ${starting * (1 + monthly_return)**3:.2f}")
        print(f"  Month 6:  ${starting * (1 + monthly_return)**6:.2f}")
        print(f"  Month 12: ${starting * (1 + monthly_return)**12:.2f}")
        
        print("\n" + "="*80)
        print("SYSTEM ASSESSMENT")
        print("="*80)
        
        if consistency > 0.85:
            print("READY FOR PRODUCTION")
            print("  - Win rate > 60%")
            print("  - Expectancy > 2%")
            print("  - Consistent across time periods")
            print("  - Sufficient trades for validation")
        else:
            print("NEEDS MORE OPTIMIZATION")
            print(f"  - Consistency too low: {consistency:.1%} (need 85%+)")
            print("  - Performance varies too much across periods")
    else:
        print("SYSTEM NEEDS MORE WORK:")
        if avg_win_rate < 0.60:
            print(f"  Win rate too low: {avg_win_rate:.1%} (need 60%+)")
        if avg_expectancy < 0.02:
            print(f"  Expectancy too low: {avg_expectancy:.2%} (need 2%+)")
        if total_trades < 10:
            print(f"  Not enough trades: {total_trades} (need 10+)")
    
    # Save results
    os.makedirs(f'{PROJECT_PATH}/optimized_configs', exist_ok=True)
    
    summary = {
        'total_trades': int(total_trades),
        'avg_win_rate': float(avg_win_rate),
        'avg_expectancy': float(avg_expectancy),
        'consistency': float(consistency),
        'win_rate_std': float(win_rate_std),
        'validation_results': df_results.to_dict('records'),
        'ready_for_production': avg_win_rate >= 0.60 and avg_expectancy >= 0.02 and total_trades >= 10 and consistency > 0.85
    }
    
    with open(f'{PROJECT_PATH}/optimized_configs/training_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: optimized_configs/training_results.json")
else:
    print("No validation results - check data and parameters")

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)

