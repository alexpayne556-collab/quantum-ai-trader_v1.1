"""
COLAB TEST SCRIPT - 5 CORE MODULES ONLY
========================================
Tests the 5 core modules in Colab before optimization
NO scraping, NO dashboard - just the core trading system
"""

# ============================================================================
# SETUP - RUN THIS FIRST IN COLAB
# ============================================================================

from google.colab import drive
drive.mount('/content/drive')

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set project path
PROJECT_PATH = '/content/drive/MyDrive/Quantum_AI_Cockpit'
sys.path.insert(0, f'{PROJECT_PATH}/backend/modules')

print("="*80)
print("COLAB TEST - 5 CORE MODULES")
print("="*80)
print(f"📁 Project path: {PROJECT_PATH}")

# ============================================================================
# IMPORT ONLY THE 5 CORE MODULES
# ============================================================================

print("\n[1/5] Importing core modules...")

try:
    from scanner_pro import ScannerPro
    print("  ✅ scanner_pro imported")
except Exception as e:
    print(f"  ❌ scanner_pro failed: {e}")
    raise

try:
    from risk_manager_pro import RiskManagerPro
    print("  ✅ risk_manager_pro imported")
except Exception as e:
    print(f"  ❌ risk_manager_pro failed: {e}")
    raise

try:
    from backtest_engine import BacktestEngine
    print("  ✅ backtest_engine imported")
except Exception as e:
    print(f"  ❌ backtest_engine failed: {e}")
    raise

try:
    from master_coordinator_pro_FIXED import MasterCoordinatorProFixed
    print("  ✅ master_coordinator_pro_FIXED imported")
except Exception as e:
    print(f"  ❌ master_coordinator_pro_FIXED failed: {e}")
    raise

try:
    from production_trading_system import ProductionTradingSystem
    print("  ✅ production_trading_system imported")
except Exception as e:
    print(f"  ❌ production_trading_system failed: {e}")
    raise

print("\n✅ All 5 core modules imported successfully!")

# ============================================================================
# LOAD OR CREATE TEST DATA
# ============================================================================

print("\n[2/5] Loading data...")

data_file = f'{PROJECT_PATH}/data/daily_data.parquet'

try:
    data = pd.read_parquet(data_file)
    print(f"  ✅ Loaded {len(data):,} records from {data_file}")
    print(f"     Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"     Tickers: {data['ticker'].nunique()}")
    print(f"     Columns: {list(data.columns)}")
except Exception as e:
    print(f"  ⚠️  Error loading data: {e}")
    print("  📊 Creating sample test data...")
    
    # Create realistic sample data with oversold conditions
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    
    # Create multiple tickers
    tickers = ['TEST1', 'TEST2', 'TEST3', 'TEST4', 'TEST5']
    all_data = []
    
    for ticker in tickers:
        # Create price data with some oversold conditions
        base_price = 100 + np.random.randn() * 10
        prices = base_price + np.cumsum(np.random.randn(100) * 2)
        
        # Make some periods oversold (price drops below mean)
        for i in range(30, 50):
            prices[i] = prices[i] * 0.92  # Drop 8% below mean
        
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
    print(f"  ✅ Created sample data: {len(data):,} records, {data['ticker'].nunique()} tickers")

# ============================================================================
# TEST 1: SCANNER
# ============================================================================

print("\n" + "="*80)
print("[TEST 1/5] TESTING SCANNER_PRO")
print("="*80)

try:
    scanner = ScannerPro()
    signals_df = scanner.scan_mean_reversion(data)
    
    print(f"✅ Scanner works!")
    print(f"   Signals found: {len(signals_df)}")
    
    if len(signals_df) > 0:
        print(f"\n   Sample signals:")
        print(signals_df[['ticker', 'entry_price', 'target_price', 'expected_return', 'rsi']].head().to_string())
    else:
        print("   ⚠️  No signals found (may need more oversold conditions in data)")
    
    SCANNER_WORKS = True
except Exception as e:
    print(f"❌ Scanner failed: {e}")
    import traceback
    traceback.print_exc()
    SCANNER_WORKS = False

# ============================================================================
# TEST 2: RISK MANAGER
# ============================================================================

print("\n" + "="*80)
print("[TEST 2/5] TESTING RISK_MANAGER_PRO")
print("="*80)

try:
    risk_mgr = RiskManagerPro()
    
    # Test with mock signal
    mock_signal = {
        'ticker': 'TEST',
        'entry_price': 100.0,
        'stop_loss': 95.0,
        'target_price': 105.0,
        'expected_return': 0.05
    }
    
    position_info = risk_mgr.calculate_position_size(mock_signal, account_value=1000.0)
    
    print(f"✅ Risk Manager works!")
    print(f"   Entry Price: ${mock_signal['entry_price']:.2f}")
    print(f"   Stop Loss: ${mock_signal['stop_loss']:.2f}")
    print(f"   Shares: {position_info['shares']}")
    print(f"   Position Value: ${position_info['position_value']:.2f}")
    print(f"   Max Loss: ${position_info['max_loss']:.2f}")
    print(f"   Risk %: {position_info['risk_pct']:.2f}%")
    
    RISK_MANAGER_WORKS = True
except Exception as e:
    print(f"❌ Risk Manager failed: {e}")
    import traceback
    traceback.print_exc()
    RISK_MANAGER_WORKS = False

# ============================================================================
# TEST 3: BACKTEST ENGINE
# ============================================================================

print("\n" + "="*80)
print("[TEST 3/5] TESTING BACKTEST_ENGINE")
print("="*80)

try:
    backtester = BacktestEngine(scanner=scanner)
    results = backtester.backtest_mean_reversion(data, lookback_days=30)
    
    if results:
        print(f"✅ Backtest Engine works!")
        print(f"   Trades: {results['trades']}")
        print(f"   Win Rate: {results['win_rate']:.1%}")
        print(f"   Avg Win: {results['avg_win']:.2%}")
        print(f"   Avg Loss: {results['avg_loss']:.2%}")
        print(f"   Expectancy: {results['expectancy']:.2%}")
        
        if results['trades'] > 0:
            print(f"\n   Sample trades:")
            print(results['trade_data'].head().to_string())
    else:
        print("✅ Backtest Engine works!")
        print("   ⚠️  No trades found in test period (expected with sample data)")
    
    BACKTEST_WORKS = True
except Exception as e:
    print(f"❌ Backtest Engine failed: {e}")
    import traceback
    traceback.print_exc()
    BACKTEST_WORKS = False

# ============================================================================
# TEST 4: MASTER COORDINATOR
# ============================================================================

print("\n" + "="*80)
print("[TEST 4/5] TESTING MASTER_COORDINATOR")
print("="*80)

try:
    coordinator = MasterCoordinatorProFixed()
    
    # Save test data for coordinator
    import os
    os.makedirs(f'{PROJECT_PATH}/data', exist_ok=True)
    data.to_parquet(f'{PROJECT_PATH}/data/daily_data.parquet')
    
    signals = coordinator.run_daily()
    
    print(f"✅ Master Coordinator works!")
    print(f"   Signals generated: {len(signals)}")
    
    if len(signals) > 0:
        print(f"\n   Sample signals:")
        for i, sig in enumerate(signals[:3], 1):
            print(f"   {i}. {sig.get('ticker', 'UNKNOWN')}: ${sig.get('entry_price', 0):.2f}")
    else:
        print("   ⚠️  No signals generated (may need more data or oversold conditions)")
    
    COORDINATOR_WORKS = True
except Exception as e:
    print(f"❌ Master Coordinator failed: {e}")
    import traceback
    traceback.print_exc()
    COORDINATOR_WORKS = False

# ============================================================================
# TEST 5: PRODUCTION TRADING SYSTEM
# ============================================================================

print("\n" + "="*80)
print("[TEST 5/5] TESTING PRODUCTION_TRADING_SYSTEM")
print("="*80)

try:
    prod_system = ProductionTradingSystem()
    
    mock_signals = [
        {
            'ticker': 'TEST1',
            'entry_price': 100,
            'shares': 10,
            'target_price': 105,
            'stop_loss': 95,
            'expected_return': 0.05,
            'strategy': 'mean_reversion',
            'position_value': 1000,
            'max_loss': 50,
            'risk_pct': 5
        },
        {
            'ticker': 'TEST2',
            'entry_price': 50,
            'shares': 20,
            'target_price': 52,
            'stop_loss': 48,
            'expected_return': 0.04,
            'strategy': 'mean_reversion',
            'position_value': 1000,
            'max_loss': 40,
            'risk_pct': 4
        }
    ]
    
    prod_system.paper_trade(mock_signals)
    
    print(f"✅ Production Trading System works!")
    print(f"   Paper trades logged: {len(mock_signals)}")
    
    # Check if log file was created
    log_file = f'{PROJECT_PATH}/logs/paper_trades.txt'
    if os.path.exists(log_file):
        print(f"   Log file created: {log_file}")
        with open(log_file, 'r') as f:
            lines = f.readlines()
            print(f"   Log entries: {len(lines)}")
    
    PRODUCTION_WORKS = True
except Exception as e:
    print(f"❌ Production Trading System failed: {e}")
    import traceback
    traceback.print_exc()
    PRODUCTION_WORKS = False

# ============================================================================
# FINAL RESULTS
# ============================================================================

print("\n" + "="*80)
print("FINAL TEST RESULTS")
print("="*80)

results = {
    'Scanner': SCANNER_WORKS,
    'Risk Manager': RISK_MANAGER_WORKS,
    'Backtest Engine': BACKTEST_WORKS,
    'Master Coordinator': COORDINATOR_WORKS,
    'Production System': PRODUCTION_WORKS
}

for module, status in results.items():
    status_icon = "✅" if status else "❌"
    print(f"{status_icon} {module}: {'PASS' if status else 'FAIL'}")

all_passed = all(results.values())

print("\n" + "="*80)
if all_passed:
    print("✅ ALL 5 MODULES PASSED - SYSTEM READY FOR OPTIMIZATION!")
    print("\n📋 NEXT STEPS:")
    print("   1. Run parameter optimization")
    print("   2. Test on different time periods")
    print("   3. Calculate performance metrics")
    print("   4. Generate optimization report")
else:
    print("❌ SOME MODULES FAILED - FIX ERRORS BEFORE OPTIMIZATION")
    print("\n⚠️  Review error messages above and fix issues")

print("="*80)

