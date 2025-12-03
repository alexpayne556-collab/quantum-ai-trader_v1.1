"""
🎓 TRAIN SYSTEM WHILE OPTIMIZING
=================================
Run this to train your current 9 modules while you work on:
- Optimizing the 26 additional modules
- Building the AI Meta-Recommender
- Researching best practices

This will run in the background and generate trained weights!
"""

import sys
from pathlib import Path

print("="*80)
print("🎓 TRAINING INSTITUTIONAL SYSTEM")
print("="*80)

# Setup
MODULES_DIR = Path('/content/drive/MyDrive/QuantumAI/backend/modules')
sys.path.insert(0, str(MODULES_DIR))

# Clear cache
for module in list(sys.modules.keys()):
    if any(x in module.lower() for x in ['backtest', 'ensemble', 'institutional']):
        del sys.modules[module]

print("\n📦 Importing modules...")
from BACKTEST_INSTITUTIONAL_ENSEMBLE import BacktestEngine

print("✅ Modules imported\n")

print("="*80)
print("🧪 STARTING BACKTEST")
print("="*80)
print("\n⏱️  This will take 5-10 minutes...")
print("💡 Progress will show below")
print("💡 You can work on other things while this runs!\n")

try:
    # Create backtest engine
    backtest = BacktestEngine()
    
    # Run backtest
    results = backtest.run_backtest()
    
    if results:
        print("\n" + "="*80)
        print("✅ BACKTEST COMPLETE!")
        print("="*80)
        
        # Print results
        backtest.print_results(results)
        
        # Save everything
        backtest.save_results(results)
        backtest.ensemble.save_weights('ensemble_weights.json')
        
        print("\n" + "="*80)
        print("📁 SAVED FILES")
        print("="*80)
        print("✅ backtest_results.json - Performance metrics")
        print("✅ backtest_trades.csv - All trades logged")
        print("✅ ensemble_weights.json - Trained weights!")
        
        print("\n" + "="*80)
        print("📊 QUICK SUMMARY")
        print("="*80)
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.1%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.1%}")
        print(f"Total Return: {results['total_return']:.2f}%")
        
        if results['win_rate'] >= 0.60:
            print("\n🎉 TARGET MET! System achieved 60%+ win rate!")
        else:
            print(f"\n🎯 Building up... System will improve with more trades")
            print(f"   Currently: {results['win_rate']:.1%} → Target: 60-70%")
        
        print("\n" + "="*80)
        print("🚀 NEXT STEPS")
        print("="*80)
        print("1. ✅ Review results in backtest_results.json")
        print("2. ✅ Check trade details in backtest_trades.csv")
        print("3. ✅ Load trained weights in dashboard")
        print("4. ✅ Continue optimizing other modules")
        print("5. ✅ Integrate AI Meta-Recommender")
        
except Exception as e:
    print(f"\n❌ Backtest error: {str(e)}")
    print("\nTroubleshooting:")
    print("1. Check all files are uploaded")
    print("2. Verify no syntax errors")
    print("3. Try clearing cache and rerunning")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("Training complete! Continue your optimization work.")
print("="*80)

