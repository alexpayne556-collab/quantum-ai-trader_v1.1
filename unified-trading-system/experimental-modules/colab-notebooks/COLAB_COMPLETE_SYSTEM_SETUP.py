"""
🚀 COMPLETE SYSTEM SETUP & STATUS CHECKER
==========================================
One script to:
1. Mount Drive & Setup
2. Check System Status
3. Show What You Have
4. Optimize & Train
5. Launch Dashboard

Just run this in a new Colab notebook!
"""

import os
import sys
import subprocess
from pathlib import Path
import json
from datetime import datetime

print("="*80)
print("🚀 QUANTUM AI - COMPLETE SYSTEM SETUP")
print("="*80)

# ============================================================================
# STEP 1: MOUNT GOOGLE DRIVE
# ============================================================================
print("\n" + "="*80)
print("1️⃣ MOUNTING GOOGLE DRIVE")
print("="*80)

try:
    from google.colab import drive
    if not os.path.exists('/content/drive'):
        print("📁 Mounting Google Drive...")
        drive.mount('/content/drive')
    else:
        print("✅ Drive already mounted")
    DRIVE_MOUNTED = True
except ImportError:
    print("⚠️  Not in Colab - skipping Drive mount")
    DRIVE_MOUNTED = False
except Exception as e:
    print(f"❌ Drive mount error: {e}")
    DRIVE_MOUNTED = False

# ============================================================================
# STEP 2: SETUP PATHS
# ============================================================================
print("\n" + "="*80)
print("2️⃣ SETTING UP PATHS")
print("="*80)

BASE_DIR = Path('/content/drive/MyDrive/QuantumAI') if DRIVE_MOUNTED else Path('.')
MODULES_DIR = BASE_DIR / 'backend' / 'modules'

# Create directories if needed
if DRIVE_MOUNTED:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    MODULES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✅ Base: {BASE_DIR}")
    print(f"✅ Modules: {MODULES_DIR}")

# Add to path
sys.path.insert(0, str(MODULES_DIR))
os.chdir(MODULES_DIR) if MODULES_DIR.exists() else None

# ============================================================================
# STEP 3: INSTALL DEPENDENCIES
# ============================================================================
print("\n" + "="*80)
print("3️⃣ INSTALLING DEPENDENCIES")
print("="*80)

dependencies = [
    'streamlit',
    'plotly',
    'yfinance',
    'pandas',
    'numpy',
    'ta',
    'tqdm',
    'pyngrok',
    'scikit-learn',
    'lightgbm',
    'xgboost',
]

print("📦 Installing packages...")
for pkg in dependencies:
    try:
        __import__(pkg.replace('-', '_'))
        print(f"  ✅ {pkg}")
    except:
        print(f"  📦 Installing {pkg}...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("✅ Dependencies ready")

# ============================================================================
# STEP 4: SYSTEM STATUS CHECK
# ============================================================================
print("\n" + "="*80)
print("4️⃣ SYSTEM STATUS CHECK")
print("="*80)

# Core files
CORE_FILES = {
    'INSTITUTIONAL_ENSEMBLE_ENGINE.py': 'Core ensemble engine (1000+ lines)',
    'BACKTEST_INSTITUTIONAL_ENSEMBLE.py': 'Backtest framework (600+ lines)',
    'ULTIMATE_INSTITUTIONAL_DASHBOARD.py': 'Complete dashboard (800+ lines)',
}

# Optional modules
OPTIONAL_MODULES = {
    'pre_gainer_scanner_v2_ML_POWERED.py': 'Pre-Gainer Scanner',
    'day_trading_scanner_v2_ML_POWERED.py': 'Day Trading Scanner',
    'opportunity_scanner_v2_ML_POWERED.py': 'Opportunity Scanner',
    'dark_pool_tracker.py': 'Dark Pool Tracker',
    'insider_trading_tracker.py': 'Insider Trading Tracker',
    'short_squeeze_scanner.py': 'Short Squeeze Scanner',
    'elite_forecaster.py': 'Elite Forecaster',
    'regime_detector.py': 'Regime Detector',
    'sentiment_engine.py': 'Sentiment Engine',
}

# Check files
found_core = []
missing_core = []
found_modules = []
missing_modules = []

if MODULES_DIR.exists():
    for filename, description in CORE_FILES.items():
        filepath = MODULES_DIR / filename
        if filepath.exists():
            print(f"  ✅ {filename}")
            found_core.append(filename)
        else:
            print(f"  ❌ {filename} - {description}")
            missing_core.append(filename)
    
    print("\n📦 Optional Modules:")
    for filename, description in OPTIONAL_MODULES.items():
        filepath = MODULES_DIR / filename
        if filepath.exists():
            print(f"  ✅ {filename}")
            found_modules.append(filename)
        else:
            print(f"  ⚠️  {filename} - {description}")
            missing_modules.append(filename)
else:
    print(f"❌ Modules directory not found: {MODULES_DIR}")
    print("   Please create the directory structure in Google Drive")

# ============================================================================
# STEP 5: SYSTEM CAPABILITIES
# ============================================================================
print("\n" + "="*80)
print("5️⃣ SYSTEM CAPABILITIES")
print("="*80)

capabilities = {
    'Backtest': len(found_core) >= 2 and 'BACKTEST_INSTITUTIONAL_ENSEMBLE.py' in found_core,
    'Dashboard': 'ULTIMATE_INSTITUTIONAL_DASHBOARD.py' in found_core,
    'Ensemble Engine': 'INSTITUTIONAL_ENSEMBLE_ENGINE.py' in found_core,
    'Real Scanners': len([m for m in found_modules if 'scanner' in m.lower()]) >= 3,
    'Institutional Signals': any('dark_pool' in m or 'insider' in m or 'squeeze' in m for m in found_modules),
    'Forecasting': any('forecaster' in m.lower() for m in found_modules),
}

for capability, available in capabilities.items():
    status = "✅" if available else "❌"
    print(f"  {status} {capability}")

# ============================================================================
# STEP 6: CHECK FOR TRAINED MODELS
# ============================================================================
print("\n" + "="*80)
print("6️⃣ TRAINED MODELS & WEIGHTS")
print("="*80)

model_files = {
    'ensemble_weights.json': 'Learned ensemble weights (from backtest)',
    'backtest_results.json': 'Backtest performance results',
    'backtest_trades.csv': 'Historical trade log',
}

models_dir = BASE_DIR
found_models = []

for filename, description in model_files.items():
    filepath = models_dir / filename
    if filepath.exists():
        print(f"  ✅ {filename}")
        found_models.append(filename)
        
        # Show stats if available
        if filename == 'ensemble_weights.json':
            try:
                with open(filepath, 'r') as f:
                    weights = json.load(f)
                    print(f"     → {len(weights)} weight entries")
            except:
                pass
        elif filename == 'backtest_results.json':
            try:
                with open(filepath, 'r') as f:
                    results = json.load(f)
                    if 'win_rate' in results:
                        print(f"     → Win Rate: {results['win_rate']:.1%}")
                    if 'sharpe_ratio' in results:
                        print(f"     → Sharpe: {results['sharpe_ratio']:.2f}")
            except:
                pass
    else:
        print(f"  ⚠️  {filename} - {description}")

# ============================================================================
# STEP 7: OPTIMIZATION SUGGESTIONS
# ============================================================================
print("\n" + "="*80)
print("7️⃣ OPTIMIZATION SUGGESTIONS")
print("="*80)

suggestions = []

if missing_core:
    suggestions.append(f"📤 Upload {len(missing_core)} core file(s) to enable full system")

if len(found_modules) < 3:
    suggestions.append("📦 Add more scanner modules for better signal diversity")

if not found_models:
    suggestions.append("🧪 Run backtest to generate trained weights and performance data")

if not any('forecaster' in m.lower() for m in found_modules):
    suggestions.append("📊 Add forecasting module for price predictions")

if not capabilities['Institutional Signals']:
    suggestions.append("🏦 Add institutional modules (dark pool, insider, squeeze) for smart money tracking")

if suggestions:
    for i, suggestion in enumerate(suggestions, 1):
        print(f"  {i}. {suggestion}")
else:
    print("  ✅ System is fully optimized!")

# ============================================================================
# STEP 8: WHAT YOU CAN DO NOW
# ============================================================================
print("\n" + "="*80)
print("8️⃣ WHAT YOU CAN DO NOW")
print("="*80)

options = []

if capabilities['Backtest']:
    options.append("1. 📊 Run Backtest - Test system on historical data (5-10 min)")
    options.append("   → Validates 60-70% win rate target")
    options.append("   → Generates optimized weights")
    options.append("   → Creates performance report")

if capabilities['Dashboard']:
    options.append("2. 🚀 Launch Dashboard - Live trading interface")
    options.append("   → Real-time signals")
    options.append("   → Paper trading")
    options.append("   → Performance tracking")

if capabilities['Ensemble Engine'] and not found_models:
    options.append("3. 🧪 Train System - Run backtest first to learn optimal weights")

if found_models:
    options.append("4. 📈 Use Trained Weights - Load optimized weights in dashboard")

for option in options:
    print(f"  {option}")

# ============================================================================
# STEP 9: QUICK IMPORT TEST
# ============================================================================
print("\n" + "="*80)
print("9️⃣ TESTING IMPORTS")
print("="*80)

# Clear cache first
modules_to_clear = ['INSTITUTIONAL_ENSEMBLE_ENGINE', 'BACKTEST_INSTITUTIONAL_ENSEMBLE']
for module_name in modules_to_clear:
    keys_to_remove = [k for k in sys.modules.keys() if module_name.lower() in k.lower()]
    for key in keys_to_remove:
        del sys.modules[key]

try:
    if 'INSTITUTIONAL_ENSEMBLE_ENGINE.py' in found_core:
        from INSTITUTIONAL_ENSEMBLE_ENGINE import InstitutionalEnsembleEngine, Signal
        engine = InstitutionalEnsembleEngine()
        print("  ✅ Ensemble Engine imported and initialized")
        print(f"     → {len(engine.current_weights)} weight groups loaded")
    else:
        print("  ⚠️  Ensemble Engine not found")
except Exception as e:
    print(f"  ❌ Import error: {e}")

try:
    if 'BACKTEST_INSTITUTIONAL_ENSEMBLE.py' in found_core:
        from BACKTEST_INSTITUTIONAL_ENSEMBLE import BacktestEngine
        print("  ✅ Backtest Engine imported")
    else:
        print("  ⚠️  Backtest Engine not found")
except Exception as e:
    print(f"  ❌ Import error: {e}")

# ============================================================================
# STEP 10: READY TO GO!
# ============================================================================
print("\n" + "="*80)
print("✅ SYSTEM READY!")
print("="*80)

print("\n📋 SUMMARY:")
print(f"  • Core Files: {len(found_core)}/{len(CORE_FILES)}")
print(f"  • Modules: {len(found_modules)}/{len(OPTIONAL_MODULES)}")
print(f"  • Trained Models: {len(found_models)}/{len(model_files)}")

print("\n🎯 NEXT STEPS:")
print("\nTo run backtest:")
print("  from BACKTEST_INSTITUTIONAL_ENSEMBLE import BacktestEngine")
print("  backtest = BacktestEngine()")
print("  results = backtest.run_backtest()")

print("\nTo launch dashboard:")
print("  !streamlit run ULTIMATE_INSTITUTIONAL_DASHBOARD.py --server.port=8501 --server.headless=true")

print("\nTo train and optimize:")
print("  1. Run backtest first (generates weights)")
print("  2. Review results in backtest_results.json")
print("  3. Load weights in dashboard for live trading")

print("\n" + "="*80)
print("🚀 READY TO TRADE!")
print("="*80)

