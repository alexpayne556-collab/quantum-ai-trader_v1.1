"""
🧪 TEST FIXED MODULES IN COLAB
================================
Tests each module to see if it can be imported
"""

import os
import sys

print("🧪 TESTING FIXED MODULES")
print("="*80)

# Setup paths
MODULES_DIR = '/content/drive/MyDrive/QuantumAI/backend/modules'
sys.path.insert(0, MODULES_DIR)
os.chdir(MODULES_DIR)

# Install dependencies
print("\n📦 Installing dependencies...")
import subprocess
subprocess.run(['pip', 'install', '-q', 'prophet', 'lightgbm', 'xgboost', 'ta', 'scikit-learn'], 
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("✅ Dependencies installed")

import importlib.util

# ============================================================================
# TEST 1: ELITE FORECASTER
# ============================================================================
print("\n1️⃣ Testing elite_forecaster.py...")
print("-" * 80)

try:
    spec = importlib.util.spec_from_file_location("elite_forecaster", "elite_forecaster.py")
    ef_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ef_module)
    
    print("✅ Module loaded successfully!")
    
    # Find class
    classes = [name for name in dir(ef_module) if not name.startswith('_')]
    class_names = [name for name in classes if 'Forecast' in name or 'Elite' in name]
    print(f"   Classes found: {class_names}")
    
    if hasattr(ef_module, 'EliteForecaster'):
        print("✅ Found EliteForecaster class")
        forecaster = ef_module.EliteForecaster()
        print(f"✅ Instantiated: {type(forecaster).__name__}")
    else:
        print("⚠️  EliteForecaster not found, but module loaded")
        
except Exception as e:
    print(f"❌ ERROR: {type(e).__name__}: {str(e)}")
    import traceback
    print("\nFirst 10 lines of traceback:")
    traceback.print_exc(limit=10)

# ============================================================================
# TEST 2: PRE-GAINER SCANNER
# ============================================================================
print("\n2️⃣ Testing pre_gainer_scanner_v2_ML_POWERED.py...")
print("-" * 80)

try:
    spec = importlib.util.spec_from_file_location("pre_gainer", "pre_gainer_scanner_v2_ML_POWERED.py")
    pg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pg_module)
    
    print("✅ Module loaded successfully!")
    
    if hasattr(pg_module, 'PreGainerScannerV2'):
        print("✅ Found PreGainerScannerV2 class")
        scanner = pg_module.PreGainerScannerV2()
        print(f"✅ Instantiated: {type(scanner).__name__}")
    else:
        classes = [name for name in dir(pg_module) if 'Scanner' in name or 'Gainer' in name]
        print(f"⚠️  Classes found: {classes}")
        
except Exception as e:
    print(f"❌ ERROR: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc(limit=10)

# ============================================================================
# TEST 3: DAY TRADING SCANNER
# ============================================================================
print("\n3️⃣ Testing day_trading_scanner_v2_ML_POWERED.py...")
print("-" * 80)

try:
    spec = importlib.util.spec_from_file_location("day_trading", "day_trading_scanner_v2_ML_POWERED.py")
    dt_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dt_module)
    
    print("✅ Module loaded successfully!")
    
    if hasattr(dt_module, 'DayTradingScannerV2'):
        print("✅ Found DayTradingScannerV2 class")
        scanner = dt_module.DayTradingScannerV2()
        print(f"✅ Instantiated: {type(scanner).__name__}")
    else:
        classes = [name for name in dir(dt_module) if 'Scanner' in name or 'Trading' in name]
        print(f"⚠️  Classes found: {classes}")
        
except Exception as e:
    print(f"❌ ERROR: {type(e).__name__}: {str(e)}")

# ============================================================================
# TEST 4-7: OTHER SCANNERS
# ============================================================================
print("\n4️⃣ Testing other scanners...")
print("-" * 80)

other_scanners = [
    ('swing_trading_scanner_v2_ML_POWERED.py', 'SwingTradingScannerV2'),
    ('momentum_tracker_v2_ML_POWERED.py', 'MomentumTrackerV2'),
    ('volume_breakout_scanner_v2_ML_POWERED.py', 'VolumeBreakoutScannerV2'),
    ('pattern_recognition_scanner_v2_ML_POWERED.py', 'PatternRecognitionScannerV2')
]

for scanner_file, class_name in other_scanners:
    if os.path.exists(scanner_file):
        try:
            module_name = scanner_file.replace('.py', '')
            spec = importlib.util.spec_from_file_location(module_name, scanner_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if hasattr(module, class_name):
                print(f"✅ {scanner_file}: Found {class_name}")
            else:
                print(f"⚠️  {scanner_file}: Loaded but {class_name} not found")
        except Exception as e:
            print(f"❌ {scanner_file}: {type(e).__name__}: {str(e)}")
    else:
        print(f"⚠️  {scanner_file}: File not found")

print("\n" + "="*80)
print("🎯 SUMMARY")
print("="*80)
print("\nIf you see ✅ for the modules you need, they're ready to use!")
print("If you see ❌, paste the errors here and we'll fix them.")

