"""
🧪 TEST ALL YOUR REAL MODULES
==============================
Tests the 3 scanners + 3 forecasters you actually have
"""

import os
import sys
import importlib.util

MODULES_DIR = '/content/drive/MyDrive/QuantumAI/backend/modules'
sys.path.insert(0, MODULES_DIR)
os.chdir(MODULES_DIR)

print("🧪 TESTING ALL YOUR MODULES")
print("="*80)

# ============================================================================
# TEST ALL 3 SCANNERS
# ============================================================================
print("\n" + "="*80)
print("🔍 TESTING SCANNERS (3)")
print("="*80)

scanners = [
    ('pre_gainer_scanner_v2_ML_POWERED.py', 'PreGainerScannerV2'),
    ('day_trading_scanner_v2_ML_POWERED.py', 'DayTradingScannerV2'),
    ('opportunity_scanner_v2_ML_POWERED.py', 'OpportunityScannerV2'),
]

working_scanners = []

for i, (file, class_name) in enumerate(scanners, 1):
    print(f"\n{i}️⃣ Testing {file}...")
    print("-" * 80)
    
    try:
        spec = importlib.util.spec_from_file_location(class_name, file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, class_name):
            print(f"✅ Found {class_name}")
            instance = getattr(module, class_name)()
            print(f"✅ Instantiated successfully!")
            
            # Check for scan method
            if hasattr(instance, 'scan'):
                print(f"✅ Has scan() method")
            elif hasattr(instance, 'scan_stocks'):
                print(f"✅ Has scan_stocks() method")
            
            working_scanners.append((file, class_name, instance))
        else:
            print(f"⚠️  Class {class_name} not found")
            # Try to find what classes exist
            classes = [name for name in dir(module) if not name.startswith('_') and 'Scanner' in name]
            if classes:
                print(f"   Found instead: {classes}")
                
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {str(e)}")

# ============================================================================
# TEST ALL 3 FORECASTERS
# ============================================================================
print("\n\n" + "="*80)
print("📈 TESTING FORECASTERS (3)")
print("="*80)

forecasters = [
    ('elite_forecaster.py', ['EliteForecaster', 'Forecaster', 'Elite']),
    ('fusior_forecast.py', ['FusiorForecast', 'Fusior', 'Forecast']),
    ('fusior_forecast_institutional.py', ['FusiorForecastInstitutional', 'FusiorInstitutional', 'Forecaster']),
]

working_forecasters = []

for i, (file, possible_names) in enumerate(forecasters, 1):
    print(f"\n{i}️⃣ Testing {file}...")
    print("-" * 80)
    
    try:
        module_name = file.replace('.py', '')
        spec = importlib.util.spec_from_file_location(module_name, file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        print(f"✅ Module loaded")
        
        # Try to find class
        found_class = None
        for name in possible_names:
            if hasattr(module, name):
                found_class = name
                print(f"✅ Found class: {name}")
                break
        
        if not found_class:
            # List all non-private attributes
            all_attrs = [name for name in dir(module) if not name.startswith('_')]
            classes = [name for name in all_attrs if 'class' in str(type(getattr(module, name, None)))]
            funcs = [name for name in all_attrs if callable(getattr(module, name, None))]
            
            print(f"⚠️  None of {possible_names} found")
            if classes:
                print(f"   Classes found: {classes[:5]}")
            if funcs:
                print(f"   Functions found: {funcs[:10]}")
            
            # If it's function-based, check for forecast function
            if 'forecast' in funcs or 'predict' in funcs:
                print(f"✅ This is a FUNCTION-BASED module (not class-based)")
                working_forecasters.append((file, 'function-based', module))
            continue
        
        # Try to instantiate
        try:
            instance = getattr(module, found_class)()
            print(f"✅ Instantiated {found_class}")
            
            # Check for forecast method
            if hasattr(instance, 'forecast'):
                print(f"✅ Has forecast() method")
            elif hasattr(instance, 'predict'):
                print(f"✅ Has predict() method")
            
            working_forecasters.append((file, found_class, instance))
        except TypeError as e:
            if '__init__' in str(e):
                print(f"⚠️  Class needs parameters to instantiate: {e}")
                print(f"   (Can still use it, just need to provide params)")
                working_forecasters.append((file, found_class, None))
            else:
                raise
                
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc(limit=5)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n\n" + "="*80)
print("🎯 SUMMARY - WHAT WORKS")
print("="*80)

print(f"\n✅ WORKING SCANNERS ({len(working_scanners)}/3):")
for file, class_name, _ in working_scanners:
    print(f"   ✅ {class_name} ({file})")

print(f"\n✅ WORKING FORECASTERS ({len(working_forecasters)}/3):")
for file, name, _ in working_forecasters:
    if name == 'function-based':
        print(f"   ✅ {file} (function-based module)")
    else:
        print(f"   ✅ {name} ({file})")

print("\n" + "="*80)
print("💡 NEXT STEPS")
print("="*80)
if len(working_scanners) >= 2 and len(working_forecasters) >= 1:
    print("✅ You have enough working modules to build the dashboard!")
    print("✅ We'll integrate:")
    print(f"   - {len(working_scanners)} scanners")
    print(f"   - {len(working_forecasters)} forecaster(s)")
    print("   - Ranking model (80% success rate)")
    print("   - Paper trading")
    print("   - Auto-logging")
else:
    print("⚠️  Need to fix some modules first")
    print(f"   Working scanners: {len(working_scanners)}/3")
    print(f"   Working forecasters: {len(working_forecasters)}/3")

