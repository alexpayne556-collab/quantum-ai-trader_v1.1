"""
🔄 FORCE RELOAD DATA_ORCHESTRATOR - Clear Cache & Verify
========================================================
This aggressively clears all cached modules and forces a fresh import
"""

import sys
import importlib
from pathlib import Path

print("="*80)
print("🔄 FORCE RELOAD DATA_ORCHESTRATOR")
print("="*80)

# Setup paths
BASE_DIR = Path('/content/drive/MyDrive/QuantumAI')
MODULES_DIR = BASE_DIR / 'backend' / 'modules'
sys.path.insert(0, str(MODULES_DIR))

# ============================================================================
# STEP 1: CLEAR ALL CACHED MODULES
# ============================================================================
print("\n1️⃣ Clearing ALL cached modules...")

# Find all modules related to data_orchestrator
modules_to_remove = []
for key in list(sys.modules.keys()):
    if 'data_orchestrator' in key.lower() or 'orchestrator' in key.lower():
        modules_to_remove.append(key)

for key in modules_to_remove:
    del sys.modules[key]
    print(f"  ✅ Removed: {key}")

print(f"✅ Cleared {len(modules_to_remove)} cached module(s)")

# Also clear any .pyc files
import os
import glob

pyc_files = glob.glob(str(MODULES_DIR / '**/*.pyc'), recursive=True)
pyc_files += glob.glob(str(MODULES_DIR / '**/__pycache__'), recursive=True)

for pyc in pyc_files:
    try:
        if os.path.isfile(pyc):
            os.remove(pyc)
        elif os.path.isdir(pyc):
            import shutil
            shutil.rmtree(pyc)
    except:
        pass

if pyc_files:
    print(f"✅ Cleared {len(pyc_files)} .pyc/cache files")

# ============================================================================
# STEP 2: VERIFY FILE HAS METHODS
# ============================================================================
print("\n2️⃣ Verifying file content...")

data_orch_file = MODULES_DIR / 'data_orchestrator.py'

if not data_orch_file.exists():
    print(f"❌ File not found: {data_orch_file}")
    sys.exit(1)

with open(data_orch_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Check for required components
checks = {
    'ScalarExtractor class': 'class ScalarExtractor:' in content,
    'DataOrchestrator class': 'class DataOrchestrator(' in content,
    'get_returns method': 'def get_returns(self, df: pd.DataFrame' in content,
    'get_ma method': 'def get_ma(self, df: pd.DataFrame' in content,
    'to_scalar method': 'def to_scalar(self, value)' in content,
    'scalar in __init__': 'self.scalar = ScalarExtractor()' in content,
}

print("\n📋 File content check:")
all_present = True
for check, present in checks.items():
    if present:
        print(f"  ✅ {check}")
    else:
        print(f"  ❌ {check} - MISSING!")
        all_present = False

if not all_present:
    print("\n❌ File is missing required components!")
    print("   Run COLAB_PATCH_DATA_ORCHESTRATOR.py first")
    sys.exit(1)

# ============================================================================
# STEP 3: FORCE FRESH IMPORT
# ============================================================================
print("\n3️⃣ Forcing fresh import...")

# Change to modules directory
original_dir = os.getcwd()
os.chdir(MODULES_DIR)

try:
    # Use importlib to force reload
    import importlib.util
    
    spec = importlib.util.spec_from_file_location(
        "data_orchestrator",
        data_orch_file
    )
    
    # Remove from cache if exists
    if 'data_orchestrator' in sys.modules:
        del sys.modules['data_orchestrator']
    
    # Load fresh
    module = importlib.util.module_from_spec(spec)
    sys.modules['data_orchestrator'] = module
    spec.loader.exec_module(module)
    
    print("✅ Module loaded fresh")
    
    # Import classes
    DataOrchestrator = module.DataOrchestrator
    ScalarExtractor = module.ScalarExtractor
    
    print("✅ Classes imported")
    
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    os.chdir(original_dir)

# ============================================================================
# STEP 4: TEST INSTANTIATION
# ============================================================================
print("\n4️⃣ Testing instantiation...")

try:
    orch = DataOrchestrator()
    print("✅ DataOrchestrator instantiated")
    
    # Test methods
    print("\n5️⃣ Testing methods...")
    
    methods_to_test = [
        ('get_returns', 'get_returns'),
        ('get_ma', 'get_ma'),
        ('to_scalar', 'to_scalar'),
        ('get_last_close', 'get_last_close'),
        ('get_last_open', 'get_last_open'),
        ('get_last_high', 'get_last_high'),
        ('get_last_low', 'get_last_low'),
        ('get_last_volume', 'get_last_volume'),
        ('get_volume_ratio', 'get_volume_ratio'),
        ('clean_module_output', 'clean_module_output'),
    ]
    
    all_work = True
    for name, method_name in methods_to_test:
        if hasattr(orch, method_name):
            method = getattr(orch, method_name)
            if callable(method):
                print(f"  ✅ {name} - callable")
            else:
                print(f"  ⚠️  {name} - exists but not callable")
                all_work = False
        else:
            print(f"  ❌ {name} - NOT FOUND!")
            all_work = False
    
    # Test scalar
    if hasattr(orch, 'scalar'):
        print(f"  ✅ scalar attribute exists")
        if hasattr(orch.scalar, 'to_scalar'):
            print(f"  ✅ scalar.to_scalar() method exists")
        else:
            print(f"  ⚠️  scalar exists but missing to_scalar method")
    else:
        print(f"  ❌ scalar attribute - NOT FOUND!")
        all_work = False
    
    if all_work:
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print("\n🎯 DataOrchestrator is ready to use!")
        print("\n📋 The module is now loaded fresh in memory.")
        print("   You can now run your backtest:")
        print("\n   %run COLAB_LAUNCH_INSTITUTIONAL_SYSTEM.py")
    else:
        print("\n⚠️  Some methods are missing or not callable")
        print("   This might indicate a structural issue with the class")
        
except Exception as e:
    print(f"\n❌ Test error: {e}")
    import traceback
    traceback.print_exc()
    print("\n💡 Try restarting runtime: Runtime → Restart runtime")

print("\n" + "="*80)

