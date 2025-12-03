"""
🔍 VERIFY DATA_ORCHESTRATOR - Quick Check
========================================
Run this to verify your data_orchestrator.py has all required methods
"""

import sys
from pathlib import Path
from datetime import datetime

print("="*80)
print("🔍 VERIFYING DATA_ORCHESTRATOR")
print("="*80)

# Setup paths
BASE_DIR = Path('/content/drive/MyDrive/QuantumAI')
MODULES_DIR = BASE_DIR / 'backend' / 'modules'
sys.path.insert(0, str(MODULES_DIR))

# Check file exists
data_orch_file = MODULES_DIR / 'data_orchestrator.py'
print(f"\n📁 File location: {data_orch_file}")

if not data_orch_file.exists():
    print("❌ FILE NOT FOUND!")
    print(f"\n📋 Upload data_orchestrator.py to: {MODULES_DIR}")
    sys.exit(1)

# Check file content
print("\n🔍 Checking file content...")
with open(data_orch_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Show file size
file_size = len(content)
print(f"📊 File size: {file_size:,} characters ({file_size/1024:.1f} KB)")
print(f"📊 File lines: {len(content.splitlines())}")

required_items = {
    'ScalarExtractor': 'ScalarExtractor class',
    'get_returns': 'get_returns method',
    'get_ma': 'get_ma method',
    'to_scalar': 'to_scalar method',
    'get_last_close': 'get_last_close method',
    'clean_module_output': 'clean_module_output method',
}

print("\n📋 Required items:")
missing = []
for item, description in required_items.items():
    if item in content:
        # Show line number where found
        lines = content.splitlines()
        line_nums = [i+1 for i, line in enumerate(lines) if item in line]
        if line_nums:
            print(f"  ✅ {description} (found at lines: {line_nums[:3]})")
    else:
        print(f"  ❌ {description} - MISSING!")
        missing.append(item)

# Check for class definition
print("\n🔍 Checking class structure...")
if 'class DataOrchestrator' in content:
    print("  ✅ DataOrchestrator class found")
    # Check inheritance
    if 'DataOrchestrator_v84' in content:
        print("  ✅ Inherits from DataOrchestrator_v84")
    else:
        print("  ⚠️  Inheritance structure may be different")
else:
    print("  ❌ DataOrchestrator class NOT FOUND!")
    missing.append('DataOrchestrator')

if missing:
    print(f"\n❌ FILE IS OUTDATED! Missing {len(missing)} item(s)")
    print("\n📋 ACTION REQUIRED:")
    print("   1. Upload the UPDATED data_orchestrator.py from your local machine")
    print(f"   2. Local path: backend/modules/data_orchestrator.py")
    print(f"   3. Upload to: {MODULES_DIR}")
    print("   4. RESTART RUNTIME (Runtime → Restart runtime)")
    print("   5. Re-run this verification")
    sys.exit(1)

# Try importing
print("\n🔍 Testing import...")
try:
    # Clear ALL cached versions (more aggressive)
    modules_to_clear = [k for k in sys.modules.keys() if 'data_orchestrator' in k.lower() or 'orchestrator' in k.lower()]
    for mod in modules_to_clear:
        del sys.modules[mod]
    print(f"  🧹 Cleared {len(modules_to_clear)} cached module(s)")
    
    from data_orchestrator import DataOrchestrator, ScalarExtractor
    print("✅ Import successful")
    
    # Check if ScalarExtractor is accessible
    if ScalarExtractor:
        print("✅ ScalarExtractor class accessible")
    else:
        print("❌ ScalarExtractor class not accessible")
    
    # Test instantiation
    print("\n🔍 Testing instantiation...")
    orch = DataOrchestrator()
    print("✅ DataOrchestrator instantiated")
    
    # Check what type it is
    print(f"  📊 Type: {type(orch).__name__}")
    print(f"  📊 MRO: {[c.__name__ for c in type(orch).__mro__]}")
    
    # Test methods
    print("\n🔍 Testing methods...")
    methods_to_test = [
        'get_returns',
        'get_ma',
        'to_scalar',
        'get_last_close',
        'get_last_open',
        'get_last_high',
        'get_last_low',
        'get_last_volume',
        'get_volume_ratio',
        'clean_module_output',
    ]
    
    all_present = True
    for method in methods_to_test:
        if hasattr(orch, method):
            print(f"  ✅ {method}")
        else:
            print(f"  ❌ {method} - MISSING!")
            all_present = False
    
    # Test ScalarExtractor
    if hasattr(orch, 'scalar'):
        print(f"  ✅ scalar (ScalarExtractor instance)")
    else:
        print(f"  ❌ scalar - MISSING!")
        all_present = False
    
    if all_present:
        print("\n" + "="*80)
        print("✅ ALL CHECKS PASSED!")
        print("="*80)
        print("\n🎯 Your data_orchestrator.py is up to date!")
        print("   You can now run the backtest or dashboard.")
    else:
        print("\n❌ Some methods are missing!")
        print("   RESTART RUNTIME and re-upload the file.")
        
except Exception as e:
    print(f"\n❌ Import/Test error: {str(e)}")
    import traceback
    print("\n📋 Full error traceback:")
    traceback.print_exc()
    
    # Additional diagnostics
    print("\n🔍 Additional diagnostics:")
    print(f"  📁 File exists: {data_orch_file.exists()}")
    if data_orch_file.exists():
        stat = data_orch_file.stat()
        print(f"  📊 File modified: {datetime.fromtimestamp(stat.st_mtime)}")
        print(f"  📊 File size: {stat.st_size} bytes")
    
    print("\n💡 SOLUTIONS:")
    print("   1. RESTART RUNTIME (Runtime → Restart runtime)")
    print("   2. Re-upload data_orchestrator.py to Drive")
    print("   3. Make sure file is in: /content/drive/MyDrive/QuantumAI/backend/modules/")
    print("   4. Re-run this verification script")

