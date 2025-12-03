"""
🔧 COLAB: RELOAD FIXED MODULES
================================
Run this in a NEW Colab cell after Google Drive sync completes!
"""

print("🔄 Reloading fixed modules...")

import sys
import importlib

# Reload the 3 fixed modules
modules_to_reload = [
    'pattern_integration_layer',
    'volume_profile_analyzer',
]

for mod_name in modules_to_reload:
    if mod_name in sys.modules:
        importlib.reload(sys.modules[mod_name])
        print(f"   ✅ Reloaded: {mod_name}")
    else:
        print(f"   ⚠️  Not loaded yet: {mod_name}")

print("\n✅ Modules reloaded! Now test again:")
print("=" * 60)
print()
print("# Test fixed system:")
print("from master_analysis_engine import analyze_stock")
print("result = await analyze_stock('NVDA', 50000, 21, verbose=True)")
print()
print("=" * 60)

