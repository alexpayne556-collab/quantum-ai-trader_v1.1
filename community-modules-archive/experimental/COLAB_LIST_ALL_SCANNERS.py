"""
📋 LIST ALL AVAILABLE SCANNERS
===============================
Find out what scanner files actually exist
"""

import os
import glob

MODULES_DIR = '/content/drive/MyDrive/QuantumAI/backend/modules'
os.chdir(MODULES_DIR)

print("📋 ALL FILES IN MODULES DIRECTORY")
print("="*80)

# Get all Python files
py_files = sorted(glob.glob('*.py'))

print(f"\n📊 Found {len(py_files)} Python files:\n")

scanner_files = []
forecaster_files = []
other_files = []

for f in py_files:
    if 'scanner' in f.lower():
        scanner_files.append(f)
        print(f"🔍 SCANNER: {f}")
    elif 'forecast' in f.lower():
        forecaster_files.append(f)
        print(f"📈 FORECASTER: {f}")
    else:
        other_files.append(f)
        print(f"📄 OTHER: {f}")

print("\n" + "="*80)
print("📊 SUMMARY")
print("="*80)
print(f"🔍 Scanners: {len(scanner_files)}")
print(f"📈 Forecasters: {len(forecaster_files)}")
print(f"📄 Other: {len(other_files)}")

print("\n" + "="*80)
print("🎯 SCANNERS WE HAVE:")
print("="*80)
for f in scanner_files:
    print(f"  ✅ {f}")

print("\n" + "="*80)
print("🎯 FORECASTERS WE HAVE:")
print("="*80)
for f in forecaster_files:
    print(f"  ✅ {f}")

# Check for model files
print("\n" + "="*80)
print("🤖 CHECKING FOR MODEL FILES (.pkl)")
print("="*80)

pkl_files = sorted(glob.glob('*.pkl'))
if pkl_files:
    print(f"Found {len(pkl_files)} model files:")
    for f in pkl_files[:20]:  # Show first 20
        print(f"  🤖 {f}")
    if len(pkl_files) > 20:
        print(f"  ... and {len(pkl_files) - 20} more")
else:
    print("⚠️  NO .pkl MODEL FILES FOUND")
    print("   This is why the scanners show model warnings.")
    print("   Solution: Use the ranking model instead.")

