"""
🧹 REMOVE REDUNDANT MODULES - Quick Colab Command
==================================================
Run this in Colab to remove all redundant/obsolete modules
"""

import re
from pathlib import Path

print("="*80)
print("🧹 REMOVING REDUNDANT MODULES")
print("="*80)

BASE_DIR = Path('/content/drive/MyDrive/QuantumAI')
MODULES_DIR = BASE_DIR / 'backend' / 'modules'
backtest_file = MODULES_DIR / 'BACKTEST_INSTITUTIONAL_ENSEMBLE.py'

if not backtest_file.exists():
    print(f"❌ File not found: {backtest_file}")
    print(f"   Looking in: {MODULES_DIR}")
    sys.exit(1)

print(f"\n📁 File: {backtest_file}")

# Read file
with open(backtest_file, 'r', encoding='utf-8') as f:
    content = f.read()

original_length = len(content)
print(f"   Original size: {original_length:,} characters")

# ============================================================================
# REMOVE REDUNDANT MODULE INITIALIZATIONS
# ============================================================================
print("\n1️⃣ Removing redundant module initializations...")

redundant_patterns = [
    # Obsolete scanner initializations
    (r'self\.pregainer = MockPatternScanner\([^)]+\)\s*\n', 'pregainer'),
    (r'self\.day_trading = MockPatternScanner\([^)]+\)\s*\n', 'day_trading'),
    (r'self\.opportunity = MockPatternScanner\([^)]+\)\s*\n', 'opportunity'),
    (r'self\.squeeze = MockShortSqueezeScanner\([^)]+\)\s*\n', 'squeeze'),
]

removed_count = 0
for pattern, name in redundant_patterns:
    matches = len(re.findall(pattern, content))
    if matches > 0:
        content = re.sub(pattern, '', content)
        removed_count += matches
        print(f"   ✅ Removed {name} ({matches} occurrence(s))")

# ============================================================================
# REMOVE REDUNDANT CLASS DEFINITIONS
# ============================================================================
print("\n2️⃣ Removing redundant class definitions...")

class_patterns = [
    (r'class MockPatternScanner:.*?return \{\'signal\': \'NEUTRAL\', \'confidence\': 0\.5\}', 'MockPatternScanner'),
    (r'class MockShortSqueezeScanner:.*?return \{\'signal\': \'LOW_SQUEEZE\', \'confidence\': 0\.4\}', 'MockShortSqueezeScanner'),
]

for pattern, name in class_patterns:
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, '', content, flags=re.DOTALL)
        print(f"   ✅ Removed {name} class definition")

# ============================================================================
# UPDATE MODULE TRACKING
# ============================================================================
print("\n3️⃣ Updating module tracking...")

# Remove redundant modules from signals_by_module
old_pattern = r"self\.signals_by_module = \{module: \[\] for module in \[.*?\]\}"
new_tracking = """self.signals_by_module = {module: [] for module in [
            'dark_pool', 'insider_trading', 'sentiment',
            'unified_scanner', 'pump_detection'
        ]}"""

if re.search(old_pattern, content, re.DOTALL):
    content = re.sub(old_pattern, new_tracking, content, flags=re.DOTALL)
    print("   ✅ Updated signals_by_module tracking")

# ============================================================================
# REMOVE REDUNDANT SIGNAL CALLS IN _generate_signals
# ============================================================================
print("\n4️⃣ Cleaning up signal generation...")

# Remove calls to obsolete scanners
signal_patterns = [
    (r'self\.pregainer\.scan\([^)]+\)', 'pregainer.scan'),
    (r'self\.day_trading\.scan\([^)]+\)', 'day_trading.scan'),
    (r'self\.opportunity\.scan\([^)]+\)', 'opportunity.scan'),
    (r'self\.squeeze\.analyze_ticker\([^)]+\)', 'squeeze.analyze_ticker'),
]

for pattern, name in signal_patterns:
    matches = len(re.findall(pattern, content))
    if matches > 0:
        print(f"   ⚠️  Found {matches} call(s) to {name} - may need manual cleanup")

# ============================================================================
# WRITE UPDATED FILE
# ============================================================================
new_length = len(content)
reduction = original_length - new_length

with open(backtest_file, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"\n   ✅ File updated")
print(f"   Size reduction: {reduction:,} characters ({reduction/original_length*100:.1f}%)")

# ============================================================================
# VERIFICATION
# ============================================================================
print("\n" + "="*80)
print("🔍 VERIFICATION")
print("="*80)

with open(backtest_file, 'r') as f:
    content_check = f.read()

checks = {
    'pregainer removed': 'self.pregainer =' not in content_check,
    'day_trading removed': 'self.day_trading =' not in content_check,
    'opportunity removed': 'self.opportunity =' not in content_check,
    'squeeze removed': 'self.squeeze =' not in content_check,
    'MockPatternScanner removed': 'class MockPatternScanner' not in content_check,
    'MockShortSqueezeScanner removed': 'class MockShortSqueezeScanner' not in content_check,
}

print("\n✅ Cleanup Checks:")
all_passed = True
for check_name, passed in checks.items():
    status = "✅" if passed else "❌"
    print(f"   {status} {check_name}")
    if not passed:
        all_passed = False

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📋 SUMMARY")
print("="*80)

if all_passed:
    print("""
✅ Redundant Modules Removed!

Removed:
   ❌ MockPatternScanner (pregainer, day_trading, opportunity)
   ❌ MockShortSqueezeScanner (squeeze)
   ❌ All redundant initializations
   ❌ All redundant class definitions

Next Steps:
   1. Restart runtime (Runtime → Restart runtime)
   2. Test system still works
   3. Integrate UnifiedMomentumScannerV3 (if not already done)
   4. Update dashboard to use new modules

System is now cleaner and ready for unified scanner!
""")
else:
    print("""
⚠️  Some redundant modules may still exist.
Please check the verification results above.
You may need to manually remove remaining references.
""")

print("="*80)
print("✅ CLEANUP COMPLETE!")
print("="*80)

