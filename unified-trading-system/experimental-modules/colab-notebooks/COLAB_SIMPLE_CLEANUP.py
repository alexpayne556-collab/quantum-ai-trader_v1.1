"""
🧹 SIMPLE CLEANUP - Remove Redundant Modules Only
===================================================
Safe cleanup that only removes redundant modules, doesn't modify methods
"""

import re
from pathlib import Path

print("="*80)
print("🧹 SIMPLE CLEANUP - Removing Redundant Modules")
print("="*80)

BASE_DIR = Path('/content/drive/MyDrive/QuantumAI')
MODULES_DIR = BASE_DIR / 'backend' / 'modules'
backtest_file = MODULES_DIR / 'BACKTEST_INSTITUTIONAL_ENSEMBLE.py'

if not backtest_file.exists():
    print(f"❌ File not found: {backtest_file}")
    import sys
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
# REMOVE REDUNDANT CLASS DEFINITIONS (if they exist)
# ============================================================================
print("\n2️⃣ Removing redundant class definitions...")

# More flexible pattern matching
class_patterns = [
    (r'class MockPatternScanner:.*?(?=\nclass |\ndef |\Z)', 'MockPatternScanner'),
    (r'class MockShortSqueezeScanner:.*?(?=\nclass |\ndef |\Z)', 'MockShortSqueezeScanner'),
]

for pattern, name in class_patterns:
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, '', content, flags=re.DOTALL)
        print(f"   ✅ Removed {name} class definition")

# ============================================================================
# UPDATE MODULE TRACKING (if it exists)
# ============================================================================
print("\n3️⃣ Updating module tracking...")

# Try to update signals_by_module if it has old modules
if "'pregainer'" in content or "'day_trading'" in content or "'opportunity'" in content:
    # Find and replace the signals_by_module line
    old_pattern = r"self\.signals_by_module = \{module: \[\] for module in \[[^\]]*'pregainer'[^\]]*\]\}"
    new_tracking = """self.signals_by_module = {module: [] for module in [
            'dark_pool', 'insider_trading', 'sentiment',
            'unified_scanner', 'pump_detection'
        ]}"""
    
    if re.search(old_pattern, content, re.DOTALL):
        content = re.sub(old_pattern, new_tracking, content, flags=re.DOTALL)
        print("   ✅ Updated signals_by_module tracking")
    else:
        print("   ⚠️  Could not find exact pattern, but old modules may still be referenced")

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
   ❌ pregainer initialization
   ❌ day_trading initialization
   ❌ opportunity initialization
   ❌ squeeze initialization

Next Steps:
   1. Restart runtime (Runtime → Restart runtime)
   2. Test system still works
   3. If _generate_signals has errors, you may need to manually update it
   4. Integrate UnifiedMomentumScannerV3 (if not already done)

System is now cleaner!
""")
else:
    print("""
⚠️  Some redundant modules may still exist.
Please check the verification results above.
""")

print("="*80)
print("✅ CLEANUP COMPLETE!")
print("="*80)

