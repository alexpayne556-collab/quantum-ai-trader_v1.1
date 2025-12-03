"""
🧹 CLEANUP OBSOLETE MODULES
===========================
Removes legacy/redundant modules that have been replaced by optimized systems

OBSOLETE MODULES TO REMOVE:
- MockPatternScanner (pregainer, day_trading, opportunity) → Replaced by CombinedScannerModule
- MockShortSqueezeScanner → Not used effectively
- Old mock modules → Replaced by OptimizedEnsembleTrader + PumpBreakoutEarlyWarning

KEEP:
- Dark pool (52.5% win rate - your best signal)
- Insider trading (will be improved)
- Sentiment (confirmation signal)
- OptimizedEnsembleTrader (master system)
- PumpBreakoutEarlyWarning (early detection)
- AI Recommender (main interface)
"""

import sys
from pathlib import Path
import re

print("="*80)
print("🧹 CLEANING UP OBSOLETE MODULES")
print("="*80)

BASE_DIR = Path('/content/drive/MyDrive/QuantumAI')
MODULES_DIR = BASE_DIR / 'backend' / 'modules'
backtest_file = MODULES_DIR / 'BACKTEST_INSTITUTIONAL_ENSEMBLE.py'

if not backtest_file.exists():
    print(f"❌ File not found: {backtest_file}")
    sys.exit(1)

print(f"\n📁 File: {backtest_file}")

# ============================================================================
# STEP 1: Remove obsolete MockPatternScanner classes
# ============================================================================
print("\n1️⃣ Removing obsolete MockPatternScanner classes...")

with open(backtest_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Remove MockPatternScanner class definition
pattern_scanner_pattern = r'class MockPatternScanner:.*?return \{\'signal\': \'NEUTRAL\', \'confidence\': 0\.5\}'
content = re.sub(pattern_scanner_pattern, '', content, flags=re.DOTALL)

# Remove MockShortSqueezeScanner class
squeeze_scanner_pattern = r'class MockShortSqueezeScanner:.*?return \{\'signal\': \'LOW_SQUEEZE\', \'confidence\': 0\.4\}'
content = re.sub(squeeze_scanner_pattern, '', content, flags=re.DOTALL)

print("   ✅ Removed obsolete scanner classes")

# ============================================================================
# STEP 2: Remove obsolete module initializations
# ============================================================================
print("\n2️⃣ Removing obsolete module initializations...")

# Remove pregainer, day_trading, opportunity, squeeze
obsolete_inits = [
    r'self\.pregainer = MockPatternScanner\(.*?\)\n',
    r'self\.day_trading = MockPatternScanner\(.*?\)\n',
    r'self\.opportunity = MockPatternScanner\(.*?\)\n',
    r'self\.squeeze = MockShortSqueezeScanner\(.*?\)\n',
]

for pattern in obsolete_inits:
    content = re.sub(pattern, '', content)

print("   ✅ Removed obsolete module initializations")

# ============================================================================
# STEP 3: Update signals_by_module to remove obsolete modules
# ============================================================================
print("\n3️⃣ Updating signals_by_module tracking...")

# Replace old module list with new one
old_modules_pattern = r"self\.signals_by_module = \{module: \[\] for module in \[.*?'pregainer', 'day_trading', 'opportunity', 'sentiment'.*?\]\}"
new_modules = """self.signals_by_module = {module: [] for module in [
            'dark_pool', 'insider_trading', 'sentiment',
            'combined_scanners', 'pump_detection'
        ]}"""

content = re.sub(old_modules_pattern, new_modules, content, flags=re.DOTALL)

print("   ✅ Updated module tracking")

# ============================================================================
# STEP 4: Update _generate_signals to use optimized system only
# ============================================================================
print("\n4️⃣ Updating _generate_signals to remove obsolete scanner calls...")

# Find and remove obsolete scanner calls
obsolete_scanner_calls = [
    r"# Patterns.*?self\.pregainer.*?self\.opportunity.*?\)",
    r"for scanner_name, scanner in \[.*?\('opportunity', self\.opportunity\).*?\]:",
]

for pattern in obsolete_scanner_calls:
    content = re.sub(pattern, '', content, flags=re.DOTALL)

print("   ✅ Removed obsolete scanner calls")

# ============================================================================
# STEP 5: Add comment about signal generation (not execution)
# ============================================================================
print("\n5️⃣ Adding signal generation focus comments...")

# Add header comment to _generate_signals
if 'def _generate_signals' in content:
    signals_header = """    def _generate_signals(self, date, data_dict, capital, current_positions) -> List[Tuple]:
        \"\"\"
        Generate trading signals and recommendations (NOT autonomous execution)
        
        This is a RESEARCH TOOL for finding stocks:
        - Signals are generated for analysis
        - Recommendations are provided via AI Recommender
        - Actual execution happens manually on Robinhood (external)
        - Paper trading available on dashboard for testing only
        \"\"\"
"""
    # Replace method definition
    content = re.sub(
        r'def _generate_signals\(self, date, data_dict, capital, current_positions\) -> List\[Tuple\]:.*?"""',
        signals_header.replace('"""', ''),
        content,
        flags=re.DOTALL
    )

print("   ✅ Added signal generation focus comments")

# Write updated file
with open(backtest_file, 'w', encoding='utf-8') as f:
    f.write(content)

print("\n   ✅ Backtest file cleaned up")

# ============================================================================
# VERIFICATION
# ============================================================================
print("\n" + "="*80)
print("🔍 VERIFICATION")
print("="*80)

with open(backtest_file, 'r') as f:
    content_check = f.read()

checks = {
    'MockPatternScanner removed': 'class MockPatternScanner' not in content_check,
    'MockShortSqueezeScanner removed': 'class MockShortSqueezeScanner' not in content_check,
    'pregainer removed': 'self.pregainer =' not in content_check,
    'day_trading removed': 'self.day_trading =' not in content_check,
    'opportunity removed': 'self.opportunity =' not in content_check,
    'squeeze removed': 'self.squeeze =' not in content_check,
}

print("\n✅ Cleanup Checks:")
for check_name, passed in checks.items():
    status = "✅" if passed else "❌"
    print(f"   {status} {check_name}")

all_passed = all(checks.values())

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📋 CLEANUP SUMMARY")
print("="*80)

if all_passed:
    print("""
✅ Cleanup Complete!

Removed Obsolete Modules:
   ❌ MockPatternScanner (pregainer, day_trading, opportunity)
      → Replaced by CombinedScannerModule in OptimizedEnsembleTrader
   
   ❌ MockShortSqueezeScanner
      → Not used effectively, removed

Updated System:
   ✅ Uses OptimizedEnsembleTrader (master system)
   ✅ Uses PumpBreakoutEarlyWarning (early detection)
   ✅ Focuses on signal generation (not execution)
   ✅ AI Recommender is main interface

Next Steps:
   1. Update _generate_signals to use OptimizedEnsembleTrader only
   2. Ensure CombinedScannerModule is used (not individual scanners)
   3. Update dashboard to show signals/recommendations clearly
   4. Remove any execution logic (keep paper trading for testing only)
""")
else:
    print("""
⚠️  Some cleanup steps may need manual review.
Please check the verification results above.
""")

print("="*80)
print("✅ CLEANUP COMPLETE!")
print("="*80)

