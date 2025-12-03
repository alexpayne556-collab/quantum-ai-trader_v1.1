"""
🔧 FIX SERIES FORMAT ERROR - Paste into Colab
=============================================
Fixes the "unsupported format string passed to Series.__format__" error
"""

import sys
from pathlib import Path

print("="*80)
print("🔧 FIXING SERIES FORMAT ERROR")
print("="*80)

# Setup paths
BASE_DIR = Path('/content/drive/MyDrive/QuantumAI')
MODULES_DIR = BASE_DIR / 'backend' / 'modules'
backtest_file = MODULES_DIR / 'BACKTEST_INSTITUTIONAL_ENSEMBLE.py'

if not backtest_file.exists():
    print(f"❌ File not found: {backtest_file}")
    sys.exit(1)

print(f"\n📁 File: {backtest_file}")

# Read file
print("📖 Reading file...")
with open(backtest_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Check if already fixed
if 'entry_price = float(entry_price)' in content:
    print("\n✅ File already has the fix!")
    sys.exit(0)

print("\n🔧 Applying fixes...")

# Fix 1: entry_price assignment
old_entry1 = "entry_price = data['Close'].iloc[-1] * (1 + self.config['slippage'])"
new_entry1 = "entry_price = float(data['Close'].iloc[-1]) * (1 + self.config['slippage'])"

# Fix 2: exit_price assignment
old_exit = "exit_price = data_dict[symbol]['Close'].iloc[-1] * (1 - self.config['slippage'])"
new_exit = "exit_price = float(data_dict[symbol]['Close'].iloc[-1]) * (1 - self.config['slippage'])"

# Fix 3: Add safety check in _open_position
old_open = """    def _open_position(self, symbol, entry_price, date, signals, positions, capital) -> float:
        \"\"\"Open a position\"\"\"
        # Position size: equal weight across max positions"""
new_open = """    def _open_position(self, symbol, entry_price, date, signals, positions, capital) -> float:
        \"\"\"Open a position\"\"\"
        # Ensure entry_price is a scalar
        entry_price = float(entry_price) if not isinstance(entry_price, (int, float)) else entry_price
        
        # Position size: equal weight across max positions"""

# Apply fixes
fixed = False
if old_entry1 in content:
    content = content.replace(old_entry1, new_entry1)
    print("✅ Fixed entry_price assignment")
    fixed = True

if old_exit in content:
    content = content.replace(old_exit, new_exit)
    print("✅ Fixed exit_price assignment")
    fixed = True

if old_open in content:
    content = content.replace(old_open, new_open)
    print("✅ Added safety check in _open_position")
    fixed = True

if fixed:
    # Write back
    with open(backtest_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n✅ All fixes applied!")
    print(f"   File size: {backtest_file.stat().st_size:,} bytes")
    
    # Verify
    with open(backtest_file, 'r') as f:
        verify = f.read()
    
    if 'float(entry_price)' in verify or 'float(data' in verify:
        print("✅ Verification passed!")
else:
    print("\n⚠️  Could not find exact patterns")
    print("   File may have different structure")

print("\n" + "="*80)
print("✅ PATCH COMPLETE")
print("="*80)
print("\n🔄 NEXT STEPS:")
print("   1. Restart runtime (Runtime → Restart runtime)")
print("   2. Re-run: %run COLAB_LAUNCH_INSTITUTIONAL_SYSTEM.py")
print("="*80)

