"""
🔧 FIX INDENTATION ERROR
========================
Fixes the indentation error in BACKTEST_INSTITUTIONAL_ENSEMBLE.py
"""

import sys
from pathlib import Path

print("="*80)
print("🔧 FIXING INDENTATION ERROR")
print("="*80)

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
    lines = f.readlines()

# Find the problematic area around line 374
print("\n🔍 Checking lines 370-380...")
for i in range(369, min(380, len(lines))):
    print(f"  {i+1:3d}: {repr(lines[i][:80])}")

# Find the _generate_signals method
print("\n🔍 Finding _generate_signals method...")
method_start = None
for i, line in enumerate(lines):
    if 'def _generate_signals' in line:
        method_start = i
        print(f"   Found at line {i+1}: {line.strip()}")
        break

if method_start is None:
    print("❌ Could not find _generate_signals method")
    sys.exit(1)

# Check if there's a docstring issue
print(f"\n🔍 Checking method structure...")
if method_start + 1 < len(lines):
    next_line = lines[method_start + 1]
    print(f"   Line {method_start + 2}: {repr(next_line[:80])}")
    
    # Check if docstring is properly indented
    if next_line.strip().startswith('"""') and not next_line.startswith('        '):
        print("   ⚠️  Docstring not properly indented - fixing...")
        # Fix indentation
        if next_line.strip().startswith('"""'):
            lines[method_start + 1] = '        """' + next_line.strip()[3:] + '\n'
        else:
            lines[method_start + 1] = '        ' + next_line.lstrip()

# Check for empty function body
print(f"\n🔍 Checking for empty function body...")
# Look for the next def or class after _generate_signals
method_end = None
for i in range(method_start + 1, len(lines)):
    stripped = lines[i].strip()
    if stripped.startswith('def ') or stripped.startswith('class '):
        method_end = i
        break
    # Check if we have actual code (not just whitespace/comments)
    if stripped and not stripped.startswith('#') and not stripped.startswith('"""'):
        # Found actual code
        if method_end is None:
            method_end = i + 1

# Check if method body is missing
if method_end and method_end <= method_start + 2:
    print("   ⚠️  Method body appears to be missing or malformed")
    
    # Check what's actually there
    print(f"\n   Lines {method_start + 1} to {method_start + 5}:")
    for i in range(method_start, min(method_start + 5, len(lines))):
        print(f"      {i+1}: {repr(lines[i])}")

# Try to fix by ensuring proper structure
print("\n🔧 Attempting to fix...")

# Read full content
with open(backtest_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Check if the method was partially replaced
if '"""Generate signals using institutional-grade system (Phase 1)"""' in content:
    print("   ✅ Found institutional version - checking structure...")
    
    # Find the method definition
    pattern = r'def _generate_signals\(self, date, data_dict, capital, current_positions\) -> List\[Tuple\]:\s*\n\s*"""Generate signals using institutional-grade system \(Phase 1\)"""'
    
    if re.search(pattern, content):
        print("   ⚠️  Method has docstring but no body - this is the problem!")
        
        # Find where the old method ends
        old_method_pattern = r'(def _generate_signals\(self, date, data_dict, capital, current_positions\) -> List\[Tuple\]:.*?)(?=\n    def |\nclass |\Z)'
        old_match = re.search(old_method_pattern, content, re.DOTALL)
        
        if old_match:
            # The method exists but may be malformed
            method_content = old_match.group(1)
            
            # Check if it has a body
            if method_content.count('\n') < 5:
                print("   ❌ Method body is missing - need to restore")
                
                # Restore from backup if available
                backup_file = backtest_file.with_suffix('.py.backup')
                if backup_file.exists():
                    print(f"   📖 Restoring from backup: {backup_file.name}")
                    with open(backup_file, 'r', encoding='utf-8') as f:
                        backup_content = f.read()
                    
                    # Find the original method
                    original_pattern = r'(def _generate_signals\(self, date, data_dict, capital, current_positions\) -> List\[Tuple\]:.*?)(?=\n    def |\nclass |\Z)'
                    original_match = re.search(original_pattern, backup_content, re.DOTALL)
                    
                    if original_match:
                        # Replace the broken method with original
                        content = content[:old_match.start()] + original_match.group(1) + content[old_match.end():]
                        print("   ✅ Restored original method from backup")
                    else:
                        print("   ❌ Could not find original method in backup")
                else:
                    print("   ❌ No backup found - need manual fix")

# Write fixed content
print("\n💾 Writing fixed file...")
with open(backtest_file, 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ File written")

# Verify syntax
print("\n🔍 Verifying syntax...")
try:
    compile(content, backtest_file, 'exec')
    print("✅ Syntax is valid!")
except SyntaxError as e:
    print(f"❌ Syntax error still exists: {e}")
    print(f"   Line {e.lineno}: {e.text}")
    print("\n💡 You may need to restore from backup manually")

print("\n" + "="*80)
print("✅ FIX COMPLETE")
print("="*80)
print("\n🔄 NEXT STEPS:")
print("   1. If syntax error persists, restore from backup:")
print(f"      {backtest_file.with_suffix('.py.backup')}")
print("   2. Then re-run the integration script")
print("="*80)

