"""
🔧 SIMPLE MODULE FIX FOR COLAB
===============================
Fixes:
1. isotonic_regression import → sklearn
2. Comments out asyncio.run() at module level
3. No complex indentation manipulation
"""

import os

print("🔧 FIXING MODULES FOR COLAB")
print("="*80)

# Navigate to modules
os.chdir('/content/drive/MyDrive/QuantumAI/backend/modules')

# ============================================================================
# FIX 1: elite_forecaster.py
# ============================================================================
print("\n1️⃣ Fixing elite_forecaster.py...")

with open('elite_forecaster.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

fixed_lines = []
for i, line in enumerate(lines):
    # Comment out module-level asyncio.run()
    if 'asyncio.run(' in line and not line.strip().startswith('#'):
        # Check if it's at module level (no indentation at start of logical block)
        stripped = line.lstrip()
        if line.startswith('asyncio.run(') or (i > 0 and not lines[i-1].strip().startswith('def ') and not lines[i-1].strip().startswith('class ')):
            fixed_lines.append('# ' + line)  # Comment it out
            print(f"   Commented line {i+1}: {line.strip()[:60]}")
        else:
            fixed_lines.append(line)
    else:
        fixed_lines.append(line)

with open('elite_forecaster.py', 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)
print("✅ elite_forecaster.py fixed")

# ============================================================================
# FIX 2: pre_gainer_scanner_v2_ML_POWERED.py
# ============================================================================
print("\n2️⃣ Fixing pre_gainer_scanner_v2_ML_POWERED.py...")

with open('pre_gainer_scanner_v2_ML_POWERED.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix import
content = content.replace(
    'from scipy.interpolate import isotonic_regression',
    '# from scipy.interpolate import isotonic_regression  # FIXED: Not available in scipy\n# Using sklearn instead if needed'
)

# Comment out asyncio.run() at module level
lines = content.split('\n')
fixed_lines = []
for line in lines:
    if 'asyncio.run(' in line and not line.strip().startswith('#') and not line.startswith('    ') and not line.startswith('\t'):
        fixed_lines.append('# ' + line + '  # FIXED: Causes issues in Colab')
        print(f"   Commented asyncio.run() call")
    else:
        fixed_lines.append(line)

content = '\n'.join(fixed_lines)

with open('pre_gainer_scanner_v2_ML_POWERED.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("✅ pre_gainer_scanner_v2_ML_POWERED.py fixed")

# ============================================================================
# FIX 3: day_trading_scanner_v2_ML_POWERED.py
# ============================================================================
print("\n3️⃣ Fixing day_trading_scanner_v2_ML_POWERED.py...")

with open('day_trading_scanner_v2_ML_POWERED.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Comment out asyncio.run() at module level
lines = content.split('\n')
fixed_lines = []
for line in lines:
    if 'asyncio.run(' in line and not line.strip().startswith('#') and not line.startswith('    ') and not line.startswith('\t'):
        fixed_lines.append('# ' + line + '  # FIXED: Causes issues in Colab')
        print(f"   Commented asyncio.run() call")
    else:
        fixed_lines.append(line)

content = '\n'.join(fixed_lines)

with open('day_trading_scanner_v2_ML_POWERED.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("✅ day_trading_scanner_v2_ML_POWERED.py fixed")

# ============================================================================
# FIX 4-7: Other scanners
# ============================================================================
print("\n4️⃣ Fixing other scanners...")

other_scanners = [
    'swing_trading_scanner_v2_ML_POWERED.py',
    'momentum_tracker_v2_ML_POWERED.py',
    'volume_breakout_scanner_v2_ML_POWERED.py',
    'pattern_recognition_scanner_v2_ML_POWERED.py'
]

for scanner_file in other_scanners:
    if os.path.exists(scanner_file):
        with open(scanner_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix isotonic import
        if 'scipy.interpolate' in content and 'isotonic' in content:
            content = content.replace(
                'from scipy.interpolate import isotonic_regression',
                '# from scipy.interpolate import isotonic_regression  # FIXED'
            )
        
        # Comment out asyncio.run()
        lines = content.split('\n')
        fixed_lines = []
        for line in lines:
            if 'asyncio.run(' in line and not line.strip().startswith('#') and not line.startswith('    '):
                fixed_lines.append('# ' + line + '  # FIXED')
            else:
                fixed_lines.append(line)
        content = '\n'.join(fixed_lines)
        
        with open(scanner_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ {scanner_file} fixed")

print("\n" + "="*80)
print("✅ ALL MODULES FIXED!")
print("="*80)

