"""
🔍 DEBUG ELITE FORECASTER
=========================
Find out what's in the elite_forecaster.py file
"""

import os
os.chdir('/content/drive/MyDrive/QuantumAI/backend/modules')

print("🔍 ANALYZING elite_forecaster.py")
print("="*80)

with open('elite_forecaster.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"📄 Total lines: {len(lines)}")
print("\n" + "="*80)
print("🔍 SEARCHING FOR CLASS DEFINITIONS")
print("="*80)

class_lines = []
for i, line in enumerate(lines, 1):
    if 'class ' in line and not line.strip().startswith('#'):
        class_lines.append((i, line.strip()))
        print(f"Line {i}: {line.strip()}")

if not class_lines:
    print("❌ NO CLASS DEFINITIONS FOUND!")
    print("\nThis might be a function-based module, not a class-based one.")
    print("\n" + "="*80)
    print("🔍 SEARCHING FOR FUNCTION DEFINITIONS")
    print("="*80)
    
    func_count = 0
    for i, line in enumerate(lines, 1):
        if line.strip().startswith('def ') and not line.strip().startswith('# def'):
            print(f"Line {i}: {line.strip()[:70]}")
            func_count += 1
            if func_count >= 10:
                print(f"... and {sum(1 for l in lines if l.strip().startswith('def '))} total functions")
                break

print("\n" + "="*80)
print("📝 FIRST 50 LINES OF FILE:")
print("="*80)
for i, line in enumerate(lines[:50], 1):
    print(f"{i:3d}: {line.rstrip()}")

print("\n" + "="*80)
print("📝 LINES 490-510 (where asyncio.run was):")
print("="*80)
for i, line in enumerate(lines[489:510], 490):
    print(f"{i:3d}: {line.rstrip()}")

