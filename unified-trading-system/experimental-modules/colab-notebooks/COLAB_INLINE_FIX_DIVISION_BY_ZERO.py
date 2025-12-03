"""
🔧 INLINE FIX - Division by Zero
=================================
Paste this entire code block into a Colab cell and run it
"""

import sys
from pathlib import Path

print("="*80)
print("🔧 FIXING DIVISION BY ZERO ERROR")
print("="*80)

# Setup paths
BASE_DIR = Path('/content/drive/MyDrive/QuantumAI')
MODULES_DIR = BASE_DIR / 'backend' / 'modules'
ensemble_file = MODULES_DIR / 'INSTITUTIONAL_ENSEMBLE_ENGINE.py'

if not ensemble_file.exists():
    print(f"❌ File not found: {ensemble_file}")
    sys.exit(1)

print(f"\n📁 File: {ensemble_file}")

# Read file
print("📖 Reading file...")
with open(ensemble_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Check if already fixed
if 'denominator < 1e-10' in content and 'confidence = max(0.01, min(0.99, signal.confidence))' in content:
    print("\n✅ File already has the fix!")
    sys.exit(0)

# Find and replace the problematic code
print("\n🔧 Applying fix...")

# Pattern 1: Try exact match
old_pattern1 = """                likelihood_ratio = (signal.confidence * signal_accuracy) / \\
                                 ((1 - signal.confidence) * (1 - signal_accuracy))"""

new_pattern1 = """                # Safety check: prevent division by zero
                confidence = max(0.01, min(0.99, signal.confidence))
                accuracy = max(0.01, min(0.99, signal_accuracy))
                denominator = (1 - confidence) * (1 - accuracy)
                if denominator < 1e-10:
                    likelihood_ratio = 100.0 if confidence > 0.9 or accuracy > 0.9 else 1.0
                else:
                    likelihood_ratio = (confidence * accuracy) / denominator"""

# Pattern 2: Without backslash
old_pattern2 = """                likelihood_ratio = (signal.confidence * signal_accuracy) / 
                                 ((1 - signal.confidence) * (1 - signal_accuracy))"""

new_pattern2 = """                # Safety check: prevent division by zero
                confidence = max(0.01, min(0.99, signal.confidence))
                accuracy = max(0.01, min(0.99, signal_accuracy))
                denominator = (1 - confidence) * (1 - accuracy)
                if denominator < 1e-10:
                    likelihood_ratio = 100.0 if confidence > 0.9 or accuracy > 0.9 else 1.0
                else:
                    likelihood_ratio = (confidence * accuracy) / denominator"""

# Pattern 3: Single line
old_pattern3 = """likelihood_ratio = (signal.confidence * signal_accuracy) / ((1 - signal.confidence) * (1 - signal_accuracy))"""

new_pattern3 = """# Safety check: prevent division by zero
                confidence = max(0.01, min(0.99, signal.confidence))
                accuracy = max(0.01, min(0.99, signal_accuracy))
                denominator = (1 - confidence) * (1 - accuracy)
                if denominator < 1e-10:
                    likelihood_ratio = 100.0 if confidence > 0.9 or accuracy > 0.9 else 1.0
                else:
                    likelihood_ratio = (confidence * accuracy) / denominator"""

# Also fix the posterior update
old_posterior = """                posterior = (posterior * weighted_lr) / \\
                           (posterior * weighted_lr + (1 - posterior))"""

new_posterior = """                denominator_posterior = posterior * weighted_lr + (1 - posterior)
                if denominator_posterior < 1e-10:
                    posterior = 0.99
                else:
                    posterior = (posterior * weighted_lr) / denominator_posterior"""

# Try each pattern
fixed = False
for old, new in [(old_pattern1, new_pattern1), (old_pattern2, new_pattern2), (old_pattern3, new_pattern3)]:
    if old in content:
        content = content.replace(old, new)
        fixed = True
        print("✅ Found and fixed likelihood_ratio calculation")
        break

# Fix posterior update
if old_posterior in content:
    content = content.replace(old_posterior, new_posterior)
    print("✅ Fixed posterior update")
elif "posterior = (posterior * weighted_lr) /" in content:
    # Try without backslash
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if "posterior = (posterior * weighted_lr) /" in line:
            # Replace the next few lines
            new_lines = lines[:i] + [
                "                denominator_posterior = posterior * weighted_lr + (1 - posterior)",
                "                if denominator_posterior < 1e-10:",
                "                    posterior = 0.99",
                "                else:",
                "                    posterior = (posterior * weighted_lr) / denominator_posterior"
            ]
            # Find where this block ends
            j = i + 1
            while j < len(lines) and (lines[j].strip() == '' or lines[j].strip().startswith('(') or 'posterior' in lines[j]):
                j += 1
            content = '\n'.join(new_lines + lines[j:])
            print("✅ Fixed posterior update (alternative method)")
            break

if fixed:
    # Write back
    with open(ensemble_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n✅ Fix applied successfully!")
    print(f"   File size: {ensemble_file.stat().st_size:,} bytes")
    
    # Verify
    with open(ensemble_file, 'r') as f:
        verify = f.read()
    
    if 'denominator < 1e-10' in verify:
        print("✅ Verification passed - fix is in place")
    else:
        print("⚠️  Verification: pattern found but may need manual check")
else:
    print("\n⚠️  Could not find exact pattern")
    print("   The file structure may be different")
    print("\n💡 Manual fix needed:")
    print("   1. Open INSTITUTIONAL_ENSEMBLE_ENGINE.py")
    print("   2. Find the 'fuse' method in BayesianConfidenceFusion class")
    print("   3. Replace the likelihood_ratio calculation with safety checks")

print("\n" + "="*80)
print("✅ PATCH COMPLETE")
print("="*80)
print("\n🔄 NEXT STEPS:")
print("   1. Restart runtime (Runtime → Restart runtime)")
print("   2. Re-run the launcher")
print("="*80)

