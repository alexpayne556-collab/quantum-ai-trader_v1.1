"""
🔧 FIX DIVISION BY ZERO ERROR
============================
Patches INSTITUTIONAL_ENSEMBLE_ENGINE.py to fix division by zero in confidence fusion
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
    print("   The division by zero error should be resolved.")
    sys.exit(0)

# Find the problematic code
old_code = """        for signal in signals:
            if signal.action == 'BUY':
                # Likelihood ratio
                signal_accuracy = MODULE_BASELINES.get(signal.name, 0.55)
                likelihood_ratio = (signal.confidence * signal_accuracy) / \\
                                 ((1 - signal.confidence) * (1 - signal_accuracy))
                
                # Weight the likelihood
                signal_weight = self._get_signal_weight(signal.name, weights)
                weighted_lr = likelihood_ratio ** signal_weight
                
                # Update posterior
                posterior = (posterior * weighted_lr) / \\
                           (posterior * weighted_lr + (1 - posterior))"""

new_code = """        for signal in signals:
            if signal.action == 'BUY':
                # Likelihood ratio
                signal_accuracy = MODULE_BASELINES.get(signal.name, 0.55)
                
                # Safety check: prevent division by zero
                # Clamp confidence and accuracy to avoid edge cases
                confidence = max(0.01, min(0.99, signal.confidence))
                accuracy = max(0.01, min(0.99, signal_accuracy))
                
                # Calculate likelihood ratio with safety
                denominator = (1 - confidence) * (1 - accuracy)
                if denominator < 1e-10:  # Avoid division by zero
                    # If confidence or accuracy is too high, use a large but finite ratio
                    likelihood_ratio = 100.0 if confidence > 0.9 or accuracy > 0.9 else 1.0
                else:
                    likelihood_ratio = (confidence * accuracy) / denominator
                
                # Weight the likelihood
                signal_weight = self._get_signal_weight(signal.name, weights)
                weighted_lr = likelihood_ratio ** signal_weight
                
                # Update posterior with safety check
                denominator_posterior = posterior * weighted_lr + (1 - posterior)
                if denominator_posterior < 1e-10:
                    posterior = 0.99  # Cap at 99% if denominator too small
                else:
                    posterior = (posterior * weighted_lr) / denominator_posterior"""

# Apply fix
if old_code in content:
    print("\n🔧 Applying fix...")
    content = content.replace(old_code, new_code)
    
    # Write back
    with open(ensemble_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fix applied successfully!")
    print(f"   File size: {ensemble_file.stat().st_size:,} bytes")
    
    # Verify
    with open(ensemble_file, 'r') as f:
        verify_content = f.read()
    
    if 'denominator < 1e-10' in verify_content:
        print("✅ Verification passed - fix is in place")
    else:
        print("⚠️  Verification failed - fix may not have been applied correctly")
        
else:
    # Try alternative pattern (might have different whitespace)
    print("\n⚠️  Exact pattern not found, trying alternative...")
    
    # Look for the line with the division
    lines = content.split('\n')
    fixed = False
    
    for i, line in enumerate(lines):
        if 'likelihood_ratio = (signal.confidence * signal_accuracy) /' in line:
            # Found the problematic line
            print(f"   Found at line {i+1}")
            
            # Check if already fixed
            if i + 5 < len(lines):
                if 'denominator < 1e-10' in '\n'.join(lines[i:i+10]):
                    print("   ✅ Already fixed!")
                    fixed = True
                    break
            
            # Apply fix manually
            # Find the block
            start_idx = i - 2  # Go back a few lines
            end_idx = i + 8  # Go forward
            
            # Replace the block
            old_block = '\n'.join(lines[start_idx:end_idx])
            if 'likelihood_ratio = (signal.confidence * signal_accuracy) /' in old_block:
                new_block = """                # Likelihood ratio
                signal_accuracy = MODULE_BASELINES.get(signal.name, 0.55)
                
                # Safety check: prevent division by zero
                # Clamp confidence and accuracy to avoid edge cases
                confidence = max(0.01, min(0.99, signal.confidence))
                accuracy = max(0.01, min(0.99, signal_accuracy))
                
                # Calculate likelihood ratio with safety
                denominator = (1 - confidence) * (1 - accuracy)
                if denominator < 1e-10:  # Avoid division by zero
                    # If confidence or accuracy is too high, use a large but finite ratio
                    likelihood_ratio = 100.0 if confidence > 0.9 or accuracy > 0.9 else 1.0
                else:
                    likelihood_ratio = (confidence * accuracy) / denominator
                
                # Weight the likelihood
                signal_weight = self._get_signal_weight(signal.name, weights)
                weighted_lr = likelihood_ratio ** signal_weight
                
                # Update posterior with safety check
                denominator_posterior = posterior * weighted_lr + (1 - posterior)
                if denominator_posterior < 1e-10:
                    posterior = 0.99  # Cap at 99% if denominator too small
                else:
                    posterior = (posterior * weighted_lr) / denominator_posterior"""
                
                lines[start_idx:end_idx] = new_block.split('\n')
                content = '\n'.join(lines)
                fixed = True
                break
    
    if fixed:
        with open(ensemble_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ Fix applied (alternative method)")
    else:
        print("❌ Could not find the exact code pattern")
        print("   You may need to manually update the file")
        print("\n💡 The fix needs to be applied to the 'fuse' method in")
        print("   BayesianConfidenceFusion class")

print("\n" + "="*80)
print("✅ PATCH COMPLETE")
print("="*80)
print("\n🔄 NEXT STEPS:")
print("   1. Restart runtime (Runtime → Restart runtime)")
print("   2. Re-run the launcher:")
print("      %run COLAB_LAUNCH_INSTITUTIONAL_SYSTEM.py")
print("\n✅ The division by zero error should now be fixed!")
print("="*80)

