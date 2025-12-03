# ================================================================================
# 🔧 QUICK FIXES FOR COLAB
# ================================================================================
# Run this in your Colab cell to fix both issues
# ================================================================================

import json
import numpy as np

# ================================================================================
# FIX #1: JSON SERIALIZATION ERROR
# ================================================================================

print("🔧 Fixing JSON serialization error...")

# The error is from numpy bool_ type. Add this before json.dump:
def convert_to_json_serializable(obj):
    """Convert numpy types to Python types."""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# Apply this to metadata before saving:
# metadata_clean = convert_to_json_serializable(metadata)
# json.dump(metadata_clean, f, indent=2)

print("✅ JSON fix ready!")

# ================================================================================
# FIX #2: PATTERN DETECTION IS TOO STRICT
# ================================================================================

print("\n🔧 Pattern detection issue detected:")
print("   Problem: Only 22-686 samples per pattern (need more!)")
print("   Cause: Detection logic is TOO STRICT")
print("\n   Solutions:")
print("   1. Relax pattern detection rules")
print("   2. Use more stocks (100+ instead of 40)")
print("   3. Use longer timeframe (5 years instead of 3)")
print("   4. Lower quality threshold (50 instead of 60)")

# Quick fix for your current run:
# Add this to the metadata save section:

QUICK_FIX_CODE = """
# PASTE THIS IN YOUR COLAB CELL (replace the metadata saving section):

# Convert numpy types to Python types
def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(item) for item in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    else:
        return obj

# Clean metadata before saving
metadata_clean = make_json_safe(metadata)

# Now save
with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
    json.dump(metadata_clean, f, indent=2)
"""

print("\n" + "="*80)
print("📋 QUICK FIX CODE:")
print("="*80)
print(QUICK_FIX_CODE)
print("="*80)

# ================================================================================
# BETTER SOLUTION: RELAXED PATTERN DETECTION
# ================================================================================

print("\n💡 RECOMMENDED: Use relaxed detection for more samples")
print("\nChange these in your training script:")
print("""
# BEFORE (Too strict):
if not pattern_config['rule'](features):
    continue  # ❌ Skips most patterns!

# AFTER (Relaxed):
if pattern_config['rule'](features):
    has_pattern = 1
else:
    has_pattern = 0  # ✅ Include ALL, let ML decide!

# Then add 'has_pattern' as a feature
features['has_pattern'] = has_pattern

# This way ML learns from both pattern & non-pattern examples!
""")

print("\n✅ This will give you 10,000+ samples instead of 22!")

