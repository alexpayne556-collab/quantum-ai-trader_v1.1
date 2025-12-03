"""
🔧 QUICK PATCH - Fix Backtest Ranking Issue
============================================
Run this in Colab to patch the file in-place
"""

import os

print("🔧 PATCHING BACKTEST FILE...")
print("="*80)

filepath = '/content/drive/MyDrive/QuantumAI/backend/modules/BACKTEST_INSTITUTIONAL_ENSEMBLE.py'

# Read file
with open(filepath, 'r') as f:
    content = f.read()

# Apply fix
old_code = """            # Simple momentum-based prediction
            returns_20d = (data['Close'].iloc[-1] / data['Close'].iloc[-20] - 1)
            predictions[symbol] = returns_20d * 0.5  # Predict mean reversion"""

new_code = """            # Simple momentum-based prediction
            returns_20d = (data['Close'].iloc[-1] / data['Close'].iloc[-20] - 1)
            # Convert to float to avoid pandas Series issues
            predictions[symbol] = float(returns_20d * 0.5)  # Predict mean reversion"""

if old_code in content:
    content = content.replace(old_code, new_code)
    
    # Write back
    with open(filepath, 'w') as f:
        f.write(content)
    
    print("✅ PATCH APPLIED!")
    print("\nNow re-run the launcher:")
    print("%run COLAB_LAUNCH_INSTITUTIONAL_SYSTEM.py")
else:
    print("⚠️  Code already patched or pattern not found")
    print("Trying alternative fix...")
    
    # Alternative: just replace the problematic line
    content = content.replace(
        'predictions[symbol] = returns_20d * 0.5',
        'predictions[symbol] = float(returns_20d * 0.5)'
    )
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print("✅ Alternative fix applied!")

print("="*80)

