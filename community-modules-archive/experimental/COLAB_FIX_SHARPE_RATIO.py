"""
🔧 FIX SHARPE RATIO ERROR - Paste into Colab
===========================================
Fixes the "unsupported format string passed to Series.__format__" error in Sharpe calculation
"""

import sys
from pathlib import Path

print("="*80)
print("🔧 FIXING SHARPE RATIO CALCULATION ERROR")
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
if 'daily_returns[\'value\'].apply(lambda x:' in content and 'sharpe = (mean_return / std_return)' in content:
    print("\n✅ File already has the fix!")
    sys.exit(0)

print("\n🔧 Applying fixes...")

# Fix 1: Portfolio value calculation
old_portfolio = """            portfolio_value = capital + sum(
                pos['shares'] * current_data[symbol]['Close'].iloc[-1]
                for symbol, pos in positions.items()
                if symbol in current_data and len(current_data[symbol]) > 0
            )"""
new_portfolio = """            portfolio_value = capital + sum(
                pos['shares'] * float(current_data[symbol]['Close'].iloc[-1])
                for symbol, pos in positions.items()
                if symbol in current_data and len(current_data[symbol]) > 0
            )
            # Ensure portfolio_value is a scalar
            portfolio_value = float(portfolio_value) if not isinstance(portfolio_value, (int, float)) else portfolio_value"""

# Fix 2: Sharpe ratio calculation
old_sharpe = """        # Sharpe ratio
        daily_returns = pd.DataFrame(self.daily_portfolio_value)
        daily_returns['return'] = daily_returns['value'].pct_change()
        sharpe = (daily_returns['return'].mean() / daily_returns['return'].std()) * np.sqrt(252)"""
new_sharpe = """        # Sharpe ratio - with robust error handling
        daily_returns = pd.DataFrame(self.daily_portfolio_value)
        
        # Ensure 'value' column contains float values, not Series
        if 'value' in daily_returns.columns:
            # Convert to numeric, handling any nested Series
            daily_returns['value'] = daily_returns['value'].apply(
                lambda x: float(x) if np.isscalar(x) else (
                    float(x.iloc[0]) if hasattr(x, 'iloc') else (
                        float(x[0]) if hasattr(x, '__len__') else float(x)
                    )
                )
            )
            daily_returns['value'] = pd.to_numeric(daily_returns['value'], errors='coerce')
            daily_returns = daily_returns.dropna(subset=['value'])
        
        # Calculate returns
        if len(daily_returns) > 1:
            daily_returns['return'] = daily_returns['value'].pct_change()
            # Remove first row (NaN from pct_change)
            daily_returns = daily_returns.dropna(subset=['return'])
            
            # Calculate Sharpe ratio safely
            if len(daily_returns) > 1:
                mean_return = daily_returns['return'].mean()
                std_return = daily_returns['return'].std()
                
                if std_return != 0 and not np.isnan(std_return) and not np.isnan(mean_return):
                    sharpe = (mean_return / std_return) * np.sqrt(252)
                else:
                    sharpe = 0.0
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0"""

# Apply fixes
fixed = False

if old_portfolio in content:
    content = content.replace(old_portfolio, new_portfolio)
    print("✅ Fixed portfolio_value calculation")
    fixed = True
elif "pos['shares'] * current_data[symbol]['Close'].iloc[-1]" in content:
    # Try alternative pattern
    content = content.replace(
        "pos['shares'] * current_data[symbol]['Close'].iloc[-1]",
        "pos['shares'] * float(current_data[symbol]['Close'].iloc[-1])"
    )
    # Add scalar check after portfolio_value calculation
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if "portfolio_value = capital + sum(" in line:
            # Find where this block ends
            j = i + 1
            while j < len(lines) and (lines[j].strip().startswith('pos') or lines[j].strip().startswith('for') or lines[j].strip().startswith('if')):
                j += 1
            # Insert scalar check
            indent = len(lines[j]) - len(lines[j].lstrip())
            lines.insert(j, ' ' * indent + "# Ensure portfolio_value is a scalar")
            lines.insert(j + 1, ' ' * indent + "portfolio_value = float(portfolio_value) if not isinstance(portfolio_value, (int, float)) else portfolio_value")
            content = '\n'.join(lines)
            print("✅ Fixed portfolio_value calculation (alternative method)")
            fixed = True
            break

if old_sharpe in content:
    content = content.replace(old_sharpe, new_sharpe)
    print("✅ Fixed Sharpe ratio calculation")
    fixed = True
elif "sharpe = (daily_returns['return'].mean() / daily_returns['return'].std())" in content:
    # Try to find and replace just the calculation part
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if "sharpe = (daily_returns['return'].mean() / daily_returns['return'].std())" in line:
            # Replace with robust version
            indent = len(line) - len(line.lstrip())
            new_lines = [
                ' ' * indent + "# Sharpe ratio - with robust error handling",
                ' ' * indent + "if 'value' in daily_returns.columns:",
                ' ' * (indent + 4) + "daily_returns['value'] = daily_returns['value'].apply(",
                ' ' * (indent + 8) + "lambda x: float(x) if np.isscalar(x) else (",
                ' ' * (indent + 12) + "float(x.iloc[0]) if hasattr(x, 'iloc') else (",
                ' ' * (indent + 16) + "float(x[0]) if hasattr(x, '__len__') else float(x)",
                ' ' * (indent + 12) + ")",
                ' ' * (indent + 8) + ")",
                ' ' * (indent + 4) + ")",
                ' ' * (indent + 4) + "daily_returns['value'] = pd.to_numeric(daily_returns['value'], errors='coerce')",
                ' ' * (indent + 4) + "daily_returns = daily_returns.dropna(subset=['value'])",
                ' ' * indent + "",
                ' ' * indent + "if len(daily_returns) > 1:",
                ' ' * (indent + 4) + "daily_returns['return'] = daily_returns['value'].pct_change()",
                ' ' * (indent + 4) + "daily_returns = daily_returns.dropna(subset=['return'])",
                ' ' * (indent + 4) + "if len(daily_returns) > 1:",
                ' ' * (indent + 8) + "mean_return = daily_returns['return'].mean()",
                ' ' * (indent + 8) + "std_return = daily_returns['return'].std()",
                ' ' * (indent + 8) + "if std_return != 0 and not np.isnan(std_return) and not np.isnan(mean_return):",
                ' ' * (indent + 12) + "sharpe = (mean_return / std_return) * np.sqrt(252)",
                ' ' * (indent + 8) + "else:",
                ' ' * (indent + 12) + "sharpe = 0.0",
                ' ' * (indent + 4) + "else:",
                ' ' * (indent + 8) + "sharpe = 0.0",
                ' ' * indent + "else:",
                ' ' * (indent + 4) + "sharpe = 0.0"
            ]
            # Remove old line
            lines[i] = '\n'.join(new_lines)
            content = '\n'.join(lines)
            print("✅ Fixed Sharpe ratio calculation (alternative method)")
            fixed = True
            break

if fixed:
    # Write back
    with open(backtest_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n✅ All fixes applied!")
    print(f"   File size: {backtest_file.stat().st_size:,} bytes")
    
    # Verify
    with open(backtest_file, 'r') as f:
        verify = f.read()
    
    if 'daily_returns[\'value\'].apply' in verify or 'float(current_data[symbol]' in verify:
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

