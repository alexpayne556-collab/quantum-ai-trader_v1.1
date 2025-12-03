"""
🔧 FIX _generate_signals METHOD BODY
=====================================
Directly fixes the method by finding it and ensuring it has a complete body
"""

import sys
from pathlib import Path

print("="*80)
print("🔧 FIXING _generate_signals METHOD BODY")
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

print(f"   Total lines: {len(lines)}")

# Find the method definition
method_start = None
for i, line in enumerate(lines):
    if 'def _generate_signals(self, date, data_dict, capital, current_positions)' in line:
        method_start = i
        print(f"\n✅ Found method definition at line {i+1}")
        print(f"   Content: {line.strip()}")
        break

if method_start is None:
    print("\n❌ Could not find _generate_signals method")
    sys.exit(1)

# Check the next few lines
print(f"\n🔍 Checking lines {method_start+1} to {method_start+5}:")
for i in range(method_start, min(method_start + 10, len(lines))):
    marker = ">>> " if i == method_start else "    "
    print(f"{marker}Line {i+1:4d}: {repr(lines[i])}")

# Find where the method ends (next def or class at same or less indentation)
method_end = None
method_indent = len(lines[method_start]) - len(lines[method_start].lstrip())

for i in range(method_start + 1, len(lines)):
    line = lines[i]
    stripped = line.lstrip()
    
    # Skip empty lines and comments
    if not stripped or stripped.startswith('#'):
        continue
    
    # Check if this is a new method/class (same or less indentation)
    if stripped.startswith('def ') or stripped.startswith('class '):
        indent = len(line) - len(stripped)
        if indent <= method_indent:
            method_end = i
            break

if method_end is None:
    method_end = len(lines)
    print(f"\n   Method extends to end of file (line {len(lines)})")
else:
    print(f"\n   Method ends at line {method_end}")

# Check if method body exists
method_body_lines = lines[method_start+1:method_end]
method_body_text = ''.join(method_body_lines)

has_body = 'ranking_predictions = self.ranking_model.predict' in method_body_text

print(f"\n📊 Method analysis:")
print(f"   Has method body: {has_body}")
print(f"   Body length: {len(method_body_text)} chars")

if not has_body or len(method_body_text.strip()) < 50:
    print("\n⚠️  Method body is missing or incomplete - restoring...")
    
    # Complete method body
    method_body = """        \"\"\"Generate signals for current date\"\"\"
        # Run ranking model (Tier 1)
        ranking_predictions = self.ranking_model.predict(
            list(data_dict.keys()),
            data_dict
        )
        
        # Filter universe
        top_stocks = self.ensemble.filter_universe(ranking_predictions)
        
        # Exclude stocks we already have
        candidates = [s for s in top_stocks if s not in current_positions]
        
        new_positions = []
        
        for symbol in candidates:
            if len(new_positions) + len(current_positions) >= self.config['max_positions']:
                break
            
            data = data_dict[symbol]
            if len(data) < 20:
                continue
            
            # Gather signals (Tier 2)
            signals = []
            
            # Dark pool
            dp_result = self.dark_pool.analyze_ticker(symbol, data)
            if dp_result['signal'] == 'BUY':
                signals.append(Signal('dark_pool', 'BUY', dp_result['confidence'], dp_result))
                self.signals_by_module['dark_pool'].append((date, symbol, dp_result['confidence']))
            
            # Insider
            insider_result = self.insider.analyze_ticker(symbol, data)
            if insider_result['signal'] == 'BUY':
                signals.append(Signal('insider_trading', 'BUY', insider_result['confidence'], insider_result))
                self.signals_by_module['insider_trading'].append((date, symbol, insider_result['confidence']))
            
            # Patterns
            for scanner_name, scanner in [
                ('pregainer', self.pregainer),
                ('day_trading', self.day_trading),
                ('opportunity', self.opportunity)
            ]:
                pattern_result = scanner.scan(symbol, data)
                if pattern_result['signal'] == 'BUY':
                    signals.append(Signal(scanner_name, 'BUY', pattern_result['confidence']))
                    self.signals_by_module[scanner_name].append((date, symbol, pattern_result['confidence']))
            
            # Sentiment
            sentiment_score = self.sentiment.analyze(symbol, data)
            if sentiment_score > 0.6:
                signals.append(Signal('sentiment', 'BUY', sentiment_score))
                self.signals_by_module['sentiment'].append((date, symbol, sentiment_score))
            
            # Regime
            regime_result = self.regime.detect_regime(data)
            
            # Get ranking percentile
            ranking_percentile = self.ensemble.universe_filter.get_percentile(
                symbol, ranking_predictions
            )
            
            # Evaluate through ensemble
            decision = self.ensemble.evaluate_stock(
                symbol=symbol,
                signals=signals,
                ranking_percentile=ranking_percentile,
                regime=regime_result['regime']
            )
            
            # Only take BUY_FULL signals for backtest
            if decision['action'] == 'BUY_FULL' and len(signals) >= 2:
                entry_price = float(data['Close'].iloc[-1]) * (1 + self.config['slippage'])
                new_positions.append((symbol, entry_price, signals))
        
        return new_positions
"""
    
    # Replace the method body
    new_lines = (
        lines[:method_start+1] +  # Keep method definition
        method_body.splitlines(keepends=True) +  # Add complete body
        lines[method_end:]  # Keep rest of file
    )
    
    print("   ✅ Method body restored")
else:
    print("\n✅ Method body exists - checking for syntax issues...")
    
    # Check if docstring has wrong indentation
    if method_start + 1 < len(lines):
        docstring_line = lines[method_start + 1]
        if '"""' in docstring_line:
            # Check indentation
            expected_indent = '        '  # 8 spaces
            if not docstring_line.startswith(expected_indent):
                print("   ⚠️  Fixing docstring indentation...")
                lines[method_start + 1] = expected_indent + docstring_line.lstrip()
                new_lines = lines
                print("   ✅ Fixed")
            else:
                print("   ✅ Docstring indentation is correct")
                new_lines = lines
        else:
            new_lines = lines
    else:
        new_lines = lines

# Write fixed file
print("\n💾 Writing fixed file...")
with open(backtest_file, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("✅ File written")

# Verify syntax
print("\n🔍 Verifying syntax...")
try:
    compile(''.join(new_lines), str(backtest_file), 'exec')
    print("✅ Syntax is valid!")
except SyntaxError as e:
    print(f"❌ Syntax error: {e}")
    if hasattr(e, 'lineno') and e.lineno:
        line_num = e.lineno - 1
        if 0 <= line_num < len(new_lines):
            print(f"   Line {e.lineno}: {repr(new_lines[line_num])}")
    if hasattr(e, 'text') and e.text:
        print(f"   Problem text: {repr(e.text)}")

print("\n" + "="*80)
print("✅ FIX COMPLETE")
print("="*80)
print("\n🔄 NEXT STEPS:")
print("   1. Restart runtime (Runtime → Restart runtime)")
print("   2. Re-run your launcher")
print("="*80)

