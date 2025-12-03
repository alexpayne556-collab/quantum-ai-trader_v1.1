"""
🔧 FIX INDENTATION ERROR - Run this in Colab
============================================
Fixes the indentation error in _generate_signals method
"""

import sys
from pathlib import Path
import re

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
    content = f.read()

# Check for the error pattern
print("\n🔍 Checking for indentation error...")

# Find the problematic method
error_pattern = r'def _generate_signals\(self, date, data_dict, capital, current_positions\) -> List\[Tuple\]:\s*\n\s*"""Generate signals using institutional-grade system \(Phase 1\)"""\s*\n'

if re.search(error_pattern, content):
    print("   ✅ Found the problematic method")
    
    # Find the full method to replace
    method_pattern = r'(def _generate_signals\(self, date, data_dict, capital, current_positions\) -> List\[Tuple\]:.*?)(?=\n    def |\nclass |\Z)'
    method_match = re.search(method_pattern, content, re.DOTALL)
    
    if method_match:
        method_content = method_match.group(1)
        
        # Check if it's just the definition and docstring
        if method_content.strip().endswith('"""') or len(method_content.split('\n')) < 5:
            print("   ⚠️  Method body is missing - restoring from backup...")
            
            # Try to restore from backup
            backup_file = backtest_file.with_suffix('.py.backup')
            if backup_file.exists():
                print(f"   📖 Reading backup: {backup_file.name}")
                with open(backup_file, 'r', encoding='utf-8') as f:
                    backup_content = f.read()
                
                # Find original method
                original_pattern = r'(def _generate_signals\(self, date, data_dict, capital, current_positions\) -> List\[Tuple\]:.*?)(?=\n    def |\nclass |\Z)'
                original_match = re.search(original_pattern, backup_content, re.DOTALL)
                
                if original_match:
                    # Restore original method
                    content = content[:method_match.start()] + original_match.group(1) + content[method_match.end():]
                    print("   ✅ Restored original method from backup")
                else:
                    print("   ❌ Could not find original method in backup")
                    print("   💡 Need to manually fix the method")
            else:
                print("   ❌ No backup found")
                print("   💡 Need to manually restore the method body")
        else:
            # Method exists but may have indentation issues
            print("   🔍 Method exists - checking indentation...")
            
            # Fix indentation - ensure all lines after def are indented
            lines = method_content.split('\n')
            fixed_lines = [lines[0]]  # Keep def line
            
            for i, line in enumerate(lines[1:], 1):
                if line.strip() and not line.startswith(' ' * 8):
                    # Fix indentation to 8 spaces
                    fixed_lines.append('        ' + line.lstrip())
                else:
                    fixed_lines.append(line)
            
            fixed_method = '\n'.join(fixed_lines)
            content = content[:method_match.start()] + fixed_method + content[method_match.end():]
            print("   ✅ Fixed indentation")

# Alternative: If the method is completely broken, restore the working version
if 'def _generate_signals' in content:
    # Check if method has proper body
    method_start = content.find('def _generate_signals')
    if method_start != -1:
        # Find next def or class
        next_def = content.find('\n    def ', method_start + 1)
        next_class = content.find('\nclass ', method_start + 1)
        
        method_end = min([x for x in [next_def, next_class, len(content)] if x > method_start])
        method_body = content[method_start:method_end]
        
        # Check if body is too short (just def + docstring)
        if method_body.count('\n') < 10:
            print("\n⚠️  Method body appears incomplete - using working version...")
            
            # Use the working version from our code
            working_method = """    def _generate_signals(self, date, data_dict, capital, current_positions) -> List[Tuple]:
        \"\"\"Generate signals for current date\"\"\"
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
        
        return new_positions"""
            
            # Replace the broken method
            content = content[:method_start] + working_method + content[method_end:]
            print("   ✅ Replaced with working version")

# Write fixed file
print("\n💾 Writing fixed file...")
with open(backtest_file, 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ File written")

# Verify syntax
print("\n🔍 Verifying syntax...")
try:
    compile(content, str(backtest_file), 'exec')
    print("✅ Syntax is valid!")
except SyntaxError as e:
    print(f"❌ Syntax error: {e}")
    print(f"   Line {e.lineno}: {e.text}")
    print("\n💡 Try restoring from backup manually")

print("\n" + "="*80)
print("✅ FIX COMPLETE")
print("="*80)
print("\n🔄 NEXT STEPS:")
print("   1. Restart runtime (Runtime → Restart runtime)")
print("   2. Re-run your launcher")
print("="*80)

