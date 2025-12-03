"""
🔧 RESTORE _generate_signals METHOD
====================================
Completely restores the _generate_signals method with proper body
"""

import sys
from pathlib import Path
import re

print("="*80)
print("🔧 RESTORING _generate_signals METHOD")
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

# Find the method
print("\n🔍 Finding _generate_signals method...")
method_pattern = r'(def _generate_signals\(self, date, data_dict, capital, current_positions\) -> List\[Tuple\]:.*?)(?=\n    def |\nclass |\Z)'
method_match = re.search(method_pattern, content, re.DOTALL)

if method_match:
    print("   ✅ Found method")
    
    # Check if method body is missing
    method_content = method_match.group(1)
    if 'ranking_predictions = self.ranking_model.predict' not in method_content:
        print("   ⚠️  Method body is missing - restoring...")
        
        # Complete working method
        complete_method = """    def _generate_signals(self, date, data_dict, capital, current_positions) -> List[Tuple]:
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
        content = content[:method_match.start()] + complete_method + content[method_match.end():]
        print("   ✅ Method restored with complete body")
    else:
        print("   ✅ Method body exists - checking indentation...")
        
        # Just fix indentation if needed
        lines = method_content.split('\n')
        fixed_lines = []
        for i, line in enumerate(lines):
            if i == 0:
                fixed_lines.append(line)  # Keep def line
            elif i == 1 and line.strip().startswith('"""'):
                # Fix docstring indentation
                fixed_lines.append('        ' + line.lstrip())
            else:
                fixed_lines.append(line)
        
        fixed_method = '\n'.join(fixed_lines)
        if fixed_method != method_content:
            content = content[:method_match.start()] + fixed_method + content[method_match.end():]
            print("   ✅ Fixed indentation")
else:
    print("   ❌ Could not find method - file structure may be different")

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
    if hasattr(e, 'text') and e.text:
        print(f"   Line {e.lineno}: {e.text.strip()}")
    else:
        print(f"   Line {e.lineno}")

print("\n" + "="*80)
print("✅ RESTORE COMPLETE")
print("="*80)
print("\n🔄 NEXT STEPS:")
print("   1. Restart runtime (Runtime → Restart runtime)")
print("   2. Re-run your launcher")
print("="*80)

