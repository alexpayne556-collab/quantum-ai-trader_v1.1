"""
🔧 SIMPLE FIX - Indentation Error
==================================
Fixes the specific indentation error on line 375
"""

import sys
from pathlib import Path

print("="*80)
print("🔧 FIXING INDENTATION ERROR (Line 375)")
print("="*80)

BASE_DIR = Path('/content/drive/MyDrive/QuantumAI')
MODULES_DIR = BASE_DIR / 'backend' / 'modules'
backtest_file = MODULES_DIR / 'BACKTEST_INSTITUTIONAL_ENSEMBLE.py'

if not backtest_file.exists():
    print(f"❌ File not found: {backtest_file}")
    sys.exit(1)

print(f"\n📁 File: {backtest_file}")

# Read file as lines
print("📖 Reading file...")
with open(backtest_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find line 374-375 (0-indexed is 373-374)
print("\n🔍 Checking lines 373-380...")
for i in range(372, min(380, len(lines))):
    print(f"  {i+1:3d}: {repr(lines[i])}")

# Fix line 375 (index 374)
if len(lines) > 374:
    line_374 = lines[373]  # Line 374 (0-indexed: 373)
    line_375 = lines[374]  # Line 375 (0-indexed: 374)
    
    print(f"\n🔍 Line 374: {repr(line_374)}")
    print(f"🔍 Line 375: {repr(line_375)}")
    
    # Check if line 375 is the docstring with wrong indentation
    if '"""Generate signals using institutional-grade system (Phase 1)"""' in line_375:
        print("\n🔧 Fixing docstring indentation...")
        
        # The docstring should be indented with 8 spaces (2 levels)
        if not line_375.startswith('        """'):
            # Fix it
            lines[374] = '        """Generate signals using institutional-grade system (Phase 1)"""\n'
            print("   ✅ Fixed docstring indentation")
        else:
            print("   ⚠️  Docstring already has correct indentation")
            
            # The issue might be that there's no code after the docstring
            # Check if line 376 exists and has code
            if len(lines) > 375:
                line_376 = lines[375]
                print(f"🔍 Line 376: {repr(line_376)}")
                
                if not line_376.strip() or line_376.strip().startswith('def ') or line_376.strip().startswith('class '):
                    print("   ⚠️  No code after docstring - method body is missing!")
                    
                    # Need to add the method body
                    # Find where the method should end (next def or class)
                    method_end = len(lines)
                    for i in range(375, len(lines)):
                        if lines[i].strip().startswith('def ') or lines[i].strip().startswith('class '):
                            method_end = i
                            break
                    
                    # Insert the method body
                    method_body = [
                        '        # Run ranking model (Tier 1)\n',
                        '        ranking_predictions = self.ranking_model.predict(\n',
                        '            list(data_dict.keys()),\n',
                        '            data_dict\n',
                        '        )\n',
                        '        \n',
                        '        # Filter universe\n',
                        '        top_stocks = self.ensemble.filter_universe(ranking_predictions)\n',
                        '        \n',
                        '        # Exclude stocks we already have\n',
                        '        candidates = [s for s in top_stocks if s not in current_positions]\n',
                        '        \n',
                        '        new_positions = []\n',
                        '        \n',
                        '        for symbol in candidates:\n',
                        '            if len(new_positions) + len(current_positions) >= self.config[\'max_positions\']:\n',
                        '                break\n',
                        '            \n',
                        '            data = data_dict[symbol]\n',
                        '            if len(data) < 20:\n',
                        '                continue\n',
                        '            \n',
                        '            # Gather signals (Tier 2)\n',
                        '            signals = []\n',
                        '            \n',
                        '            # Dark pool\n',
                        '            dp_result = self.dark_pool.analyze_ticker(symbol, data)\n',
                        '            if dp_result[\'signal\'] == \'BUY\':\n',
                        '                signals.append(Signal(\'dark_pool\', \'BUY\', dp_result[\'confidence\'], dp_result))\n',
                        '                self.signals_by_module[\'dark_pool\'].append((date, symbol, dp_result[\'confidence\']))\n',
                        '            \n',
                        '            # Insider\n',
                        '            insider_result = self.insider.analyze_ticker(symbol, data)\n',
                        '            if insider_result[\'signal\'] == \'BUY\':\n',
                        '                signals.append(Signal(\'insider_trading\', \'BUY\', insider_result[\'confidence\'], insider_result))\n',
                        '                self.signals_by_module[\'insider_trading\'].append((date, symbol, insider_result[\'confidence\']))\n',
                        '            \n',
                        '            # Patterns\n',
                        '            for scanner_name, scanner in [\n',
                        '                (\'pregainer\', self.pregainer),\n',
                        '                (\'day_trading\', self.day_trading),\n',
                        '                (\'opportunity\', self.opportunity)\n',
                        '            ]:\n',
                        '                pattern_result = scanner.scan(symbol, data)\n',
                        '                if pattern_result[\'signal\'] == \'BUY\':\n',
                        '                    signals.append(Signal(scanner_name, \'BUY\', pattern_result[\'confidence\']))\n',
                        '                    self.signals_by_module[scanner_name].append((date, symbol, pattern_result[\'confidence\']))\n',
                        '            \n',
                        '            # Sentiment\n',
                        '            sentiment_score = self.sentiment.analyze(symbol, data)\n',
                        '            if sentiment_score > 0.6:\n',
                        '                signals.append(Signal(\'sentiment\', \'BUY\', sentiment_score))\n',
                        '                self.signals_by_module[\'sentiment\'].append((date, symbol, sentiment_score))\n',
                        '            \n',
                        '            # Regime\n',
                        '            regime_result = self.regime.detect_regime(data)\n',
                        '            \n',
                        '            # Get ranking percentile\n',
                        '            ranking_percentile = self.ensemble.universe_filter.get_percentile(\n',
                        '                symbol, ranking_predictions\n',
                        '            )\n',
                        '            \n',
                        '            # Evaluate through ensemble\n',
                        '            decision = self.ensemble.evaluate_stock(\n',
                        '                symbol=symbol,\n',
                        '                signals=signals,\n',
                        '                ranking_percentile=ranking_percentile,\n',
                        '                regime=regime_result[\'regime\']\n',
                        '            )\n',
                        '            \n',
                        '            # Only take BUY_FULL signals for backtest\n',
                        '            if decision[\'action\'] == \'BUY_FULL\' and len(signals) >= 2:\n',
                        '                entry_price = float(data[\'Close\'].iloc[-1]) * (1 + self.config[\'slippage\'])\n',
                        '                new_positions.append((symbol, entry_price, signals))\n',
                        '        \n',
                        '        return new_positions\n'
                    ]
                    
                    # Insert before method_end
                    lines = lines[:376] + method_body + lines[method_end:]
                    print("   ✅ Added method body")
            else:
                print("   ⚠️  File ends after docstring - need to add method body")
                # Add method body at end
                method_body = [
                    '        # Run ranking model (Tier 1)\n',
                    '        ranking_predictions = self.ranking_model.predict(\n',
                    '            list(data_dict.keys()),\n',
                    '            data_dict\n',
                    '        )\n',
                    '        \n',
                    '        # Filter universe\n',
                    '        top_stocks = self.ensemble.filter_universe(ranking_predictions)\n',
                    '        \n',
                    '        # Exclude stocks we already have\n',
                    '        candidates = [s for s in top_stocks if s not in current_positions]\n',
                    '        \n',
                    '        new_positions = []\n',
                    '        \n',
                    '        for symbol in candidates:\n',
                    '            if len(new_positions) + len(current_positions) >= self.config[\'max_positions\']:\n',
                    '                break\n',
                    '            \n',
                    '            data = data_dict[symbol]\n',
                    '            if len(data) < 20:\n',
                    '                continue\n',
                    '            \n',
                    '            # Gather signals (Tier 2)\n',
                    '            signals = []\n',
                    '            \n',
                    '            # Dark pool\n',
                    '            dp_result = self.dark_pool.analyze_ticker(symbol, data)\n',
                    '            if dp_result[\'signal\'] == \'BUY\':\n',
                    '                signals.append(Signal(\'dark_pool\', \'BUY\', dp_result[\'confidence\'], dp_result))\n',
                    '                self.signals_by_module[\'dark_pool\'].append((date, symbol, dp_result[\'confidence\']))\n',
                    '            \n',
                    '            # Insider\n',
                    '            insider_result = self.insider.analyze_ticker(symbol, data)\n',
                    '            if insider_result[\'signal\'] == \'BUY\':\n',
                    '                signals.append(Signal(\'insider_trading\', \'BUY\', insider_result[\'confidence\'], insider_result))\n',
                    '                self.signals_by_module[\'insider_trading\'].append((date, symbol, insider_result[\'confidence\']))\n',
                    '            \n',
                    '            # Patterns\n',
                    '            for scanner_name, scanner in [\n',
                    '                (\'pregainer\', self.pregainer),\n',
                    '                (\'day_trading\', self.day_trading),\n',
                    '                (\'opportunity\', self.opportunity)\n',
                    '            ]:\n',
                    '                pattern_result = scanner.scan(symbol, data)\n',
                    '                if pattern_result[\'signal\'] == \'BUY\':\n',
                    '                    signals.append(Signal(scanner_name, \'BUY\', pattern_result[\'confidence\']))\n',
                    '                    self.signals_by_module[scanner_name].append((date, symbol, pattern_result[\'confidence\']))\n',
                    '            \n',
                    '            # Sentiment\n',
                    '            sentiment_score = self.sentiment.analyze(symbol, data)\n',
                    '            if sentiment_score > 0.6:\n',
                    '                signals.append(Signal(\'sentiment\', \'BUY\', sentiment_score))\n',
                    '                self.signals_by_module[\'sentiment\'].append((date, symbol, sentiment_score))\n',
                    '            \n',
                    '            # Regime\n',
                    '            regime_result = self.regime.detect_regime(data)\n',
                    '            \n',
                    '            # Get ranking percentile\n',
                    '            ranking_percentile = self.ensemble.universe_filter.get_percentile(\n',
                    '                symbol, ranking_predictions\n',
                    '            )\n',
                    '            \n',
                    '            # Evaluate through ensemble\n',
                    '            decision = self.ensemble.evaluate_stock(\n',
                    '                symbol=symbol,\n',
                    '                signals=signals,\n',
                    '                ranking_percentile=ranking_percentile,\n',
                    '                regime=regime_result[\'regime\']\n',
                    '            )\n',
                    '            \n',
                    '            # Only take BUY_FULL signals for backtest\n',
                    '            if decision[\'action\'] == \'BUY_FULL\' and len(signals) >= 2:\n',
                    '                entry_price = float(data[\'Close\'].iloc[-1]) * (1 + self.config[\'slippage\'])\n',
                    '                new_positions.append((symbol, entry_price, signals))\n',
                    '        \n',
                    '        return new_positions\n'
                ]
                lines.extend(method_body)
                print("   ✅ Added method body at end")

# Write fixed file
print("\n💾 Writing fixed file...")
with open(backtest_file, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("✅ File written")

# Verify syntax
print("\n🔍 Verifying syntax...")
try:
    content = ''.join(lines)
    compile(content, str(backtest_file), 'exec')
    print("✅ Syntax is valid!")
except SyntaxError as e:
    print(f"❌ Syntax error: {e}")
    print(f"   Line {e.lineno}: {e.text if e.text else 'N/A'}")

print("\n" + "="*80)
print("✅ FIX COMPLETE")
print("="*80)

