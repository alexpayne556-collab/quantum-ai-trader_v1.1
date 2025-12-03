# ============================================================================
# 🔧 INLINE FIX FOR _generate_signals - Paste this directly into Colab
# ============================================================================

import sys
from pathlib import Path

print("="*80)
print("🔧 FIXING _generate_signals METHOD")
print("="*80)

BASE_DIR = Path('/content/drive/MyDrive/QuantumAI')
MODULES_DIR = BASE_DIR / 'backend' / 'modules'
backtest_file = MODULES_DIR / 'BACKTEST_INSTITUTIONAL_ENSEMBLE.py'

if not backtest_file.exists():
    print(f"❌ File not found: {backtest_file}")
else:
    # Read entire file
    with open(backtest_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the method and replace it completely
    import re
    
    # Pattern to match the entire method (from def to next def/class)
    pattern = r'(    def _generate_signals\(self, date, data_dict, capital, current_positions\) -> List\[Tuple\]:.*?)(?=\n    def |\nclass |\Z)'
    
    # Complete replacement method
    replacement = """    def _generate_signals(self, date, data_dict, capital, current_positions) -> List[Tuple]:
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
    
    # Replace in content
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        print("✅ Method found and replaced")
    else:
        print("⚠️  Method pattern not found - trying alternative approach...")
        # Try finding just the def line and replacing everything after it
        def_pattern = r'(    def _generate_signals\(self, date, data_dict, capital, current_positions\) -> List\[Tuple\]:)'
        if re.search(def_pattern, content):
            # Find the def line
            match = re.search(def_pattern, content)
            start_pos = match.start()
            
            # Find where next method/class starts
            remaining = content[start_pos:]
            next_def = re.search(r'\n    def |\nclass ', remaining)
            if next_def:
                end_pos = start_pos + next_def.start()
            else:
                end_pos = len(content)
            
            # Replace
            content = content[:start_pos] + replacement + '\n' + content[end_pos:]
            print("✅ Method replaced using alternative method")
        else:
            print("❌ Could not find method definition")
            sys.exit(1)
    
    # Write back
    with open(backtest_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ File written")
    
    # Verify
    try:
        compile(content, str(backtest_file), 'exec')
        print("✅ Syntax verified!")
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        print(f"   Line {e.lineno}")

print("\n" + "="*80)
print("✅ FIX COMPLETE")
print("="*80)
print("\n🔄 Restart runtime and re-run launcher")
print("="*80)

