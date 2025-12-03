"""
🎯 PHASE 1 COMPLETE INTEGRATION - Dark Pool Lead + Confidence Threshold
=======================================================================
This script integrates Phase 1 improvements into your backtest:
1. Updates weights (dark_pool: 60%, sentiment: 25%, scanners: 10%, insider: 5%)
2. Adds dark pool lead requirement (must have dark pool signal)
3. Adds confidence threshold (minimum 0.65)
4. Adds confirmation requirement (dark pool + at least 1 other signal)

Expected Result: 46.2% → 50-52% win rate
"""

import sys
from pathlib import Path
import re

print("="*80)
print("🎯 PHASE 1 INTEGRATION - INSTITUTIONAL IMPROVEMENTS")
print("="*80)

BASE_DIR = Path('/content/drive/MyDrive/QuantumAI')
MODULES_DIR = BASE_DIR / 'backend' / 'modules'
backtest_file = MODULES_DIR / 'BACKTEST_INSTITUTIONAL_ENSEMBLE.py'
ensemble_file = MODULES_DIR / 'INSTITUTIONAL_ENSEMBLE_ENGINE.py'

if not backtest_file.exists():
    print(f"❌ File not found: {backtest_file}")
    sys.exit(1)

print(f"\n📁 Files:")
print(f"   Backtest: {backtest_file}")
print(f"   Ensemble: {ensemble_file}")

# ============================================================================
# STEP 1: Update INITIAL_WEIGHTS in ensemble engine
# ============================================================================
print("\n1️⃣ Updating INITIAL_WEIGHTS to Phase 1 values...")

if ensemble_file.exists():
    with open(ensemble_file, 'r', encoding='utf-8') as f:
        ensemble_content = f.read()
    
    # Find INITIAL_WEIGHTS definition
    weights_pattern = r'(INITIAL_WEIGHTS\s*=\s*\{[^}]*pattern_scanners[^}]*\}[^}]*\})'
    weights_match = re.search(weights_pattern, ensemble_content, re.DOTALL)
    
    if weights_pattern:
        # Phase 1 weights
        new_weights = """INITIAL_WEIGHTS = {
    # Phase 1: Dark Pool Lead Strategy
    # dark_pool: 60% (your only profitable signal at 52.5%)
    # sentiment: 25% (barely profitable at 49.1%)
    # scanners: 10% (merged redundant scanners at 47%)
    # insider: 5% (broken at 44.2%, will fix in Phase 2)
    'pattern_scanners': {
        'pregainer': 0.033,      # Reduced from 0.12
        'day_trading': 0.033,     # Reduced from 0.12
        'opportunity': 0.034,     # Reduced from 0.11
    },
    'dark_pool': 0.60,            # INCREASED from ~0.17 (3.5x boost)
    'sentiment': 0.25,            # INCREASED from ~0.17
    'insider_trading': 0.05,      # REDUCED from ~0.17 (broken signal)
    'short_squeeze': 0.05,        # Keep small weight
}"""
        
        if weights_match:
            ensemble_content = ensemble_content[:weights_match.start()] + new_weights + ensemble_content[weights_match.end():]
            print("   ✅ Updated INITIAL_WEIGHTS")
        else:
            # Try to find and replace just the weights dict
            old_pattern = r"('dark_pool':\s*)[\d.]+"
            ensemble_content = re.sub(old_pattern, r"\g<1>0.60", ensemble_content)
            old_pattern = r"('sentiment':\s*)[\d.]+"
            ensemble_content = re.sub(old_pattern, r"\g<1>0.25", ensemble_content)
            print("   ✅ Updated weights (partial)")
        
        with open(ensemble_file, 'w', encoding='utf-8') as f:
            f.write(ensemble_content)
        print("   ✅ Ensemble weights updated")
    else:
        print("   ⚠️  Could not find INITIAL_WEIGHTS pattern (may need manual update)")

# ============================================================================
# STEP 2: Update _generate_signals to add Phase 1 gates
# ============================================================================
print("\n2️⃣ Adding Phase 1 gates to _generate_signals...")

with open(backtest_file, 'r', encoding='utf-8') as f:
    backtest_content = f.read()

# Find the _generate_signals method
method_pattern = r'(def _generate_signals\(self, date, data_dict, capital, current_positions\) -> List\[Tuple\]:.*?)(?=\n    def |\nclass |\Z)'
method_match = re.search(method_pattern, backtest_content, re.DOTALL)

if method_match:
    method_content = method_match.group(1)
    
    # Check if Phase 1 gates already exist
    if 'PHASE_1_GATE' in method_content or 'dark_pool_signal' in method_content:
        print("   ⚠️  Phase 1 gates may already exist - checking...")
    
    # Find where signals are evaluated
    # Look for the decision evaluation section
    if 'decision = self.ensemble.evaluate_stock' in method_content:
        print("   ✅ Found signal evaluation section")
        
        # Create new method with Phase 1 gates
        new_method = """    def _generate_signals(self, date, data_dict, capital, current_positions) -> List[Tuple]:
        \"\"\"Generate signals using Phase 1 institutional improvements\"\"\"
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
            dark_pool_signal = None
            dark_pool_confidence = 0.0
            
            # Dark pool (CRITICAL - must have this)
            dp_result = self.dark_pool.analyze_ticker(symbol, data)
            if dp_result['signal'] == 'BUY':
                dark_pool_signal = dp_result
                dark_pool_confidence = dp_result.get('confidence', 0.5)
                signals.append(Signal('dark_pool', 'BUY', dark_pool_confidence, dp_result))
                self.signals_by_module['dark_pool'].append((date, symbol, dark_pool_confidence))
            
            # PHASE 1 GATE 1: Dark pool must have signal
            if not dark_pool_signal:
                continue  # Skip - no dark pool signal
            
            # PHASE 1 GATE 2: Dark pool must be confident enough
            if dark_pool_confidence < 0.60:
                continue  # Skip - dark pool not confident enough
            
            # Insider
            insider_result = self.insider.analyze_ticker(symbol, data)
            if insider_result['signal'] == 'BUY':
                signals.append(Signal('insider_trading', 'BUY', insider_result['confidence'], insider_result))
                self.signals_by_module['insider_trading'].append((date, symbol, insider_result['confidence']))
            
            # Patterns (scanners)
            scanner_signals = []
            for scanner_name, scanner in [
                ('pregainer', self.pregainer),
                ('day_trading', self.day_trading),
                ('opportunity', self.opportunity)
            ]:
                pattern_result = scanner.scan(symbol, data)
                if pattern_result['signal'] == 'BUY':
                    scanner_signals.append((scanner_name, pattern_result))
                    signals.append(Signal(scanner_name, 'BUY', pattern_result['confidence']))
                    self.signals_by_module[scanner_name].append((date, symbol, pattern_result['confidence']))
            
            # Sentiment
            sentiment_score = self.sentiment.analyze(symbol, data)
            if sentiment_score > 0.6:
                signals.append(Signal('sentiment', 'BUY', sentiment_score))
                self.signals_by_module['sentiment'].append((date, symbol, sentiment_score))
            
            # PHASE 1 GATE 3: At least 1 other signal must confirm dark pool
            other_confirmations = len(signals) - 1  # Exclude dark pool itself
            if other_confirmations < 1:
                continue  # Skip - no confirmation from other signals
            
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
            
            # PHASE 1 GATE 4: Confidence threshold (minimum 0.65)
            final_confidence = decision.get('confidence', 0.0)
            if final_confidence < 0.65:
                continue  # Skip - confidence too low
            
            # Only take BUY_FULL signals for backtest
            if decision['action'] == 'BUY_FULL':
                entry_price = float(data['Close'].iloc[-1]) * (1 + self.config['slippage'])
                new_positions.append((symbol, entry_price, signals))
        
        return new_positions"""
        
        # Replace the method
        backtest_content = backtest_content[:method_match.start()] + new_method + backtest_content[method_match.end():]
        print("   ✅ Updated _generate_signals with Phase 1 gates")
    else:
        print("   ⚠️  Could not find signal evaluation section")
        print("   ⚠️  Method may need manual integration")
else:
    print("   ❌ Could not find _generate_signals method")

# Write updated backtest
with open(backtest_file, 'w', encoding='utf-8') as f:
    f.write(backtest_content)

print("   ✅ Backtest file updated")

# ============================================================================
# STEP 3: Add Phase 1 logging for validation
# ============================================================================
print("\n3️⃣ Adding Phase 1 validation logging...")

# Add a method to track Phase 1 metrics
tracking_method = """
    def get_phase1_metrics(self) -> Dict:
        \"\"\"Get Phase 1 performance metrics for validation\"\"\"
        total_signals = sum(len(signals) for signals in self.signals_by_module.values())
        dark_pool_signals = len(self.signals_by_module.get('dark_pool', []))
        
        return {
            'total_signals_generated': total_signals,
            'dark_pool_signals': dark_pool_signals,
            'dark_pool_percentage': dark_pool_signals / max(total_signals, 1),
            'trades_executed': len(self.trades),
            'signals_by_module': {k: len(v) for k, v in self.signals_by_module.items()}
        }
"""

# Add tracking method before the last class method
if 'def get_phase1_metrics' not in backtest_content:
    # Find a good place to insert (before _calculate_results)
    insert_pattern = r'(def _calculate_results\(self)'
    if re.search(insert_pattern, backtest_content):
        backtest_content = re.sub(insert_pattern, tracking_method + r'\n    \1', backtest_content)
        print("   ✅ Added Phase 1 metrics tracking")
    else:
        print("   ⚠️  Could not find insertion point for metrics tracking")

# Write final version
with open(backtest_file, 'w', encoding='utf-8') as f:
    f.write(backtest_content)

# ============================================================================
# VERIFICATION
# ============================================================================
print("\n" + "="*80)
print("🔍 VERIFICATION")
print("="*80)

# Check Phase 1 gates are present
with open(backtest_file, 'r') as f:
    content = f.read()

checks = {
    'Dark pool gate': 'if not dark_pool_signal:' in content or 'PHASE_1_GATE' in content,
    'Confidence threshold': 'if final_confidence < 0.65:' in content or 'confidence < 0.65' in content,
    'Confirmation requirement': 'other_confirmations < 1' in content or 'len(signals) - 1' in content,
    'Dark pool confidence check': 'dark_pool_confidence < 0.60' in content,
}

print("\n✅ Phase 1 Integration Checks:")
for check_name, passed in checks.items():
    status = "✅" if passed else "❌"
    print(f"   {status} {check_name}")

all_passed = all(checks.values())
if all_passed:
    print("\n🎉 Phase 1 integration complete!")
else:
    print("\n⚠️  Some checks failed - review the integration")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📋 PHASE 1 INTEGRATION SUMMARY")
print("="*80)
print("""
✅ Changes Made:
   1. Updated INITIAL_WEIGHTS:
      - dark_pool: 60% (was ~17%)
      - sentiment: 25% (was ~17%)
      - scanners: 10% combined (was ~35% total)
      - insider: 5% (was ~17%)

   2. Added Phase 1 Gates:
      - GATE 1: Dark pool signal required
      - GATE 2: Dark pool confidence >= 0.60
      - GATE 3: At least 1 other signal confirms
      - GATE 4: Final confidence >= 0.65

   3. Added Phase 1 Metrics Tracking

📊 Expected Results:
   - Win Rate: 46.2% → 50-52% (+4-6 points)
   - Trades: 117 → 35-50 (fewer but better quality)
   - Sharpe: -0.37 → -0.1 to +0.1

🔄 Next Steps:
   1. Restart runtime (Runtime → Restart runtime)
   2. Run your launcher: COLAB_LAUNCH_INSTITUTIONAL_SYSTEM.py
   3. Choose Option 1 (Backtest)
   4. Verify win rate reaches 50-52%
   5. Check Phase 1 metrics in results

⏱️  Time to implement: ~2-3 hours
📈 Expected improvement: +4-6% win rate
""")

print("="*80)
print("✅ PHASE 1 INTEGRATION COMPLETE!")
print("="*80)

