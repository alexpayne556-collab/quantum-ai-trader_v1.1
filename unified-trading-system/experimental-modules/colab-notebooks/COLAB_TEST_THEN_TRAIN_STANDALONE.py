"""
🧪 TEST THEN TRAIN - COMPLETE STANDALONE VERSION
==================================================
Works entirely inline - no external dependencies
Uses Colab paths correctly
Handles missing modules gracefully
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import datetime
import json
import sys

print("="*80)
print("🧪 TEST THEN TRAIN - STANDALONE")
print("="*80)

# ============================================================================
# CONFIGURATION (COLAB PATHS)
# ============================================================================

# Colab paths
BASE_DIR = Path('/content/drive/MyDrive/QuantumAI')
MODULES_DIR = BASE_DIR / 'backend' / 'modules'

# Add to path
if str(MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(MODULES_DIR))

# Test configuration
TEST_SYMBOLS = ['AAPL', 'TSLA', 'NVDA', 'AMD', 'MSFT', 'GOOGL', 'META', 'AMZN']
LOOKBACK_DAYS = 90

print(f"\n📊 Configuration:")
print(f"   Modules Dir: {MODULES_DIR}")
print(f"   Test Symbols: {', '.join(TEST_SYMBOLS)}")
print(f"   Lookback: {LOOKBACK_DAYS} days")

# ============================================================================
# INLINE TESTING FRAMEWORK
# ============================================================================

class ModuleTester:
    """Inline testing framework - no external dependencies"""
    
    def __init__(self):
        self.results = {}
    
    def test_module(self, module_name: str, module_instance, 
                   test_symbols: list, lookback_days: int) -> dict:
        """Test a module's signal quality"""
        print(f"\n🧪 Testing {module_name}...")
        
        signals = []
        outcomes = []
        
        for symbol in test_symbols:
            try:
                # Get historical data
                data = yf.download(symbol, period=f'{lookback_days}d', 
                                 progress=False, auto_adjust=True)
                
                if len(data) < 30:
                    continue
                
                # Get signal from module
                signal = self._get_signal(module_instance, symbol, data)
                
                if signal and signal.get('signal') in ['BUY', 'LONG']:
                    # Check outcome
                    entry_price = float(data['Close'].iloc[-1])
                    future_price = float(data['Close'].iloc[-5]) if len(data) >= 5 else entry_price
                    
                    gain_pct = (future_price - entry_price) / entry_price
                    was_correct = gain_pct > 0.02  # 2% gain = correct
                    
                    signals.append({
                        'symbol': symbol,
                        'entry_price': entry_price,
                        'future_price': future_price,
                        'gain_pct': gain_pct,
                        'was_correct': was_correct,
                        'confidence': signal.get('confidence', 0.5)
                    })
                    
                    outcomes.append({
                        'was_correct': was_correct,
                        'gain_pct': gain_pct,
                        'confidence': signal.get('confidence', 0.5)
                    })
            
            except Exception as e:
                print(f"   ⚠️  Error testing {symbol}: {e}")
                continue
        
        # Calculate metrics
        if len(outcomes) == 0:
            print(f"   ⚠️  No signals generated")
            return {}
        
        correct = sum(1 for o in outcomes if o['was_correct'])
        win_rate = correct / len(outcomes)
        
        correct_gains = [o['gain_pct'] for o in outcomes if o['was_correct']]
        wrong_gains = [o['gain_pct'] for o in outcomes if not o['was_correct']]
        
        avg_gain_correct = np.mean(correct_gains) if correct_gains else 0
        avg_loss_wrong = np.mean(wrong_gains) if wrong_gains else 0
        avg_confidence = np.mean([o['confidence'] for o in outcomes])
        
        results = {
            'win_rate': win_rate,
            'avg_confidence': avg_confidence,
            'signal_count': len(outcomes),
            'correct_signals': correct,
            'false_positives': len(outcomes) - correct,
            'avg_gain_when_correct': avg_gain_correct,
            'avg_loss_when_wrong': avg_loss_wrong,
            'signals': signals
        }
        
        self.results[module_name] = results
        
        print(f"   ✅ Win Rate: {win_rate:.1%} ({correct}/{len(outcomes)})")
        print(f"   ✅ Avg Confidence: {avg_confidence:.1%}")
        
        return results
    
    def _get_signal(self, module_instance, symbol: str, data: pd.DataFrame):
        """Get signal from module (handles different interfaces)"""
        try:
            if hasattr(module_instance, 'analyze_symbol'):
                return module_instance.analyze_symbol(symbol)
            elif hasattr(module_instance, 'analyze_ticker'):
                return module_instance.analyze_ticker(symbol, data)
            elif hasattr(module_instance, 'scan'):
                return module_instance.scan(symbol, data)
            elif hasattr(module_instance, 'detect_early_accumulation'):
                volume_data = data['Volume'] if 'Volume' in data.columns else pd.Series([0])
                return module_instance.detect_early_accumulation(symbol, volume_data)
            else:
                return None
        except Exception as e:
            return None
    
    def get_performance_report(self) -> pd.DataFrame:
        """Get performance report"""
        if not self.results:
            return pd.DataFrame()
        
        report_data = []
        for module_name, results in self.results.items():
            report_data.append({
                'Module': module_name,
                'Win Rate': f"{results['win_rate']:.1%}",
                'Signals': results['signal_count'],
                'Correct': results['correct_signals'],
                'Avg Confidence': f"{results['avg_confidence']:.1%}",
                'Avg Gain (Correct)': f"{results['avg_gain_when_correct']:+.1%}",
                'Avg Loss (Wrong)': f"{results['avg_loss_when_wrong']:+.1%}"
            })
        
        return pd.DataFrame(report_data).sort_values('Win Rate', ascending=False)

# ============================================================================
# INLINE TRAINING FRAMEWORK
# ============================================================================

class SimpleTrainer:
    """Simple trainer - calculates optimal weights"""
    
    def __init__(self):
        self.training_history = []
    
    def calculate_optimal_weights(self, performance_data: dict) -> dict:
        """Calculate optimal weights based on performance"""
        optimal_weights = {}
        total_reliability = 0
        
        for module_name, perf in performance_data.items():
            win_rate = perf.get('win_rate', 0.5)
            signal_count = perf.get('signal_count', 0)
            
            # Reliability = win_rate * sample_factor
            sample_factor = min(signal_count / 20.0, 1.5)
            reliability = win_rate * sample_factor
            
            optimal_weights[module_name] = reliability
            total_reliability += reliability
        
        # Normalize
        if total_reliability > 0:
            optimal_weights = {k: v/total_reliability for k, v in optimal_weights.items()}
        else:
            optimal_weights = {k: 1.0/len(performance_data) for k in performance_data.keys()}
        
        return optimal_weights

# ============================================================================
# STEP 1: TEST MODULES (BASELINE)
# ============================================================================

print("\n" + "="*80)
print("STEP 1: TESTING MODULES (BASELINE)")
print("="*80)

tester = ModuleTester()
baseline_results = {}

# Try to test available modules
modules_to_test = [
    ('dark_pool', 'dark_pool_tracker', 'DarkPoolTracker'),
    ('insider_trading', 'insider_trading_tracker', 'InsiderTradingTracker'),
    ('unified_scanner', 'unified_momentum_scanner_v3', 'UnifiedMomentumScannerV3'),
    ('sentiment', 'sentiment_engine', 'SentimentEngine'),
]

for module_name, module_file, module_class in modules_to_test:
    try:
        # Try to import
        module = __import__(module_file, fromlist=[module_class])
        module_instance = getattr(module, module_class)()
        
        # Test it
        results = tester.test_module(module_name, module_instance, TEST_SYMBOLS, LOOKBACK_DAYS)
        if results:
            baseline_results[module_name] = results
    
    except ImportError as e:
        print(f"   ⚠️  {module_name}: Module not found ({module_file})")
    except Exception as e:
        print(f"   ⚠️  {module_name}: Error - {e}")

# Display baseline
print("\n" + "="*80)
print("📊 BASELINE PERFORMANCE RESULTS")
print("="*80)

baseline_df = tester.get_performance_report()
if not baseline_df.empty:
    print("\n" + baseline_df.to_string(index=False))
else:
    print("\n⚠️  No test results available")
    print("   💡 Make sure modules are in the modules directory")

print(f"\n✅ Baseline testing complete!")
print(f"   Tested {len(baseline_results)} modules")

# ============================================================================
# STEP 2: CALCULATE OPTIMAL WEIGHTS
# ============================================================================

print("\n" + "="*80)
print("STEP 2: CALCULATING OPTIMAL WEIGHTS")
print("="*80)

trainer = SimpleTrainer()

# Prepare performance data
performance_data = {}
for module_name, results in baseline_results.items():
    performance_data[module_name] = {
        'win_rate': results.get('win_rate', 0.5),
        'signal_count': results.get('signal_count', 0)
    }

# Calculate optimal weights
optimal_weights = trainer.calculate_optimal_weights(performance_data)

print("\n✅ Optimal Weights Calculated:")
for module, weight in sorted(optimal_weights.items(), key=lambda x: x[1], reverse=True):
    wr = baseline_results[module].get('win_rate', 0)
    print(f"   {module}: {weight:.1%} (win rate: {wr:.1%})")

# ============================================================================
# STEP 3: SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("STEP 3: SAVING RESULTS")
print("="*80)

# Save results
results_file = MODULES_DIR / 'test_train_results.json'

results_data = {
    'baseline': {
        module: {
            'win_rate': results.get('win_rate', 0),
            'avg_confidence': results.get('avg_confidence', 0),
            'signal_count': results.get('signal_count', 0)
        }
        for module, results in baseline_results.items()
    },
    'optimal_weights': optimal_weights,
    'timestamp': datetime.now().isoformat(),
    'test_symbols': TEST_SYMBOLS,
    'lookback_days': LOOKBACK_DAYS
}

try:
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"   ✅ Results saved to: {results_file}")
except Exception as e:
    print(f"   ⚠️  Could not save results: {e}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("📋 TEST-THEN-TRAIN SUMMARY")
print("="*80)

print(f"""
✅ Workflow Complete!

Testing Results:
   ✅ Tested {len(baseline_results)} modules
   ✅ Baseline performance established
   ✅ Optimal weights calculated

Optimal Weights:
{chr(10).join([f"   '{k}': {v:.3f}," for k, v in sorted(optimal_weights.items(), key=lambda x: x[1], reverse=True)])}

Next Steps:
   1. Use optimal_weights in your ensemble
   2. Run backtest with trained weights
   3. Generate recommendations
   4. Monitor performance
   5. Retrain weekly/monthly

💡 To use these weights:
   optimal_weights = {optimal_weights}
""")

print("="*80)
print("✅ TEST-THEN-TRAIN COMPLETE!")
print("="*80)

# Return for use
print("\n💡 Results available:")
print(f"   baseline_results = {baseline_results}")
print(f"   optimal_weights = {optimal_weights}")

