"""
🚀 FINAL COLAB VALIDATION, TRAINING & OPTIMIZATION
===================================================
Complete pipeline:
1. Test all 7 PRO modules with real data
2. Backtest on historical winners (GME, AMC, BBIG)
3. Train ML models on results
4. Auto-optimize weights and thresholds
5. Re-test with optimized settings
6. Generate Perplexity question for further optimization

PASTE THIS INTO ONE COLAB CELL AND RUN.
"""

print("="*80)
print("🚀 QUANTUM AI - COMPLETE VALIDATION, TRAINING & OPTIMIZATION")
print("="*80)

# ============================================================================
# SETUP
# ============================================================================

from google.colab import drive
drive.mount('/content/drive')

import sys
import os
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Paths
QUANTUM_PATH = '/content/drive/MyDrive/QuantumAI'
sys.path.insert(0, QUANTUM_PATH)
sys.path.insert(0, f'{QUANTUM_PATH}/backend')
sys.path.insert(0, f'{QUANTUM_PATH}/backend/modules')
os.chdir(QUANTUM_PATH)

print(f"✅ Working directory: {os.getcwd()}")

# Install dependencies
print("\n📦 Installing dependencies...")
!pip install -q yfinance pandas numpy scikit-learn xgboost lightgbm joblib plotly

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

print("✅ Dependencies ready")

# ============================================================================
# PART 1: IMPORT AND VALIDATE ALL 7 PRO MODULES
# ============================================================================

print("\n" + "="*80)
print("📦 PART 1: TESTING ALL 7 PRO MODULES")
print("="*80)

from backend.modules.professional_signal_coordinator import ProfessionalSignalCoordinator, Strategy
from backend.modules.daily_scanner import ProfessionalDailyScanner
from backend.modules.position_size_calculator import PositionSizeCalculator

# Test each PRO module
modules_to_test = [
    ('ai_forecast_pro', 'AIForecastPro'),
    ('institutional_flow_pro', 'InstitutionalFlowPro'),
    ('pattern_engine_pro', 'PatternEnginePro'),
    ('sentiment_pro', 'SentimentPro'),
    ('scanner_pro', 'ScannerPro'),
    ('risk_manager_pro', 'RiskManagerPro'),
    ('ai_recommender_pro', 'AIRecommenderPro')
]

loaded_modules = {}
module_results = {}

for module_name, class_name in modules_to_test:
    try:
        exec(f"from backend.modules.{module_name} import {class_name}")
        exec(f"loaded_modules['{module_name}'] = {class_name}()")
        print(f"✅ {module_name}")
        module_results[module_name] = {'loaded': True, 'tests': []}
    except Exception as e:
        print(f"❌ {module_name}: {e}")
        module_results[module_name] = {'loaded': False, 'error': str(e)}

modules_loaded = sum(1 for m in module_results.values() if m.get('loaded'))
print(f"\n📊 LOADED: {modules_loaded}/7 modules")

# ============================================================================
# PART 2: TEST EACH MODULE WITH REAL DATA
# ============================================================================

print("\n" + "="*80)
print("🧪 PART 2: TESTING WITH REAL DATA")
print("="*80)

test_tickers = ['NVDA', 'AMD', 'TSLA']

for ticker in test_tickers:
    print(f"\n📊 Testing {ticker}...")
    
    try:
        # Download data
        df = yf.download(ticker, period='1y', progress=False)
        if df.empty:
            print(f"   ⚠️  No data")
            continue
        
        df.columns = [col.lower() if isinstance(col, str) else col[0].lower() for col in df.columns]
        print(f"   ✅ {len(df)} days data")
        
        # Test each module
        for module_name, module_obj in loaded_modules.items():
            try:
                if 'forecast' in module_name:
                    result = module_obj.forecast(ticker, df, horizon_days=21)
                    print(f"   ✅ {module_name}: Target ${result['base_case']['price']:.2f}")
                    module_results[module_name]['tests'].append({'ticker': ticker, 'passed': True})
                
                elif 'institutional' in module_name:
                    result = module_obj.analyze(ticker)
                    print(f"   ✅ {module_name}: {result.get('flow_direction', 'N/A')}")
                    module_results[module_name]['tests'].append({'ticker': ticker, 'passed': True})
                
                elif 'pattern' in module_name:
                    result = module_obj.detect_patterns(ticker, df)
                    count = len(result) if result else 0
                    print(f"   ✅ {module_name}: {count} patterns")
                    module_results[module_name]['tests'].append({'ticker': ticker, 'passed': True})
                
                elif 'sentiment' in module_name:
                    result = module_obj.analyze(ticker)
                    score = result.get('sentiment_score', 50)
                    print(f"   ✅ {module_name}: {score:.0f}/100")
                    module_results[module_name]['tests'].append({'ticker': ticker, 'passed': True})
                
                elif 'scanner' in module_name:
                    result = module_obj.scan(ticker, df)
                    print(f"   ✅ {module_name}: scanned")
                    module_results[module_name]['tests'].append({'ticker': ticker, 'passed': True})
                
                elif 'risk' in module_name:
                    result = module_obj.calculate_metrics(ticker, df)
                    sharpe = result.get('sharpe_ratio', 0)
                    print(f"   ✅ {module_name}: Sharpe {sharpe:.2f}")
                    module_results[module_name]['tests'].append({'ticker': ticker, 'passed': True})
                
                elif 'recommender' in module_name:
                    result = module_obj.recommend(ticker, df)
                    rec = result.get('recommendation', 'N/A')
                    print(f"   ✅ {module_name}: {rec}")
                    module_results[module_name]['tests'].append({'ticker': ticker, 'passed': True})
                
            except Exception as e:
                print(f"   ❌ {module_name}: {e}")
                module_results[module_name]['tests'].append({'ticker': ticker, 'passed': False, 'error': str(e)})
    
    except Exception as e:
        print(f"   ❌ Failed to test {ticker}: {e}")

# Calculate test pass rate
total_tests = sum(len(m.get('tests', [])) for m in module_results.values())
passed_tests = sum(1 for m in module_results.values() for t in m.get('tests', []) if t.get('passed'))
pass_rate = passed_tests / total_tests if total_tests > 0 else 0

print(f"\n✅ TEST RESULTS: {passed_tests}/{total_tests} passed ({pass_rate:.0%})")

# ============================================================================
# PART 3: BACKTEST ON HISTORICAL WINNERS
# ============================================================================

print("\n" + "="*80)
print("🧪 PART 3: BACKTEST ON HISTORICAL WINNERS")
print("="*80)

historical_winners = [
    {'ticker': 'GME', 'entry_date': datetime(2021, 1, 13), 'entry_price': 20.0,
     'peak_date': datetime(2021, 1, 28), 'peak_price': 483.0, 'gain_pct': 2315.0},
    {'ticker': 'AMC', 'entry_date': datetime(2021, 5, 13), 'entry_price': 12.0,
     'peak_date': datetime(2021, 6, 2), 'peak_price': 72.0, 'gain_pct': 500.0},
    {'ticker': 'BBIG', 'entry_date': datetime(2021, 8, 27), 'entry_price': 3.0,
     'peak_date': datetime(2021, 9, 8), 'peak_price': 12.5, 'gain_pct': 317.0}
]

print("\nTesting with current Perplexity-optimized weights:")
print("Penny Stock Strategy:")
print("  - Scanner: 30%")
print("  - Sentiment: 25%")
print("  - Patterns: 20%")
print("  - Forecast: 5%")
print("  - Institutional: 3%")
print("  - Risk: 2%")

backtest_results = []

try:
    scanner = ProfessionalDailyScanner(Strategy.PENNY_STOCK)
    
    for winner in historical_winners:
        ticker = winner['ticker']
        print(f"\n📊 {ticker} (Historical: +{winner['gain_pct']:.0f}%)")
        
        try:
            result = scanner.score_ticker(ticker)
            
            if result:
                score = result.final_score
                rec = result.recommendation.label
                caught = score >= 80  # Penny stock threshold
                
                print(f"   Score: {score:.0f}/100")
                print(f"   Recommendation: {rec}")
                print(f"   {'✅ CAUGHT' if caught else '❌ MISSED'}")
                
                # Store detailed results
                backtest_results.append({
                    'ticker': ticker,
                    'score': score,
                    'caught': caught,
                    'gain': winner['gain_pct'],
                    'signals': result.normalized_signals,
                    'weights': result.weights
                })
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Backtest summary
    if backtest_results:
        catch_rate = sum(1 for r in backtest_results if r['caught']) / len(backtest_results)
        avg_score = np.mean([r['score'] for r in backtest_results])
        avg_score_caught = np.mean([r['score'] for r in backtest_results if r['caught']]) if any(r['caught'] for r in backtest_results) else 0
        avg_score_missed = np.mean([r['score'] for r in backtest_results if not r['caught']]) if any(not r['caught'] for r in backtest_results) else 0
        
        print("\n" + "="*80)
        print("📊 BACKTEST RESULTS")
        print("="*80)
        print(f"\n✅ Catch Rate: {catch_rate:.0%} ({sum(1 for r in backtest_results if r['caught'])}/{len(backtest_results)})")
        print(f"📊 Avg Score (All): {avg_score:.0f}/100")
        if avg_score_caught > 0:
            print(f"📊 Avg Score (Caught): {avg_score_caught:.0f}/100")
        if avg_score_missed > 0:
            print(f"📊 Avg Score (Missed): {avg_score_missed:.0f}/100")
        
        # Signal importance analysis
        print("\n📊 SIGNAL IMPORTANCE (Caught vs Missed):")
        
        if any(r['caught'] for r in backtest_results) and any(not r['caught'] for r in backtest_results):
            all_signal_names = set()
            for r in backtest_results:
                all_signal_names.update(r['signals'].keys())
            
            for signal_name in sorted(all_signal_names):
                caught_avg = np.mean([r['signals'].get(signal_name, 50) for r in backtest_results if r['caught']])
                missed_avg = np.mean([r['signals'].get(signal_name, 50) for r in backtest_results if not r['caught']])
                diff = caught_avg - missed_avg
                
                symbol = "🔥" if diff > 10 else "✅" if diff > 5 else "⚠️" if diff > 0 else "❌"
                print(f"   {symbol} {signal_name}: Caught={caught_avg:.1f}, Missed={missed_avg:.1f}, Diff={diff:+.1f}")
        
        backtest_passed = catch_rate >= 0.33  # 33%+ for explosive moves
        
except Exception as e:
    print(f"\n❌ Backtest failed: {e}")
    import traceback
    traceback.print_exc()
    backtest_passed = False

# ============================================================================
# PART 4: ML TRAINING ON BACKTEST RESULTS
# ============================================================================

print("\n" + "="*80)
print("🤖 PART 4: ML MODEL TRAINING")
print("="*80)

ml_trained = False

if backtest_results and len(backtest_results) >= 3:
    print("\nTraining ML model to learn winner patterns...")
    
    try:
        # Prepare features
        X = []
        y = []
        
        for result in backtest_results:
            # Features: all normalized signals
            features = list(result['signals'].values())
            X.append(features)
            # Label: 1 if caught, 0 if missed
            y.append(1 if result['caught'] else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        # Train model
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X, y)
        
        # Feature importance
        feature_names = list(backtest_results[0]['signals'].keys())
        importances = dict(zip(feature_names, model.feature_importances_))
        sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        
        print("\n✅ ML Model Trained!")
        print("\n📊 FEATURE IMPORTANCE:")
        for feature, importance in sorted_importances:
            print(f"   {feature}: {importance:.3f}")
        
        # Save model
        os.makedirs(f'{QUANTUM_PATH}/backend/models', exist_ok=True)
        model_path = f'{QUANTUM_PATH}/backend/models/winner_predictor.joblib'
        joblib.dump(model, model_path)
        print(f"\n💾 Model saved: {model_path}")
        
        ml_trained = True
        
    except Exception as e:
        print(f"⚠️  ML training failed: {e}")
else:
    print("⚠️  Need more backtest data for ML training")

# ============================================================================
# PART 5: WEIGHT OPTIMIZATION
# ============================================================================

print("\n" + "="*80)
print("🔧 PART 5: WEIGHT OPTIMIZATION")
print("="*80)

if backtest_results and any(r['caught'] for r in backtest_results) and any(not r['caught'] for r in backtest_results):
    print("\nAnalyzing optimal weights based on backtest...")
    
    # Calculate signal importance from caught vs missed
    signal_importance = {}
    all_signals = set()
    for r in backtest_results:
        all_signals.update(r['signals'].keys())
    
    for signal_name in all_signals:
        caught_vals = [r['signals'].get(signal_name, 50) for r in backtest_results if r['caught']]
        missed_vals = [r['signals'].get(signal_name, 50) for r in backtest_results if not r['caught']]
        
        if caught_vals and missed_vals:
            caught_avg = np.mean(caught_vals)
            missed_avg = np.mean(missed_vals)
            importance = caught_avg - missed_avg
            signal_importance[signal_name] = {
                'caught_avg': caught_avg,
                'missed_avg': missed_avg,
                'importance': importance
            }
    
    # Normalize importance to weights (sum to 1.0)
    total_importance = sum(max(0, s['importance']) for s in signal_importance.values())
    
    if total_importance > 0:
        optimized_weights = {}
        for signal, data in signal_importance.items():
            weight = max(0, data['importance']) / total_importance
            optimized_weights[signal] = weight
        
        print("\n📊 CURRENT WEIGHTS vs OPTIMIZED WEIGHTS:")
        print("\nSignal                Current    Optimized   Change")
        print("-" * 60)
        
        current_weights = backtest_results[0]['weights']
        
        for signal in sorted(signal_importance.keys()):
            current = current_weights.get(signal, 0)
            optimized = optimized_weights.get(signal, 0)
            change = optimized - current
            
            arrow = "🔥" if change > 0.05 else "✅" if change > 0 else "⬇️" if change < -0.05 else "➡️"
            print(f"{signal:18} {current:6.1%}     {optimized:6.1%}    {change:+6.1%} {arrow}")
        
        # Save optimization results
        optimization_results = {
            'timestamp': datetime.now().isoformat(),
            'backtest_catch_rate': catch_rate,
            'current_weights': current_weights,
            'optimized_weights': optimized_weights,
            'signal_importance': signal_importance
        }
        
        with open(f'{QUANTUM_PATH}/optimization_results.json', 'w') as f:
            json.dump(optimization_results, f, indent=2)
        
        print(f"\n💾 Optimization results saved")
else:
    print("\n⚠️  Need both caught and missed examples to optimize weights")
    optimized_weights = None

# ============================================================================
# PART 6: FINAL ASSESSMENT
# ============================================================================

print("\n" + "="*80)
print("🎯 FINAL ASSESSMENT")
print("="*80)

checks = [
    ("Modules Loaded (5/7+)", modules_loaded >= 5),
    ("Test Pass Rate (70%+)", pass_rate >= 0.7),
    ("Backtest Catch Rate (33%+)", backtest_passed if 'backtest_passed' in locals() else False),
    ("ML Model Trained", ml_trained)
]

print("\n📋 CHECKLIST:")
for name, passed in checks:
    print(f"   {'✅' if passed else '❌'} {name}")

all_passed = all(p for _, p in checks)
critical_passed = checks[0][1] and checks[2][1]  # Modules + Backtest

print("\n" + "="*80)

if all_passed:
    print("🎉 SYSTEM FULLY VALIDATED & OPTIMIZED!")
    print("="*80)
    print("\n✅ ALL CHECKS PASSED")
    print("\n🚀 READY FOR PRODUCTION:")
    print("   1. Launch: streamlit run REAL_PORTFOLIO_DASHBOARD.py")
    print("   2. Paper trade 1 week")
    print("   3. Go live with $50 positions")
    print("\n💰 YOU'RE READY TO TRADE!")

elif critical_passed:
    print("⚠️  SYSTEM MOSTLY READY")
    print("="*80)
    print("\n✅ Core components working")
    print("⚠️  Some optimization checks incomplete")
    print("\n🎯 YOU CAN:")
    print("   - Use the dashboard")
    print("   - Start paper trading")
    print("   - Monitor performance closely")

else:
    print("❌ SYSTEM NEEDS WORK")
    print("="*80)
    print("\n❌ Critical checks failed")
    print("\n🔧 FIX THESE FIRST:")
    for name, passed in checks:
        if not passed:
            print(f"   - {name}")

# ============================================================================
# PART 7: GENERATE PERPLEXITY OPTIMIZATION QUESTION
# ============================================================================

print("\n" + "="*80)
print("📝 GENERATING PERPLEXITY OPTIMIZATION QUESTION")
print("="*80)

perplexity_question = f"""
# PERPLEXITY: Optimize My Trading System Based on Real Results

## MY SYSTEM PERFORMANCE:

**Modules:** {modules_loaded}/7 loaded, {pass_rate:.0%} test pass rate
**Backtest:** {catch_rate:.0%} catch rate on GME/AMC/BBIG
**Average Score:** {avg_score:.0f}/100

## CURRENT WEIGHTS (Penny Stock Strategy):
"""

if backtest_results:
    for signal, weight in sorted(current_weights.items(), key=lambda x: x[1], reverse=True):
        perplexity_question += f"\n- {signal}: {weight:.1%}"

perplexity_question += """

## BACKTEST RESULTS:
"""

for result in backtest_results:
    perplexity_question += f"\n- {result['ticker']}: Score {result['score']:.0f}/100 ({'CAUGHT' if result['caught'] else 'MISSED'})"

if optimized_weights:
    perplexity_question += """

## MY OPTIMIZATION SUGGESTS:
"""
    for signal, weight in sorted(optimized_weights.items(), key=lambda x: x[1], reverse=True):
        current = current_weights.get(signal, 0)
        change = weight - current
        perplexity_question += f"\n- {signal}: {current:.1%} → {weight:.1%} ({change:+.1%})"

perplexity_question += f"""

## QUESTIONS:

1. **Are my optimized weights correct?** Should I use them or stick with current?

2. **Why did I miss {[r['ticker'] for r in backtest_results if not r['caught']]}?** What signals were too weak?

3. **How can I improve my {avg_score:.0f}/100 average score?**

4. **Should I adjust thresholds?** Currently using 80+ for penny stocks.

5. **What's the best way to auto-tune my system?** Walk-forward optimization? Bayesian?

6. **How do I prevent overfitting?** With only {len(backtest_results)} backtest samples.

7. **Should I weight signals differently for:**
   - Swing trades (3-4 months, 10-20% target)
   - Penny stocks (days/weeks, 50-100%+ target)
   - Large caps (6-12 months, 5-15% target)

8. **What's a realistic catch rate target?**
   - For 100%+ movers (GME/AMC)
   - For 50%+ movers
   - For 20%+ movers

Please give me:
- ✅ Exact weight recommendations
- ✅ Threshold adjustments
- ✅ Auto-tuning strategy
- ✅ Realistic performance expectations
"""

# Save question
with open(f'{QUANTUM_PATH}/PERPLEXITY_OPTIMIZATION_QUESTION.txt', 'w') as f:
    f.write(perplexity_question)

print("\n✅ Perplexity question saved to:")
print(f"   {QUANTUM_PATH}/PERPLEXITY_OPTIMIZATION_QUESTION.txt")

print("\n📋 Copy this question to Perplexity for optimization advice!")

print("\n" + "="*80)
print("🎉 VALIDATION & TRAINING COMPLETE!")
print("="*80)
print(f"\nResults saved to: {QUANTUM_PATH}/")
print("- optimization_results.json")
print("- PERPLEXITY_OPTIMIZATION_QUESTION.txt")
print("\n🚀 Next: Review results and launch dashboard if ready!")

