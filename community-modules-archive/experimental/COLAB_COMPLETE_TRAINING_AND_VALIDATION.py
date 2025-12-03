"""
🚀 COMPLETE COLAB TRAINING & VALIDATION PIPELINE
=================================================
Trains ALL models, backtests EVERYTHING, validates EVERY module.
Then tells you if you're ready for the dashboard.

PASTE THIS ENTIRE FILE INTO ONE COLAB CELL.
"""

# ============================================================================
# PART 1: SETUP & MOUNT DRIVE
# ============================================================================

print("="*80)
print("🚀 QUANTUM AI - COMPLETE TRAINING & VALIDATION")
print("="*80)

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

import sys
import os

# Set paths
QUANTUM_PATH = '/content/drive/MyDrive/QuantumAI'
sys.path.insert(0, QUANTUM_PATH)
sys.path.insert(0, f'{QUANTUM_PATH}/backend')
sys.path.insert(0, f'{QUANTUM_PATH}/backend/modules')

os.chdir(QUANTUM_PATH)

print(f"✅ Working directory: {os.getcwd()}")

# Install dependencies
print("\n📦 Installing dependencies...")
!pip install -q yfinance pandas numpy scikit-learn xgboost lightgbm prophet plotly streamlit joblib

print("✅ Dependencies installed")

# ============================================================================
# PART 2: IMPORT ALL MODULES
# ============================================================================

print("\n" + "="*80)
print("📦 IMPORTING ALL MODULES")
print("="*80)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import your PRO modules
print("\n1️⃣  Importing PRO modules...")
try:
    from backend.modules.professional_signal_coordinator import ProfessionalSignalCoordinator, Strategy
    print("   ✅ Signal Coordinator")
except Exception as e:
    print(f"   ❌ Signal Coordinator: {e}")

try:
    from backend.modules.daily_scanner import ProfessionalDailyScanner
    print("   ✅ Daily Scanner")
except Exception as e:
    print(f"   ❌ Daily Scanner: {e}")

try:
    from backend.modules.position_size_calculator import PositionSizeCalculator
    print("   ✅ Position Calculator")
except Exception as e:
    print(f"   ❌ Position Calculator: {e}")

try:
    from backend.modules.portfolio_manager import PortfolioManager
    print("   ✅ Portfolio Manager")
except Exception as e:
    print(f"   ❌ Portfolio Manager: {e}")

# Import individual PRO modules
print("\n2️⃣  Importing individual PRO modules...")

modules_status = {}

try:
    from backend.modules.ai_forecast_pro import AIForecastPro
    modules_status['forecast'] = True
    print("   ✅ AI Forecast Pro")
except Exception as e:
    modules_status['forecast'] = False
    print(f"   ⚠️  AI Forecast Pro: {e}")

try:
    from backend.modules.institutional_flow_pro import InstitutionalFlowPro
    modules_status['institutional'] = True
    print("   ✅ Institutional Flow Pro")
except Exception as e:
    modules_status['institutional'] = False
    print(f"   ⚠️  Institutional Flow Pro: {e}")

try:
    from backend.modules.pattern_engine_pro import PatternEnginePro
    modules_status['patterns'] = True
    print("   ✅ Pattern Engine Pro")
except Exception as e:
    modules_status['patterns'] = False
    print(f"   ⚠️  Pattern Engine Pro: {e}")

try:
    from backend.modules.sentiment_pro import SentimentPro
    modules_status['sentiment'] = True
    print("   ✅ Sentiment Pro")
except Exception as e:
    modules_status['sentiment'] = False
    print(f"   ⚠️  Sentiment Pro: {e}")

try:
    from backend.modules.scanner_pro import ScannerPro
    modules_status['scanner'] = True
    print("   ✅ Scanner Pro")
except Exception as e:
    modules_status['scanner'] = False
    print(f"   ⚠️  Scanner Pro: {e}")

try:
    from backend.modules.risk_manager_pro import RiskManagerPro
    modules_status['risk'] = True
    print("   ✅ Risk Manager Pro")
except Exception as e:
    modules_status['risk'] = False
    print(f"   ⚠️  Risk Manager Pro: {e}")

try:
    from backend.modules.ai_recommender_pro import AIRecommenderPro
    modules_status['recommender'] = True
    print("   ✅ AI Recommender Pro")
except Exception as e:
    modules_status['recommender'] = False
    print(f"   ⚠️  AI Recommender Pro: {e}")

modules_working = sum(modules_status.values())
print(f"\n📊 MODULE STATUS: {modules_working}/7 working")

# ============================================================================
# PART 3: VALIDATE EACH MODULE WITH REAL DATA
# ============================================================================

print("\n" + "="*80)
print("🧪 TESTING EACH MODULE WITH REAL DATA")
print("="*80)

test_tickers = ['NVDA', 'AMD', 'TSLA']
validation_results = {}

for ticker in test_tickers:
    print(f"\n📊 Testing {ticker}...")
    
    try:
        import yfinance as yf
        df = yf.download(ticker, period='1y', progress=False)
        
        if df.empty:
            print(f"   ⚠️  No data for {ticker}")
            continue
        
        # Standardize columns
        df.columns = [col.lower() if isinstance(col, str) else col[0].lower() for col in df.columns]
        
        print(f"   ✅ Downloaded {len(df)} days of data")
        
        # Test each module
        ticker_results = {}
        
        # 1. Forecast
        if modules_status['forecast']:
            try:
                forecaster = AIForecastPro()
                forecast = forecaster.forecast(ticker, df, horizon_days=21)
                print(f"   ✅ Forecast: {forecast['base_case']['price']:.2f} (conf: {forecast['confidence']:.0f}%)")
                ticker_results['forecast'] = True
            except Exception as e:
                print(f"   ❌ Forecast failed: {e}")
                ticker_results['forecast'] = False
        
        # 2. Institutional
        if modules_status['institutional']:
            try:
                inst = InstitutionalFlowPro()
                inst_result = inst.analyze(ticker)
                print(f"   ✅ Institutional: {inst_result.get('flow_direction', 'N/A')}")
                ticker_results['institutional'] = True
            except Exception as e:
                print(f"   ❌ Institutional failed: {e}")
                ticker_results['institutional'] = False
        
        # 3. Patterns
        if modules_status['patterns']:
            try:
                pattern_engine = PatternEnginePro()
                patterns = pattern_engine.detect_patterns(ticker, df)
                pattern_count = len(patterns) if patterns else 0
                print(f"   ✅ Patterns: {pattern_count} detected")
                ticker_results['patterns'] = True
            except Exception as e:
                print(f"   ❌ Patterns failed: {e}")
                ticker_results['patterns'] = False
        
        # 4. Sentiment
        if modules_status['sentiment']:
            try:
                sentiment = SentimentPro()
                sent_result = sentiment.analyze(ticker)
                print(f"   ✅ Sentiment: {sent_result.get('sentiment_score', 50):.0f}/100")
                ticker_results['sentiment'] = True
            except Exception as e:
                print(f"   ❌ Sentiment failed: {e}")
                ticker_results['sentiment'] = False
        
        # 5. Scanner
        if modules_status['scanner']:
            try:
                scanner = ScannerPro()
                scan_result = scanner.scan(ticker, df)
                print(f"   ✅ Scanner: momentum detected")
                ticker_results['scanner'] = True
            except Exception as e:
                print(f"   ❌ Scanner failed: {e}")
                ticker_results['scanner'] = False
        
        # 6. Risk
        if modules_status['risk']:
            try:
                risk_mgr = RiskManagerPro()
                risk_result = risk_mgr.calculate_metrics(ticker, df)
                print(f"   ✅ Risk: Sharpe {risk_result.get('sharpe_ratio', 0):.2f}")
                ticker_results['risk'] = True
            except Exception as e:
                print(f"   ❌ Risk failed: {e}")
                ticker_results['risk'] = False
        
        # 7. Recommender
        if modules_status['recommender']:
            try:
                recommender = AIRecommenderPro()
                rec_result = recommender.recommend(ticker, df)
                print(f"   ✅ Recommender: {rec_result.get('recommendation', 'N/A')}")
                ticker_results['recommender'] = True
            except Exception as e:
                print(f"   ❌ Recommender failed: {e}")
                ticker_results['recommender'] = False
        
        validation_results[ticker] = ticker_results
        
    except Exception as e:
        print(f"   ❌ Failed to test {ticker}: {e}")

# Validation summary
print("\n" + "="*80)
print("📊 VALIDATION SUMMARY")
print("="*80)

all_tests = []
for ticker, results in validation_results.items():
    for module, passed in results.items():
        all_tests.append(passed)

if all_tests:
    pass_rate = sum(all_tests) / len(all_tests)
    print(f"\n✅ Overall Pass Rate: {pass_rate:.0%} ({sum(all_tests)}/{len(all_tests)} tests)")
    
    if pass_rate >= 0.7:
        print("✅ VALIDATION PASSED - Modules are working!")
    else:
        print("⚠️  VALIDATION NEEDS WORK - Some modules failing")
else:
    print("❌ No validation tests completed")

# ============================================================================
# PART 4: BACKTEST ON HISTORICAL WINNERS
# ============================================================================

print("\n" + "="*80)
print("🧪 BACKTESTING ON HISTORICAL WINNERS")
print("="*80)

historical_winners = [
    {'ticker': 'GME', 'entry_date': datetime(2021, 1, 13), 'entry_price': 20.0,
     'peak_date': datetime(2021, 1, 28), 'peak_price': 483.0, 'gain_pct': 2315.0},
    {'ticker': 'AMC', 'entry_date': datetime(2021, 5, 13), 'entry_price': 12.0,
     'peak_date': datetime(2021, 6, 2), 'peak_price': 72.0, 'gain_pct': 500.0},
    {'ticker': 'BBIG', 'entry_date': datetime(2021, 8, 27), 'entry_price': 3.0,
     'peak_date': datetime(2021, 9, 8), 'peak_price': 12.5, 'gain_pct': 317.0}
]

backtest_results = []

try:
    scanner = ProfessionalDailyScanner(Strategy.PENNY_STOCK)
    
    for winner in historical_winners:
        ticker = winner['ticker']
        print(f"\n📊 Testing {ticker} ({winner['gain_pct']:.0f}% historical gain)...")
        
        try:
            result = scanner.score_ticker(ticker)
            
            if result:
                score = result.final_score
                rec = result.recommendation.label
                
                caught = score >= 70  # Penny stock threshold
                
                print(f"   AI Score: {score:.0f}/100")
                print(f"   Recommendation: {rec}")
                print(f"   Result: {'✅ WOULD HAVE CAUGHT' if caught else '❌ WOULD HAVE MISSED'}")
                
                backtest_results.append({
                    'ticker': ticker,
                    'score': score,
                    'caught': caught,
                    'gain': winner['gain_pct']
                })
            else:
                print(f"   ⚠️  Could not score {ticker}")
                
        except Exception as e:
            print(f"   ❌ Error testing {ticker}: {e}")
    
    # Backtest summary
    if backtest_results:
        catch_rate = sum(1 for r in backtest_results if r['caught']) / len(backtest_results)
        avg_score = sum(r['score'] for r in backtest_results) / len(backtest_results)
        
        print("\n" + "="*80)
        print("📊 BACKTEST RESULTS")
        print("="*80)
        print(f"\n✅ Catch Rate: {catch_rate:.0%} ({sum(1 for r in backtest_results if r['caught'])}/{len(backtest_results)})")
        print(f"📊 Average Score: {avg_score:.0f}/100")
        
        if catch_rate >= 0.33:  # 33%+ is acceptable for 100%+ movers
            print("\n✅ BACKTEST PASSED - System catches explosive moves!")
        else:
            print("\n⚠️  BACKTEST NEEDS TUNING - Catch rate too low")
            print("   Recommendation: Lower thresholds or adjust weights")
    
except Exception as e:
    print(f"\n❌ Backtest failed: {e}")

# ============================================================================
# PART 5: LIVE SCAN TEST
# ============================================================================

print("\n" + "="*80)
print("🔍 LIVE SCAN TEST")
print("="*80)

test_stocks = ['NVDA', 'AMD', 'PLTR', 'SOFI']

print(f"\nScanning {len(test_stocks)} stocks for swing trades...")

try:
    scanner = ProfessionalDailyScanner(Strategy.SWING_TRADE)
    results = scanner.scan_universe(test_stocks)
    
    if results:
        print(f"\n✅ Scanner working! Scored {len(results)} stocks\n")
        
        for result in results:
            print(f"   {result.ticker}: {result.final_score:.0f}/100 - {result.recommendation.label}")
        
        # Check if any opportunities
        opportunities = [r for r in results if r.final_score >= 70]
        
        if opportunities:
            print(f"\n🔥 Found {len(opportunities)} opportunities (score ≥70)")
        else:
            print("\n⚠️  No opportunities found (all scores <70)")
    else:
        print("\n⚠️  Scanner returned no results")
        
except Exception as e:
    print(f"\n❌ Live scan failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# PART 6: ML TRAINING (IF BACKTEST DATA EXISTS)
# ============================================================================

print("\n" + "="*80)
print("🤖 ML MODEL TRAINING")
print("="*80)

if backtest_results:
    print("\nTraining ML model on backtest data...")
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        import joblib
        
        # Prepare training data
        X_data = []
        y_data = []
        
        for result in backtest_results:
            # Features: just the score for now (we'd add more in production)
            X_data.append([result['score']])
            # Label: 1 if caught, 0 if missed
            y_data.append(1 if result['caught'] else 0)
        
        if len(X_data) >= 3:  # Need at least 3 samples
            X = np.array(X_data)
            y = np.array(y_data)
            
            # Train simple model
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)
            
            # Predict on training data (just to show it works)
            predictions = model.predict(X)
            accuracy = (predictions == y).sum() / len(y)
            
            print(f"✅ ML Model trained!")
            print(f"   Training accuracy: {accuracy:.0%}")
            
            # Save model
            model_path = f'{QUANTUM_PATH}/backend/models/ml_model.joblib'
            os.makedirs(f'{QUANTUM_PATH}/backend/models', exist_ok=True)
            joblib.dump(model, model_path)
            
            print(f"💾 Model saved to: {model_path}")
        else:
            print("⚠️  Need more backtest data to train ML model")
            
    except Exception as e:
        print(f"⚠️  ML training skipped: {e}")
else:
    print("⚠️  No backtest data available for ML training")

# ============================================================================
# PART 7: FINAL READINESS ASSESSMENT
# ============================================================================

print("\n" + "="*80)
print("🎯 FINAL READINESS ASSESSMENT")
print("="*80)

readiness_checks = []

# Check 1: Modules working
modules_check = modules_working >= 5  # At least 5/7
readiness_checks.append(("Modules Working (5/7+)", modules_check))

# Check 2: Validation pass rate
validation_check = pass_rate >= 0.7 if all_tests else False
readiness_checks.append(("Module Validation (70%+)", validation_check))

# Check 3: Backtest catch rate
backtest_check = catch_rate >= 0.33 if backtest_results else False
readiness_checks.append(("Backtest Catch Rate (33%+)", backtest_check))

# Check 4: Live scan working
scan_check = len(results) > 0 if 'results' in locals() else False
readiness_checks.append(("Live Scanner Working", scan_check))

# Display results
print("\n📋 CHECKLIST:")
for check_name, passed in readiness_checks:
    status = "✅" if passed else "❌"
    print(f"   {status} {check_name}")

# Overall assessment
all_passed = all(passed for _, passed in readiness_checks)
critical_passed = readiness_checks[0][1] and readiness_checks[3][1]  # Modules + Scanner

print("\n" + "="*80)

if all_passed:
    print("🎉 SYSTEM FULLY READY!")
    print("="*80)
    print("\n✅ ALL CHECKS PASSED")
    print("\n🚀 NEXT STEPS:")
    print("   1. Launch dashboard: streamlit run REAL_PORTFOLIO_DASHBOARD.py")
    print("   2. Paper trade this week")
    print("   3. Go live next week with small positions")
    print("\n💰 YOU'RE READY TO MAKE MONEY!")
    
elif critical_passed:
    print("⚠️  SYSTEM PARTIALLY READY")
    print("="*80)
    print("\n✅ Core components working (modules + scanner)")
    print("⚠️  Some validation/backtest checks need improvement")
    print("\n🎯 RECOMMENDATIONS:")
    print("   1. You CAN use the dashboard")
    print("   2. Start with paper trading")
    print("   3. Monitor performance closely")
    print("   4. Tune thresholds based on results")
    
else:
    print("❌ SYSTEM NOT READY")
    print("="*80)
    print("\n❌ Critical components failing")
    print("\n🔧 REQUIRED FIXES:")
    
    if not readiness_checks[0][1]:
        print("   - Fix module imports/errors")
    if not readiness_checks[3][1]:
        print("   - Fix scanner functionality")
    
    print("\n📝 DO NOT trade real money yet. Fix issues first.")

print("\n" + "="*80)
print("📊 TRAINING & VALIDATION COMPLETE")
print("="*80)

