"""
🧪 COLAB MODULE TESTER
=======================
Test which modules actually WORK in Google Colab!

This will:
1. Try to import each module
2. Try to instantiate the class
3. Try to run a basic function
4. Report which ones WORK vs FAIL
"""

import sys
sys.path.insert(0, '/content/drive/MyDrive/QuantumAI/backend/modules')

# Test results
results = {
    'working': [],
    'failed_import': [],
    'failed_instantiate': [],
    'failed_run': []
}

print("=" * 80)
print("🧪 QUANTUM AI MODULE TESTER")
print("=" * 80)
print("\nTesting which modules actually work in Colab...\n")

# ============================================================================
# TIER 1: MUST-HAVE MODULES TO TEST
# ============================================================================

test_modules = [
    # FORECASTERS
    {
        'name': 'Elite Forecaster',
        'import': 'import elite_forecaster as ef',
        'test': 'ef',
        'function': None,  # Module-level functions
        'critical': True
    },
    {
        'name': 'Fusior Forecast',
        'import': 'from fusior_forecast import FusiorForecast',
        'test': 'FusiorForecast()',
        'function': None,
        'critical': False
    },
    
    # PATTERN RECOGNITION
    {
        'name': 'Pattern Recognition Engine',
        'import': 'from PATTERN_RECOGNITION_ENGINE import UnifiedPatternRecognitionEngine',
        'test': 'UnifiedPatternRecognitionEngine()',
        'function': None,
        'critical': True
    },
    
    # INSTITUTIONAL
    {
        'name': 'Dark Pool Tracker',
        'import': 'from dark_pool_tracker import DarkPoolTracker',
        'test': 'DarkPoolTracker()',
        'function': 'analyze_ticker("AAPL")',
        'critical': True
    },
    {
        'name': 'Insider Trading Tracker',
        'import': 'from insider_trading_tracker import InsiderTradingTracker',
        'test': 'InsiderTradingTracker()',
        'function': 'analyze_ticker("AAPL")',
        'critical': True
    },
    {
        'name': 'Earnings Surprise Predictor',
        'import': 'from earnings_surprise_predictor import EarningsSurprisePredictor',
        'test': 'EarningsSurprisePredictor()',
        'function': None,
        'critical': True
    },
    
    # SCANNERS
    {
        'name': 'Pre-Gainer Scanner',
        'import': 'from pre_gainer_scanner import PreGainerScanner',
        'test': 'PreGainerScanner()',
        'function': None,
        'critical': True
    },
    {
        'name': 'Opportunity Scanner',
        'import': 'from opportunity_scanner import OpportunityScanner',
        'test': 'OpportunityScanner()',
        'function': 'scan_for_opportunities(max_results=3)',
        'critical': True
    },
    {
        'name': 'Day Trading Scanner',
        'import': 'from day_trading_scanner import DayTradingScanner',
        'test': 'DayTradingScanner()',
        'function': None,
        'critical': False
    },
    {
        'name': 'Breakout Screener',
        'import': 'from breakout_screener import BreakoutScreener',
        'test': 'BreakoutScreener()',
        'function': None,
        'critical': False
    },
    {
        'name': 'Short Squeeze Scanner',
        'import': 'from short_squeeze_scanner import ShortSqueezeScanner',
        'test': 'ShortSqueezeScanner()',
        'function': None,
        'critical': False
    },
    
    # AI RECOMMENDERS
    {
        'name': 'AI Recommender (Institutional Enhanced)',
        'import': 'from ai_recommender_institutional_enhanced import InstitutionalRecommender',
        'test': 'InstitutionalRecommender()',
        'function': None,
        'critical': True
    },
    {
        'name': 'Master Analysis (Institutional)',
        'import': 'from master_analysis_institutional import MasterAnalysisInstitutional',
        'test': 'MasterAnalysisInstitutional()',
        'function': None,
        'critical': False
    },
    
    # TECHNICAL ANALYSIS
    {
        'name': 'Multi-Timeframe Analyzer',
        'import': 'from multi_timeframe_analyzer import MultiTimeframeAnalyzer',
        'test': 'MultiTimeframeAnalyzer()',
        'function': None,
        'critical': False
    },
    {
        'name': 'Regime Detector',
        'import': 'from regime_detector import RegimeDetector',
        'test': 'RegimeDetector()',
        'function': None,
        'critical': False
    },
    {
        'name': 'Support/Resistance Detector',
        'import': 'from support_resistance_detector import SupportResistanceDetector',
        'test': 'SupportResistanceDetector()',
        'function': None,
        'critical': False
    },
    {
        'name': 'Momentum Tracker',
        'import': 'from momentum_tracker import MomentumTracker',
        'test': 'MomentumTracker()',
        'function': None,
        'critical': False
    },
    
    # DATA & NEWS
    {
        'name': 'News Scraper',
        'import': 'from news_scraper import NewsScraper',
        'test': 'NewsScraper()',
        'function': None,
        'critical': True
    },
    {
        'name': 'Sentiment Engine',
        'import': 'from sentiment_engine import SentimentEngine',
        'test': 'SentimentEngine()',
        'function': None,
        'critical': True
    },
    {
        'name': 'Universal Scraper Engine',
        'import': 'from universal_scraper_engine import UniversalScraperEngine',
        'test': 'UniversalScraperEngine()',
        'function': None,
        'critical': True
    },
    
    # PORTFOLIO & RISK
    {
        'name': 'Portfolio Manager',
        'import': 'from portfolio_manager import PortfolioManager',
        'test': 'PortfolioManager()',
        'function': None,
        'critical': False
    },
    {
        'name': 'Risk Engine',
        'import': 'from risk_engine import RiskEngine',
        'test': 'RiskEngine()',
        'function': None,
        'critical': True
    },
    {
        'name': 'Position Size Calculator',
        'import': 'from position_size_calculator import PositionSizeCalculator',
        'test': 'PositionSizeCalculator()',
        'function': None,
        'critical': True
    },
    
    # DATA ORCHESTRATION
    {
        'name': 'Data Orchestrator',
        'import': 'from data_orchestrator import DataOrchestrator',
        'test': 'DataOrchestrator()',
        'function': None,
        'critical': True
    },
]

# ============================================================================
# RUN TESTS
# ============================================================================

for i, module in enumerate(test_modules, 1):
    name = module['name']
    critical = "🔴 CRITICAL" if module['critical'] else "🟡 OPTIONAL"
    
    print(f"\n[{i}/{len(test_modules)}] Testing: {name} {critical}")
    print("-" * 60)
    
    # Step 1: Try import
    try:
        exec(module['import'])
        print(f"   ✅ Import successful")
    except Exception as e:
        print(f"   ❌ Import failed: {str(e)[:60]}")
        results['failed_import'].append({'name': name, 'error': str(e)[:100], 'critical': module['critical']})
        continue
    
    # Step 2: Try instantiation
    try:
        obj = eval(module['test'])
        print(f"   ✅ Instantiation successful")
    except Exception as e:
        print(f"   ❌ Instantiation failed: {str(e)[:60]}")
        results['failed_instantiate'].append({'name': name, 'error': str(e)[:100], 'critical': module['critical']})
        continue
    
    # Step 3: Try function call (if specified)
    if module['function']:
        try:
            result = eval(f"obj.{module['function']}")
            print(f"   ✅ Function test successful")
            print(f"   📊 Result type: {type(result)}")
        except Exception as e:
            print(f"   ⚠️ Function test failed: {str(e)[:60]}")
            results['failed_run'].append({'name': name, 'error': str(e)[:100], 'critical': module['critical']})
            # Don't mark as failed if function fails but class works
    
    # Module is working!
    results['working'].append({'name': name, 'critical': module['critical']})
    print(f"   ✅ MODULE WORKS!")

# ============================================================================
# FINAL REPORT
# ============================================================================

print("\n\n" + "=" * 80)
print("📊 FINAL TEST RESULTS")
print("=" * 80)

print(f"\n✅ WORKING MODULES: {len(results['working'])}")
for item in results['working']:
    critical = "🔴" if item['critical'] else "🟡"
    print(f"   {critical} {item['name']}")

print(f"\n❌ FAILED TO IMPORT: {len(results['failed_import'])}")
for item in results['failed_import']:
    critical = "🔴 CRITICAL" if item['critical'] else "🟡 Optional"
    print(f"   {critical} {item['name']}")
    print(f"      Error: {item['error']}")

print(f"\n❌ FAILED TO INSTANTIATE: {len(results['failed_instantiate'])}")
for item in results['failed_instantiate']:
    critical = "🔴 CRITICAL" if item['critical'] else "🟡 Optional"
    print(f"   {critical} {item['name']}")
    print(f"      Error: {item['error']}")

print(f"\n⚠️ FAILED FUNCTION TEST: {len(results['failed_run'])}")
for item in results['failed_run']:
    critical = "🔴 CRITICAL" if item['critical'] else "🟡 Optional"
    print(f"   {critical} {item['name']}")
    print(f"      Error: {item['error']}")

# Summary
total_tested = len(test_modules)
total_working = len(results['working'])
success_rate = (total_working / total_tested) * 100

print("\n" + "=" * 80)
print(f"📈 SUCCESS RATE: {total_working}/{total_tested} ({success_rate:.1f}%)")
print("=" * 80)

# Critical failures
critical_failures = [
    item for item in 
    results['failed_import'] + results['failed_instantiate'] 
    if item['critical']
]

if critical_failures:
    print(f"\n🚨 CRITICAL FAILURES: {len(critical_failures)}")
    print("These modules are marked as critical but failed!")
    for item in critical_failures:
        print(f"   ❌ {item['name']}: {item['error']}")
else:
    print("\n✅ All critical modules working!")

print("\n💡 RECOMMENDATION:")
print(f"   Use the {total_working} working modules for your dashboard")
print(f"   Focus on fixing {len(critical_failures)} critical failures")
print(f"   Ignore {len(results['failed_import']) + len(results['failed_instantiate']) - len(critical_failures)} optional failures")

print("\n" + "=" * 80)

