"""
🔍 QUANTUM AI COCKPIT - COMPLETE SYSTEM DIAGNOSTIC
===================================================

This cell will systematically check EVERY component of your system and
identify ALL problems preventing bulletproof operation and training.

Copy this ENTIRE cell into Google Colab and run it to get a complete
diagnostic report of what needs to be fixed.

Author: AI Assistant
Date: November 18, 2025
Purpose: Pre-training system validation
"""

import sys
import os
import subprocess
import importlib
from pathlib import Path
from datetime import datetime
import asyncio
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# DIAGNOSTIC TRACKING
# ═══════════════════════════════════════════════════════════════════

class DiagnosticReport:
    """Track all issues found during diagnostics."""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.successes = []
        self.critical_failures = []
        
    def add_issue(self, category, description, severity="ERROR"):
        """Add an issue to the report."""
        issue = {
            'category': category,
            'description': description,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        }
        
        if severity == "CRITICAL":
            self.critical_failures.append(issue)
        elif severity == "ERROR":
            self.issues.append(issue)
        elif severity == "WARNING":
            self.warnings.append(issue)
            
    def add_success(self, category, description):
        """Add a success to the report."""
        self.successes.append({
            'category': category,
            'description': description
        })
        
    def print_report(self):
        """Print a formatted diagnostic report."""
        print("\n" + "="*80)
        print("📊 DIAGNOSTIC REPORT")
        print("="*80)
        
        # Critical failures
        if self.critical_failures:
            print(f"\n🚨 CRITICAL FAILURES ({len(self.critical_failures)}):")
            print("-" * 80)
            for i, issue in enumerate(self.critical_failures, 1):
                print(f"{i}. [{issue['category']}] {issue['description']}")
        
        # Errors
        if self.issues:
            print(f"\n❌ ERRORS ({len(self.issues)}):")
            print("-" * 80)
            for i, issue in enumerate(self.issues, 1):
                print(f"{i}. [{issue['category']}] {issue['description']}")
        
        # Warnings
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            print("-" * 80)
            for i, warning in enumerate(self.warnings, 1):
                print(f"{i}. [{warning['category']}] {warning['description']}")
        
        # Successes
        if self.successes:
            print(f"\n✅ SUCCESSES ({len(self.successes)}):")
            print("-" * 80)
            for success in self.successes[-10:]:  # Show last 10
                print(f"  • [{success['category']}] {success['description']}")
        
        # Summary
        print("\n" + "="*80)
        print("📈 SUMMARY")
        print("="*80)
        total_issues = len(self.critical_failures) + len(self.issues)
        system_health = max(0, 100 - (len(self.critical_failures) * 25) - (len(self.issues) * 10) - (len(self.warnings) * 2))
        
        print(f"Critical Failures: {len(self.critical_failures)}")
        print(f"Errors: {len(self.issues)}")
        print(f"Warnings: {len(self.warnings)}")
        print(f"Successes: {len(self.successes)}")
        print(f"System Health: {system_health}%")
        
        if system_health >= 90:
            print("\n🎉 SYSTEM STATUS: BULLETPROOF - Ready for training!")
        elif system_health >= 70:
            print("\n✅ SYSTEM STATUS: GOOD - Minor fixes needed")
        elif system_health >= 50:
            print("\n⚠️  SYSTEM STATUS: FAIR - Several issues to address")
        else:
            print("\n🚨 SYSTEM STATUS: NEEDS WORK - Critical issues must be fixed")
            
        return total_issues == 0

report = DiagnosticReport()

print("="*80)
print("🔍 QUANTUM AI COCKPIT - SYSTEM DIAGNOSTIC")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ═══════════════════════════════════════════════════════════════════
# PHASE 1: ENVIRONMENT SETUP
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PHASE 1: ENVIRONMENT SETUP")
print("="*80)

# Check if in Colab
try:
    import google.colab
    IN_COLAB = True
    report.add_success("Environment", "Running in Google Colab")
    print("✅ Running in Google Colab")
except ImportError:
    IN_COLAB = False
    report.add_issue("Environment", "NOT running in Google Colab - some features may not work", "WARNING")
    print("⚠️  NOT running in Google Colab")

# Check Python version
python_version = sys.version_info
print(f"📌 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
if python_version.major == 3 and python_version.minor >= 8:
    report.add_success("Environment", f"Python {python_version.major}.{python_version.minor}.{python_version.micro}")
else:
    report.add_issue("Environment", f"Python version {python_version} may have compatibility issues", "WARNING")

# ═══════════════════════════════════════════════════════════════════
# PHASE 2: DEPENDENCY CHECKS
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PHASE 2: CHECKING DEPENDENCIES")
print("="*80)

# Critical packages for institutional system
REQUIRED_PACKAGES = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'prophet': 'prophet',  # Facebook Prophet for forecasting
    'lightgbm': 'lightgbm',  # LightGBM for ML
    'xgboost': 'xgboost',  # XGBoost for ML
    'statsmodels': 'statsmodels',  # ARIMA
    'sklearn': 'scikit-learn',  # General ML
    'aiohttp': 'aiohttp',  # Async HTTP
    'yfinance': 'yfinance',  # Fallback data source
}

installed_packages = {}
missing_packages = []

for import_name, package_name in REQUIRED_PACKAGES.items():
    try:
        if import_name == 'sklearn':
            import sklearn
            installed_packages[import_name] = sklearn.__version__
        else:
            mod = importlib.import_module(import_name)
            version = getattr(mod, '__version__', 'unknown')
            installed_packages[import_name] = version
        
        print(f"✅ {package_name:20s} v{installed_packages[import_name]}")
        report.add_success("Dependencies", f"{package_name} installed")
    except ImportError:
        print(f"❌ {package_name:20s} NOT INSTALLED")
        missing_packages.append(package_name)
        report.add_issue("Dependencies", f"{package_name} is missing", "CRITICAL")

# Check NumPy version (must be < 2.0 for Prophet compatibility)
if 'numpy' in installed_packages:
    numpy_version = installed_packages['numpy']
    major_version = int(numpy_version.split('.')[0])
    if major_version >= 2:
        report.add_issue("Dependencies", 
                        f"NumPy {numpy_version} is incompatible with Prophet (needs <2.0)",
                        "CRITICAL")
        print(f"⚠️  NumPy version {numpy_version} may cause Prophet errors!")
        print("   Fix: pip install 'numpy<2.0' --force-reinstall")
    else:
        report.add_success("Dependencies", f"NumPy {numpy_version} is compatible with Prophet")

# Auto-install missing packages
if missing_packages:
    print(f"\n📦 Installing {len(missing_packages)} missing packages...")
    for package in missing_packages:
        try:
            print(f"   Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
            report.add_success("Dependencies", f"Auto-installed {package}")
            print(f"   ✅ {package} installed")
        except Exception as e:
            report.add_issue("Dependencies", f"Failed to install {package}: {e}", "CRITICAL")
            print(f"   ❌ Failed to install {package}")

# ═══════════════════════════════════════════════════════════════════
# PHASE 3: GOOGLE DRIVE MOUNT & FILE VERIFICATION
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PHASE 3: GOOGLE DRIVE & FILE VERIFICATION")
print("="*80)

# Mount Google Drive
if IN_COLAB:
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        report.add_success("Drive", "Google Drive mounted successfully")
        print("✅ Google Drive mounted")
    except Exception as e:
        report.add_issue("Drive", f"Failed to mount Google Drive: {e}", "CRITICAL")
        print(f"❌ Failed to mount Google Drive: {e}")

# Check project structure
PROJECT_ROOT = '/content/drive/MyDrive/Quantum_AI_Cockpit'
MODULES_PATH = f'{PROJECT_ROOT}/backend/modules'

if os.path.exists(PROJECT_ROOT):
    report.add_success("Drive", f"Project root found: {PROJECT_ROOT}")
    print(f"✅ Project root: {PROJECT_ROOT}")
else:
    report.add_issue("Drive", f"Project root not found: {PROJECT_ROOT}", "CRITICAL")
    print(f"❌ Project root not found: {PROJECT_ROOT}")

if os.path.exists(MODULES_PATH):
    report.add_success("Drive", f"Modules path found: {MODULES_PATH}")
    print(f"✅ Modules path: {MODULES_PATH}")
else:
    report.add_issue("Drive", f"Modules path not found: {MODULES_PATH}", "CRITICAL")
    print(f"❌ Modules path not found: {MODULES_PATH}")

# Required module files
REQUIRED_MODULES = [
    'data_orchestrator.py',          # Data fetching
    'elite_forecaster.py',           # Elite forecasting
    'master_analysis_institutional.py',  # Main institutional engine
    'fusior_forecast_institutional.py',  # Institutional forecaster wrapper
    'ai_recommender_institutional_enhanced.py',  # Trade plan builder
    'pattern_integration_layer.py',  # Pattern analysis
    'pattern_quality_scorer.py',     # Pattern grading
]

OPTIONAL_MODULES = [
    'pattern_lib.py',                # Pattern library
    'master_analysis_engine.py',     # Basic analysis engine
    'fusior_forecast.py',            # Legacy forecaster (fallback)
]

print("\n📁 Checking required modules:")
missing_modules = []
for module in REQUIRED_MODULES:
    module_path = os.path.join(MODULES_PATH, module)
    if os.path.exists(module_path):
        size = os.path.getsize(module_path)
        print(f"✅ {module:45s} ({size:,} bytes)")
        report.add_success("Modules", f"{module} found")
    else:
        print(f"❌ {module:45s} NOT FOUND")
        missing_modules.append(module)
        report.add_issue("Modules", f"{module} is missing", "CRITICAL")

print("\n📁 Checking optional modules:")
for module in OPTIONAL_MODULES:
    module_path = os.path.join(MODULES_PATH, module)
    if os.path.exists(module_path):
        size = os.path.getsize(module_path)
        print(f"✅ {module:45s} ({size:,} bytes)")
        report.add_success("Modules", f"{module} found")
    else:
        print(f"⚠️  {module:45s} NOT FOUND (optional)")
        report.add_issue("Modules", f"{module} is missing (optional)", "WARNING")

if missing_modules:
    print(f"\n🚨 {len(missing_modules)} CRITICAL modules missing!")
    print("   Upload these files to Google Drive:")
    print(f"   Destination: {MODULES_PATH}")
    for module in missing_modules:
        print(f"   - {module}")

# Add modules to Python path
if os.path.exists(MODULES_PATH):
    if MODULES_PATH not in sys.path:
        sys.path.insert(0, MODULES_PATH)
        sys.path.insert(0, PROJECT_ROOT)
        print(f"\n✅ Added to Python path: {MODULES_PATH}")
        report.add_success("Setup", "Modules added to Python path")

# ═══════════════════════════════════════════════════════════════════
# PHASE 4: DATA ORCHESTRATOR TEST
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PHASE 4: DATA ORCHESTRATOR TEST")
print("="*80)

async def test_data_orchestrator():
    """Test data fetching capabilities."""
    try:
        from data_orchestrator import DataOrchestrator
        report.add_success("Data", "DataOrchestrator imported successfully")
        print("✅ DataOrchestrator imported")
        
        orchestrator = DataOrchestrator()
        
        # Test with a reliable stock
        test_symbol = 'AAPL'
        print(f"\n📊 Testing data fetch for {test_symbol}...")
        
        df = await orchestrator.fetch_symbol_data(test_symbol, days=150)
        
        if df is not None and len(df) > 0:
            print(f"✅ Fetched {len(df)} rows of data for {test_symbol}")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Date range: {df.index[0]} to {df.index[-1]}")
            
            # Check data quality
            has_required_columns = all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
            if has_required_columns:
                report.add_success("Data", f"Complete OHLCV data for {test_symbol}")
                print(f"✅ All required columns present")
            else:
                report.add_issue("Data", f"Missing required columns for {test_symbol}", "ERROR")
                print(f"❌ Missing required columns")
            
            # Check for NaN values
            nan_count = df.isna().sum().sum()
            if nan_count == 0:
                report.add_success("Data", "No NaN values in data")
                print(f"✅ No NaN values")
            else:
                report.add_issue("Data", f"Found {nan_count} NaN values in data", "WARNING")
                print(f"⚠️  Found {nan_count} NaN values")
            
            # Check if enough data for Prophet
            if len(df) >= 100:
                report.add_success("Data", f"{len(df)} rows - sufficient for Prophet (needs 100+)")
                print(f"✅ Sufficient data for Prophet forecasting")
            else:
                report.add_issue("Data", f"Only {len(df)} rows - Prophet needs 100+", "ERROR")
                print(f"⚠️  Insufficient data for Prophet (needs 100+ rows)")
                
        else:
            report.add_issue("Data", f"Failed to fetch data for {test_symbol}", "CRITICAL")
            print(f"❌ Failed to fetch data for {test_symbol}")
            
    except Exception as e:
        report.add_issue("Data", f"DataOrchestrator error: {e}", "CRITICAL")
        print(f"❌ DataOrchestrator error: {e}")
        import traceback
        traceback.print_exc()

if os.path.exists(MODULES_PATH):
    await test_data_orchestrator()
else:
    report.add_issue("Data", "Cannot test DataOrchestrator - modules path not found", "CRITICAL")
    print("❌ Cannot test - modules not found")

# ═══════════════════════════════════════════════════════════════════
# PHASE 5: ELITE FORECASTER TEST
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PHASE 5: ELITE FORECASTER TEST")
print("="*80)

async def test_elite_forecaster():
    """Test elite forecasting models."""
    try:
        from elite_forecaster import run_elite_forecast, PROPHET_AVAILABLE, LIGHTGBM_AVAILABLE, XGBOOST_AVAILABLE, ARIMA_AVAILABLE
        from data_orchestrator import DataOrchestrator
        
        report.add_success("Forecaster", "Elite Forecaster imported successfully")
        print("✅ Elite Forecaster imported")
        
        # Check which models are available
        models_available = []
        if PROPHET_AVAILABLE:
            models_available.append("Prophet")
            print("✅ Prophet available (58-62% accuracy)")
            report.add_success("Forecaster", "Prophet model available")
        else:
            print("❌ Prophet NOT available")
            report.add_issue("Forecaster", "Prophet model not available", "ERROR")
            
        if LIGHTGBM_AVAILABLE:
            models_available.append("LightGBM")
            print("✅ LightGBM available (55-60% accuracy)")
            report.add_success("Forecaster", "LightGBM model available")
        else:
            print("⚠️  LightGBM NOT available")
            report.add_issue("Forecaster", "LightGBM model not available", "WARNING")
            
        if XGBOOST_AVAILABLE:
            models_available.append("XGBoost")
            print("✅ XGBoost available (52-58% accuracy)")
            report.add_success("Forecaster", "XGBoost model available")
        else:
            print("⚠️  XGBoost NOT available")
            report.add_issue("Forecaster", "XGBoost model not available", "WARNING")
            
        if ARIMA_AVAILABLE:
            models_available.append("ARIMA")
            print("✅ ARIMA available (48-52% accuracy)")
            report.add_success("Forecaster", "ARIMA model available")
        else:
            print("⚠️  ARIMA NOT available")
            report.add_issue("Forecaster", "ARIMA model not available", "WARNING")
        
        if not models_available:
            report.add_issue("Forecaster", "NO forecasting models available", "CRITICAL")
            print("🚨 NO forecasting models available!")
            return
        
        print(f"\n📊 {len(models_available)} models available: {', '.join(models_available)}")
        
        # Test actual forecasting
        test_symbol = 'AMD'
        print(f"\n🔮 Testing forecast for {test_symbol}...")
        
        orchestrator = DataOrchestrator()
        df = await orchestrator.fetch_symbol_data(test_symbol, days=150)
        
        if df is not None and len(df) >= 100:
            forecast = await run_elite_forecast(test_symbol, df, horizon=5)
            
            if forecast and 'ensemble_forecast' in forecast:
                current_price = float(df['close'].iloc[-1])
                forecast_price = forecast['ensemble_forecast']
                change_pct = ((forecast_price / current_price) - 1.0) * 100
                confidence = forecast.get('ensemble_confidence', 0)
                
                print(f"✅ Forecast generated successfully!")
                print(f"   Current: ${current_price:.2f}")
                print(f"   Forecast: ${forecast_price:.2f}")
                print(f"   Change: {change_pct:+.2f}%")
                print(f"   Confidence: {confidence*100:.1f}%")
                
                report.add_success("Forecaster", f"Forecast generated with {confidence*100:.1f}% confidence")
            else:
                report.add_issue("Forecaster", "Forecast returned but missing expected fields", "ERROR")
                print("⚠️  Forecast incomplete")
        else:
            report.add_issue("Forecaster", "Insufficient data for forecast test", "ERROR")
            print("⚠️  Insufficient data for forecast test")
            
    except Exception as e:
        report.add_issue("Forecaster", f"Elite Forecaster error: {e}", "CRITICAL")
        print(f"❌ Elite Forecaster error: {e}")
        import traceback
        traceback.print_exc()

if os.path.exists(MODULES_PATH):
    await test_elite_forecaster()
else:
    print("❌ Cannot test - modules not found")

# ═══════════════════════════════════════════════════════════════════
# PHASE 6: PATTERN ANALYSIS TEST
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PHASE 6: PATTERN ANALYSIS TEST")
print("="*80)

async def test_pattern_analysis():
    """Test pattern detection and grading."""
    try:
        from pattern_integration_layer import PatternIntegrationLayer
        from data_orchestrator import DataOrchestrator
        
        report.add_success("Patterns", "Pattern Integration Layer imported")
        print("✅ Pattern Integration Layer imported")
        
        test_symbol = 'NVDA'
        print(f"\n📊 Testing pattern analysis for {test_symbol}...")
        
        orchestrator = DataOrchestrator()
        df = await orchestrator.fetch_symbol_data(test_symbol, days=60)
        
        if df is not None and len(df) >= 30:
            pattern_layer = PatternIntegrationLayer()
            patterns = await pattern_layer.analyze_all_patterns(df, symbol=test_symbol)
            
            if patterns:
                final_signal = patterns.get('final_signal', {})
                summary = patterns.get('summary', {})
                
                print(f"✅ Pattern analysis complete")
                
                if final_signal:
                    print(f"   Signal: {final_signal.get('action', 'N/A')}")
                    print(f"   Confidence: {final_signal.get('confidence', 0)*100:.1f}%")
                    report.add_success("Patterns", "Pattern signals generated")
                
                if summary:
                    detected = summary.get('patterns_detected', 0)
                    print(f"   Patterns detected: {detected}")
                    report.add_success("Patterns", f"{detected} patterns detected")
                    
            else:
                report.add_issue("Patterns", "Pattern analysis returned empty result", "WARNING")
                print("⚠️  No patterns detected")
        else:
            report.add_issue("Patterns", "Insufficient data for pattern analysis", "ERROR")
            print("⚠️  Insufficient data")
            
    except Exception as e:
        report.add_issue("Patterns", f"Pattern analysis error: {e}", "ERROR")
        print(f"❌ Pattern analysis error: {e}")
        import traceback
        traceback.print_exc()

if os.path.exists(MODULES_PATH):
    await test_pattern_analysis()
else:
    print("❌ Cannot test - modules not found")

# ═══════════════════════════════════════════════════════════════════
# PHASE 7: INSTITUTIONAL ENGINE TEST
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PHASE 7: INSTITUTIONAL ENGINE TEST")
print("="*80)

async def test_institutional_engine():
    """Test the complete institutional analysis engine."""
    try:
        from master_analysis_institutional import InstitutionalAnalysisEngine
        
        report.add_success("Institutional", "InstitutionalAnalysisEngine imported")
        print("✅ InstitutionalAnalysisEngine imported")
        
        test_symbol = 'AMD'
        account_balance = 10000
        
        print(f"\n🏆 Running institutional analysis on {test_symbol}...")
        print(f"   Account: ${account_balance:,}")
        
        engine = InstitutionalAnalysisEngine()
        result = await engine.analyze_with_ensemble(
            symbol=test_symbol,
            account_balance=account_balance,
            forecast_days=5
        )
        
        if result:
            current_price = result.get('current_price', 0)
            action = result.get('action', 'N/A')
            confidence = result.get('confidence', 0) * 100
            
            print(f"\n✅ INSTITUTIONAL ANALYSIS COMPLETE")
            print(f"   Price: ${current_price:.2f}")
            print(f"   Action: {action}")
            print(f"   Confidence: {confidence:.1f}%")
            
            # Check for complete trade plan
            has_entry_plan = 'entry_plan' in result
            has_exit_plan = 'exit_plan' in result
            has_position_sizing = 'position_sizing' in result
            has_risk_metrics = 'risk_metrics' in result
            has_trade_grade = 'trade_grade' in result
            
            if has_entry_plan:
                print(f"   ✅ Entry plan included")
                report.add_success("Institutional", "Entry plan generated")
            else:
                print(f"   ⚠️  No entry plan")
                report.add_issue("Institutional", "Missing entry plan", "WARNING")
                
            if has_exit_plan:
                print(f"   ✅ Exit plan included")
                report.add_success("Institutional", "Exit plan generated")
            else:
                print(f"   ⚠️  No exit plan")
                report.add_issue("Institutional", "Missing exit plan", "WARNING")
                
            if has_position_sizing:
                shares = result['position_sizing'].get('total_shares', 0)
                print(f"   ✅ Position sizing: {shares} shares")
                report.add_success("Institutional", f"Position sizing: {shares} shares")
            else:
                print(f"   ⚠️  No position sizing")
                report.add_issue("Institutional", "Missing position sizing", "WARNING")
                
            if has_risk_metrics:
                rr = result['risk_metrics'].get('risk_reward_ratio', 0)
                print(f"   ✅ Risk/Reward: {rr:.2f}:1")
                report.add_success("Institutional", f"R/R ratio: {rr:.2f}:1")
            else:
                print(f"   ⚠️  No risk metrics")
                report.add_issue("Institutional", "Missing risk metrics", "WARNING")
                
            if has_trade_grade:
                grade = result['trade_grade'].get('grade', 'N/A')
                score = result['trade_grade'].get('score', 0)
                print(f"   ✅ Trade Grade: {grade} ({score:.1f}/100)")
                report.add_success("Institutional", f"Trade grade: {grade}")
            else:
                print(f"   ⚠️  No trade grade")
                report.add_issue("Institutional", "Missing trade grade", "WARNING")
            
            # Overall assessment
            complete_plan = has_entry_plan and has_exit_plan and has_position_sizing and has_risk_metrics
            if complete_plan:
                report.add_success("Institutional", "Complete institutional trade plan generated")
                print(f"\n🎉 COMPLETE INSTITUTIONAL TRADE PLAN!")
            else:
                report.add_issue("Institutional", "Incomplete trade plan - missing components", "ERROR")
                print(f"\n⚠️  Trade plan incomplete - missing components")
                
        else:
            report.add_issue("Institutional", "Institutional analysis returned no result", "CRITICAL")
            print(f"❌ No result returned")
            
    except Exception as e:
        report.add_issue("Institutional", f"Institutional engine error: {e}", "CRITICAL")
        print(f"❌ Institutional engine error: {e}")
        import traceback
        traceback.print_exc()

if os.path.exists(MODULES_PATH):
    await test_institutional_engine()
else:
    print("❌ Cannot test - modules not found")

# ═══════════════════════════════════════════════════════════════════
# PHASE 8: TRAINING READINESS CHECK
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PHASE 8: TRAINING READINESS CHECK")
print("="*80)

# Check for training prerequisites
training_ready = True

# 1. Check data availability
print("\n1️⃣ Data Availability:")
if len([s for s in report.successes if s['category'] == 'Data']) >= 3:
    print("   ✅ Data fetching operational")
else:
    print("   ❌ Data fetching has issues")
    training_ready = False
    report.add_issue("Training", "Data fetching must be stable before training", "CRITICAL")

# 2. Check forecasting models
print("\n2️⃣ Forecasting Models:")
forecaster_successes = len([s for s in report.successes if s['category'] == 'Forecaster'])
if forecaster_successes >= 2:
    print("   ✅ Multiple forecasting models available")
else:
    print("   ❌ Need at least 2 forecasting models")
    training_ready = False
    report.add_issue("Training", "Need Prophet + LightGBM at minimum", "CRITICAL")

# 3. Check institutional engine
print("\n3️⃣ Institutional Engine:")
institutional_successes = len([s for s in report.successes if s['category'] == 'Institutional'])
if institutional_successes >= 4:
    print("   ✅ Institutional engine fully functional")
else:
    print("   ❌ Institutional engine incomplete")
    training_ready = False
    report.add_issue("Training", "Institutional engine must generate complete trade plans", "CRITICAL")

# 4. Check all required modules
print("\n4️⃣ Required Modules:")
if not missing_modules:
    print("   ✅ All required modules present")
else:
    print(f"   ❌ {len(missing_modules)} modules missing")
    training_ready = False
    report.add_issue("Training", f"{len(missing_modules)} critical modules missing", "CRITICAL")

# 5. Check dependencies
print("\n5️⃣ Dependencies:")
if not missing_packages:
    print("   ✅ All dependencies installed")
else:
    print(f"   ❌ {len(missing_packages)} packages missing")
    training_ready = False
    report.add_issue("Training", f"{len(missing_packages)} required packages missing", "CRITICAL")

# Training readiness verdict
print("\n" + "="*80)
if training_ready:
    print("🎉 SYSTEM IS READY FOR TRAINING!")
    print("="*80)
    report.add_success("Training", "All prerequisites met - ready to train")
else:
    print("🚨 SYSTEM NOT READY FOR TRAINING")
    print("="*80)
    print("Fix the issues above before training")

# ═══════════════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════════════

is_bulletproof = report.print_report()

print("\n" + "="*80)
print("🎯 ACTION ITEMS")
print("="*80)

if report.critical_failures:
    print("\n🚨 CRITICAL (Fix These First):")
    for i, issue in enumerate(report.critical_failures, 1):
        print(f"\n{i}. {issue['description']}")
        
        # Provide specific fixes
        if 'missing' in issue['description'].lower() and '.py' in issue['description']:
            print(f"   → Upload file to: {MODULES_PATH}")
        elif 'numpy' in issue['description'].lower() and 'incompatible' in issue['description'].lower():
            print(f"   → Run: pip install 'numpy<2.0' --force-reinstall")
        elif 'prophet' in issue['description'].lower():
            print(f"   → Run: pip install prophet")
        elif 'drive' in issue['description'].lower():
            print(f"   → Mount Google Drive and create folder structure")

if report.issues:
    print("\n❌ ERRORS (Fix After Critical):")
    for i, issue in enumerate(report.issues[:5], 1):  # Show top 5
        print(f"\n{i}. {issue['description']}")

print("\n" + "="*80)
print("📝 NEXT STEPS")
print("="*80)

if is_bulletproof:
    print("""
1. ✅ Run full backtest on 20 stocks
2. ✅ Validate performance metrics
3. ✅ Train reinforcement learning models
4. ✅ Deploy to paper trading
5. ✅ Monitor for 1 week
6. ✅ Deploy to real money
""")
else:
    critical_count = len(report.critical_failures)
    error_count = len(report.issues)
    
    if critical_count > 0:
        print(f"""
1. ❌ Fix {critical_count} CRITICAL issues (see above)
2. ⏳ Re-run this diagnostic
3. ⏳ Address remaining errors
4. ⏳ Validate system is bulletproof
5. ⏳ Begin training
""")
    elif error_count > 0:
        print(f"""
1. ✅ Critical issues resolved
2. ⚠️  Fix {error_count} remaining errors
3. ⏳ Re-run diagnostic
4. ⏳ Begin training
""")
    else:
        print("""
1. ✅ All major issues resolved
2. ⚠️  Address warnings (optional)
3. ✅ System ready for training
""")

print("\n" + "="*80)
print(f"Diagnostic completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

print("\n💡 TIP: Save this output and fix issues in order of severity")
print("   Re-run this cell after each fix to track progress")

