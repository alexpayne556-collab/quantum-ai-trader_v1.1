"""
🔍 QUANTUM AI COCKPIT - SYSTEM DIAGNOSTIC (AUTO-FIX NUMPY)
===========================================================

This version automatically fixes the NumPy/Prophet compatibility issue
before running diagnostics.

Copy this ENTIRE cell into Google Colab and run it.
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

print("="*80)
print("🔍 QUANTUM AI COCKPIT - SYSTEM DIAGNOSTIC (AUTO-FIX)")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ═══════════════════════════════════════════════════════════════════
# PHASE 0: AUTO-FIX NUMPY/PROPHET COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PHASE 0: AUTO-FIXING NUMPY/PROPHET COMPATIBILITY")
print("="*80)

# Check current NumPy version
try:
    import numpy as np
    numpy_version = np.__version__
    major_version = int(numpy_version.split('.')[0])
    
    print(f"📌 Current NumPy version: {numpy_version}")
    
    if major_version >= 2:
        print("⚠️  NumPy 2.0+ detected - incompatible with Prophet")
        print("🔧 Auto-fixing: Downgrading NumPy to <2.0...")
        
        # Downgrade NumPy
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy<2.0"])
        print("✅ NumPy downgraded")
        
        # Force reimport
        import importlib
        importlib.reload(np)
        
        print("⚠️  IMPORTANT: Restart runtime for changes to take full effect")
        print("   Runtime → Restart runtime, then re-run this cell")
    else:
        print(f"✅ NumPy {numpy_version} is compatible with Prophet")
        
except Exception as e:
    print(f"❌ Error checking NumPy: {e}")

# Install Prophet and other forecasters
print("\n📦 Installing forecasting libraries...")

required_libs = ['prophet', 'lightgbm', 'xgboost', 'statsmodels', 'yfinance', 'aiohttp']
for lib in required_libs:
    try:
        print(f"   Installing {lib}...", end=" ")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", lib])
        print("✅")
    except Exception as e:
        print(f"❌ {e}")

print("\n✅ Phase 0 complete - Dependencies installed")

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

# ═══════════════════════════════════════════════════════════════════
# PHASE 1: ENVIRONMENT CHECK
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PHASE 1: ENVIRONMENT CHECK")
print("="*80)

# Check if in Colab
try:
    import google.colab
    IN_COLAB = True
    report.add_success("Environment", "Running in Google Colab")
    print("✅ Running in Google Colab")
except ImportError:
    IN_COLAB = False
    report.add_issue("Environment", "NOT running in Google Colab", "WARNING")
    print("⚠️  NOT running in Google Colab")

# Check Python version
python_version = sys.version_info
print(f"📌 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
report.add_success("Environment", f"Python {python_version.major}.{python_version.minor}.{python_version.micro}")

# ═══════════════════════════════════════════════════════════════════
# PHASE 2: VERIFY DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PHASE 2: VERIFYING DEPENDENCIES")
print("="*80)

# Test each package
print("\n📦 Testing installed packages:")

# NumPy
try:
    import numpy as np
    print(f"✅ NumPy v{np.__version__}")
    if int(np.__version__.split('.')[0]) >= 2:
        report.add_issue("Dependencies", f"NumPy {np.__version__} may still cause issues - restart runtime", "WARNING")
    else:
        report.add_success("Dependencies", f"NumPy {np.__version__} compatible")
except Exception as e:
    print(f"❌ NumPy: {e}")
    report.add_issue("Dependencies", "NumPy import failed", "CRITICAL")

# Pandas
try:
    import pandas as pd
    print(f"✅ Pandas v{pd.__version__}")
    report.add_success("Dependencies", "Pandas installed")
except Exception as e:
    print(f"❌ Pandas: {e}")
    report.add_issue("Dependencies", "Pandas import failed", "CRITICAL")

# Prophet
try:
    from prophet import Prophet
    print(f"✅ Prophet (working)")
    report.add_success("Dependencies", "Prophet working")
except Exception as e:
    print(f"❌ Prophet: {e}")
    report.add_issue("Dependencies", f"Prophet failed: {e}", "CRITICAL")
    print("   → Restart runtime and re-run this cell")

# LightGBM
try:
    import lightgbm
    print(f"✅ LightGBM v{lightgbm.__version__}")
    report.add_success("Dependencies", "LightGBM installed")
except Exception as e:
    print(f"⚠️  LightGBM: {e}")
    report.add_issue("Dependencies", "LightGBM not available", "WARNING")

# XGBoost
try:
    import xgboost
    print(f"✅ XGBoost v{xgboost.__version__}")
    report.add_success("Dependencies", "XGBoost installed")
except Exception as e:
    print(f"⚠️  XGBoost: {e}")
    report.add_issue("Dependencies", "XGBoost not available", "WARNING")

# Statsmodels
try:
    import statsmodels
    print(f"✅ Statsmodels v{statsmodels.__version__}")
    report.add_success("Dependencies", "Statsmodels installed")
except Exception as e:
    print(f"⚠️  Statsmodels: {e}")
    report.add_issue("Dependencies", "Statsmodels not available", "WARNING")

# YFinance
try:
    import yfinance
    print(f"✅ YFinance v{yfinance.__version__}")
    report.add_success("Dependencies", "YFinance installed")
except Exception as e:
    print(f"⚠️  YFinance: {e}")
    report.add_issue("Dependencies", "YFinance not available", "WARNING")

# ═══════════════════════════════════════════════════════════════════
# PHASE 3: GOOGLE DRIVE & FILES
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PHASE 3: GOOGLE DRIVE & FILE VERIFICATION")
print("="*80)

# Mount Google Drive
if IN_COLAB:
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        report.add_success("Drive", "Google Drive mounted")
        print("✅ Google Drive mounted")
    except Exception as e:
        report.add_issue("Drive", f"Failed to mount: {e}", "CRITICAL")
        print(f"❌ Failed to mount: {e}")

# Check project structure
PROJECT_ROOT = '/content/drive/MyDrive/Quantum_AI_Cockpit'
MODULES_PATH = f'{PROJECT_ROOT}/backend/modules'

if os.path.exists(PROJECT_ROOT):
    report.add_success("Drive", "Project root found")
    print(f"✅ Project root: {PROJECT_ROOT}")
else:
    report.add_issue("Drive", "Project root not found", "CRITICAL")
    print(f"❌ Project root not found: {PROJECT_ROOT}")
    print(f"   → Create folder in Drive: Quantum_AI_Cockpit/backend/modules")

if os.path.exists(MODULES_PATH):
    report.add_success("Drive", "Modules path found")
    print(f"✅ Modules path: {MODULES_PATH}")
    
    # List modules
    try:
        modules = [f for f in os.listdir(MODULES_PATH) if f.endswith('.py')]
        print(f"\n📁 Found {len(modules)} Python modules:")
        for module in sorted(modules)[:10]:  # Show first 10
            print(f"   • {module}")
    except Exception as e:
        print(f"   ⚠️  Could not list modules: {e}")
else:
    report.add_issue("Drive", "Modules path not found", "CRITICAL")
    print(f"❌ Modules path not found: {MODULES_PATH}")

# ═══════════════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════════════

report.print_report()

print("\n" + "="*80)
print("🎯 NEXT STEPS")
print("="*80)

if report.critical_failures:
    print("\n🚨 CRITICAL ISSUES TO FIX:")
    for i, issue in enumerate(report.critical_failures, 1):
        print(f"\n{i}. {issue['description']}")
        
        if 'Prophet' in issue['description']:
            print("   → Solution: Runtime → Restart runtime, then re-run this cell")
        elif 'Project root not found' in issue['description']:
            print("   → Solution: Create folder in Google Drive:")
            print("      My Drive/Quantum_AI_Cockpit/backend/modules/")
        elif 'Modules path not found' in issue['description']:
            print("   → Solution: Upload your Python modules to:")
            print("      My Drive/Quantum_AI_Cockpit/backend/modules/")
else:
    print("\n✅ No critical issues!")
    print("   Next: Upload your Python modules and run full diagnostic")

print("\n" + "="*80)
print(f"Diagnostic completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

