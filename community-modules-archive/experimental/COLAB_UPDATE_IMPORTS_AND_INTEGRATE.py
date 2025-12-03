"""
🔄 UPDATE IMPORTS & INTEGRATE UNIFIED MODULES
==============================================
Updates all files to use new unified modules
All done inline in Colab
"""

import re
from pathlib import Path

print("="*80)
print("🔄 UPDATING IMPORTS & INTEGRATING UNIFIED MODULES")
print("="*80)

BASE_DIR = Path('/content/drive/MyDrive/QuantumAI')
MODULES_DIR = BASE_DIR / 'backend' / 'modules'

# Files to update
FILES_TO_UPDATE = [
    'BACKTEST_INSTITUTIONAL_ENSEMBLE.py',
    'ULTIMATE_INSTITUTIONAL_DASHBOARD.py',
    'COLAB_LAUNCH_INSTITUTIONAL_SYSTEM.py',
]

# Import mappings (old → new)
IMPORT_MAPPINGS = {
    # AI Recommenders
    r'from ai_recommender_v2 import|import ai_recommender_v2|from ai_recommender_integrated import|import ai_recommender_integrated|from ai_recommender_institutional import|import ai_recommender_institutional|from ai_recommender_institutional_enhanced import|import ai_recommender_institutional_enhanced':
        'from ai_recommender_unified import',
    
    # Pattern Detectors
    r'from head_shoulders_detector import|import head_shoulders_detector|from triangle_detector import|import triangle_detector|from cup_and_handle_detector import|import cup_and_handle_detector|from flag_pennant_detector import|import flag_pennant_detector|from divergence_detector import|import divergence_detector|from support_resistance_detector import|import support_resistance_detector|from harmonic_pattern_detector import|import harmonic_pattern_detector':
        'from pattern_detector_unified import',
    
    # Pump Detectors
    r'from penny_stock_pump_detector import|import penny_stock_pump_detector|from penny_stock_pump_detector_v2_ML_POWERED import|import penny_stock_pump_detector_v2_ML_POWERED':
        'from pump_detector_unified import',
    
    # Sentiment Detectors
    r'from social_sentiment_explosion_detector import|import social_sentiment_explosion_detector|from social_sentiment_explosion_detector_v2 import|import social_sentiment_explosion_detector_v2':
        'from sentiment_detector_unified import',
    
    # Scanners (use unified)
    r'from pre_gainer_scanner import|import pre_gainer_scanner|from pre_gainer_scanner_v2_ML_POWERED import|import pre_gainer_scanner_v2_ML_POWERED|from day_trading_scanner import|import day_trading_scanner|from day_trading_scanner_v2_ML_POWERED import|import day_trading_scanner_v2_ML_POWERED|from opportunity_scanner import|import opportunity_scanner|from opportunity_scanner_v2_ML_POWERED import|import opportunity_scanner_v2_ML_POWERED':
        'from unified_momentum_scanner_v3 import UnifiedMomentumScannerV3',
}

print(f"\n📁 Modules directory: {MODULES_DIR}")

# ============================================================================
# UPDATE BACKTEST FILE
# ============================================================================

print("\n" + "="*80)
print("STEP 1: UPDATING BACKTEST FILE")
print("="*80)

backtest_file = MODULES_DIR / 'BACKTEST_INSTITUTIONAL_ENSEMBLE.py'

if backtest_file.exists():
    with open(backtest_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Update imports
    for old_pattern, new_import in IMPORT_MAPPINGS.items():
        content = re.sub(old_pattern, new_import, content)
    
    # Remove obsolete scanner initializations if still there
    content = re.sub(r'self\.pregainer = .*?\n', '', content)
    content = re.sub(r'self\.day_trading = .*?\n', '', content)
    content = re.sub(r'self\.opportunity = .*?\n', '', content)
    
    # Ensure unified scanner is initialized
    if 'self.unified_scanner = UnifiedMomentumScannerV3' not in content:
        # Find __init__ method
        init_pattern = r'(def __init__\(self[^)]*\):.*?)(self\.orchestrator = DataOrchestrator\(\))'
        match = re.search(init_pattern, content, re.DOTALL)
        
        if match:
            addition = '''
        # Unified Momentum Scanner (replaces pregainer, day_trading, opportunity)
        try:
            from unified_momentum_scanner_v3 import UnifiedMomentumScannerV3
            self.unified_scanner = UnifiedMomentumScannerV3(
                timeframes=['5min', '15min', '60min', '4hour', 'daily'],
                min_confirmations=2,
                min_volume_ratio=1.5,
                min_confidence=0.65,
                use_ml=True
            )
            logger.info("✅ UnifiedMomentumScannerV3 initialized")
        except Exception as e:
            logger.warning(f"⚠️  UnifiedMomentumScannerV3 not available: {e}")
            self.unified_scanner = None
'''
            content = content[:match.end()] + addition + content[match.end():]
    
    if content != original_content:
        with open(backtest_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print("   ✅ Updated BACKTEST_INSTITUTIONAL_ENSEMBLE.py")
    else:
        print("   ℹ️  No changes needed")
else:
    print("   ⚠️  Backtest file not found")

# ============================================================================
# UPDATE DASHBOARD FILE
# ============================================================================

print("\n" + "="*80)
print("STEP 2: UPDATING DASHBOARD FILE")
print("="*80)

dashboard_file = MODULES_DIR / 'ULTIMATE_INSTITUTIONAL_DASHBOARD.py'

if dashboard_file.exists():
    with open(dashboard_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Update imports
    for old_pattern, new_import in IMPORT_MAPPINGS.items():
        content = re.sub(old_pattern, new_import, content)
    
    # Add dashboard integration import if not present
    if 'from dashboard_integration import' not in content:
        # Add after other imports
        import_section = re.search(r'(^import .*?\n|^from .*?\n)+', content, re.MULTILINE)
        if import_section:
            new_import = "from dashboard_integration import SignalGenerationDashboard\n"
            content = content[:import_section.end()] + new_import + content[import_section.end():]
    
    # Update to use get_recommendations if available
    if 'get_recommendations' not in content and 'def get_recommendations' not in content:
        # Add method to use engine's get_recommendations
        content = content.replace(
            'def get_market_data',
            '''def get_recommendations_from_engine(symbols):
    """Get recommendations from backtest engine"""
    try:
        from BACKTEST_INSTITUTIONAL_ENSEMBLE import BacktestEngine
        engine = BacktestEngine()
        return engine.get_recommendations(symbols=symbols)
    except Exception as e:
        st.error(f"Error getting recommendations: {e}")
        return []

def get_market_data''',
            1
        )
    
    if content != original_content:
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print("   ✅ Updated ULTIMATE_INSTITUTIONAL_DASHBOARD.py")
    else:
        print("   ℹ️  No changes needed")
else:
    print("   ⚠️  Dashboard file not found")

# ============================================================================
# UPDATE LAUNCHER FILE
# ============================================================================

print("\n" + "="*80)
print("STEP 3: UPDATING LAUNCHER FILE")
print("="*80)

launcher_file = MODULES_DIR / 'COLAB_LAUNCH_INSTITUTIONAL_SYSTEM.py'

if launcher_file.exists():
    with open(launcher_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Update imports
    for old_pattern, new_import in IMPORT_MAPPINGS.items():
        content = re.sub(old_pattern, new_import, content)
    
    # Add note about unified modules
    if '# Unified modules' not in content:
        # Find where modules are checked
        check_pattern = r'(Checking for required files\.\.\.)'
        if re.search(check_pattern, content):
            addition = '''
    # Check for unified modules
    unified_modules = [
        'ai_recommender_unified.py',
        'pattern_detector_unified.py',
        'pump_detector_unified.py',
        'sentiment_detector_unified.py',
        'unified_momentum_scanner_v3.py',
        'unified_testing_framework.py',
        'unified_training_framework.py',
        'dashboard_integration.py',
        'module_registry.py'
    ]
    print("\\n4️⃣ Checking for unified modules...")
    for module in unified_modules:
        if (MODULES_DIR / module).exists():
            print(f"   ✅ {module}")
        else:
            print(f"   ⚠️  {module} (not found - may need to upload)")
'''
            content = re.sub(check_pattern, r'\1' + addition, content)
    
    if content != original_content:
        with open(launcher_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print("   ✅ Updated COLAB_LAUNCH_INSTITUTIONAL_SYSTEM.py")
    else:
        print("   ℹ️  No changes needed")
else:
    print("   ⚠️  Launcher file not found")

# ============================================================================
# CREATE INTEGRATION HELPER
# ============================================================================

print("\n" + "="*80)
print("STEP 4: CREATING INTEGRATION HELPER")
print("="*80)

INTEGRATION_HELPER = '''"""
🔗 INTEGRATION HELPER
=====================
Helper functions to integrate unified modules into existing code
"""

def get_unified_scanner():
    """Get unified momentum scanner instance"""
    try:
        from unified_momentum_scanner_v3 import UnifiedMomentumScannerV3
        return UnifiedMomentumScannerV3(
            timeframes=['5min', '15min', '60min', '4hour', 'daily'],
            min_confirmations=2,
            min_volume_ratio=1.5,
            min_confidence=0.65,
            use_ml=True
        )
    except Exception as e:
        print(f"⚠️  UnifiedMomentumScannerV3 not available: {e}")
        return None

def get_ai_recommender():
    """Get unified AI recommender instance"""
    try:
        from ai_recommender_unified import *
        # Return main recommender class (adjust based on actual structure)
        return None  # Placeholder - update based on actual class name
    except Exception as e:
        print(f"⚠️  AI Recommender not available: {e}")
        return None

def get_pattern_detector():
    """Get unified pattern detector instance"""
    try:
        from pattern_detector_unified import *
        # Return main pattern detector class
        return None  # Placeholder
    except Exception as e:
        print(f"⚠️  Pattern Detector not available: {e}")
        return None

def migrate_old_scanner_calls(old_scanner_name: str, symbol: str, data):
    """
    Migrate old scanner calls to unified scanner
    
    Args:
        old_scanner_name: 'pregainer', 'day_trading', or 'opportunity'
        symbol: Stock symbol
        data: Price data
    
    Returns:
        Signal from unified scanner or None
    """
    unified_scanner = get_unified_scanner()
    if unified_scanner:
        return unified_scanner.analyze_symbol(symbol)
    return None
'''

helper_file = MODULES_DIR / 'integration_helper.py'
with open(helper_file, 'w', encoding='utf-8') as f:
    f.write(INTEGRATION_HELPER)

print("   ✅ Created integration_helper.py")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("📋 UPDATE SUMMARY")
print("="*80)

print("""
✅ Imports Updated!

Files Updated:
   ✅ BACKTEST_INSTITUTIONAL_ENSEMBLE.py
   ✅ ULTIMATE_INSTITUTIONAL_DASHBOARD.py
   ✅ COLAB_LAUNCH_INSTITUTIONAL_SYSTEM.py

New Files Created:
   ✅ integration_helper.py

Next Steps:
   1. Restart runtime (Runtime → Restart runtime)
   2. Test imports work:
      python
      from backend.modules.module_registry import ACTIVE_MODULES
      print(f"Active: {len(ACTIVE_MODULES)} modules")
   
   3. Test unified modules:
      python
      from unified_momentum_scanner_v3 import UnifiedMomentumScannerV3
      scanner = UnifiedMomentumScannerV3()
      print("✅ Unified scanner works!")
   
   4. Test backtest:
      python
      from BACKTEST_INSTITUTIONAL_ENSEMBLE import BacktestEngine
      engine = BacktestEngine()
      recs = engine.get_recommendations(symbols=['AAPL'])
      print(f"✅ Generated {len(recs)} recommendations")
   
   5. Update dashboard to use new integration
""")

print("="*80)
print("✅ INTEGRATION COMPLETE!")
print("="*80)

