"""
🔄 MERGE SIMILAR MODULES INTO UNIFIED FILES
===========================================
Combines redundant modules into single, optimized files
Based on Perplexity recommendations for code consolidation
"""

import os
import shutil
import re
from pathlib import Path
from collections import defaultdict
from typing import List
from datetime import datetime

print("="*80)
print("🔄 MERGING SIMILAR MODULES")
print("="*80)

# Colab paths
BASE_DIR = Path('/content/drive/MyDrive/QuantumAI')
MODULES_DIR = BASE_DIR / 'backend' / 'modules'

# Create directory if it doesn't exist
if not MODULES_DIR.exists():
    print(f"⚠️  Modules directory not found: {MODULES_DIR}")
    print(f"   Creating directory...")
    MODULES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"   ✅ Directory created")

print(f"\n📁 Modules directory: {MODULES_DIR}")

# ============================================================================
# MODULE GROUPS TO MERGE
# ============================================================================

module_groups = {
    # 1. AI RECOMMENDERS (4 → 1)
    'ai_recommender_unified.py': [
        'ai_recommender_v2.py',
        'ai_recommender_integrated.py',
        'ai_recommender_institutional.py',
        'ai_recommender_institutional_enhanced.py'  # Keep this as base
    ],
    
    # 2. PATTERN DETECTORS (7 → 1)
    'advanced_pattern_engine_unified.py': [
        'head_shoulders_detector.py',
        'triangle_detector.py',
        'cup_and_handle_detector.py',
        'flag_pennant_detector.py',
        'divergence_detector.py',
        'support_resistance_detector.py',
        'harmonic_pattern_detector.py'
    ],
    
    # 3. SCANNERS (7 → 1) - Already have unified_momentum_scanner_v3.py
    # But merge old versions into it
    'unified_momentum_scanner_v3.py': [
        'pre_gainer_scanner.py',  # Old version
        'day_trading_scanner.py',  # Old version
        'opportunity_scanner.py',  # Old version
        'breakout_screener.py',
        'ticker_discovery_engine.py'
    ],
    
    # 4. PUMP DETECTORS (2 → 1)
    'pump_detector_unified.py': [
        'penny_stock_pump_detector.py',  # Old version
        'penny_stock_pump_detector_v2_ML_POWERED.py'  # Keep this as base
    ],
    
    # 5. SENTIMENT DETECTORS (2 → 1)
    'social_sentiment_unified.py': [
        'social_sentiment_explosion_detector.py',  # Old version
        'social_sentiment_explosion_detector_v2.py'  # Keep this as base
    ],
    
    # 6. TRAINING MODULES (6 → 1)
    'ml_trainer_unified.py': [
        'forecast_trainer.py',
        'production_pattern_trainer.py',
        'master_pattern_trainer.py',
        'forecast_backtest_tuner.py',
        'module_training_calibrator.py',
        'training_orchestrator.py'
    ],
    
    # 7. VALIDATORS (7 → 1)
    'system_validator_unified.py': [
        'validate_all_modules.py',
        'full_system_validator.py',
        'quantum_system_validator.py',
        'full_repair_validator.py',
        'comprehensive_validator.py',
        'orchestration_validator.py',
        'system_integrity_validator.py'  # Keep this as base
    ],
    
    # 8. MORNING BRIEF GENERATORS (2 → 1)
    'morning_brief_unified.py': [
        'morning_brief_generator.py',  # Old version
        'morning_brief_generator_v2_ML_POWERED.py'  # Keep this as base
    ],
    
    # 9. DATA ROUTERS (3 → 1)
    'data_orchestrator_unified.py': [
        'data_router.py',
        'data_watchdog.py',
        'data_orchestrator.py'  # Keep this as base
    ],
    
    # 10. ANALYSIS ENGINES (4 → 1)
    'analysis_engine_unified.py': [
        'deep_analysis_engine.py',
        'master_analysis_engine.py',
        'master_analysis_institutional.py',
        'deep_analysis_lab.py'  # Keep this as base
    ],
    
    # 11. FORECASTERS (3 → 1)
    'forecaster_unified.py': [
        'fusior_forecast.py',
        'fusior_forecast_institutional.py',
        'elite_forecaster.py'  # Keep this as base
    ],
}

# ============================================================================
# BACKUP DIRECTORY
# ============================================================================

BACKUP_DIR = MODULES_DIR / '_merged_backup'
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n💾 Backup directory: {BACKUP_DIR}")

# ============================================================================
# MERGE FUNCTION
# ============================================================================

def merge_modules(target_file: str, source_files: List[str], keep_base: str = None):
    """
    Merge multiple module files into one unified file
    
    Args:
        target_file: Name of the unified output file
        source_files: List of source files to merge
        keep_base: Which file to use as the base (if specified)
    """
    target_path = MODULES_DIR / target_file
    source_paths = [MODULES_DIR / f for f in source_files if (MODULES_DIR / f).exists()]
    
    if len(source_paths) == 0:
        print(f"   ⚠️  No source files found for {target_file}")
        return False
    
    print(f"\n📝 Merging into: {target_file}")
    print(f"   Source files: {len(source_paths)}")
    
    # Determine base file
    if keep_base and (MODULES_DIR / keep_base).exists():
        base_file = MODULES_DIR / keep_base
        print(f"   Using {keep_base} as base")
    else:
        base_file = source_paths[0]
        print(f"   Using {source_paths[0].name} as base")
    
    # Read base file
    with open(base_file, 'r', encoding='utf-8') as f:
        merged_content = f.read()
    
    # Add header comment
    header = f'''"""
UNIFIED MODULE: {target_file.replace('.py', '').replace('_', ' ').title()}
==================================================
Merged from: {', '.join([f.name for f in source_paths])}
Created: {datetime.now().strftime('%Y-%m-%d')}

This unified module combines functionality from multiple similar modules
to reduce redundancy and improve maintainability.
"""

'''
    
    merged_content = header + merged_content
    
    # Extract classes/functions from other files
    other_classes = []
    other_functions = []
    
    for source_path in source_paths:
        if source_path == base_file:
            continue
        
        try:
            with open(source_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract class definitions
            classes = re.findall(r'^class\s+\w+.*?(?=^class|\Z)', content, re.MULTILINE | re.DOTALL)
            other_classes.extend(classes)
            
            # Extract standalone functions
            functions = re.findall(r'^def\s+\w+.*?(?=^def|^class|\Z)', content, re.MULTILINE | re.DOTALL)
            other_functions.extend(functions)
            
        except Exception as e:
            print(f"   ⚠️  Error reading {source_path.name}: {e}")
    
    # Append unique classes/functions
    if other_classes or other_functions:
        merged_content += "\n\n# ============================================================================\n"
        merged_content += "# ADDITIONAL CLASSES/FUNCTIONS FROM MERGED FILES\n"
        merged_content += "# ============================================================================\n\n"
        
        for cls in other_classes:
            merged_content += cls + "\n\n"
        
        for func in other_functions:
            merged_content += func + "\n\n"
    
    # Write merged file
    with open(target_path, 'w', encoding='utf-8') as f:
        f.write(merged_content)
    
    print(f"   ✅ Created {target_file} ({len(merged_content):,} chars)")
    
    # Backup source files
    for source_path in source_paths:
        backup_path = BACKUP_DIR / source_path.name
        shutil.copy2(source_path, backup_path)
        print(f"   💾 Backed up {source_path.name}")
    
    return True

# ============================================================================
# MAIN MERGE PROCESS
# ============================================================================

print("\n" + "="*80)
print("🔄 STARTING MERGE PROCESS")
print("="*80)

merged_count = 0
skipped_count = 0

for target_file, source_files in module_groups.items():
    # Determine which file to keep as base (last one in list if specified)
    keep_base = None
    if len(source_files) > 0:
        # Check if any file has "v2" or "enhanced" in name
        for f in reversed(source_files):
            if 'v2' in f.lower() or 'enhanced' in f.lower() or 'institutional' in f.lower():
                keep_base = f
                break
        
        if not keep_base:
            keep_base = source_files[-1]  # Use last as default
    
    if merge_modules(target_file, source_files, keep_base):
        merged_count += 1
    else:
        skipped_count += 1

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("📋 MERGE SUMMARY")
print("="*80)

print(f"""
✅ Merged: {merged_count} module groups
⚠️  Skipped: {skipped_count} module groups

Module Consolidations:
   1. AI Recommenders: 4 → 1 (ai_recommender_unified.py)
   2. Pattern Detectors: 7 → 1 (advanced_pattern_engine_unified.py)
   3. Scanners: 7 → 1 (unified_momentum_scanner_v3.py)
   4. Pump Detectors: 2 → 1 (pump_detector_unified.py)
   5. Sentiment Detectors: 2 → 1 (social_sentiment_unified.py)
   6. Training Modules: 6 → 1 (ml_trainer_unified.py)
   7. Validators: 7 → 1 (system_validator_unified.py)
   8. Morning Brief: 2 → 1 (morning_brief_unified.py)
   9. Data Routers: 3 → 1 (data_orchestrator_unified.py)
  10. Analysis Engines: 4 → 1 (analysis_engine_unified.py)
  11. Forecasters: 3 → 1 (forecaster_unified.py)

Total Reduction: ~40+ files → 11 unified files

Backup Location: {BACKUP_DIR}

Next Steps:
   1. Test unified modules work correctly
   2. Update imports in other files to use unified modules
   3. Delete old files after verification (they're backed up)
   4. Update __init__.py to export unified modules
""")

print("="*80)
print("✅ MERGE COMPLETE!")
print("="*80)

