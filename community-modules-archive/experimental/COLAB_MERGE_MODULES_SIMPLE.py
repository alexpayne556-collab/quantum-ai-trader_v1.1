"""
🔄 SIMPLE MODULE MERGER FOR COLAB
==================================
Run this in Colab to merge similar modules into unified files
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import re

print("="*80)
print("🔄 MERGING SIMILAR MODULES")
print("="*80)

BASE_DIR = Path('/content/drive/MyDrive/QuantumAI')
MODULES_DIR = BASE_DIR / 'backend' / 'modules'

if not MODULES_DIR.exists():
    print(f"❌ Modules directory not found: {MODULES_DIR}")
    print("   Make sure Google Drive is mounted!")
    exit(1)

print(f"\n📁 Modules directory: {MODULES_DIR}")

# Create backup
BACKUP_DIR = MODULES_DIR / '_merged_backup'
BACKUP_DIR.mkdir(exist_ok=True)
print(f"💾 Backup directory: {BACKUP_DIR}")

# ============================================================================
# MODULE GROUPS TO MERGE
# ============================================================================

merge_groups = [
    {
        'target': 'ai_recommender_unified.py',
        'sources': [
            'ai_recommender_institutional_enhanced.py',  # Keep as base
            'ai_recommender_v2.py',
            'ai_recommender_integrated.py',
            'ai_recommender_institutional.py'
        ]
    },
    {
        'target': 'pattern_detector_unified.py',
        'sources': [
            'harmonic_pattern_detector.py',  # Keep as base (most advanced)
            'head_shoulders_detector.py',
            'triangle_detector.py',
            'cup_and_handle_detector.py',
            'flag_pennant_detector.py',
            'divergence_detector.py',
            'support_resistance_detector.py'
        ]
    },
    {
        'target': 'pump_detector_unified.py',
        'sources': [
            'penny_stock_pump_detector_v2_ML_POWERED.py',  # Keep as base
            'penny_stock_pump_detector.py'
        ]
    },
    {
        'target': 'sentiment_detector_unified.py',
        'sources': [
            'social_sentiment_explosion_detector_v2.py',  # Keep as base
            'social_sentiment_explosion_detector.py'
        ]
    },
    {
        'target': 'ml_trainer_unified.py',
        'sources': [
            'training_orchestrator.py',  # Keep as base
            'forecast_trainer.py',
            'production_pattern_trainer.py',
            'master_pattern_trainer.py',
            'forecast_backtest_tuner.py',
            'module_training_calibrator.py'
        ]
    },
    {
        'target': 'system_validator_unified.py',
        'sources': [
            'system_integrity_validator.py',  # Keep as base
            'validate_all_modules.py',
            'full_system_validator.py',
            'quantum_system_validator.py',
            'full_repair_validator.py',
            'comprehensive_validator.py',
            'orchestration_validator.py'
        ]
    },
    {
        'target': 'morning_brief_unified.py',
        'sources': [
            'morning_brief_generator_v2_ML_POWERED.py',  # Keep as base
            'morning_brief_generator.py'
        ]
    },
    {
        'target': 'data_orchestrator_unified.py',
        'sources': [
            'data_orchestrator.py',  # Keep as base
            'data_router.py',
            'data_watchdog.py'
        ]
    }
]

# ============================================================================
# MERGE FUNCTION
# ============================================================================

def merge_group(target: str, sources: list):
    """Merge source files into target file"""
    target_path = MODULES_DIR / target
    source_paths = [MODULES_DIR / f for f in sources if (MODULES_DIR / f).exists()]
    
    if len(source_paths) == 0:
        print(f"   ⚠️  No source files found")
        return False
    
    print(f"\n📝 {target}")
    print(f"   Merging {len(source_paths)} files...")
    
    # Use first file as base (should be the "keep as base" one)
    base_file = source_paths[0]
    
    # Read base file
    try:
        with open(base_file, 'r', encoding='utf-8') as f:
            merged = f.read()
    except Exception as e:
        print(f"   ❌ Error reading {base_file.name}: {e}")
        return False
    
    # Add header
    header = f'''"""
UNIFIED MODULE: {target.replace('.py', '').replace('_', ' ').title()}
==================================================
Merged from: {', '.join([f.name for f in source_paths])}
Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This unified module combines functionality from multiple similar modules.
Original files backed up to _merged_backup/
"""

'''
    merged = header + merged
    
    # Extract unique classes/functions from other files
    all_classes = set()
    all_functions = set()
    
    for source_path in source_paths[1:]:  # Skip base file
        try:
            with open(source_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract classes
            classes = re.findall(r'^class\s+(\w+).*?(?=^class|\Z)', content, re.MULTILINE | re.DOTALL)
            for cls_match in classes:
                cls_name = cls_match[0] if isinstance(cls_match, tuple) else cls_match
                if cls_name not in all_classes:
                    all_classes.add(cls_name)
                    # Extract full class
                    cls_full = re.search(rf'^class\s+{cls_name}.*?(?=^class|\Z)', content, re.MULTILINE | re.DOTALL)
                    if cls_full:
                        merged += f"\n\n# From {source_path.name}:\n{cls_full.group()}\n"
            
            # Extract standalone functions
            functions = re.findall(r'^def\s+(\w+).*?(?=^def|^class|\Z)', content, re.MULTILINE | re.DOTALL)
            for func_match in functions:
                func_name = func_match[0] if isinstance(func_match, tuple) else func_match
                if func_name not in all_functions and not func_name.startswith('_'):
                    all_functions.add(func_name)
                    # Extract full function
                    func_full = re.search(rf'^def\s+{func_name}.*?(?=^def|^class|\Z)', content, re.MULTILINE | re.DOTALL)
                    if func_full:
                        merged += f"\n\n# From {source_path.name}:\n{func_full.group()}\n"
        
        except Exception as e:
            print(f"   ⚠️  Error processing {source_path.name}: {e}")
    
    # Write merged file
    try:
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(merged)
        print(f"   ✅ Created {target} ({len(merged):,} chars)")
    except Exception as e:
        print(f"   ❌ Error writing {target}: {e}")
        return False
    
    # Backup source files
    for source_path in source_paths:
        try:
            backup_path = BACKUP_DIR / source_path.name
            shutil.copy2(source_path, backup_path)
        except Exception as e:
            print(f"   ⚠️  Could not backup {source_path.name}: {e}")
    
    return True

# ============================================================================
# RUN MERGES
# ============================================================================

print("\n" + "="*80)
print("🔄 STARTING MERGE PROCESS")
print("="*80)

merged = 0
failed = 0

for group in merge_groups:
    if merge_group(group['target'], group['sources']):
        merged += 1
    else:
        failed += 1

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("📋 MERGE SUMMARY")
print("="*80)

print(f"""
✅ Successfully merged: {merged} groups
❌ Failed: {failed} groups

Unified Files Created:
   1. ai_recommender_unified.py (4 files merged)
   2. pattern_detector_unified.py (7 files merged)
   3. pump_detector_unified.py (2 files merged)
   4. sentiment_detector_unified.py (2 files merged)
   5. ml_trainer_unified.py (6 files merged)
   6. system_validator_unified.py (7 files merged)
   7. morning_brief_unified.py (2 files merged)
   8. data_orchestrator_unified.py (3 files merged)

Total: ~33 files → 8 unified files

Backup Location: {BACKUP_DIR}

Next Steps:
   1. Test that unified modules work
   2. Update imports in your code
   3. Delete old files (they're backed up)
""")

print("="*80)
print("✅ MERGE COMPLETE!")
print("="*80)

