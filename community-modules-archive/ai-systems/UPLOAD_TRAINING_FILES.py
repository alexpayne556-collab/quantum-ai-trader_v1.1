"""
Upload Training Files to Google Drive
Run this from your LOCAL machine (not Colab)
"""

import os
import shutil
from pathlib import Path

# Files to upload
TRAINING_FILES = [
    'TRAIN_COMPLETE_AUTO_TUNING.py',          # Main auto-tuning script
    'TRAIN_PATTERNS_V2_INSTITUTIONAL.py',     # Institutional formulas
    'AUTO_ADJUSTMENT_SYSTEM.py',              # Auto-adjustment logic
    'PERPLEXITY_INSIGHTS_APPLIED.md',         # Research summary
    'START_TRAINING_NOW.md',                  # Instructions
    'ASK_PERPLEXITY_PATTERN_TRAINING.md',     # Perplexity prompts
]

# Optional support files
OPTIONAL_FILES = [
    'AINVEST_PATTERN_TRAINING.py',            # Original training script
    'COLAB_TRAINING_V4_OPTIMIZED.py',         # Previous version
    'TOMORROW_MORNING_PLAN.md',               # Day 2 plan
    'UI_INFLUENCES_QUICK_REF.md',             # Dashboard design
]

def list_files_to_upload():
    """Show what files are ready to upload"""
    print("="*80)
    print("📦 FILES READY TO UPLOAD TO GOOGLE DRIVE")
    print("="*80)
    
    print("\n🔥 ESSENTIAL FILES (Upload these first):\n")
    for idx, file in enumerate(TRAINING_FILES, 1):
        exists = "✅" if os.path.exists(file) else "❌ MISSING"
        size = os.path.getsize(file) / 1024 if os.path.exists(file) else 0
        print(f"  {idx}. {file:45s} {exists} ({size:.1f} KB)")
    
    print("\n📚 OPTIONAL FILES (Upload if you want reference docs):\n")
    for idx, file in enumerate(OPTIONAL_FILES, 1):
        exists = "✅" if os.path.exists(file) else "❌"
        size = os.path.getsize(file) / 1024 if os.path.exists(file) else 0
        print(f"  {idx}. {file:45s} {exists} ({size:.1f} KB)")
    
    print("\n" + "="*80)
    print("📍 UPLOAD DESTINATION:")
    print("   Google Drive → MyDrive → Quantum_AI_Cockpit")
    print("="*80)

if __name__ == "__main__":
    list_files_to_upload()
    
    print("\n🚀 NEXT STEPS:")
    print("\n1. Open Google Drive in your browser")
    print("   → drive.google.com")
    print("\n2. Navigate to (or create):")
    print("   → MyDrive/Quantum_AI_Cockpit/")
    print("\n3. Upload the ✅ ESSENTIAL FILES listed above")
    print("\n4. Open Google Colab")
    print("   → colab.research.google.com")
    print("\n5. Copy-paste the training cells (see below)")
    print("\n" + "="*80)

