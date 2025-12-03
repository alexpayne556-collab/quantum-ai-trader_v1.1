# =============================================================================
# VERIFY FILES IN GOOGLE DRIVE
# =============================================================================
# Run this in Colab AFTER uploading files to check they're in the right place

from google.colab import drive
drive.mount('/content/drive')

import os

# Check if files exist
files_to_check = [
    '/content/drive/MyDrive/QuantumAI/backend/modules/professional_signal_coordinator.py',
    '/content/drive/MyDrive/QuantumAI/backend/modules/api_integrations.py',
    '/content/drive/MyDrive/QuantumAI/backend/modules/daily_scanner.py',
    '/content/drive/MyDrive/QuantumAI/backend/modules/ai_forecast_pro.py',
    '/content/drive/MyDrive/QuantumAI/backend/modules/institutional_flow_pro.py',
    '/content/drive/MyDrive/QuantumAI/backend/modules/pattern_engine_pro.py',
    '/content/drive/MyDrive/QuantumAI/backend/modules/sentiment_pro.py',
    '/content/drive/MyDrive/QuantumAI/backend/modules/scanner_pro.py',
    '/content/drive/MyDrive/QuantumAI/backend/modules/risk_manager_pro.py',
    '/content/drive/MyDrive/QuantumAI/backend/modules/ai_recommender_pro.py'
]

print("="*70)
print("🔍 CHECKING FILES IN GOOGLE DRIVE")
print("="*70 + "\n")

missing = []
found = []

for filepath in files_to_check:
    filename = os.path.basename(filepath)
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"✅ {filename:40s} ({size:,} bytes)")
        found.append(filename)
    else:
        print(f"❌ {filename:40s} MISSING!")
        missing.append(filename)

print("\n" + "="*70)
print(f"SUMMARY: {len(found)}/10 files found")
print("="*70)

if missing:
    print(f"\n⚠️  MISSING FILES ({len(missing)}):")
    for f in missing:
        print(f"   - {f}")
    print("\nPlease upload the missing files to:")
    print("/content/drive/MyDrive/QuantumAI/backend/modules/")
else:
    print("\n🎉 ALL FILES PRESENT! Ready to run scanner!")
    
print()

