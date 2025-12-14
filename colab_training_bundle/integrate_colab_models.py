"""
Model Integration Script
Integrates trained models from Colab Pro into local system.
"""

import torch
import joblib
import json
import shutil
from pathlib import Path

print("="*70)
print("🔄 INTEGRATING COLAB PRO TRAINED MODELS")
print("="*70)

# Paths
COLAB_MODELS_DIR = "colab_trained_models"
MODELS_DIR = "models"

if not Path(COLAB_MODELS_DIR).exists():
    print("\n❌ Error: colab_trained_models/ directory not found!")
    print("   Please download models from Colab and place them in this directory.")
    exit(1)

# Create models directory
Path(MODELS_DIR).mkdir(exist_ok=True)

print("\n📦 Copying trained models...")

# 1. CNN model
if Path(f"{COLAB_MODELS_DIR}/best_cnn_model.pth").exists():
    shutil.copy(
        f"{COLAB_MODELS_DIR}/best_cnn_model.pth",
        f"{MODELS_DIR}/pattern_cnn_v2.pth"
    )
    print("   ✓ pattern_cnn_v2.pth")
else:
    print("   ⚠️ best_cnn_model.pth not found")

# 2. Numerical model
if Path(f"{COLAB_MODELS_DIR}/best_numerical_model.pkl").exists():
    shutil.copy(
        f"{COLAB_MODELS_DIR}/best_numerical_model.pkl",
        f"{MODELS_DIR}/forecast_hist_gb_v2.pkl"
    )
    print("   ✓ forecast_hist_gb_v2.pkl")
else:
    print("   ⚠️ best_numerical_model.pkl not found")

# 3. Scaler
if Path(f"{COLAB_MODELS_DIR}/feature_scaler.pkl").exists():
    shutil.copy(
        f"{COLAB_MODELS_DIR}/feature_scaler.pkl",
        f"{MODELS_DIR}/feature_scaler_v2.pkl"
    )
    print("   ✓ feature_scaler_v2.pkl")
else:
    print("   ⚠️ feature_scaler.pkl not found")

# 4. Config
if Path(f"{COLAB_MODELS_DIR}/optimized_ensemble_config_v2.json").exists():
    shutil.copy(
        f"{COLAB_MODELS_DIR}/optimized_ensemble_config_v2.json",
        "optimized_ensemble_config_v2.json"
    )
    
    # Load and display config
    with open("optimized_ensemble_config_v2.json", "r") as f:
        config = json.load(f)
    
    print("   ✓ optimized_ensemble_config_v2.json")
    print("\n📊 V2.0 RESULTS:")
    print(f"   CNN Accuracy: {config.get('cnn_accuracy', 'N/A')}%")
    print(f"   Numerical Accuracy: {config.get('numerical_accuracy', 'N/A')}%")
    print(f"   Ensemble Accuracy: {config.get('ensemble_accuracy', 'N/A')}%")
    print(f"   Win Rate: {config.get('win_rate', 'N/A')}%")
else:
    print("   ⚠️ optimized_ensemble_config_v2.json not found")

print("\n✅ Integration complete!")
print("\nNext steps:")
print("   1. Test models: python test_v2_models.py")
print("   2. Run backtest: python backtest_v2.py")
print("   3. Deploy: Update trading_orchestrator.py to use v2 models")
