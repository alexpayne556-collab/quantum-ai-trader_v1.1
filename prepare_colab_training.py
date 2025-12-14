"""
QUANTUM AI TRADER - COLAB PRO PREPARATION SCRIPT
=================================================
Prepares all necessary files for GPU training in Google Colab Pro.

Usage:
    python prepare_colab_training.py

What it does:
    1. Packages current optimized configs
    2. Creates upload bundle for Colab
    3. Generates download script for trained models
    4. Creates integration guide
"""

import json
import os
import shutil
from datetime import datetime

print("="*70)
print("🚀 COLAB PRO TRAINING - PREPARATION SCRIPT")
print("="*70)

# Create colab_bundle directory
BUNDLE_DIR = "colab_training_bundle"
if os.path.exists(BUNDLE_DIR):
    shutil.rmtree(BUNDLE_DIR)
os.makedirs(BUNDLE_DIR)

print(f"\n📦 Creating training bundle in {BUNDLE_DIR}/...")

# 1. Copy current optimized configs
print("\n1️⃣ Packaging current configurations...")
files_to_copy = [
    "optimized_signal_config.py",
    "optimized_exit_config.py",
    "optimized_stack_config.py",
    "COLAB_FULL_STACK_OPTIMIZER.ipynb",
    "DEEP_PATTERN_EVOLUTION_TRAINER.ipynb"
]

for file in files_to_copy:
    if os.path.exists(file):
        shutil.copy(file, os.path.join(BUNDLE_DIR, file))
        print(f"   ✓ {file}")

# 2. Create current_results.json summary
print("\n2️⃣ Creating results summary...")
current_results = {
    "generated_at": datetime.now().isoformat(),
    "version": "1.1",
    "signal_optimization": {
        "tier_s_signals": ["trend"],
        "tier_a_signals": ["rsi_divergence"],
        "tier_b_signals": ["dip_buy", "bounce", "momentum"],
        "disabled_signals": ["nuclear_dip", "vol_squeeze", "consolidation", "uptrend_pullback"],
        "backtest_results": {
            "win_rate": 61.7,
            "avg_return_pct": 0.82,
            "total_trades": 587
        }
    },
    "forecast_optimization": {
        "best_model": "HistGB",
        "horizon_days": 5,
        "threshold": 0.03,
        "decay_start_day": 5,
        "decay_rate": 0.2
    },
    "ai_recommender_optimization": {
        "top_features": [
            "atr_pct", "ema_50", "ema_21", "ema_8", "bb_width",
            "obv", "rsi_21", "rsi_14", "macd", "macd_signal",
            "vol_sma", "trend_long", "atr_14", "rsi_7", "cci"
        ],
        "atr_multiplier": 0.75,
        "model_params": {
            "max_iter": 358,
            "max_depth": 15,
            "learning_rate": 0.011819161482309668,
            "min_samples_leaf": 7,
            "l2_regularization": 0.48339224620158877
        }
    },
    "risk_manager_optimization": {
        "risk_per_trade": 0.015,
        "max_daily_loss": 0.01,
        "max_positions": 3
    },
    "market_regime_optimization": {
        "bull_threshold": 10.0,
        "bear_threshold": -3.0,
        "adx_trend_threshold": 30.0
    }
}

with open(os.path.join(BUNDLE_DIR, "current_results_v1.1.json"), "w") as f:
    json.dump(current_results, f, indent=2)
print("   ✓ current_results_v1.1.json")

# 3. Create Colab upload instructions
print("\n3️⃣ Creating upload instructions...")
upload_instructions = """# 📤 UPLOAD TO GOOGLE COLAB PRO

## Step 1: Upload Bundle
1. Open Google Colab: https://colab.research.google.com
2. Runtime → Change runtime type → T4 GPU or A100 GPU
3. Create new notebook or upload COLAB_PRO_VISUAL_NUMERICAL_TRAINER.ipynb
4. Upload this entire folder to Colab:
   ```python
   from google.colab import files
   uploaded = files.upload()  # Upload all files from colab_training_bundle/
   ```

## Step 2: Run Training
1. Execute all cells in order
2. Training will take ~15-30 minutes on T4 GPU
3. Watch for accuracy improvements and convergence

## Step 3: Download Trained Models
After training completes, download:
- `best_cnn_model.pth` (Visual pattern CNN)
- `best_numerical_model.pkl` (Technical indicator model)
- `feature_scaler.pkl` (Feature normalization)
- `optimized_ensemble_config_v2.json` (Final config)

## Step 4: Integrate Locally
Run: `python integrate_colab_models.py`

---

**Estimated Time:** 1-2 hours total
**Cost:** ~$0.50 (Colab Pro compute units)
**Expected Improvement:** +5-10% accuracy
"""

with open(os.path.join(BUNDLE_DIR, "UPLOAD_INSTRUCTIONS.md"), "w") as f:
    f.write(upload_instructions)
print("   ✓ UPLOAD_INSTRUCTIONS.md")

# 4. Create model integration script
print("\n4️⃣ Creating integration script...")
integration_script = '''"""
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
    print("\\n❌ Error: colab_trained_models/ directory not found!")
    print("   Please download models from Colab and place them in this directory.")
    exit(1)

# Create models directory
Path(MODELS_DIR).mkdir(exist_ok=True)

print("\\n📦 Copying trained models...")

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
    print("\\n📊 V2.0 RESULTS:")
    print(f"   CNN Accuracy: {config.get('cnn_accuracy', 'N/A')}%")
    print(f"   Numerical Accuracy: {config.get('numerical_accuracy', 'N/A')}%")
    print(f"   Ensemble Accuracy: {config.get('ensemble_accuracy', 'N/A')}%")
    print(f"   Win Rate: {config.get('win_rate', 'N/A')}%")
else:
    print("   ⚠️ optimized_ensemble_config_v2.json not found")

print("\\n✅ Integration complete!")
print("\\nNext steps:")
print("   1. Test models: python test_v2_models.py")
print("   2. Run backtest: python backtest_v2.py")
print("   3. Deploy: Update trading_orchestrator.py to use v2 models")
'''

with open(os.path.join(BUNDLE_DIR, "integrate_colab_models.py"), "w") as f:
    f.write(integration_script)
print("   ✓ integrate_colab_models.py")

# 5. Create README
print("\n5️⃣ Creating README...")
readme = """# 🚀 Colab Pro Training Bundle

This bundle contains everything needed for GPU training in Google Colab Pro.

## 📦 Contents

- `optimized_signal_config.py` - Current signal optimization results
- `optimized_exit_config.py` - Current exit strategy results  
- `optimized_stack_config.py` - Current full stack config
- `current_results_v1.1.json` - Current performance metrics
- `UPLOAD_INSTRUCTIONS.md` - Step-by-step Colab guide
- `integrate_colab_models.py` - Model integration script

## 🎯 Workflow

1. **Upload to Colab** → Follow UPLOAD_INSTRUCTIONS.md
2. **Run Training** → Execute notebook cells (~15-30 min)
3. **Download Models** → Get trained models from Colab
4. **Integrate Locally** → Run `python integrate_colab_models.py`
5. **Test & Deploy** → Validate then deploy to production

## 📈 Expected Improvements

| Metric | v1.1 | v2.0 Target |
|--------|------|-------------|
| Accuracy | 60% | 70%+ |
| Win Rate | 61.7% | 70%+ |
| Avg Return | +0.82% | +1.5%+ |

## ⚡ GPU Acceleration

- **T4 GPU:** ~20 minutes training time
- **A100 GPU:** ~10 minutes training time
- **CPU (local):** ~4 hours (not recommended)

## 🔗 Resources

- Colab Pro: https://colab.research.google.com/signup
- PyTorch Docs: https://pytorch.org/docs/
- Research Papers: See PERPLEXITY_PRO_RESEARCH.md

---

**Created:** {datetime}
**Version:** v1.1 → v2.0
**Status:** Ready for training 🚀
""".format(datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

with open(os.path.join(BUNDLE_DIR, "README.md"), "w") as f:
    f.write(readme)
print("   ✓ README.md")

print("\n" + "="*70)
print("✅ PREPARATION COMPLETE!")
print("="*70)
print(f"\n📁 Bundle location: {os.path.abspath(BUNDLE_DIR)}/")
print(f"📦 Files included: {len(os.listdir(BUNDLE_DIR))}")
print("\n🚀 Next step: Upload to Google Colab Pro and start training!")
print("   See: colab_training_bundle/UPLOAD_INSTRUCTIONS.md")
