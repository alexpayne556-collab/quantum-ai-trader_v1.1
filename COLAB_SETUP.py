# ============================================================
# 🚀 COLAB SETUP - Run this FIRST in Colab!
# ============================================================
# This clones your repo and sets up the environment

# Step 1: Clone your repo (or pull latest)
import os

REPO_URL = "https://github.com/alexpayne556-collab/quantum-ai-trader_v1.1.git"
REPO_DIR = "/content/quantum-ai-trader"

if os.path.exists(REPO_DIR):
    print("📂 Repo exists, pulling latest...")
    os.chdir(REPO_DIR)
    !git pull
else:
    print("📥 Cloning repo...")
    !git clone {REPO_URL} {REPO_DIR}
    os.chdir(REPO_DIR)

# Step 2: Add to Python path
import sys
sys.path.insert(0, REPO_DIR)

# Step 3: Install dependencies
print("\n📦 Installing dependencies...")
!pip install -q yfinance lightgbm TA-Lib python-dotenv fastapi uvicorn

# Step 4: Mount Google Drive for model saving
from google.colab import drive
drive.mount('/content/drive')

# Create model save directory
MODEL_DIR = "/content/drive/MyDrive/quantum-trader-models"
os.makedirs(MODEL_DIR, exist_ok=True)
print(f"\n✅ Model save directory: {MODEL_DIR}")

# Step 5: Verify imports
print("\n🔧 Verifying modules...")
try:
    from ultimate_signal_generator import UltimateSignalGenerator
    print("✅ UltimateSignalGenerator")
except Exception as e:
    print(f"❌ UltimateSignalGenerator: {e}")

try:
    from ultimate_forecaster import UltimateForecaster
    print("✅ UltimateForecaster")
except Exception as e:
    print(f"❌ UltimateForecaster: {e}")

try:
    from data_fetcher import DataFetcher
    print("✅ DataFetcher")
except Exception as e:
    print(f"❌ DataFetcher: {e}")

print("\n" + "="*60)
print("🎉 SETUP COMPLETE!")
print(f"📁 Working directory: {os.getcwd()}")
print(f"💾 Models will save to: {MODEL_DIR}")
print("="*60)
