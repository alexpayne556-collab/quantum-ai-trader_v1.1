# How to Download Project Files to Your Local E: Drive

## Option 1: Clone the Entire Repository (Recommended)

Open PowerShell or Command Prompt on your local machine and run:

```powershell
# Navigate to your E: drive project folder
cd E:\path\to\your\project

# If you haven't cloned yet:
git clone https://github.com/alexpayne556-collab/quantum-ai-trader_v1.1.git

# If you already have the repo, just pull latest:
cd quantum-ai-trader_v1.1
git pull origin main
```

This will give you ALL files including the trained models.

---

## Option 2: Download Specific Files via GitHub Web

1. Go to: https://github.com/alexpayne556-collab/quantum-ai-trader_v1.1
2. Click "Code" → "Download ZIP"
3. Extract to your E: drive

---

## Option 3: Download Individual Files

### Key Files to Download:

**Core Predictor:**
- `core/colab_predictor.py` - Main prediction engine

**Paper Trading System:**
- `paper_trader.py` - Paper trading with continuous learning
- `continuous_learning_runner.py` - Scheduled runner

**Trained Models:**
- `trained_models/colab/xgboost_model.pkl`
- `trained_models/colab/lightgbm_model.pkl`
- `trained_models/colab/scaler.pkl`
- `trained_models/colab/top_features.json`

**API Server:**
- `api_server.py` - FastAPI backend

---

## Option 4: Sync via VS Code (If Using GitHub Codespaces)

1. In VS Code with the Codespace open
2. Open Source Control panel (Ctrl+Shift+G)
3. Ensure all changes are committed and pushed
4. On your local machine, open the same repo
5. Pull the latest changes

---

## Option 5: Direct Download Links

After pushing to GitHub, you can download individual files by right-clicking on them in GitHub and selecting "Download".

Raw file URLs:
- https://raw.githubusercontent.com/alexpayne556-collab/quantum-ai-trader_v1.1/main/paper_trader.py
- https://raw.githubusercontent.com/alexpayne556-collab/quantum-ai-trader_v1.1/main/continuous_learning_runner.py
- https://raw.githubusercontent.com/alexpayne556-collab/quantum-ai-trader_v1.1/main/core/colab_predictor.py

---

## Running on Your Local Machine

Once you have the files on your E: drive:

```powershell
# Navigate to project
cd E:\quantum-ai-trader_v1.1

# Create virtual environment (if not exists)
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements-production.txt

# Run paper trading simulation
python continuous_learning_runner.py --simulate 30

# Or run live paper trading
python paper_trader.py --mode daily
```

---

## Required Dependencies for Local Setup

Make sure you have these installed:
- Python 3.10+
- TA-Lib (Windows binary from: https://github.com/cgohlke/talib-build/releases)

```powershell
pip install pandas numpy scikit-learn xgboost lightgbm yfinance ta-lib schedule joblib
```

For TA-Lib on Windows, download the appropriate .whl file from the link above, then:
```powershell
pip install TA_Lib-0.4.29-cp311-cp311-win_amd64.whl
```
(Replace with your Python version)
