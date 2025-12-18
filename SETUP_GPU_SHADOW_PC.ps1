# COMPLETE GPU + QUANT TRADING SETUP for Shadow PC
# Creates fresh venv with EVERYTHING needed for serious quant work
# ============================================================================

Write-Host "🚀 Complete Quant Trading + GPU Setup - Shadow PC" -ForegroundColor Cyan
Write-Host "============================================`n" -ForegroundColor Cyan

# Step 0: Clean up old venv and create fresh one
Write-Host "Step 0: Setting up fresh virtual environment..." -ForegroundColor Yellow

if (Test-Path ".\venv") {
    Write-Host "🗑️  Removing old venv to save space..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force ".\venv"
    Write-Host "✅ Old venv deleted" -ForegroundColor Green
}

Write-Host "`n📦 Creating fresh venv with Python 3.11..." -ForegroundColor Cyan
python -m venv venv

Write-Host "✅ Activating venv..." -ForegroundColor Green
& ".\venv\Scripts\Activate.ps1"

Write-Host "`n📥 Installing comprehensive package suite..." -ForegroundColor Cyan
Write-Host "   This will take 5-10 minutes`n" -ForegroundColor Yellow

# Upgrade pip first
pip install --upgrade pip setuptools wheel

Write-Host "`n1️⃣  GPU Acceleration (CuPy, cuDF, cuML)..." -ForegroundColor Yellow
pip install cupy-cuda12x cudf-cu12 cuml-cu12

Write-Host "`n2️⃣  Core Data Science..." -ForegroundColor Yellow
pip install numpy pandas scipy matplotlib seaborn plotly

Write-Host "`n3️⃣  Machine Learning..." -ForegroundColor Yellow
pip install scikit-learn xgboost lightgbm tensorflow torch torchvision

Write-Host "`n4️⃣  Financial Data & APIs..." -ForegroundColor Yellow
pip install yfinance pandas-datareader alpaca-trade-api polygon-api-client finnhub-python

Write-Host "`n5️⃣  Technical Analysis..." -ForegroundColor Yellow
pip install ta-lib ta mplfinance

Write-Host "`n6️⃣  Backtesting & Strategy..." -ForegroundColor Yellow
pip install backtrader bt vectorbt zipline-reloaded

Write-Host "`n7️⃣  Statistical Analysis..." -ForegroundColor Yellow
pip install statsmodels arch hmmlearn pymc3 arviz

Write-Host "`n8️⃣  Database & Storage..." -ForegroundColor Yellow
pip install sqlalchemy psycopg2-binary pymongo redis

Write-Host "`n9️⃣  Utilities & Async..." -ForegroundColor Yellow
pip install requests aiohttp websockets python-dotenv tqdm joblib

Write-Host "`n🔟 Jupyter & Notebooks..." -ForegroundColor Yellow
pip install jupyter jupyterlab ipywidgets notebook

Write-Host "`n✅ All packages installed!`n" -ForegroundColor Green

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "🎯 SETUP COMPLETE!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "`nYour venv is activated and ready for:" -ForegroundColor Yellow
Write-Host "  ✅ GPU acceleration (CuPy, cuDF)" -ForegroundColor Green
Write-Host "  ✅ Machine learning (sklearn, XGBoost, PyTorch)" -ForegroundColor Green
Write-Host "  ✅ Trading APIs (Alpaca, yfinance, Polygon)" -ForegroundColor Green
Write-Host "  ✅ Backtesting (vectorbt, backtrader)" -ForegroundColor Green
Write-Host "  ✅ Technical analysis (TA-Lib)" -ForegroundColor Green
Write-Host "  ✅ Statistical modeling (statsmodels, HMM)" -ForegroundColor Green
Write-Host "`nTest GPU:" -ForegroundColor Yellow
Write-Host "  python -c 'import cupy as cp; print(cp.cuda.Device().name)'" -ForegroundColor Cyan
Write-Host "`n" -ForegroundColor Yellow
exit
