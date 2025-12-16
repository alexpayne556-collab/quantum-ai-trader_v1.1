# ========================================
# FRESH START - AI Council Testing Setup
# ========================================
# Run this from wherever you cloned the repo
# It will create everything from scratch

Write-Host "=== FRESH START: Creating shadow_ai environment ===" -ForegroundColor Cyan

# 1. Create venv right here
Write-Host "`n[1/6] Creating virtual environment..." -ForegroundColor Yellow
python -m venv shadow_ai

# 2. Activate it
Write-Host "`n[2/6] Activating environment..." -ForegroundColor Yellow
.\shadow_ai\Scripts\Activate.ps1

# 3. Upgrade pip
Write-Host "`n[3/6] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# 4. Install PyTorch with CUDA 12.1 (for your RTX 2000 Ada)
Write-Host "`n[4/6] Installing PyTorch with CUDA support..." -ForegroundColor Yellow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Install all other packages
Write-Host "`n[5/6] Installing packages..." -ForegroundColor Yellow
pip install jupyter jupyterlab ipykernel
pip install yfinance plotly hmmlearn
pip install pandas numpy matplotlib scipy scikit-learn transformers accelerate

# 6. Register Jupyter kernel
Write-Host "`n[6/6] Registering Jupyter kernel..." -ForegroundColor Yellow
python -m ipykernel install --user --name shadow_ai --display-name "Python (shadow_ai)"

# Done!
Write-Host "`n=== SETUP COMPLETE ===" -ForegroundColor Green
Write-Host "`nYour GPU:" -ForegroundColor Cyan
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

Write-Host "`n=== READY TO LAUNCH ===" -ForegroundColor Green
Write-Host "Starting Jupyter Lab in 3 seconds..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

jupyter lab
