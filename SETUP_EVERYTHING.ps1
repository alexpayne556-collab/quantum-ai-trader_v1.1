# ============================================
# AI COUNCIL + OUR INNOVATIONS - COMPLETE SETUP
# For PowerShell (Windows/Shadow PC)
# ============================================

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "SETTING UP AI COUNCIL TRADING SYSTEM" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# Navigate to project
Set-Location "C:\Users\alexj\Desktop\shadow_ai\quantum-ai-trader_v1.1"

# Use EXISTING .venv from yesterday
Write-Host "Using existing .venv from yesterday..." -ForegroundColor Yellow
if (-Not (Test-Path ".venv")) {
    Write-Host "ERROR: .venv not found! Run this from the project folder." -ForegroundColor Red
    exit 1
}

# Activate venv
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
.\.venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet

# Install only MISSING dependencies (everything else already installed yesterday)
Write-Host "Checking for missing packages..." -ForegroundColor Yellow
pip install --quiet yfinance plotly hmmlearn

Write-Host "Kernel 'shadow_ai' already registered from yesterday ✓" -ForegroundColor Green

Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "✅ SETUP COMPLETE!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""
Write-Host "NEXT STEPS:" -ForegroundColor Cyan
Write-Host "1. Run: jupyter lab" -ForegroundColor White
Write-Host "2. Open: notebooks/AI_COUNCIL_COMPLETE.ipynb" -ForegroundColor White
Write-Host "3. Select kernel: Python (shadow_ai)" -ForegroundColor White
Write-Host ""
Write-Host "GPU Status:" -ForegroundColor Cyan
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

Write-Host ""
Write-Host "Ready to test ALL implementations!" -ForegroundColor Green
