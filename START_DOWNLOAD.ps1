# ============================================================================
# SHADOW PC ONE-COMMAND SETUP
# ============================================================================
# Just run this - it does EVERYTHING automatically
# ============================================================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SETTING UP DATA DOWNLOAD" -ForegroundColor Cyan  
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Auto-detect if we're in the right place
if (-not (Test-Path "research_lab")) {
    Write-Host "ERROR: Can't find research_lab folder" -ForegroundColor Red
    Write-Host "Make sure you're in: C:\Users\Shadow\quantum-ai-trader_v1.1" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Run: cd C:\Users\Shadow\quantum-ai-trader_v1.1" -ForegroundColor White
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "[1/4] Checking Python..." -ForegroundColor Cyan
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Python not found" -ForegroundColor Red
    exit 1
}
Write-Host "  OK" -ForegroundColor Green

Write-Host "[2/4] Creating venv and installing packages..." -ForegroundColor Cyan
if (-not (Test-Path "venv")) {
    python -m venv venv
}
& "venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip --quiet
pip install yfinance pandas sqlalchemy tqdm requests pyarrow --quiet
Write-Host "  Installed" -ForegroundColor Green

Write-Host "[3/4] Creating directories..." -ForegroundColor Cyan
New-Item -ItemType Directory -Path "data" -Force | Out-Null
New-Item -ItemType Directory -Path "data\backups" -Force | Out-Null
Write-Host "  Ready" -ForegroundColor Green

Write-Host "[4/4] Checking database..." -ForegroundColor Cyan
if (Test-Path "data\market_data.db") {
    $DbSize = [Math]::Round((Get-Item "data\market_data.db").Length / 1MB, 1)
    Write-Host "  Found: $DbSize MB" -ForegroundColor Green
} else {
    Write-Host "  Starting fresh" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "READY - Press ENTER to start download" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "This downloads 2 years of data for ~10,000 stocks" -ForegroundColor White
Write-Host "Takes 12-15 hours (overnight)" -ForegroundColor White
Write-Host "You can close PowerShell after it starts" -ForegroundColor Yellow
Write-Host ""
Read-Host "Press ENTER"

# Start download
Write-Host ""
Write-Host "Starting download..." -ForegroundColor Green

Start-Process -FilePath "python" -ArgumentList "-u", "research_lab\industrial_data_pipeline.py" `
    -RedirectStandardOutput "download.log" -RedirectStandardError "download_errors.log" `
    -NoNewWindow -WorkingDirectory $PWD

Start-Sleep -Seconds 3

# Check if started
$Process = Get-Process python -ErrorAction SilentlyContinue
if ($Process) {
    Write-Host ""
    Write-Host "SUCCESS - Download running!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Check progress:" -ForegroundColor Cyan
    Write-Host "  Get-Content download.log -Tail 20" -ForegroundColor White
    Write-Host ""
    Write-Host "Check size:" -ForegroundColor Cyan  
    Write-Host "  Get-Item data\market_data.db" -ForegroundColor White
    Write-Host ""
    Write-Host "Will complete overnight. You can close this window." -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "WARNING: Couldn't confirm process" -ForegroundColor Yellow
    Write-Host "Check download.log for errors" -ForegroundColor Yellow
}

Write-Host ""
Read-Host "Press Enter to exit"
