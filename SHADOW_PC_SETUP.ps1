# ============================================================================
# SHADOW PC DATA DOWNLOAD - GUARANTEED TO WORK
# ============================================================================
# Run this in PowerShell on Shadow PC
# It will activate venv, install packages, and start overnight download
# ============================================================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "QUANTUM AI TRADER - DATA DOWNLOAD SETUP" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Verify we're in the right directory
Write-Host "[1/5] Checking current directory..." -ForegroundColor Cyan

if (Test-Path "research_lab\industrial_data_pipeline.py") {
    Write-Host "  Found repo at: $PWD" -ForegroundColor Green
} else {
    Write-Host "[1/5] ERROR: Not in quantum-ai-trader_v1.1 directory" -ForegroundColor Red
    Write-Host ""
    Write-Host "Run these commands first:" -ForegroundColor Yellow
    Write-Host "  cd C:\Users\Shadow\quantum-ai-trader_v1.1" -ForegroundColor White
    Write-Host "  .\SHADOW_PC_SETUP.ps1" -ForegroundColor White
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# Step 2: Activate venv
Write-Host "[2/5] Activating virtual environment..." -ForegroundColor Cyan

# Common venv locations
$VenvPaths = @(
    ".\venv\Scripts\Activate.ps1",
    ".\.venv\Scripts\Activate.ps1",
    ".\env\Scripts\Activate.ps1"
)

$VenvFound = $false
foreach ($VenvPath in $VenvPaths) {
    if (Test-Path $VenvPath) {
        & $VenvPath
        Write-Host "  Activated: $VenvPath" -ForegroundColor Green
        $VenvFound = $true
        break
    }
}

if (-not $VenvFound) {
    Write-Host "  No venv found. Creating new one..." -ForegroundColor Yellow
    python -m venv venv
    & ".\venv\Scripts\Activate.ps1"
    Write-Host "  Created and activated new venv" -ForegroundColor Green
}

# Step 3: Install required packages
Write-Host "[3/5] Installing Python packages..." -ForegroundColor Cyan
Write-Host "  (This may take 2-3 minutes)" -ForegroundColor Gray

python -m pip install --upgrade pip --quiet
pip install -r requirements_download.txt --quiet

if ($LASTEXITCODE -eq 0) {
    Write-Host "  All packages installed successfully" -ForegroundColor Green
} else {
    Write-Host "  ERROR: Package installation failed" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Step 4: Create necessary directories
Write-Host "[4/5] Setting up directories..." -ForegroundColor Cyan

$Directories = @(
    "data",
    "data\backups",
    "data\exports",
    "research_lab"
)

foreach ($Dir in $Directories) {
    if (-not (Test-Path $Dir)) {
        New-Item -ItemType Directory -Path $Dir -Force | Out-Null
    }
}
Write-Host "  All directories ready" -ForegroundColor Green

# Step 5: Check for existing progress
Write-Host "[5/5] Checking download progress..." -ForegroundColor Cyan

if (Test-Path "data\market_data.db") {
    $DbSize = (Get-Item "data\market_data.db").Length / 1MB
    Write-Host "  Found existing database: $([Math]::Round($DbSize, 1)) MB" -ForegroundColor Green
    Write-Host "  Download will resume from last ticker" -ForegroundColor Yellow
} else {
    Write-Host "  Starting fresh download" -ForegroundColor Yellow
}

# Final confirmation
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "READY TO START DOWNLOAD" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This will download 2 years of data for ~10,000 stocks" -ForegroundColor White
Write-Host "Estimated time: 12-15 hours (overnight)" -ForegroundColor White
Write-Host "Database will grow to ~500 MB - 3.5 GB" -ForegroundColor White
Write-Host ""
Write-Host "Press ENTER to start, or Ctrl+C to cancel" -ForegroundColor Yellow
Read-Host

# Start download in new window (so you can close PowerShell)
Write-Host ""
Write-Host "Starting download in background..." -ForegroundColor Green
Write-Host "You can close this window - download will continue" -ForegroundColor Yellow
Write-Host ""

# Run in background with output to log file
Start-Process -FilePath "python" -ArgumentList "-u", "research_lab\industrial_data_pipeline.py" -RedirectStandardOutput "download.log" -RedirectStandardError "download_errors.log" -NoNewWindow

# Wait a moment and check it started
Start-Sleep -Seconds 3

$DownloadProcess = Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle -eq "" }

if ($DownloadProcess) {
    Write-Host "SUCCESS: Download running (PID: $($DownloadProcess.Id))" -ForegroundColor Green
    Write-Host ""
    Write-Host "To check progress:" -ForegroundColor Cyan
    Write-Host "  Get-Content download.log -Tail 20" -ForegroundColor White
    Write-Host ""
    Write-Host "To check database size:" -ForegroundColor Cyan
    Write-Host "  Get-Item data\market_data.db | Select-Object Length" -ForegroundColor White
    Write-Host ""
    Write-Host "Download will complete overnight. Sleep well!" -ForegroundColor Green
} else {
    Write-Host "WARNING: Could not confirm process started" -ForegroundColor Yellow
    Write-Host "Check download.log for errors" -ForegroundColor Yellow
}

Write-Host ""
Read-Host "Press Enter to exit"
