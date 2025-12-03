param(
  [string]$Ticker = "SPY",
  [switch]$UseVite
)

$ErrorActionPreference = "Stop"
Write-Host "== Quantum Trader Preview Runner ==" -ForegroundColor Cyan

# Resolve repo root reliably whether invoked via relative/absolute path
$repo = $PSScriptRoot
if (-not $repo -or $repo -eq '') {
  # Fallback: assume current directory is the repo root
  $repo = Get-Location
}
Push-Location $repo

# 1) Ensure Python venv + backend deps
if (-not (Test-Path ".venv")) {
  Write-Host "Creating Python venv..." -ForegroundColor Yellow
  python -m venv .venv
}

Write-Host "Activating venv + installing backend deps..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1
pip install -r .\backend\requirements_min.txt

# 2) Start backend in a new PowerShell window
$backendCmd = "`"$repo\.venv\Scripts\Activate.ps1`"; uvicorn backend.min_api:app --host 127.0.0.1 --port 8000 --reload"
Write-Host "Starting backend (http://127.0.0.1:8000)..." -ForegroundColor Yellow
Start-Process pwsh -ArgumentList "-NoExit","-Command", $backendCmd | Out-Null

# 3) Optionally start Vite dev server
if ($UseVite.IsPresent) {
  $viteDir = Join-Path $repo "frontend-dashboard"
  $viteCmd = "cd `"$viteDir`"; npm install; npm run dev -- --host"
  Write-Host "Starting Vite dev server (http://localhost:5173)..." -ForegroundColor Yellow
  Start-Process pwsh -ArgumentList "-NoExit","-Command", $viteCmd | Out-Null
}

# 4) Open static HTML cyberpunk preview
$preview = Join-Path $repo "frontend-dashboard\public\cyberpunk_preview.html"
if (Test-Path $preview) {
  Write-Host "Opening static preview..." -ForegroundColor Yellow
  Start-Process $preview | Out-Null
}

# 5) Quick health check (backend)
Start-Sleep -Seconds 1
try {
  $resp = Invoke-WebRequest -Uri "http://127.0.0.1:8000/healthz" -UseBasicParsing -TimeoutSec 3
  Write-Host "Backend health: $($resp.StatusCode)" -ForegroundColor Green
} catch {
  Write-Host "Backend not reachable yet; it may still be starting." -ForegroundColor DarkYellow
}

Write-Host "Done. If using Vite, browse to http://localhost:5173/" -ForegroundColor Cyan
Pop-Location
