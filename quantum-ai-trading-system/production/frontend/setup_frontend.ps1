# Quantum AI Cockpit - Frontend Setup Script (PowerShell)
# This script installs all npm dependencies

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Quantum AI Cockpit - Frontend Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Change to frontend directory
Set-Location $PSScriptRoot

# Check if node_modules exists
if (Test-Path "node_modules") {
    Write-Host "node_modules found. Updating dependencies..." -ForegroundColor Yellow
} else {
    Write-Host "Installing dependencies..." -ForegroundColor Green
}

Write-Host ""
Write-Host "Installing npm packages..." -ForegroundColor Green
npm install

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Frontend setup complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To start the frontend dev server, run:" -ForegroundColor Yellow
Write-Host "  npm run dev" -ForegroundColor White
Write-Host ""
Read-Host "Press Enter to exit"

