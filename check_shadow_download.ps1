# Shadow PC Download Diagnostics
Write-Host "=== SHADOW PC DOWNLOAD DIAGNOSTICS ===" -ForegroundColor Cyan
Write-Host ""

# Check database status
Write-Host "1. DATABASE STATUS:" -ForegroundColor Yellow
$db = Get-Item data\market_data.db -ErrorAction SilentlyContinue
if ($db) {
    $sizeMB = [Math]::Round($db.Length / 1MB, 1)
    Write-Host "   Size: $sizeMB MB" -ForegroundColor Green
    Write-Host "   Last modified: $($db.LastWriteTime)" -ForegroundColor Green
    Write-Host ""
    
    if ($sizeMB -lt 100) {
        Write-Host "   ⚠️  Only 93 MB = Original 281 tickers from codespace" -ForegroundColor Yellow
        Write-Host "   ⚠️  Download did NOT run successfully last night" -ForegroundColor Yellow
    } elseif ($sizeMB -lt 500) {
        Write-Host "   ✅ Partial download ($sizeMB MB)" -ForegroundColor Green
    } else {
        Write-Host "   ✅ COMPLETE! Full dataset downloaded" -ForegroundColor Green
    }
} else {
    Write-Host "   ❌ Database not found!" -ForegroundColor Red
}
Write-Host ""

# Check error log
Write-Host "2. ERROR LOG:" -ForegroundColor Yellow
if (Test-Path download_errors.log) {
    $errors = Get-Content download_errors.log -Tail 20
    if ($errors) {
        Write-Host $errors -ForegroundColor Red
    } else {
        Write-Host "   (empty)" -ForegroundColor Gray
    }
} else {
    Write-Host "   No error log found" -ForegroundColor Gray
}
Write-Host ""

# Check if Python is running
Write-Host "3. RUNNING PROCESSES:" -ForegroundColor Yellow
$pythonProcs = Get-Process python -ErrorAction SilentlyContinue
if ($pythonProcs) {
    Write-Host "   ✅ Python is running:" -ForegroundColor Green
    $pythonProcs | Format-Table Id, StartTime, CPU, WorkingSet -AutoSize
} else {
    Write-Host "   ❌ No Python processes running" -ForegroundColor Red
    Write-Host "   Download is NOT currently running" -ForegroundColor Red
}
Write-Host ""

# Check what's actually in the database
Write-Host "4. DATABASE CONTENTS:" -ForegroundColor Yellow
.\venv\Scripts\python.exe -c @"
import sqlite3
conn = sqlite3.connect('data/market_data.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(DISTINCT ticker) FROM daily_bars')
tickers = cursor.fetchone()[0]
cursor.execute('SELECT COUNT(*) FROM daily_bars')
bars = cursor.fetchone()[0]
print(f'   Tickers: {tickers:,}')
print(f'   Total bars: {bars:,}')
conn.close()
"@
Write-Host ""

# Recommendation
Write-Host "5. WHAT TO DO:" -ForegroundColor Cyan
Write-Host "   Since database is still 93 MB (original size), download needs to restart" -ForegroundColor White
Write-Host ""
Write-Host "   Run this to restart download:" -ForegroundColor Yellow
Write-Host "   .\START_DOWNLOAD.ps1" -ForegroundColor Green
Write-Host ""
