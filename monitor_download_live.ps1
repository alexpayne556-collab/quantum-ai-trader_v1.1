# LIVE PROGRESS MONITOR
# Run this in a separate PowerShell window to watch download progress in real-time
# Updates every 30 seconds

$dbPath = "data\market_data.db"
$targetSize = 500  # MB
$updateInterval = 30  # seconds

# Get initial size
if (-not (Test-Path $dbPath)) {
    Write-Host "❌ Database not found. Start download first!" -ForegroundColor Red
    exit
}

$initialSize = (Get-Item $dbPath).Length / 1MB
$startTime = Get-Date

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   LIVE DOWNLOAD MONITOR" -ForegroundColor White
Write-Host "   Press Ctrl+C to stop" -ForegroundColor Gray
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting monitor at: $($startTime.ToString('HH:mm:ss'))" -ForegroundColor Yellow
Write-Host "Initial size: $([math]::Round($initialSize, 2)) MB" -ForegroundColor Gray
Write-Host ""

$iteration = 0

while ($true) {
    $iteration++
    $currentTime = Get-Date
    $currentSize = (Get-Item $dbPath).Length / 1MB
    $progress = [math]::Round(($currentSize / $targetSize) * 100, 1)
    
    # Calculate download rate
    $elapsedMinutes = ($currentTime - $startTime).TotalMinutes
    if ($elapsedMinutes -gt 0) {
        $downloadedMB = $currentSize - $initialSize
        $rateMBPerMin = $downloadedMB / $elapsedMinutes
        $rateMBPerHour = $rateMBPerMin * 60
    } else {
        $rateMBPerMin = 0
        $rateMBPerHour = 0
    }
    
    # Estimate completion
    $remainingMB = $targetSize - $currentSize
    if ($rateMBPerMin -gt 0 -and $remainingMB -gt 0) {
        $remainingMinutes = $remainingMB / $rateMBPerMin
        $remainingHours = [math]::Round($remainingMinutes / 60, 1)
        $eta = $currentTime.AddMinutes($remainingMinutes)
    } else {
        $remainingHours = "Unknown"
        $eta = "Unknown"
    }
    
    # Progress bar
    $barLength = 50
    $filled = [math]::Floor(($progress / 100) * $barLength)
    $empty = $barLength - $filled
    $bar = "█" * $filled + "░" * $empty
    
    # Clear previous line (for updating in place)
    if ($iteration -gt 1) {
        # Move cursor up to overwrite previous output
        $Host.UI.RawUI.CursorPosition = @{X=0; Y=$Host.UI.RawUI.CursorPosition.Y - 7}
    }
    
    # Display current status
    $timestamp = $currentTime.ToString("HH:mm:ss")
    Write-Host "[$timestamp] " -NoNewline -ForegroundColor DarkGray
    Write-Host "Update #$iteration" -ForegroundColor Gray
    
    Write-Host ""
    Write-Host "[" -NoNewline
    Write-Host $bar -NoNewline -ForegroundColor $(if ($progress -lt 25) { "Red" } elseif ($progress -lt 75) { "Yellow" } else { "Green" })
    Write-Host "] " -NoNewline
    Write-Host "$progress%" -ForegroundColor White
    
    Write-Host ""
    Write-Host "📦 Size: " -NoNewline -ForegroundColor White
    Write-Host "$([math]::Round($currentSize, 2)) MB " -NoNewline -ForegroundColor Cyan
    Write-Host "/ $targetSize MB" -ForegroundColor Gray
    
    Write-Host "⚡ Rate: " -NoNewline -ForegroundColor White
    Write-Host "$([math]::Round($rateMBPerHour, 2)) MB/hour " -NoNewline -ForegroundColor Yellow
    Write-Host "($([math]::Round($rateMBPerMin, 3)) MB/min)" -ForegroundColor Gray
    
    Write-Host "⏱️  ETA: " -NoNewline -ForegroundColor White
    if ($remainingHours -ne "Unknown") {
        Write-Host "$remainingHours hours " -NoNewline -ForegroundColor Green
        Write-Host "($($eta.ToString('HH:mm')))" -ForegroundColor Gray
    } else {
        Write-Host "Calculating..." -ForegroundColor Yellow
    }
    
    Write-Host ""
    
    # Check if complete
    if ($progress -ge 95) {
        Write-Host ""
        Write-Host "✅ DOWNLOAD COMPLETE!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Total time: $([math]::Round($elapsedMinutes / 60, 1)) hours" -ForegroundColor Cyan
        Write-Host "Total downloaded: $([math]::Round($downloadedMB, 2)) MB" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Ready to run research pipeline!" -ForegroundColor White
        break
    }
    
    Start-Sleep -Seconds $updateInterval
}

Write-Host ""
Write-Host "Monitor stopped at: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Gray
Write-Host ""
