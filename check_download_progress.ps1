# CHECK DOWNLOAD PROGRESS
# Run this on Shadow PC to see how far along you are

$dbPath = "data\market_data.db"
$targetSize = 500  # MB (expected final size)
$targetTickers = 1300  # Expected number after quality filtering

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   DOWNLOAD PROGRESS CHECK" -ForegroundColor White
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Check if database exists
if (-not (Test-Path $dbPath)) {
    Write-Host "❌ Database not found at: $dbPath" -ForegroundColor Red
    Write-Host "   Make sure you're in the project directory" -ForegroundColor Yellow
    exit
}

# Get current size
$currentSizeMB = [math]::Round((Get-Item $dbPath).Length / 1MB, 2)
$progressPct = [math]::Round(($currentSizeMB / $targetSize) * 100, 1)

Write-Host "📊 Database Size: " -NoNewline -ForegroundColor White
Write-Host "$currentSizeMB MB " -NoNewline -ForegroundColor Green
Write-Host "/ $targetSize MB (target)" -ForegroundColor Gray

Write-Host "📈 Progress: " -NoNewline -ForegroundColor White
Write-Host "$progressPct%" -ForegroundColor $(if ($progressPct -lt 25) { "Red" } elseif ($progressPct -lt 75) { "Yellow" } else { "Green" })

# Progress bar
$barLength = 50
$filled = [math]::Floor(($progressPct / 100) * $barLength)
$empty = $barLength - $filled
$bar = "█" * $filled + "░" * $empty

Write-Host ""
Write-Host "[" -NoNewline
Write-Host $bar -NoNewline -ForegroundColor Green
Write-Host "]" -ForegroundColor White
Write-Host ""

# Query database to count tickers
Write-Host "🔍 Checking ticker count..." -ForegroundColor Yellow

try {
    Add-Type -Path "System.Data.SQLite.dll" -ErrorAction SilentlyContinue
    $connection = New-Object System.Data.SQLite.SQLiteConnection("Data Source=$dbPath")
    $connection.Open()
    
    $command = $connection.CreateCommand()
    $command.CommandText = "SELECT COUNT(DISTINCT ticker) as count FROM ohlcv"
    $reader = $command.ExecuteReader()
    
    if ($reader.Read()) {
        $tickerCount = $reader["count"]
        Write-Host "📦 Tickers in database: " -NoNewline -ForegroundColor White
        Write-Host "$tickerCount " -NoNewline -ForegroundColor Green
        Write-Host "/ ~$targetTickers (estimated)" -ForegroundColor Gray
        
        $tickerProgress = [math]::Round(($tickerCount / $targetTickers) * 100, 1)
        Write-Host "📈 Ticker progress: " -NoNewline -ForegroundColor White
        Write-Host "$tickerProgress%" -ForegroundColor $(if ($tickerProgress -lt 25) { "Red" } elseif ($tickerProgress -lt 75) { "Yellow" } else { "Green" })
    }
    
    $reader.Close()
    $connection.Close()
    
} catch {
    Write-Host "⚠️  Could not query database (SQLite driver not available)" -ForegroundColor Yellow
    Write-Host "   Using file size as progress indicator" -ForegroundColor Gray
}

Write-Host ""

# Estimate time remaining
if ($currentSizeMB -gt 10 -and $progressPct -lt 95) {
    # Rough estimate: 500 MB over 12-15 hours = ~0.6-0.7 MB/min
    $remainingMB = $targetSize - $currentSizeMB
    $estimatedMinutes = $remainingMB / 0.65  # Average rate
    $estimatedHours = [math]::Round($estimatedMinutes / 60, 1)
    
    Write-Host "⏱️  Estimated time remaining: " -NoNewline -ForegroundColor White
    Write-Host "$estimatedHours hours" -ForegroundColor Cyan
    Write-Host ""
}

# Status assessment
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray

if ($progressPct -lt 5) {
    Write-Host "⏳ Status: " -NoNewline -ForegroundColor White
    Write-Host "Just started - Long way to go!" -ForegroundColor Red
    Write-Host "   💡 Tip: This will take 12-15 hours total" -ForegroundColor Gray
}
elseif ($progressPct -lt 25) {
    Write-Host "🏃 Status: " -NoNewline -ForegroundColor White
    Write-Host "Early stages - Keep waiting" -ForegroundColor Yellow
    Write-Host "   💡 Tip: Good time to set up GPU environment" -ForegroundColor Gray
}
elseif ($progressPct -lt 50) {
    Write-Host "🔥 Status: " -NoNewline -ForegroundColor White
    Write-Host "Making progress!" -ForegroundColor Yellow
    Write-Host "   💡 Tip: Read those academic papers" -ForegroundColor Gray
}
elseif ($progressPct -lt 75) {
    Write-Host "🚀 Status: " -NoNewline -ForegroundColor White
    Write-Host "More than halfway!" -ForegroundColor Cyan
    Write-Host "   💡 Tip: Prepare your diagnostic scripts" -ForegroundColor Gray
}
elseif ($progressPct -lt 95) {
    Write-Host "🎯 Status: " -NoNewline -ForegroundColor White
    Write-Host "Almost there!" -ForegroundColor Green
    Write-Host "   💡 Tip: Get ready to run tests" -ForegroundColor Gray
}
else {
    Write-Host "✅ Status: " -NoNewline -ForegroundColor White
    Write-Host "COMPLETE!" -ForegroundColor Green
    Write-Host "   🎉 Ready to start research!" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "   Next steps:" -ForegroundColor White
    Write-Host "   1. Run: python diagnostic_tests.py" -ForegroundColor Gray
    Write-Host "   2. Check data quality" -ForegroundColor Gray
    Write-Host "   3. Test earnings edge" -ForegroundColor Gray
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
