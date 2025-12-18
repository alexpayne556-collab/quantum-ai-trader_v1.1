# SHADOW PC DATA RECOVERY & STATUS CHECK

## 🎯 WHAT YOU NEED TO DO RIGHT NOW

### STEP 1: Find Your Data on Shadow PC

Your download saved to: **`C:\Users\Shadow\quantum-ai-trader_v1.1\data\market_data.db`**

Open PowerShell on Shadow PC and run:

```powershell
cd C:\Users\Shadow\quantum-ai-trader_v1.1

# Check if database exists and how big it is
Get-Item data\market_data.db -ErrorAction SilentlyContinue | Select-Object Name, Length, LastWriteTime

# Check download logs to see what happened
Get-Content download.log -Tail 50

# Check error log
Get-Content download_errors.log -Tail 20 -ErrorAction SilentlyContinue
```

### STEP 2: Interpret the Results

**If database file exists:**
- **Under 100 MB**: Only has the 281 tickers from yesterday (download didn't run)
- **100-500 MB**: Downloaded ~300-1,300 tickers (partial success)
- **500 MB - 3.5 GB**: COMPLETE SUCCESS! Most/all tickers downloaded

**LastWriteTime tells you:**
- If it's from last night (3-4 AM): download started but stopped
- If it's recent: download might still be running!

### STEP 3: Check If Download Is Still Running

```powershell
# Check for running Python processes
Get-Process python -ErrorAction SilentlyContinue | Format-Table Id, StartTime, CPU, WorkingSet

# If you see a python process, check what it's doing
Get-Content download.log -Tail 5 -Wait
# (Press Ctrl+C to stop watching)
```

### STEP 4: Restart Download If Needed

**OPTION A: If Shadow PC is still on and you want to resume now**

```powershell
cd C:\Users\Shadow\quantum-ai-trader_v1.1

# Pull latest code (in case anything changed)
git pull

# Restart download (automatically resumes from where it stopped)
.\START_DOWNLOAD.ps1
```

**OPTION B: If Shadow PC turned off and lost progress**

DON'T WORRY! The database has backups and auto-resume:

1. **Your codespace has 281 tickers backed up** (89 MB in data/backups/)
2. **The download script auto-resumes** - it checks what's already downloaded and skips it
3. **Every ticker that finished is cached** - you won't re-download them

Just run `.\START_DOWNLOAD.ps1` again and it picks up where it left off.

---

## 🔍 HOW TO CHECK WHAT GOT DOWNLOADED

```powershell
cd C:\Users\Shadow\quantum-ai-trader_v1.1

# Activate venv
.\venv\Scripts\Activate.ps1

# Check database contents
python -c "
import sqlite3
import os

db_path = 'data/market_data.db'

if not os.path.exists(db_path):
    print('❌ Database not found!')
    exit()

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Count tickers
cursor.execute('SELECT COUNT(DISTINCT ticker) FROM daily_bars')
ticker_count = cursor.fetchone()[0]

# Count total bars
cursor.execute('SELECT COUNT(*) FROM daily_bars')
bar_count = cursor.fetchone()[0]

# Get date range
cursor.execute('SELECT MIN(date), MAX(date) FROM daily_bars')
date_range = cursor.fetchone()

# Get last 10 tickers downloaded
cursor.execute('''
    SELECT ticker, status, last_updated, error_message 
    FROM download_status 
    ORDER BY last_updated DESC 
    LIMIT 10
''')
recent = cursor.fetchall()

print(f'📊 DATABASE STATUS')
print(f'=' * 60)
print(f'Tickers downloaded: {ticker_count:,}')
print(f'Total bars: {bar_count:,}')
print(f'Date range: {date_range[0]} to {date_range[1]}')
print(f'Database size: {os.path.getsize(db_path) / 1024 / 1024:.1f} MB')
print()
print(f'Last 10 downloads:')
for r in recent:
    print(f'  {r[0]}: {r[1]} at {r[2]}')
    if r[3]:
        print(f'    Error: {r[3]}')

conn.close()
"
```

---

## 📈 CHECK ALPACA PAPER TRADES

You placed 2 paper trades yesterday:
- **HUT**: 3.74 shares @ $40.16 ($150 position)
- **RXRX**: 32.05 shares @ $4.68 ($150 position)

Both were **Vol2x+GapUp** edges with 85.4% hit rate.

### To Check Performance:

You need to set your Alpaca API keys first. If you have them:

```powershell
# On Shadow PC (PowerShell):
$env:ALPACA_API_KEY="your_key_here"
$env:ALPACA_SECRET_KEY="your_secret_here"

cd C:\Users\Shadow\quantum-ai-trader_v1.1
.\venv\Scripts\Activate.ps1

python scripts/alpaca_paper_trader.py
```

Or just login to your Alpaca account at: https://app.alpaca.markets/paper/dashboard

---

## 🚀 QUICK START DOWNLOAD AGAIN

If you're on Shadow PC right now and just want to get it running:

```powershell
# 1. Go to repo
cd C:\Users\Shadow\quantum-ai-trader_v1.1

# 2. Pull latest
git pull

# 3. Start download
.\START_DOWNLOAD.ps1
```

It will:
✅ Auto-resume from where it stopped
✅ Skip already-downloaded tickers
✅ Run in background
✅ Save progress every ticker
✅ Create logs (download.log, download_errors.log)

---

## ❓ FAQ

**Q: Will I lose data if Shadow PC turns off?**
A: NO! Every completed ticker is saved to the database immediately. You only lose the CURRENT ticker being downloaded.

**Q: How do I know if it's done?**
A: Check the log: `Get-Content download.log -Tail 1`
Should say "Download complete! Final stats: X tickers, Y bars"

**Q: Can I use the computer while it downloads?**
A: YES! It runs in background. Just don't close PowerShell or shut down.

**Q: What if I see errors in download_errors.log?**
A: Normal! Some tickers fail (delisted, no data, API errors). Script skips them and continues.

**Q: How long will it take?**
A: 12-15 hours for full 10,986 tickers. Depends on:
- How many already downloaded (resumes from there)
- Network speed
- API rate limits

**Q: Should I just let it run overnight again?**
A: YES! That's the best approach. Start it before bed, check in the morning.

---

## 📊 EXPECTED RESULTS

When complete, you should see:
- **~1,300 tickers** passing quality checks (out of 10,986 total)
- **~650,000 bars** of data (1,300 tickers × 504 days)
- **~500 MB** database size
- **2 years** of data (2023-12-18 to 2025-12-17)

Then you can run the hypothesis tests on GPU and discover real edges!
