# SHADOW PC QUICK START GUIDE

## One-Time Setup (5 minutes)

### 1. Get the code
```powershell
cd C:\Users\YourUsername
git clone https://github.com/alexpayne556-collab/quantum-ai-trader_v1.1.git
cd quantum-ai-trader_v1.1
```

### 2. Run setup script
```powershell
.\SHADOW_PC_SETUP.ps1
```

**That's it!** The script will:
- ✓ Activate your venv (or create new one)
- ✓ Install all packages
- ✓ Start overnight download
- ✓ Run in background so you can close PowerShell

---

## Download runs overnight (~12-15 hours)

The process will:
- Download 2 years of data for ~10,000 stocks
- Store in `data/market_data.db` (grows to ~500 MB - 3.5 GB)
- Auto-backup every hour to `data/backups/`
- Resume automatically if it crashes
- Log progress to `download.log`

---

## Check Progress Anytime

```powershell
# See last 20 lines of progress
Get-Content download.log -Tail 20

# Check database size
Get-Item data\market_data.db

# Count tickers downloaded
python -c "import sqlite3; print(sqlite3.connect('data/market_data.db').execute('SELECT COUNT(DISTINCT ticker) FROM daily_bars').fetchone()[0], 'tickers')"
```

---

## Troubleshooting

### If setup fails:
```powershell
# Manually create venv
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements_download.txt

# Then start download
python research_lab\industrial_data_pipeline.py
```

### If download stops:
```powershell
# Just run setup again - it resumes from last ticker
.\SHADOW_PC_SETUP.ps1
```

### If you need to start fresh:
```powershell
# Delete database and start over
Remove-Item data\market_data.db
.\SHADOW_PC_SETUP.ps1
```

---

## After Download Completes

You'll have:
- ✓ ~1,300 high-quality stocks (2 years each)
- ✓ ~650,000 price bars
- ✓ Ready for GPU-accelerated analysis
- ✓ Multiple backups in `data/backups/`

Then run analysis frameworks:
```powershell
# Hypothesis testing
python research_lab\statistical_framework.py

# Regime detection  
python research_lab\regime_detection.py

# Factor discovery (uses GPU if available)
python research_lab\factor_analysis.py
```

---

## Support Files

- `SHADOW_PC_SETUP.ps1` - Automated setup (run this)
- `requirements_download.txt` - Python packages needed
- `research_lab/industrial_data_pipeline.py` - Download engine
- `research_lab/data_cache.py` - Backup/recovery tools
- `DATA_SAFETY_REPORT.md` - Full technical details
