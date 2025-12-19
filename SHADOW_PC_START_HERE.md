# 🎯 SHADOW PC - QUICK START GUIDE

You're already in the right directory! Just follow these steps:

## Step 1: Pull Latest Code (CRITICAL!)
```powershell
git pull
```

## Step 2: Check What You Have
```powershell
ls *.py | Select-Object Name
ls market_data.db
```

## Step 3: Run GPU Tests!
```powershell
# Test GPU acceleration first:
python SHADOW_PC_GPU_TESTS.py

# This will test:
# - Calendar effects (day of week, month patterns)
# - Volatility regimes (ATR breakouts)
# - Microstructure (gaps, volume spikes, round numbers)
```

## Step 4: After Tests Complete
```powershell
git add data/*_COMPREHENSIVE.csv
git commit -m "GPU results: Calendar/Volatility/Microstructure complete"
git push
```

## What You Already Have ✅
- Python 3.11 ✅
- pandas, numpy, scipy, tqdm, numba ✅
- **CuPy for GPU!** ✅
- Git repo cloned ✅

## What You Need
- Pull latest code with `git pull`
- market_data.db (496MB) - if not present, need to transfer from Codespaces

## Expected GPU Performance
- CPU (Numba): 41M calculations/sec
- **GPU (RTX 3070): 400M-2B calculations/sec** (10-100x faster!)
- Tests should complete in 5-10 minutes instead of hours!

---
**START HERE:** Run `git pull` in PowerShell!
