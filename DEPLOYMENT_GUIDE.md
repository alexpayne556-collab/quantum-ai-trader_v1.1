# 📘 QUANTUM AI TRADER - COMPLETE DEPLOYMENT GUIDE

**Last Updated:** 2025-11-30  
**Version:** 1.1  
**Risk Level:** REAL CAPITAL - PRODUCTION SYSTEM

---

## 🎯 OVERVIEW

This guide walks you through deploying the complete 24/7 self-learning trading system from scratch. Follow these steps **in order** - no shortcuts.

**System Components:**
- **Pattern Detection:** 60+ TA-Lib patterns + custom EMA/VWAP/ORB detection
- **AI Forecasting:** Calibrated HistGradientBoostingClassifier per ticker
- **Market Regime Detection:** ADX/ATR-based strategy routing
- **Risk Management:** Position sizing, daily loss limits, drawdown protection
- **Interactive Charts:** Plotly with glowing pattern highlights
- **Production Logging:** SQLite + JSON audit trail
- **Online Learning:** Continuous model updates from live results (coming soon)

---

## 📋 PREREQUISITES

### System Requirements
- **OS:** Windows 10/11 (PowerShell)
- **RAM:** 16GB minimum, 32GB recommended
- **Storage:** 5GB free space
- **Internet:** Stable connection for API calls
- **GPU:** Not required (CPU-optimized)

### Accounts Needed
- **Google Colab Pro:** For model training (GPU optional but faster)
- **Google Drive:** 2GB storage for models/logs
- **Yahoo Finance:** Free (via yfinance library)
- *Optional: Finnhub, Alpha Vantage for real-time data*

---

## 🔧 INSTALLATION

### Step 1: Clone/Download Repository

```powershell
# Navigate to your workspace
cd E:\

# If using Git
git clone <your-repo-url> quantum-ai-trader-v1.1

# Or extract ZIP to E:\quantum-ai-trader-v1.1
```

### Step 2: Install Python Dependencies

```powershell
cd E:\quantum-ai-trader-v1.1

# Install TA-Lib (REQUIRED - technical indicators)
pip install TA-Lib

# Install all other dependencies
pip install yfinance pandas numpy scikit-learn joblib plotly apscheduler sqlite3 python-dotenv
```

**Verify TA-Lib Installation:**
```powershell
python -c "import talib; print(talib.__version__)"
# Should output: 0.6.7 or newer
```

### Step 3: Verify Directory Structure

```powershell
ls

# You should see:
# - pattern_detector.py
# - chart_engine.py
# - trading_orchestrator.py
# - production_logger.py
# - ai_recommender_tuned.py
# - COLAB_PRO_TRAINING_GUIDE.ipynb
# - models/ (create if missing)
# - data/ (create if missing)
# - frontend/charts/ (create if missing)
```

**Create missing directories:**
```powershell
mkdir -p models, data, frontend/charts, logs
```

---

## 🏋️ MODEL TRAINING (COLAB PRO)

### Step 1: Upload Notebook to Google Colab

1. Go to https://colab.research.google.com/
2. **File → Upload notebook**
3. Select `COLAB_PRO_TRAINING_GUIDE.ipynb` from your local system
4. Upgrade to **Colab Pro** (if not already):
   - Click **Upgrade** in top-right
   - $12/month for GPU access + longer runtimes

### Step 2: Mount Google Drive

Run the first cell:
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Authorize access:**
- Click the link
- Sign in to Google account
- Copy authorization code
- Paste into Colab

### Step 3: Run Training Pipeline

**Execute cells in order:**

1. **Cell 2:** Install dependencies (takes ~2 minutes)
   ```python
   !pip install TA-Lib yfinance pandas numpy scikit-learn joblib
   ```

2. **Cell 3:** Configuration (set tickers, lookback period)
   ```python
   TICKERS = ['MU', 'IONQ', 'APLD', 'ANNX', ...]  # 20 tickers
   TRAINING_PERIOD = '3y'  # 3 years of data
   FORWARD_DAYS = 7  # Predict 7-day returns
   ```

3. **Cell 7:** Train all models (**THIS IS THE BIG ONE**)
   ```python
   for ticker in TICKERS:
       result = train_ticker_model(ticker, period=TRAINING_PERIOD)
   ```

**Expected runtime:**
- **CPU:** ~40-60 minutes for 20 tickers
- **GPU (T4):** ~20-30 minutes
- **GPU (A100):** ~10-15 minutes

**Output will show:**
```
==============================================================
Training: MU
==============================================================
Data: 756 rows from 2022-11-30 to 2025-11-30
Features: 52, Samples: 700
Label distribution: {0: 180, 1: 340, 2: 180}
  Fold 1: Accuracy = 0.642
  Fold 2: Accuracy = 0.658
  ...
✅ Walk-forward validation: 0.650 ± 0.012

Top 10 features:
  rsi_14              : 0.0845
  macd_hist           : 0.0721
  adx                 : 0.0689
  ...

💾 Model saved: /content/drive/MyDrive/quantum_trader/models/MU_tuned.pkl
```

### Step 4: Download Models to Local System

After training completes:

1. **In Google Drive:**
   - Navigate to `My Drive → quantum_trader → models/`
   - You'll see 20 files: `MU_tuned.pkl`, `IONQ_tuned.pkl`, etc.

2. **Download all:**
   - Select all `.pkl` files (Ctrl+A or Cmd+A)
   - Right-click → **Download**
   - Save to: `E:\quantum-ai-trader-v1.1\models\`

3. **Verify local files:**
   ```powershell
   ls E:\quantum-ai-trader-v1.1\models\*.pkl
   # Should show 20 files
   ```

**File sizes:** Each `.pkl` file should be 500KB-2MB. If much smaller, training may have failed.

---

## 🚀 DEPLOYMENT

### Step 1: Test Pattern Detection

```powershell
cd E:\quantum-ai-trader-v1.1
python pattern_detector.py
```

**Expected output:**
```
MU: 108 patterns detected (36ms)
  TA-Lib: 108
  EMA ribbon: 15
  VWAP pullback: 3
  ORB: 2
```

If you see `EMA ribbon: 0`, the fix wasn't applied. Let me know.

### Step 2: Test Chart Generation

```powershell
python chart_engine.py
```

**Expected output:**
```
✅ MU chart saved to frontend/charts/MU_chart.html (108 patterns)
✅ IONQ chart saved to frontend/charts/IONQ_chart.html (96 patterns)
...
```

**Open charts:**
```powershell
start frontend/charts/MU_chart.html
```

You should see interactive Plotly chart with:
- Candlesticks
- 7 EMA ribbons (gradient colors)
- Pattern overlays (semi-transparent rectangles)
- Volume bars below
- RSI + MACD subplots

### Step 3: Start Trading Orchestrator

```powershell
python trading_orchestrator.py
```

**Expected startup log:**
```
============================================================
Initializing Trading Orchestrator
============================================================
✅ Orchestrator initialized successfully
📊 Tracking 20 tickers
💰 Account size: $10,000.00
⚠️  Risk per trade: 1.0%
🔄 Update interval: 5 minutes
✓ Loaded model for MU
✓ Loaded model for IONQ
...
Loaded 20/20 AI models

============================================================
🚀 STARTING TRADING SYSTEM
============================================================
✅ Scheduler started
   → Analysis cycle: Every 5 minutes
   → Daily tasks: 4:00 PM EST
   → Weekly tasks: Sunday 6:00 PM EST

============================================================
🔄 Starting Analysis Cycle - 2025-11-30 14:23:15
============================================================
🎯 MU: BUY signal - Confidence: 78.5%
   Entry: $94.32 | Stop: $91.15 | Target: $98.76
   Confluence: CDLENGULFING, EMA_RIBBON_BULLISH, FORECAST_BULLISH
...
✅ Cycle complete: 3 signals generated
💼 Account equity: $10,000.00
📊 Daily P&L: $0.00
============================================================
```

### Step 4: Monitor Logs

**Real-time log viewing:**
```powershell
# Open new PowerShell window
cd E:\quantum-ai-trader-v1.1
Get-Content -Path "logs/orchestrator.log" -Wait
```

**Production database:**
```powershell
# Query recent signals
sqlite3 data/production.db "SELECT * FROM signals ORDER BY timestamp DESC LIMIT 10;"
```

---

## 🔍 MONITORING & MAINTENANCE

### Daily Tasks (Automated at 4 PM EST)

The orchestrator runs these automatically:
1. Reset daily P&L counters
2. Generate updated charts for all tickers
3. Save charts to `frontend/charts/`
4. Log daily performance summary

**Manual trigger:**
```python
# In Python REPL
from trading_orchestrator import TradingOrchestrator
orchestrator = TradingOrchestrator()
import asyncio
asyncio.run(orchestrator.run_daily_tasks())
```

### Weekly Tasks (Automated Sunday 6 PM EST)

**Model Retraining:**
1. Open `COLAB_PRO_TRAINING_GUIDE.ipynb` in Colab
2. Run **Cell 7** (Train all models)
3. Download updated `.pkl` files to `models/`
4. Restart orchestrator:
   ```powershell
   # Press Ctrl+C to stop
   python trading_orchestrator.py
   ```

**Why retrain weekly?**
- Market regimes change
- New patterns emerge
- Model drift correction

### Performance Metrics to Track

**In production database:**
```sql
-- Win rate per ticker
SELECT ticker, 
       COUNT(*) as total_trades,
       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate,
       AVG(pnl) as avg_pnl
FROM trades
WHERE exit_timestamp IS NOT NULL
GROUP BY ticker
ORDER BY win_rate DESC;

-- Pattern success rate
SELECT primary_pattern,
       COUNT(*) as occurrences,
       AVG(pattern_confidence) as avg_confidence,
       SUM(CASE WHEN executed = 1 THEN 1 ELSE 0 END) as executed_count
FROM signals
GROUP BY primary_pattern
ORDER BY occurrences DESC;
```

---

## ⚙️ CONFIGURATION

### Edit `trading_orchestrator.py` Configuration

```python
# Top of file (lines 50-60):
TICKERS = ['MU', 'IONQ', 'APLD', 'ANNX', ...]  # Add/remove tickers
UPDATE_INTERVAL_MINUTES = 5  # Change scan frequency
ACCOUNT_SIZE = 10000.0  # Your starting capital
RISK_PER_TRADE = 0.01  # 1% risk per trade (adjust carefully)
MAX_DAILY_LOSS_PCT = 0.02  # 2% max daily loss (circuit breaker)
MIN_CONFIDENCE = 0.70  # 70% minimum to trade (raise to 0.75 for fewer signals)
```

**Conservative settings (less risk):**
```python
RISK_PER_TRADE = 0.005  # 0.5%
MAX_DAILY_LOSS_PCT = 0.01  # 1%
MIN_CONFIDENCE = 0.75  # 75%
```

**Aggressive settings (more risk):**
```python
RISK_PER_TRADE = 0.02  # 2%
MAX_DAILY_LOSS_PCT = 0.03  # 3%
MIN_CONFIDENCE = 0.65  # 65%
```

---

## 🐛 TROUBLESHOOTING

### Issue: "ModuleNotFoundError: No module named 'talib'"

**Solution:**
```powershell
pip install TA-Lib --upgrade
```

If that fails:
1. Download binary from: https://github.com/cgohlke/talib-build/releases
2. Install: `pip install <downloaded-file>.whl`

### Issue: "No models loaded (0/20)"

**Solution:**
- Verify `.pkl` files exist: `ls models/*.pkl`
- Check file sizes (should be 500KB-2MB each)
- Re-run Colab training if files are missing/corrupted

### Issue: "Daily loss limit hit" on first day

**Solution:**
- This is a safety feature - check `data/production.db` for trades
- Review `daily_pnl` in risk manager
- If false trigger, restart orchestrator to reset counters

### Issue: Charts show no patterns

**Solution:**
```powershell
# Re-run pattern detector
python pattern_detector.py

# Check output - should see "EMA ribbon: X" (not 0)
# If 0, contact support
```

---

## 📊 EXPECTED PERFORMANCE

Based on walk-forward backtests:

| Metric | Target | Acceptable | Poor |
|--------|--------|------------|------|
| Model Accuracy | >65% | 60-65% | <60% |
| Win Rate | >55% | 50-55% | <50% |
| Sharpe Ratio | >1.5 | 1.0-1.5 | <1.0 |
| Max Drawdown | <15% | 15-25% | >25% |
| Avg Trade Return | >1.5% | 1.0-1.5% | <1.0% |

**First week expectations:**
- 10-20 signals generated
- 5-10 trades executed (manual execution recommended initially)
- Win rate may vary (sample size too small)

**First month expectations:**
- 50-100 signals
- 30-60 trades
- Win rate stabilizing around 55-60%

---

## 🔒 RISK DISCLAIMERS

**REAL CAPITAL WARNING:**
- This system trades REAL MONEY
- Past performance ≠ future results
- Markets can change rapidly
- Always monitor manually for first 2 weeks
- Start with small capital ($1,000-$5,000)
- Increase only after validation

**Circuit Breakers:**
The system has built-in safety limits:
1. **Daily loss limit:** 2% of account (halts trading)
2. **Max drawdown:** 25% (halts trading)
3. **Position size cap:** 5% of account per trade

**Do NOT disable these** unless you understand the risks.

---

## 📞 SUPPORT

### Log Files

Always include these when reporting issues:
- `logs/orchestrator.log` (last 100 lines)
- `data/production.db` (SQLite database)
- Model accuracy from Colab training output

### Next Steps After Deployment

1. **Week 1:** Paper trade (monitor signals, don't execute)
2. **Week 2:** Execute 1-2 trades manually to validate
3. **Week 3:** Semi-automated (review signals before execution)
4. **Week 4+:** Fully automated with daily monitoring

---

## ✅ DEPLOYMENT CHECKLIST

- [ ] Python 3.11 installed
- [ ] TA-Lib installed and verified
- [ ] All dependencies installed (`pip install ...`)
- [ ] Directories created (`models/`, `data/`, `frontend/charts/`)
- [ ] Colab Pro account created
- [ ] Training notebook uploaded to Colab
- [ ] Models trained (20 `.pkl` files)
- [ ] Models downloaded to `models/` directory
- [ ] Pattern detector tested (`python pattern_detector.py`)
- [ ] Chart engine tested (`python chart_engine.py`)
- [ ] Orchestrator started (`python trading_orchestrator.py`)
- [ ] First analysis cycle completed (5 min wait)
- [ ] Production logs verified (`data/production.db` exists)
- [ ] Risk limits configured in `trading_orchestrator.py`
- [ ] Weekly retraining reminder set (calendar)

---

**🎯 YOU'RE READY TO TRADE!**

System will scan every 5 minutes, generate signals, and log to `data/production.db`.

**Good luck and trade safe! 📈**
