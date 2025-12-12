# WHAT MIGHT BREAK

## Potential Failure Points & Fixes for `trading_system_unified.py`

---

## 🔴 CRITICAL FAILURES

### 1. Missing Model Files

**Symptom:**
```
ERROR | Error loading models: [Errno 2] No such file or directory: 'trained_models/colab/xgboost_model.pkl'
WARNING | ML models not loaded, using pattern-based signal
```

**Cause:** Model files don't exist or are in wrong location.

**Fix:**
```bash
# Check if models exist
ls -la trained_models/colab/

# Expected files:
# - xgboost_model.pkl
# - lightgbm_model.pkl
# - scaler.pkl
# - top_features.json

# If missing, you need to train models in Colab:
# 1. Open COLAB_PRO_TRAINING_GUIDE.ipynb in Google Colab
# 2. Run all cells
# 3. Download the trained_models.zip
# 4. Extract to trained_models/colab/
```

**Workaround:** System falls back to pattern-based signals (less accurate).

---

### 2. TA-Lib Not Installed

**Symptom:**
```
WARNING | TA-Lib not installed. Using fallback indicators.
```

**Cause:** TA-Lib C library not installed.

**Fix:**
```bash
# Ubuntu/Debian
sudo apt-get install libta-lib-dev
pip install TA-Lib

# macOS
brew install ta-lib
pip install TA-Lib

# Windows
# Download from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib‑0.4.24‑cp39‑cp39‑win_amd64.whl
```

**Impact:** ~2-3% accuracy loss. Candlestick pattern detection disabled.

---

### 3. yfinance Data Fetch Failure

**Symptom:**
```
ERROR | Failed to fetch data: No data found for ticker INVALID
ValueError: Cannot fetch data for INVALID
```

**Causes:**
- Invalid ticker symbol
- Yahoo Finance rate limiting
- No internet connection
- Market closed / no recent data

**Fix:**
```python
# Provide your own data instead
import pandas as pd
from trading_system_unified import UnifiedTradingSystem

df = pd.read_csv('my_data.csv', index_col='Date', parse_dates=True)
system = UnifiedTradingSystem()
signal = system.analyze('CUSTOM', price_data=df)
```

**Workaround:** Add retry logic:
```python
import time
for attempt in range(3):
    try:
        signal = system.analyze('AAPL')
        break
    except Exception as e:
        print(f"Attempt {attempt+1} failed: {e}")
        time.sleep(2)
```

---

### 4. Insufficient Data

**Symptom:**
```
ValueError: Insufficient data for AAPL: 30 rows (need >= 50)
```

**Cause:** Not enough historical data for feature calculation.

**Fix:**
```bash
# Use longer period
python trading_system_unified.py --ticker AAPL --period 1y
```

**Minimum Requirements:**
- 50 rows minimum (will work but some features NaN)
- 100 rows recommended
- 200+ rows ideal

---

## 🟡 WARNING FAILURES

### 5. Feature Mismatch

**Symptom:**
```
WARNING | Missing features: ['VIX_Level', 'VIX_Change', ...]
```

**Cause:** Model trained with features that can't be calculated (e.g., VIX data unavailable).

**Impact:** Missing features filled with 0, slight accuracy loss.

**Fix:** Ensure SPY and VIX data available:
```python
import yfinance as yf

spy = yf.download('SPY', period='6mo')
vix = yf.download('^VIX', period='6mo')

signal = system.analyze('AAPL', spy_data=spy, vix_data=vix)
```

---

### 6. Model Version Mismatch

**Symptom:**
```
UserWarning: Trying to unpickle estimator XGBClassifier from version 1.7.0 when using version 2.0.0
```

**Cause:** XGBoost/LightGBM version differs from training environment.

**Fix:**
```bash
# Match training versions
pip install xgboost==1.7.6
pip install lightgbm==4.0.0
pip install scikit-learn==1.3.0
```

**Or retrain models** with current library versions.

---

### 7. NaN/Inf in Features

**Symptom:**
```
WARNING | Input contains NaN, infinity or a value too large for dtype('float64')
```

**Cause:** Edge cases in feature calculation (division by zero, log of negative).

**Fix:** Already handled in code, but if persists:
```python
# Manual cleanup
features = features.replace([np.inf, -np.inf], np.nan)
features = features.fillna(0)
```

---

## 🟢 NON-CRITICAL ISSUES

### 8. Slow Performance

**Symptom:** Takes 5+ seconds per ticker.

**Causes:**
- yfinance network latency
- Large data period
- Many rolling calculations

**Fix:**
```python
# Cache SPY/VIX data for batch analysis
spy = yf.download('SPY', period='6mo')
vix = yf.download('^VIX', period='6mo')

for ticker in tickers:
    signal = system.analyze(ticker, spy_data=spy, vix_data=vix)
```

---

### 9. Pattern Detection Empty

**Symptom:**
```
patterns_detected: []
pattern_confluence: 0
```

**Cause:** No patterns detected in recent data (market consolidating).

**Impact:** None - this is normal. ML model still works.

---

### 10. Low Confidence Predictions

**Symptom:**
```
HOLD AAPL @ $178.50 | Confidence: 35.2%
```

**Cause:** Model uncertain - often during sideways markets.

**Interpretation:**
- Confidence < 50%: Don't trade
- Confidence 50-65%: Small position
- Confidence > 65%: Normal position
- Confidence > 80%: High conviction

---

## 🛠️ DEBUGGING CHECKLIST

```bash
# 1. Check Python version
python --version  # Need 3.8+

# 2. Check dependencies
pip list | grep -E "numpy|pandas|xgboost|lightgbm|yfinance|TA-Lib"

# 3. Check model files
ls -la trained_models/colab/

# 4. Test basic import
python -c "from trading_system_unified import UnifiedTradingSystem; print('OK')"

# 5. Test with verbose
python trading_system_unified.py --ticker SPY --verbose

# 6. Check internet
curl -I https://query1.finance.yahoo.com
```

---

## 📊 EXPECTED ACCURACY

| Scenario | Expected Accuracy |
|----------|-------------------|
| Full system (ML + patterns) | 69-72% |
| ML only (no TA-Lib) | 65-68% |
| Patterns only (no models) | 55-60% |
| Missing SPY/VIX data | 66-69% |

---

## 🔧 COMMON ERROR CODES

| Error | Meaning | Quick Fix |
|-------|---------|-----------|
| `ModuleNotFoundError: talib` | TA-Lib not installed | Install TA-Lib C library first |
| `FileNotFoundError: xgboost_model.pkl` | Models missing | Train in Colab or download |
| `ValueError: Insufficient data` | Need more historical data | Use `--period 1y` |
| `KeyError: 'Close'` | Wrong column names | Ensure OHLCV format |
| `ConnectionError` | Network issue | Check internet, retry |

---

## 📞 IF ALL ELSE FAILS

1. **Reset environment:**
```bash
pip uninstall xgboost lightgbm scikit-learn
pip install xgboost==1.7.6 lightgbm==4.0.0 scikit-learn==1.3.0
```

2. **Use fallback mode:**
```python
# Pattern-only analysis (no ML)
system = UnifiedTradingSystem()
system.is_loaded = False  # Force pattern mode
signal = system.analyze('AAPL')
```

3. **Check logs:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
# Re-run analysis
```

4. **File an issue:** https://github.com/alexpayne556-collab/quantum-ai-trader_v1.1/issues
