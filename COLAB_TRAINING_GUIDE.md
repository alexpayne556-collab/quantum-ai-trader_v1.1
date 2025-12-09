# Underdog Trading System - Colab Pro Training Guide

## Overview
Train the complete 3-model ensemble on **Alpha 76 watchlist** using Colab Pro's T4 GPU.

**Your Edge**: Intelligence, not speed. Multi-model ensemble trained on 1.3M+ data points beats hedge funds on small/mid-cap stocks.

---

## Prerequisites

### 1. Colab Pro Subscription ($10/month)
- T4 GPU (16GB VRAM)
- 25GB RAM
- 24-hour runtime limit

### 2. Google Drive
- 5GB free space for models and data
- Will be used to save trained models

### 3. GitHub Repository Access
- https://github.com/alexpayne556-collab/quantum-ai-trader_v1.1

---

## Quick Start (5 Steps)

### Step 1: Upload Notebook to Colab
1. Go to https://colab.research.google.com
2. File → Upload notebook
3. Select `notebooks/UNDERDOG_COLAB_TRAINER.ipynb`

### Step 2: Enable T4 GPU
1. Runtime → Change runtime type
2. Hardware accelerator: **T4 GPU**
3. Runtime shape: **Standard** (25GB RAM)
4. Save

### Step 3: Run All Cells
1. Runtime → Run all
2. Authorize Google Drive access when prompted
3. **Expected runtime**: 2-4 hours

### Step 4: Monitor Training
Check progress in each cell:
- Data download: 10-15 minutes (76 tickers × 2 years × 1hr bars)
- Feature calculation: 20-30 minutes (30+ indicators per ticker)
- Model training: 60-90 minutes (XGBoost GPU, RF, GB)
- Backtest: 5-10 minutes

### Step 5: Download Models
After training completes:
1. Go to Google Drive: `My Drive/underdog_trader/deploy`
2. Download entire folder
3. Copy to local `quantum-ai-trader_v1.1/models/`

---

## Training Pipeline Details

### Data Sources (All Free)
- **yfinance**: Historical OHLCV data
- **Period**: 2 years (Dec 2023 - Dec 2025)
- **Interval**: 1-hour bars
- **Tickers**: 76 (Alpha 76 watchlist)
- **Expected rows**: ~1.3M (76 tickers × 2 years × ~8,500 bars/ticker)

### Feature Engineering (30+ Features)
**Momentum Indicators**:
- RSI (14-period)
- MACD (12,26,9)
- Stochastic (14,3,3)
- Williams %R
- ROC (5, 10, 20 periods)

**Trend Indicators**:
- EMAs (8, 21, 50, 200)
- Bollinger Bands (20, 2std)
- ADX (14)
- ATR (14)

**Volume Indicators**:
- OBV (On Balance Volume)
- VWAP
- Volume ratios
- Volume trend

**Microstructure Proxies** (No Level 2 data needed):
- Spread proxy (high-low range)
- Amihud illiquidity
- Order imbalance proxy
- Price pressure
- Roll spread estimator

**Price Patterns**:
- Candle body size
- Upper/lower shadows
- Consecutive bull/bear bars
- Higher highs / lower lows

### Model Architecture

**Model 1: XGBoost (GPU-Accelerated)**
- Objective: Multi-class classification (BUY/HOLD/SELL)
- Tree method: `gpu_hist` (T4 GPU)
- Params: max_depth=6, learning_rate=0.1, n_estimators=200
- Training time: ~30-45 minutes

**Model 2: Random Forest (CPU)**
- Params: n_estimators=200, max_depth=10
- Uses all CPU cores (`n_jobs=-1`)
- Training time: ~15-20 minutes

**Model 3: Gradient Boosting (CPU)**
- Params: n_estimators=200, max_depth=5, learning_rate=0.1
- Training time: ~10-15 minutes

**Ensemble Voting**:
- 3/3 agree → STRONG signal (confidence: 1.0)
- 2/3 agree → MODERATE signal (confidence: 0.67-0.90)
- 1/3 agree → WEAK signal (confidence: <0.67)

### Label Creation
**Forward-looking returns** (5-bar horizon):
- BUY (label=2): Forward return > +2%
- HOLD (label=1): Forward return between -2% and +2%
- SELL (label=0): Forward return < -2%

### Train/Validation Split
- **Time-based split** (no shuffle)
- Train: First 80% of data (chronological)
- Validation: Last 20% of data (most recent)
- Prevents lookahead bias

---

## Expected Performance Targets

### Minimum Acceptable Performance
- **Accuracy**: >60% on validation set
- **Precision**: >0.60 (avoid false positives)
- **Recall**: >0.55 (catch real opportunities)
- **ROC-AUC**: >0.65 (better than random)

### Realistic Performance (After Full Training)
- **Accuracy**: 55-65%
- **Win rate** (high-confidence BUY): 60-70%
- **Average return** (5-bar forward): 1.5-3%

### Why This Works
- **Alpha 76 edge**: High-velocity small/mid-caps (less competition)
- **Regime filtering**: Only trade in favorable regimes
- **3-model consensus**: Reduces overfitting
- **Microstructure features**: Proxy for institutional activity

---

## Troubleshooting

### "Out of Memory" Error
**Solution**: Reduce data size
```python
# In Step 5, use shorter period
df = yf.download(ticker, period='1y', interval='1h')  # 1 year instead of 2
```

### "Disconnected from Runtime"
**Solution**: Colab Pro allows 24-hour runtimes, but:
- Stay active (don't close browser)
- Run cells periodically to keep session alive
- Save checkpoints to Drive

### "GPU Not Detected"
**Solution**: 
1. Runtime → Change runtime type → T4 GPU
2. Restart runtime
3. Run `!nvidia-smi` to verify

### "Module Not Found" Errors
**Solution**: Install missing packages
```python
!pip install xgboost scikit-learn yfinance pandas numpy
```

### Training Taking Too Long (>6 hours)
**Possible causes**:
1. Using CPU instead of GPU (check Runtime type)
2. Too many tickers (reduce to 50 instead of 76)
3. Too much data (use 1 year instead of 2 years)

---

## After Training: Next Steps

### 1. Validate Models
Run backtest on 2024 data:
- High-confidence signals should have >60% win rate
- Average return should be >1.5% (5-bar forward)

### 2. Download Models
```
Google Drive/underdog_trader/deploy/
├── models/
│   ├── xgboost.pkl
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   ├── scaler.pkl
│   └── metadata.pkl
├── multi_model_ensemble.py
├── feature_engine.py
├── regime_classifier.py
└── README.md
```

### 3. Copy to Local Repository
```bash
cp -r ~/Downloads/deploy /workspaces/quantum-ai-trader_v1.1/models/underdog_v1
```

### 4. Test Locally
```python
from multi_model_ensemble import MultiModelEnsemble

ensemble = MultiModelEnsemble()
ensemble.load('models/underdog_v1')

# Make prediction
prediction = ensemble.predict(features_df)
print(prediction['signal'], prediction['confidence'])
```

### 5. Build Live Trading Engine
Now that models are trained, integrate into live trading:
- Real-time data streaming (Alpaca/Polygon)
- Regime-aware position sizing
- Risk management (stop losses, max positions)
- Trade logging and monitoring

---

## Cost Breakdown

### One-Time Setup
- Colab Pro: $10/month (one month minimum)
- Data: $0 (yfinance is free)
- Training compute: Included in Colab Pro

### Ongoing Production
- Alpaca broker: $0/month (commission-free)
- Data streaming: $0 (use yfinance or Alpaca free tier)
- Cloud hosting: $0 (Railway.app free tier)

**Total monthly cost**: $10 (Colab Pro only)

---

## Advanced: Retraining Schedule

### Weekly Retraining (Recommended)
- Friday night: Retrain on latest data
- Weekend: Validate new models
- Monday: Deploy updated models

### Why Weekly?
- Market regimes shift (VIX, SPY trends)
- New catalysts (earnings, FDA approvals)
- Feature drift detection

### Automation (Future Enhancement)
```python
# Schedule retraining via GitHub Actions
# Trigger Colab training automatically
# Download models to production server
```

---

## Performance Metrics to Track

### During Training (Logged Automatically)
- Accuracy, Precision, Recall, ROC-AUC per model
- Training time per model
- Feature importance (XGBoost)

### Post-Training (Backtest)
- Win rate (high-confidence signals)
- Average return (5-bar forward)
- Sharpe ratio (if position sizing applied)
- Maximum drawdown

### Production (Live Trading)
- Daily P&L
- Win rate vs backtest
- Feature drift warnings
- Regime classification accuracy

---

## FAQ

**Q: Can I use free Colab instead of Colab Pro?**
A: Yes, but:
- No GPU priority (training takes 10x longer)
- 12-hour session limit (may not finish)
- Frequent disconnects

**Q: Can I train on fewer tickers?**
A: Yes, start with 20-30 high-quality tickers (RKLB, IONQ, SOFI, APP, VKTX)

**Q: Do I need to retrain every week?**
A: For optimal performance, yes. Markets change. But monthly retraining still works.

**Q: Can I train on daily bars instead of hourly?**
A: Yes, but you'll have less data (~500 bars/ticker vs 8,500). Model may overfit.

**Q: What if accuracy is <50%?**
A: Check:
1. Label distribution (should be ~33% BUY, 33% HOLD, 33% SELL)
2. Feature NaN count (should be 0 after cleaning)
3. Training/validation split (should be time-based)

---

## Success Criteria ✅

You're ready for production when:
- ✅ All 3 models trained successfully
- ✅ Validation accuracy >60%
- ✅ Backtest win rate >60% (high-confidence signals)
- ✅ Models saved to Google Drive
- ✅ Downloaded to local repository
- ✅ Integration test passes locally

**Next**: Build live trading engine to use these models in production.

---

## Support

- GitHub Issues: https://github.com/alexpayne556-collab/quantum-ai-trader_v1.1/issues
- Documentation: See `START_HERE.md` in repository
- Colab Docs: https://colab.research.google.com/notebooks/gpu.ipynb

**Remember**: Intelligence edge, not speed edge. You have tools hedge funds don't. 🚀
