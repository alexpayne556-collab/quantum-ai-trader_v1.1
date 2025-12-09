# 🚀 SYSTEM READY FOR COLAB TRAINING

**Date**: December 9, 2025  
**Status**: ✅ ALL SYSTEMS GO

---

## ✅ API Keys Verified (9/11 Working)

### Market Data (5 sources active)
- ✅ **Twelve Data**: 800/day, 8/min (PRIMARY - best rate limit)
- ✅ **Finnhub**: 60/min (SECONDARY - real-time)
- ✅ **Alpha Vantage**: 25/day (BACKUP)
- ✅ **EODHD**: 20/day (BACKUP)
- ✅ **yfinance**: Unlimited (FALLBACK - always works)
- ❌ Polygon: 5/min (key expired, but not critical)
- ❌ FMP: 250/day (key expired, but not critical)

### Regime Detection
- ✅ **FRED**: Unlimited (VIX, yields, economic data)

### Paper Trading (Week 3-4)
- ✅ **Alpaca**: Paper account active ($100k virtual cash)
  - API Key: PKRNFP4NMO4O2CDYRRBGLH2EFU
  - Endpoint: https://paper-api.alpaca.markets

### AI Assistance
- ✅ **Perplexity AI**: For optimization guidance
- ✅ **OpenAI**: Backup AI

---

## 🔄 Data Source Rotation System

**File**: `src/python/data_source_manager.py`

**Priority Order** (automatic failover):
1. Twelve Data (800/day) → Best rate limit
2. Finnhub (60/min) → Real-time, high frequency
3. yfinance (unlimited) → Always available fallback
4. Alpha Vantage (25/day) → Backup source
5. EODHD (20/day) → Last resort

**Features**:
- ✅ Automatic rate limit tracking
- ✅ Smart source rotation when limits hit
- ✅ Per-minute and per-day limit management
- ✅ FRED integration for macro data
- ✅ Usage statistics monitoring

**Tested**: December 9, 2025
- Fetched 1000 bars for AAPL (Twelve Data)
- Retrieved VIX data (FRED: 15.41)
- Rate limit tracking: 1/800 used (0.1%)

---

## 📋 Training Checklist

### Tonight (2-4 hours)
- [x] API keys configured and verified
- [x] Data source rotation system built
- [ ] Update Colab notebook with max excursion targeting
- [ ] Upload notebook to Colab Pro
- [ ] Enable T4 GPU runtime
- [ ] Run training (76 tickers × 2 years × 1hr bars)
- [ ] Download trained models to Google Drive

### Week 1 (Dec 10-16)
- [ ] Hyperparameter optimization (RandomizedSearchCV)
- [ ] Target: Improve precision from 55% → 65-70%
- [ ] Ask Perplexity for XGBoost parameter guidance

### Week 2 (Dec 17-23)
- [ ] Feature selection (49 → 30 best features)
- [ ] Label engineering (test 3/5/10 bar horizons)
- [ ] Ensemble weight optimization
- [ ] Target: Improve precision to 68-72%

### Week 3-4 (Dec 24-Jan 6)
- [ ] Build alpha_76_trader.py for Alpaca paper trading
- [ ] Execute trades with trained models ($100k virtual)
- [ ] Track: Win rate ≥58%, Sharpe ≥1.5, Max DD <20%
- [ ] IF profitable → Build Spark frontend
- [ ] IF not profitable → Iterate optimization

---

## 🎯 Training Strategy (User-Defined)

### 1. Ticker Rotation
**Strategy**: Semi-active with "Survivor Rule"
- Weekly check: volume >$1M, ATR >2%
- Replace dead tickers automatically
- Maintain 76-ticker portfolio

### 2. Forward Horizon
**Strategy**: Max excursion in next 5 bars
```python
# Did price hit +3% at ANY point in next 5 bars?
indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=5)
df['future_high'] = df['High'].rolling(window=indexer).max()
df['target_return'] = (df['future_high'] - df['Close']) / df['Close']
df['target'] = (df['target_return'] > 0.03).astype(int)  # BUY if hit +3%
```
**Rationale**: "As a trader, you can sell early. If it hits +3% in Hour 2, you take profit."

### 3. Label Thresholds
**Strategy**: Asymmetric +3%/-2%
- BUY: Price hits +3% in next 5 bars
- SELL: Price drops -2% in next 5 bars
- HOLD: Neither threshold met
**Rationale**: Maintains 1.5:1 risk-reward ratio

### 4. Class Imbalance
**Strategy**: Ignore HOLD, optimize BUY precision
- Use `scale_pos_weight` in XGBoost
- Focus metric: Precision on BUY class >60%
- Optimize for high-confidence signals only

### 5. Regime-Adaptive
**Strategy**: Single ensemble with regime_id as feature
- NOT 3 separate models per regime
- Add `regime_id` as 50th feature column
- Let model learn regime patterns

### 6. Walk-Forward Validation
**Strategy**: 3-fold TimeSeriesSplit
- Fold 1: Bull market (2024 Q1-Q2)
- Fold 2: Choppy market (2024 Q3)
- Fold 3: Volatile market (2024 Q4)

### 7. Success Metrics
- **Precision**: >60% (not just accuracy)
- **Win Rate**: ≥58%
- **Sharpe Ratio**: ≥1.5
- **Max Drawdown**: <20% (kill switch)

---

## 📦 Files Ready for Upload to Colab

### Core Modules (copy to Colab)
```bash
src/python/multi_model_ensemble.py      # 3-model voting ensemble
src/python/feature_engine.py            # 49 technical indicators
src/python/regime_classifier.py         # 10 market regimes
src/python/data_source_manager.py       # NEW: API rotation system
```

### Notebook
```bash
notebooks/UNDERDOG_COLAB_TRAINER.ipynb  # Training pipeline (needs max excursion update)
```

### Environment Variables (copy .env to Colab)
```bash
.env                                    # All 10 API keys
```

---

## 🔧 Next Steps (IMMEDIATE)

### Step 1: Update Colab Notebook (15 min)
**File**: `notebooks/UNDERDOG_COLAB_TRAINER.ipynb`

**Current target calculation**:
```python
# WRONG: Only looks at 5th hour close
target = (close[t+5] - close[t]) / close[t]
```

**Update to max excursion**:
```python
# CORRECT: Looks at HIGHEST point in next 5 bars
indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=5)
df['future_high'] = df['High'].rolling(window=indexer).max()
df['future_low'] = df['Low'].rolling(window=indexer).min()
df['max_excursion_up'] = (df['future_high'] - df['Close']) / df['Close']
df['max_excursion_down'] = (df['Close'] - df['future_low']) / df['Close']

# Asymmetric thresholds: +3% BUY, -2% SELL
df['target'] = 1  # Default: HOLD
df.loc[df['max_excursion_up'] >= 0.03, 'target'] = 2  # BUY
df.loc[df['max_excursion_down'] >= 0.02, 'target'] = 0  # SELL
```

### Step 2: Upload to Colab Pro (5 min)
1. Open https://colab.research.google.com
2. Upload `UNDERDOG_COLAB_TRAINER.ipynb`
3. Runtime → Change runtime type → T4 GPU
4. Upload 4 Python files to Colab session
5. Create `.env` file in Colab with API keys

### Step 3: Run Training (2-4 hours)
1. Execute all cells sequentially
2. Monitor GPU usage (nvidia-smi)
3. Download models when complete:
   - `xgboost.pkl`
   - `random_forest.pkl`
   - `gradient_boosting.pkl`
   - `scaler.pkl`
   - `metadata.pkl`
   - `training_metrics.json`

### Step 4: Validate Results (30 min)
Expected baseline metrics:
- Validation precision: 55-60%
- ROC-AUC: 0.65-0.70
- Training time: 2-4 hours on T4 GPU

If precision <50%: Check class imbalance settings
If OOM error: Reduce batch size or use fewer tickers

---

## 💾 Data Coverage Estimate

**76 tickers × 2 years × 1hr bars**:
- Trading hours: 6.5 hours/day
- Trading days: ~252/year
- Total bars per ticker: 252 × 2 × 6.5 = 3,276 bars
- Total dataset: 76 × 3,276 = **248,976 rows**

**With 49 features**:
- Memory: ~250k rows × 49 features × 8 bytes = 98 MB (fits in GPU)
- Training time: 2-4 hours on T4 GPU

**Rate limit consumption**:
- Twelve Data: 76 tickers = 76/800 = 9.5% daily limit
- If Twelve Data exhausted, auto-rotate to Finnhub (60/min = 1.3 min for 76 tickers)
- Fallback to yfinance (unlimited)

---

## 🎓 Philosophy

**"Intelligence edge, not speed edge"**

1. **Week 1-2**: Make it PROFITABLE (>60% precision)
2. **Week 3-4**: Validate with paper trading (Alpaca)
3. **Month 2+**: IF profitable, THEN build Spark frontend
4. **Otherwise**: Iterate optimization until signals are worthy

**No frontend until live win rate ≥58%**

---

## 📞 Support Resources

### API Documentation
- Twelve Data: https://twelvedata.com/docs
- Finnhub: https://finnhub.io/docs/api
- FRED: https://fred.stlouisfed.org/docs/api
- Alpaca: https://alpaca.markets/docs
- Perplexity: https://docs.perplexity.ai

### Colab Resources
- GPU Guide: https://colab.research.google.com/notebooks/gpu.ipynb
- Drive Integration: https://colab.research.google.com/notebooks/io.ipynb

### Training Help
Ask Perplexity (share API_KEYS_STATUS.md):
1. "What are optimal XGBoost hyperparameters for high-volatility small-cap stocks?"
2. "How do I handle 15% minority class in 3-class classification?"
3. "Which technical indicators predict biotech sector momentum best?"

---

## ✅ FINAL STATUS

**System Status**: 🟢 READY FOR TRAINING

**Blockers**: NONE
- ✅ 9/11 API keys working (5 market data sources)
- ✅ Data rotation system tested and working
- ✅ All modules built and integration tested
- ✅ Alpaca paper trading configured
- ✅ FRED regime detection ready

**Next Action**: Update Colab notebook target calculation → Upload → Train (TONIGHT)

**Expected Timeline**:
- Tonight: Baseline training (2-4 hours)
- Week 1: Hyperparameter optimization → 65-70% precision
- Week 2: Feature engineering → 68-72% precision
- Week 3-4: Paper trading validation → 58%+ live win rate
- Month 2: IF profitable, build Spark frontend

---

**Date Updated**: December 9, 2025 9:47 PM  
**Verified By**: Comprehensive API test (verify_all_apis.py)  
**Training Status**: Ready to begin TONIGHT 🚀
