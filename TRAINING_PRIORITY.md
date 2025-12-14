# 🎯 TRAINING PRIORITY - Make It Profitable First

## Current Status
✅ Research complete - 3 core modules built  
✅ Alpha 76 watchlist defined (76 high-growth tickers)  
✅ Colab Pro training pipeline ready  
❌ Models untrained - 0% accuracy  
❌ Not profitable yet  

## Priority: Train → Optimize → Validate → THEN Frontend

Frontend integration can wait **weeks**. We need profitable signals first.

---

## Week 1-2: Training & Initial Optimization

### Step 1: Colab Pro Training (Tonight/Tomorrow)
**Goal**: Get baseline performance from 3-model ensemble

1. Upload `notebooks/UNDERDOG_COLAB_TRAINER.ipynb` to Colab Pro
2. Enable T4 GPU runtime
3. Train on 76 tickers × 2 years × 1hr bars = 1.3M rows
4. **Target Metrics**:
   - Validation accuracy: **60-65%** (baseline)
   - Win rate (backtest): **55-60%**
   - ROC-AUC: **0.65-0.70**

**Expected Runtime**: 2-4 hours on T4 GPU

**Download from Colab**:
```
models/underdog_v1/
  ├── xgboost.pkl
  ├── random_forest.pkl
  ├── gradient_boosting.pkl
  ├── scaler.pkl
  ├── metadata.pkl
  └── training_metrics.json
```

---

### Step 2: Hyperparameter Optimization (Days 2-3)
**Goal**: Improve accuracy from 60% → 68-72%

#### XGBoost GPU Tuning
Current defaults (untested):
```python
xgb_params = {
    'n_estimators': 200,
    'max_depth': 8,
    'learning_rate': 0.05,
    'tree_method': 'gpu_hist'
}
```

**Optimize using GridSearch**:
```python
param_grid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [6, 8, 10, 12],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2]
}
```

**Expected gain**: +3-5% accuracy

---

#### Random Forest Tuning
Current defaults:
```python
rf_params = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 10,
    'min_samples_leaf': 4
}
```

**Optimize**:
```python
param_grid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [12, 15, 20, None],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [2, 4, 8],
    'max_features': ['sqrt', 'log2', 0.5]
}
```

**Expected gain**: +2-4% accuracy

---

#### Gradient Boosting Tuning
Current defaults:
```python
gb_params = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 6
}
```

**Optimize**:
```python
param_grid = {
    'n_estimators': [200, 300, 500],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'max_depth': [4, 6, 8],
    'subsample': [0.8, 0.9, 1.0],
    'min_samples_split': [2, 5, 10]
}
```

**Expected gain**: +2-3% accuracy

---

### Step 3: Feature Engineering Optimization (Days 3-5)
**Goal**: Find optimal feature subset

Current: 49 features (may have redundancy/noise)

**Feature Selection Methods**:
1. **Mutual Information** - Remove low-info features
2. **Correlation Analysis** - Remove highly correlated pairs (r > 0.95)
3. **SHAP Values** - Keep top 30 most important features
4. **Recursive Feature Elimination** - Iteratively remove weakest

**Expected gain**: +2-5% accuracy

---

### Step 4: Label Engineering (Days 5-7)
**Goal**: Optimize forward horizon and thresholds

Current labels:
```python
# 5-bar forward return (5 hours)
future_return = (close[t+5] - close[t]) / close[t]

if future_return > 0.03:  # +3%
    label = 2  # BUY
elif future_return < -0.02:  # -2%
    label = 0  # SELL
else:
    label = 1  # HOLD
```

**Optimize**:
- **Horizon**: Test 3-bar (3h), 5-bar (5h), 10-bar (10h)
- **Thresholds**: Test (2%, 1.5%), (3%, 2%), (4%, 3%)
- **Risk-reward ratio**: Optimize asymmetric thresholds

**Expected gain**: +3-7% accuracy

---

## Week 2-3: Advanced Optimization

### Step 5: Ensemble Weighting
Current: Simple majority voting (1:1:1)

**Optimize weights based on per-model accuracy**:
```python
# If XGBoost = 68%, RF = 64%, GB = 62%
# Weight: XGBoost 50%, RF 30%, GB 20%
weights = [0.50, 0.30, 0.20]
```

**Expected gain**: +1-3% accuracy

---

### Step 6: Regime-Adaptive Training
**Goal**: Train separate models per regime

Instead of 1 ensemble for all regimes:
- **Train 3 ensembles**: BULL models, BEAR models, CHOPPY models
- Switch models based on current regime
- Each optimized for specific market conditions

**Expected gain**: +5-10% accuracy (significant!)

---

### Step 7: Walk-Forward Validation
**Goal**: Prevent overfitting, validate temporal robustness

Current: Single train/test split (may overfit)

**Walk-Forward**:
```
Train: 2023-01 to 2023-12 → Test: 2024-01 to 2024-03
Train: 2023-04 to 2024-03 → Test: 2024-04 to 2024-06
Train: 2023-07 to 2024-06 → Test: 2024-07 to 2024-09
...
```

**Expected gain**: -2-5% accuracy (reality check, but honest)

---

## Week 3-4: Paper Trading Validation

### Step 8: Live Paper Trading
**Goal**: Validate on unseen 2025 data

1. Run predictions on live Alpha 76 tickers
2. Track signals for 2 weeks (no real money)
3. Measure:
   - Win rate (actual)
   - Average return
   - Sharpe ratio
   - Max drawdown
   - Signal quality vs backtest

**Success Criteria**:
- Win rate ≥ 58% (slightly lower than backtest OK)
- Average return ≥ 1.5% per winning trade
- Sharpe ≥ 1.5
- Max drawdown ≤ 15%

**If fails**: Return to optimization (Steps 2-7)

---

## Week 5+: Deploy or Iterate

### Decision Point: Is It Profitable?

**If YES (Win rate ≥ 58%, Sharpe ≥ 1.5)**:
- Start with small capital ($1k-5k)
- Monitor for 1 month live
- Scale up gradually
- **THEN** build frontend for monitoring

**If NO**:
- Analyze failure modes
- Optimize weak areas
- Add new features (order flow, dark pool, sentiment)
- Try alternative models (LightGBM, CatBoost, Neural Nets)
- Consider longer horizons (1-day instead of 5-hour)

---

## Optimization Toolkit (Already Built)

File: `optimization_toolkit.py` (existing in workspace)

**Tools Available**:
- Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- Feature selection (RFECV, SelectKBest, SHAP)
- Model comparison
- Cross-validation strategies
- Performance visualization

**Use it!**

---

## Training Checklist (Next 2 Weeks)

### Week 1
- [ ] Day 1: Upload to Colab Pro, train baseline (2-4 hours)
- [ ] Day 2: Download models, run first backtest
- [ ] Day 3: XGBoost hyperparameter tuning
- [ ] Day 4: Random Forest tuning
- [ ] Day 5: Gradient Boosting tuning
- [ ] Day 6: Feature selection (reduce 49 → 30 best)
- [ ] Day 7: Retrain with optimal params + features

**Target after Week 1**: 65-70% accuracy, 60% win rate

### Week 2
- [ ] Day 8: Label engineering (test different horizons)
- [ ] Day 9: Test asymmetric thresholds (risk-reward)
- [ ] Day 10: Ensemble weight optimization
- [ ] Day 11: Start regime-adaptive training
- [ ] Day 12: Validate regime models
- [ ] Day 13: Walk-forward validation setup
- [ ] Day 14: Full walk-forward test

**Target after Week 2**: 68-72% accuracy, 62-65% win rate

### Week 3-4
- [ ] Paper trading (2 weeks minimum)
- [ ] Track every signal, measure actual performance
- [ ] Compare backtest vs live results
- [ ] Identify failure modes
- [ ] Final adjustments

**Target after Week 3-4**: Live win rate ≥ 58%, ready for capital

---

## Performance Targets (Realistic)

### Baseline (Week 1)
- Validation Accuracy: 60-65%
- Backtest Win Rate: 55-60%
- ROC-AUC: 0.65-0.70
- Sharpe Ratio: 1.2-1.5

### After Optimization (Week 2)
- Validation Accuracy: 68-72%
- Backtest Win Rate: 62-65%
- ROC-AUC: 0.72-0.78
- Sharpe Ratio: 1.8-2.2

### Live Trading (Week 3-4)
- Actual Win Rate: 58-62% (slightly lower than backtest is normal)
- Average Return: 1.5-2.5% per winner
- Sharpe Ratio: 1.5-2.0
- Max Drawdown: 10-15%

### Production Target (Month 2+)
- Win Rate: 58%+ sustained
- Monthly Return: 8-15%
- Sharpe: 1.8+
- Max Drawdown: <20%

---

## When to Build Frontend?

**NOT NOW!**

**Build frontend when**:
- ✅ Live win rate ≥ 58% for 2+ weeks
- ✅ Sharpe ratio ≥ 1.5
- ✅ Confident in signals
- ✅ Ready to deploy capital

**Reason**: What's the point of a beautiful dashboard showing unprofitable signals?

First: Make it profitable  
Then: Make it scalable  
Finally: Make it pretty

---

## Next Immediate Action

**RIGHT NOW**: Upload `notebooks/UNDERDOG_COLAB_TRAINER.ipynb` to Colab Pro and start training.

Everything else waits for these models to show they can make money.

No frontend. No fancy features. Just **profitable predictions**.

Let's make this thing print money first! 💰
