# Quick Reference: ML Trading System Optimization (43% → 60%+)

## 📊 DIAGNOSTIC: What's Wrong with Current 43%?

### Top Suspects (Run These First)

```python
# 1. Check for look-ahead bias
# Plot feature vs forward return - should show NO correlation at t=0
import matplotlib.pyplot as plt
plt.scatter(X['rsi_14'], y, alpha=0.1)
plt.axhline(0.33, color='r', linestyle='--', label='Random')
plt.title('RSI vs Forward Return (should be ~random)')
plt.show()

# 2. Check label balance
unique, counts = np.unique(y, return_counts=True)
print(pd.Series(counts, index=['HOLD', 'BUY', 'SELL']))
# If one class >70%, you have imbalance problem

# 3. Check for data leakage
# Feature at day N should NOT depend on return at day N
corr = X.iloc[:100].corrwith(y[:100])
print(corr.nlargest(5))
# If any feature has corr > 0.5, it's leaking

# 4. Validate with purged CV (should drop accuracy 3-5%)
```

## 🎯 ONE-DAY QUICK WIN: 43% → 48%+

### Step 1: Fix Focal Loss (5 min)
```python
from torch.nn.functional import cross_entropy

# Replace standard loss with focal
def focal_loss(inputs, targets, gamma=2.0, alpha=None):
    ce = cross_entropy(inputs, targets, reduction='none')
    p = torch.exp(-ce)
    loss = (1 - p) ** gamma * ce
    if alpha:
        loss = loss * alpha[targets]
    return loss.mean()
```

**Expected Gain:** +3-5% on minority classes (BUY/SELL)

### Step 2: Use SHAP to Find Top 20 Features (10 min)
```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
feature_importance = np.abs(shap_values).mean(axis=0)
top_20 = X.columns[np.argsort(feature_importance)[-20:]]

# Retrain with only top 20
X_clean = X[top_20]
```

**Expected Gain:** +2-3% by reducing noise

### Step 3: Add Class Weights to XGBoost (2 min)
```python
# Count label distribution
unique, counts = np.unique(y, return_counts=True)
class_weights = 1.0 / (counts / counts.min())

model = xgb.XGBClassifier(
    scale_pos_weight=class_weights[1] / class_weights[0]  # For binary
)
```

**Expected Gain:** +2-3% on imbalanced classes

**Total Expected:** 43% → 50-51%

---

## 🚀 ONE-WEEK SPRINT: 43% → 55%+

### Day 1: Implement Purged Cross-Validation
- Replace your CV with purged CV
- You'll likely see accuracy drop to 40-42%
- This is the TRUTH; your 43% had leakage
- Now you have realistic baseline

### Days 2-3: Multi-Task Learning
- Use PyTorch code from `colab-trading-impl.py` Block 7
- Train model to predict: (1) direction, (2) magnitude, (3) confidence
- Share encoder layers for better generalization

**Expected Gain:** +4-6%

### Days 4-5: Feature Engineering Redux
- Remove redundant indicators (EMA5/10/20 are 95% correlated)
- Add: Order flow imbalance, volatility regime, cross-asset correlation
- Keep only SHAP top-30

**Expected Gain:** +2-3%

### Days 6-7: Evaluate & Deploy
- Run full Combinatorial Purged CV
- Calculate Sharpe ratio, max drawdown
- Deploy to frontend

**Expected Accuracy:** 50-55%
**Expected Sharpe:** 0.7-1.0

---

## 🏆 ONE-MONTH GRIND: 43% → 60-65%+

### Week 1: Foundation
- Purged CV + Focal Loss + SHAP selection = 50-52%

### Week 2: Multi-Task
- Add magnitude + confidence prediction = 52-55%

### Week 3: Architecture
- Implement TFT (Temporal Fusion Transformer)
- Or CNN-GRU-XGBoost hybrid
- Test on validation set = 55-60%

### Week 4: Ensemble
- Combine TFT + XGBoost meta-learner
- Add regime detection (bull/bear)
- Dynamic feature selection per regime = 60-65%

**Total Expected:** 60-65% accuracy, 1.0-1.5 Sharpe ratio

---

## ⚙️ ARCHITECTURE DECISION TREE

```
START: What's your compute?
  ├─ GPU (T4/A100): Use TFT or Informer
  ├─ CPU only: Use LightGBM + SHAP
  └─ Balanced: Use CNN-GRU-XGBoost hybrid

Is accuracy your only metric?
  ├─ YES: Push for 65%+ (harder)
  └─ NO: Optimize Sharpe ratio (easier, >1.0 is good)

Do you have alternative data (sentiment, options)?
  ├─ YES: Add FinBERT + multi-modal transformer = +5-7%
  └─ NO: Stick to price/volume = 55-60% realistic

Do you need interpretability?
  ├─ YES: Use TFT (attention weights) + SHAP
  └─ NO: Use TabNet or raw neural networks

Timeline pressure?
  ├─ 1 week: Just fix CV + focal loss + SHAP
  ├─ 2-3 weeks: Add multi-task + light TFT
  └─ 1 month+: Full TFT + ensemble + regime
```

---

## 📈 EXPECTED ACCURACY BY TECHNIQUE

| Technique | Implementation Time | Expected Gain | Notes |
|-----------|---|---|---|
| Fix look-ahead bias (CPCV) | 2 hours | -3 to -5% | Reveals truth |
| Focal loss | 30 min | +3-5% | Fixes imbalance |
| SHAP feature selection | 1 hour | +2-3% | Reduces noise |
| Multi-task learning | 1 day | +4-6% | Best bang for buck |
| TFT architecture | 2-3 days | +5-8% | Captures sequences |
| Regime detection | 1 day | +2-4% | Context-aware |
| Ensemble | 1 day | +2-3% | Combines strengths |
| Sentiment data | 3-5 days | +3-5% | If available |

**Total Realistic Stacking:** 43% → 55-60% (3-4 weeks)

---

## 🔴 RED FLAGS TO AVOID

### Feature Engineering
- ❌ Don't use 150 features; 30-50 optimal for financial data
- ❌ Don't include future information (close tomorrow, earnings dates)
- ❌ Don't use indicator crosses in training that happen after label

### Model Training
- ❌ Don't use standard K-fold (use purged CV)
- ❌ Don't train on 2016-2022 bull market only
- ❌ Don't ignore class imbalance (HOLD>70% in most data)
- ❌ Don't hyperparameter-tune on test set

### Evaluation
- ❌ Don't evaluate on accuracy alone (Sharpe ratio matters more)
- ❌ Don't ignore transaction costs (0.05% kills profitability)
- ❌ Don't backtest on survivorship-biased data

### Deployment
- ❌ Don't trade signals with <55% confidence
- ❌ Don't retrain infrequently (model decays weekly)
- ❌ Don't ignore regime shifts (model performs differently in bear markets)

---

## 💡 DEBUGGING CHECKLIST

**If accuracy plateaus at 48%:**
- [ ] Check Gini coefficient (data quality measure)
- [ ] Verify no future data in features
- [ ] Ensure labels are correctly shifted
- [ ] Run purged CV (not standard CV)

**If accuracy drops from 48% to 45%:**
- [ ] You likely found look-ahead bias
- [ ] Recheck feature engineering
- [ ] Use future information? (common mistake)

**If training accuracy 65% but test 48%:**
- [ ] Severe overfitting; reduce model complexity
- [ ] Use temporal dropout
- [ ] Increase regularization (L1/L2)

**If Sharpe ratio is 0.3 despite 52% accuracy:**
- [ ] Model is picking low-probability signals
- [ ] Filter by confidence (only trade >60% prob signals)
- [ ] Account for transaction costs

---

## 🎮 QUICK WINS (Ranked by Effort/Return)

1. **Purged CV** (2 hours) → Reveals true baseline
2. **Focal Loss** (30 min) → +3-5%
3. **SHAP Top-20** (1 hour) → +2-3%
4. **Remove redundant features** (30 min) → +1-2%
5. **Multi-task learning** (1 day) → +4-6%
6. **TFT architecture** (3 days) → +5-8%

**Do these in order; each builds on previous.**

---

## 📞 WHEN TO GIVE UP / PIVOT

If after implementing items 1-5 you're still at 48%:
- [ ] Consider data quality issue (survivorship bias?)
- [ ] Check if ±2% label threshold is realistic
- [ ] Verify cross-asset data is aligned
- [ ] Try different timeframe (1-day instead of 5-day)

**Alternative pivots:**
- Switch to Sharpe ratio optimization (easier than accuracy)
- Focus on risk-adjusted returns (max drawdown <15%)
- Trade only high-confidence signals (>65% probability)

---

## 🚀 DEPLOYMENT CHECKLIST

Before pushing to production:

- [ ] CPCV accuracy ≥ 50%
- [ ] Sharpe ratio ≥ 0.7
- [ ] Max drawdown ≤ 20%
- [ ] Feature stability (top-10 features same across folds)
- [ ] Model retrained weekly
- [ ] Transaction costs included in P&L
- [ ] Regime detection active (switch models in bear markets)
- [ ] Confidence filtering (skip signals <60% conf)

---

**Document:** Quick Reference v1.0  
**Last Updated:** December 2025  
**Target:** 60%+ accuracy on 5-day swing trading
