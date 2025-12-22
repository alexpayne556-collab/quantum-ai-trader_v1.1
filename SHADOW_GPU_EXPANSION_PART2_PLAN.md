# SHADOW GPU EXPANSION - PART 2 PLAN
**Machine Learning Feature Engineering & Ensemble Methods**

---

## OBJECTIVE
Test 2,000-3,000 machine learning strategies that combine discoveries from Part 1 using non-linear methods, ensemble techniques, and advanced feature engineering.

---

## KEY INSIGHTS FROM PART 1

### Top Performing Patterns:
1. **Fibonacci retracements**: 82.9% significant (highest!)
2. **Ichimoku alignment**: t=131.57 (strongest individual signal!)
3. **Volatility regimes**: 72.4% significant
4. **EMA compression**: Predicts volatility breakouts

### Why These Work:
- **Fibonacci**: Self-fulfilling prophecy + natural human pattern recognition
- **Ichimoku**: Multi-timeframe trend + momentum + mean reversion in one system
- **Volatility regimes**: Volatility clustering (GARCH effects)
- **EMA compression**: Low vol → high vol transition with directional bias

---

## PART 2 STRATEGY CATEGORIES

### 1. NON-LINEAR COMBINATIONS (500 strategies)

Use XGBoost/LightGBM to find non-linear relationships:

**Features to combine:**
- Top 20 signals from Part 1 (by t-statistic)
- Fibonacci proximity to levels (continuous, not binary)
- Ichimoku component distances (not just above/below)
- EMA distances (5-200 day spread)
- Volatility regime probabilities (not just high/low)
- RSI multi-timeframe alignment scores
- Momentum acceleration (2nd derivative)

**XGBoost strategies:**
- Binary classification (up/down next H days)
- Regression (predict forward return magnitude)
- Ranking (sort stocks by predicted return)
- Feature importance analysis (which features matter most?)

**Boosting variations:**
- Standard XGBoost
- LightGBM (faster, handles large datasets)
- CatBoost (handles categorical features)
- Gradient boosting with different loss functions

**Test configurations:**
- Different max depths (3, 5, 7, 10)
- Different learning rates (0.01, 0.05, 0.1)
- Different n_estimators (50, 100, 200)
- Different feature subsampling ratios

---

### 2. ENSEMBLE METHODS (400 strategies)

Combine multiple weak learners into strong learners:

**Bagging approaches:**
- Random forest on top 50 features from Part 1
- Extra trees (extremely randomized trees)
- Bootstrap aggregating with different samples

**Stacking approaches:**
- Level 1: Fibonacci + Ichimoku + Volatility + Momentum models
- Level 2: Meta-learner combines Level 1 predictions
- 3-layer stacking (base → meta → final)

**Voting methods:**
- Majority voting (3+ signals agree)
- Weighted voting (weight by t-statistic from Part 1)
- Ranked voting (top K predictions)

**Combinations to test:**
- Fibonacci + Ichimoku (why do both work so well?)
- Fibonacci + Volatility regime
- Ichimoku + EMA alignment
- Momentum + mean reversion (contrarian approach)
- All 5 categories weighted ensemble

---

### 3. FEATURE ENGINEERING (500 strategies)

Create advanced features from Part 1 signals:

**Interaction terms:**
- Fibonacci proximity × Volatility level
- Ichimoku alignment × Momentum strength
- EMA compression × Volume surge
- RSI oversold × High volatility
- Low volatility × Strong momentum (anomaly!)

**Polynomial features:**
- Quadratic terms (momentum²)
- Cubic terms (volatility³)
- Cross products (feature A × feature B)

**Ratio features:**
- Fibonacci level / ATR (significance relative to noise)
- Momentum / Volatility (risk-adjusted momentum)
- Volume / Average volume (relative activity)
- Price / EMA200 (trend strength)

**Time-based features:**
- Signal persistence (how many days has signal been active?)
- Signal freshness (just triggered vs ongoing)
- Signal acceleration (is signal strengthening?)
- Historical signal performance (how often does this signal win?)

**Statistical features:**
- Rolling z-scores of all features
- Percentile ranks within ticker history
- Cross-sectional ranks (vs all tickers)
- Correlation to market (SPY)

---

### 4. DIMENSIONALITY REDUCTION (300 strategies)

Compress 100+ features from Part 1 into key components:

**PCA (Principal Component Analysis):**
- Extract top 10 principal components
- Test PC1 (market factor)
- Test PC2 (size/value factor)
- Test PC3-10 (idiosyncratic factors)

**Factor analysis:**
- Extract latent factors
- Rotate factors for interpretability
- Test each factor as predictor

**t-SNE/UMAP:**
- Non-linear dimensionality reduction
- Cluster similar market states
- Test within-cluster vs across-cluster signals

**Autoencoders:**
- Neural network compression
- Latent space representation
- Decode to predict returns

---

### 5. TIME-SERIES SPECIFIC ML (300 strategies)

Leverage sequential nature of data:

**LSTM (Long Short-Term Memory):**
- Sequence-to-one prediction
- Look back 20/60/100 days
- Predict next H-day return

**GRU (Gated Recurrent Unit):**
- Simpler than LSTM, faster training
- Similar architecture variations

**Temporal Convolutional Networks:**
- 1D convolutions on time series
- Different kernel sizes (3, 5, 10)

**Attention mechanisms:**
- Self-attention on feature importance
- Multi-head attention (which timeframe matters?)
- Transformer architecture

**WaveNet-style models:**
- Dilated convolutions
- Capture long-range dependencies

---

## IMPLEMENTATION DETAILS

### Data Preparation:
```python
# Features from Part 1
features = [
    'ichimoku_bullish_alignment',
    'ichimoku_bearish_alignment', 
    'fib_golden_zone',
    'ema_compressed',
    'vol_regime_low',
    'rsi_oversold',
    # ... top 50 features by t-stat
]

# Create feature matrix
X = df[features].values
y = df['fwd_10'].values  # Example: 10-day forward return

# Split by time (no lookahead bias)
train_mask = df['date'] < '2024-07-01'
test_mask = df['date'] >= '2024-07-01'

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]
```

### XGBoost Example:
```python
import xgboost as xgb

# Binary classification (up/down)
model = xgb.XGBClassifier(
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100,
    objective='binary:logistic',
    tree_method='gpu_hist',  # GPU acceleration!
    gpu_id=0
)

model.fit(X_train, (y_train > 0).astype(int))
predictions = model.predict_proba(X_test)[:, 1]

# Test strategy: go long on top quintile predictions
signals = predictions > np.percentile(predictions, 80)
returns = y_test[signals]

# Calculate t-statistic
mean_ret, n, t_stat = calc_t_fast(returns)
```

### Ensemble Example:
```python
from sklearn.ensemble import VotingClassifier

# Combine top 3 models from Part 1
fib_model = LogisticRegression()  # Fibonacci features
ich_model = LogisticRegression()  # Ichimoku features  
vol_model = LogisticRegression()  # Volatility features

ensemble = VotingClassifier(
    estimators=[
        ('fib', fib_model),
        ('ich', ich_model),
        ('vol', vol_model)
    ],
    voting='soft',  # Use probabilities
    weights=[0.829, 0.747, 0.724]  # Weight by Part 1 hit rates!
)

ensemble.fit(X_train, y_train)
```

---

## GPU OPTIMIZATION

### PyTorch GPU Tensors:
```python
# Convert to GPU tensors for faster computation
X_gpu = torch.tensor(X, device='cuda', dtype=torch.float32)
y_gpu = torch.tensor(y, device='cuda', dtype=torch.float32)

# Batch processing
batch_size = 10000
for i in range(0, len(X_gpu), batch_size):
    X_batch = X_gpu[i:i+batch_size]
    # Process batch on GPU
```

### XGBoost GPU:
```python
# Use GPU histogram method
params = {
    'tree_method': 'gpu_hist',
    'gpu_id': 0,
    'predictor': 'gpu_predictor'
}
```

### Numba GPU Kernels:
```python
from numba import cuda

@cuda.jit
def compute_features_gpu(close, ema_5, ema_20, result):
    idx = cuda.grid(1)
    if idx < close.shape[0]:
        result[idx] = (close[idx] - ema_5[idx]) / ema_20[idx]
```

---

## EXPECTED RESULTS

### Realistic Expectations:
- Part 1 hit rate: 66.7%
- Part 2 hit rate: 70-75% (ML should improve on raw signals)
- Top models: t-stats > 150 (beating Part 1's max of 131.57)
- Feature importance insights: Understand which laws are most predictive

### Key Questions to Answer:
1. Do non-linear combinations beat linear signals?
2. Which features matter most (XGBoost feature importance)?
3. Are Fibonacci + Ichimoku complementary or redundant?
4. Can we predict magnitude (not just direction)?
5. Do ensemble methods reduce overfitting?

---

## VALIDATION STRATEGY

### Walk-Forward:
- Train on 2023-06 to 2024-06
- Test on 2024-07 to 2024-12
- Retrain quarterly

### Cross-Validation:
- Time-series split (not random!)
- Purged K-fold (remove overlapping forward returns)
- Combinatorial purged CV

### Robustness Checks:
- Parameter sensitivity (does edge survive different settings?)
- Ticker holdout (train on 80% tickers, test on 20%)
- Regime splits (bull vs bear)

---

## OUTPUT FORMAT

Same as Part 1 for consistency:
```csv
category,strategy,avg_return,n_samples,t_stat,significant,source,hold_period
ML_XGBOOST,XGB_Top20Features_MaxDepth5_H10,0.0234,125000,87.23,True,GPU_EXPANSION_PART2,10
ML_ENSEMBLE,FibIchimokuVol_WeightedVote_H15,0.0456,98000,102.45,True,GPU_EXPANSION_PART2,15
...
```

---

## TIMELINE

- **Day 1** (Dec 23): Implement XGBoost + ensemble infrastructure
- **Day 2** (Dec 24): Run 1,000 XGBoost variations overnight
- **Day 3** (Dec 25): Analyze, implement feature engineering
- **Day 4** (Dec 26): Run 1,000 feature engineering strategies overnight
- **Total**: 2,000-3,000 new strategies → **10,000 total strategies achieved!**

---

## CODE STRUCTURE

### SHADOW_GPU_EXPANSION_PART2.py:
```python
# 1. Load Part 1 results + identify top features
# 2. Build feature matrix from top 50 signals
# 3. XGBoost hyperparameter sweep (500 configs)
# 4. Ensemble methods (400 configs)
# 5. Feature engineering (500 configs)
# 6. Dimensionality reduction (300 configs)
# 7. Time-series models (300 configs)
# 8. Save to data/GPU_EXPANSION_PART2.csv
```

---

**Ready to implement? This will take us to 10,000+ strategies and reveal which combinations of laws work best together. 🚀**
