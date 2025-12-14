# Perplexity AI Research Request: Low ML Accuracy Diagnosis & Solutions

## Current Problem
We built a stock price forecasting system with 62 engineered features (Gentile + AlphaGo + Technical indicators) using XGBoost, LightGBM, and CatBoost ensemble, but we're getting **only 44-45% accuracy** on 3-class classification (SELL/HOLD/BUY), which is barely better than random guessing (33%).

## Our Current Implementation

### 1. Data & Features
- **Dataset**: 56 large-cap stocks (AAPL, MSFT, GOOGL, etc.) with 2 years of daily OHLCV data
- **Features**: 62 total
  - 16 Gentile features (margin violation detection, MA crosses, volatility adaptation)
  - 24 AlphaGo features (hierarchical game-state: board position, trend strength, volatility state, support/resistance, volume, reversion signals)
  - 22 Technical features (RSI, MACD, Bollinger Bands, Stochastic, ADX, OBV, Ichimoku, price patterns)
- **Window size**: 60 days of historical data per sample
- **Feature scaling**: StandardScaler applied to all features

### 2. Labeling Strategy (Triple Barrier)
```python
forecast_horizon = 7 days
buy_threshold = +3%   # Label = BUY (2)
sell_threshold = -3%  # Label = SELL (0)
# Otherwise = HOLD (1)
```

**Current label distribution after feature engineering:**
- SELL (0): ~20-25%
- HOLD (1): ~50-55%
- BUY (2): ~25-30%

### 3. Training Pipeline
```python
# Time-aware split (no shuffling)
Train: 70% (oldest data)
Val: 15% (middle data)  
Test: 15% (newest data)

# Class balancing
SMOTE applied to training set only with k_neighbors=3

# Models trained with Optuna hyperparameter optimization
XGBoost: 75 trials, early_stopping_rounds=50
LightGBM: 75 trials, early_stopping_rounds=50
CatBoost: 50 trials (GPU), early_stopping_rounds=50

# Ensemble: Stacking with LogisticRegression meta-learner
# Confidence calibration: IsotonicRegression on validation set
```

### 4. Results (TERRIBLE)
```
XGBoost Test Accuracy: ~44%
LightGBM Test Accuracy: ~45%
CatBoost Test Accuracy: ~43%
Ensemble Test Accuracy: ~45%

Classification Report:
              precision    recall  f1-score   support
SELL (0)         0.25      0.25      0.25       725
HOLD (1)         0.57      0.59      0.58      1899
BUY (2)          0.34      0.32      0.33       965
```

## Critical Questions for Research

### 1. **Are our thresholds fundamentally wrong?**
- Is ±3% over 7 days too aggressive/loose for stock forecasting?
- Should we use different thresholds for different volatility regimes?
- What threshold ranges do successful quant funds actually use?
- Should we use asymmetric thresholds (e.g., +2% BUY, -4% SELL)?

### 2. **Is our labeling strategy flawed?**
- Triple barrier vs. forward returns: which is more realistic?
- Should we use percentage change vs. absolute price targets?
- Are we introducing look-ahead bias somehow?
- Should we label based on risk-adjusted returns instead?

### 3. **Is SMOTE destroying our signal?**
- Does SMOTE create unrealistic synthetic samples for time-series?
- Should we use class weights instead of oversampling?
- Is there a better resampling method for financial data (ADASYN, BorderlineSMOTE)?
- Should we just accept class imbalance and use stratified sampling?

### 4. **Feature engineering issues?**
- Are 60-day windows too long/short for daily data?
- Should we use multiple timeframes (5/10/20/60 days)?
- Are we missing critical features (sentiment, volume profile, market regime)?
- Should we use feature selection to remove noise?

### 5. **Model architecture problems?**
- Are gradient boosting models appropriate for this task?
- Should we use sequential models (LSTM, Transformer) instead?
- Is our ensemble too simple (just stacking)?
- Should we use different models for different market regimes?

### 6. **Data quality concerns?**
- Is 2 years enough data? Should we use 5-10 years?
- Are we mixing different market regimes (bull/bear/sideways) badly?
- Should we train separate models per sector?
- Do we need to normalize by market (SPY returns)?

### 7. **Realistic expectations?**
- What accuracy do professional quant firms actually achieve?
- Is 70%+ accuracy on 7-day forecasts even possible?
- Should we focus on fewer, higher-confidence predictions?
- What metrics matter more than accuracy (Sharpe, win rate, profit factor)?

## What We Need From You

Please research and provide:

1. **Root cause analysis**: What's most likely causing 44% accuracy?
2. **Industry benchmarks**: What accuracy do real quant systems achieve for 7-day stock forecasts?
3. **Specific fixes ranked by impact**:
   - Most likely to improve results
   - Backed by research papers or industry practice
   - With concrete implementation details

4. **Alternative approaches**: If our fundamental architecture is wrong, what should we use instead?
   - LSTM/GRU for sequences?
   - Transformers (Temporal Fusion Transformer)?
   - Factor models + ML?
   - Regime-switching models?

5. **Realistic targets**: Given our setup (56 stocks, 7-day horizon, daily data), what accuracy/Sharpe ratio should we aim for?

6. **Quick wins**: What are the top 3 changes we can make RIGHT NOW to see improvement?

## Research Focus Areas

Please search for:
- Academic papers on stock return prediction with ML (2020-2024)
- Industry practices from top quant firms (Two Sigma, Renaissance, Citadel)
- Kaggle competitions on stock forecasting (winning solutions)
- Financial ML books (Marcos López de Prado - "Advances in Financial Machine Learning")
- Time series classification benchmarks
- Label engineering techniques for imbalanced financial data
- Feature importance studies in stock prediction

## Expected Output Format

```markdown
## Root Cause Analysis
[Main issues causing low accuracy]

## Industry Benchmarks
[What professionals actually achieve]

## Recommended Fixes (Ranked)
1. [Highest impact fix with details]
2. [Second highest impact]
3. [Third highest impact]

## Alternative Architectures
[If needed, what to try instead]

## Realistic Targets
[What we should aim for]

## Implementation Roadmap
[Step-by-step plan to fix this]
```

---

**URGENT**: We need actionable solutions to get from 44% to at least 60-65% accuracy, or understand why that's impossible and what we should do instead.
