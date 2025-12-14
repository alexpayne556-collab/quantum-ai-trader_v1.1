# 📈 Forecaster Optimization Roadmap

## Current Status

**Forecaster Performance:**
- Direction Accuracy: 57.4% (target: 65-70%)
- MAE: $20.54
- 5% Hit Rate: 59.2%
- Status: ⚠️ NEEDS IMPROVEMENT

---

## 🎯 Optimization Strategy

### Phase 1: Feature Engineering (Quickest Wins)

**Add High-Value Features:**

1. **Volume Patterns** (2-3% accuracy gain expected)
   - Volume momentum (5/10/20 day)
   - Volume vs average (10/20/50 day)
   - Money flow index
   - On-balance volume

2. **Volatility Features** (1-2% gain)
   - ATR (14-day)
   - Bollinger Band width
   - Historical volatility (10/20 day)
   - Volatility percentile

3. **Momentum Indicators** (2-3% gain)
   - RSI (14-day)
   - MACD (12,26,9)
   - Stochastic oscillator
   - Rate of change

4. **Trend Strength** (1-2% gain)
   - ADX (Average Directional Index)
   - Linear regression slope
   - R-squared of trend
   - Trend consistency score

5. **Market Context** (1-2% gain)
   - SPY correlation
   - Sector relative strength
   - VIX level
   - Market regime (from regime detector)

**Expected Total Gain: 7-12% → Target 64-69% accuracy**

---

### Phase 2: Advanced Models (Colab Pro)

**Temporal CNN-LSTM Architecture:**

```python
class TemporalForecaster(nn.Module):
    def __init__(self):
        # CNN for pattern extraction
        self.conv1 = nn.Conv1d(features, 64, kernel_size=5)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3)
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(128, 256, num_layers=2, dropout=0.3)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(256, heads=8)
        
        # Output layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 24)  # 24-day forecast
```

**Expected Gain: 3-5% → Target 67-74% accuracy**

---

### Phase 3: Ensemble Forecasting

**Multi-Horizon Ensemble:**

1. **Short-Term (1-5 days)**
   - Optimized for quick moves
   - High sensitivity to momentum
   - Uses recent data heavily

2. **Medium-Term (5-20 days)**
   - Balanced approach
   - Pattern recognition focus
   - Current forecaster here

3. **Long-Term (20-60 days)**
   - Trend-following
   - Sector rotation aware
   - Fundamental signals

**Ensemble Method:**
- Weighted average by horizon
- Confidence-based blending
- Regime-dependent weights

**Expected Gain: 2-4% → Target 69-78% accuracy**

---

## 🚀 Implementation Plan

### Week 1: Feature Engineering

```bash
# 1. Create feature engineering module
python create_advanced_features.py

# 2. Retrain forecaster with new features
python train_forecaster_v2.py --features advanced

# 3. Validate improvement
python validate_forecaster.py
```

**Files to Create:**
- `advanced_features.py` - Feature calculators
- `train_forecaster_v2.py` - Enhanced trainer
- `forecaster_v2.pkl` - New model

---

### Week 2: Temporal CNN-LSTM (Colab Pro)

```python
# Upload to Colab Pro
# COLAB_FORECASTER_OPTIMIZER.ipynb

# Train with GPU
trainer = TemporalForecasterTrainer(
    sequence_length=60,
    forecast_horizon=24,
    features=advanced_features,
    epochs=100,
    batch_size=128
)

# Save best model
torch.save(model.state_dict(), 'temporal_forecaster.pth')
```

**Expected Training Time:** 2-3 hours on T4 GPU

---

### Week 3: Ensemble Integration

```python
# Create ensemble forecaster
class EnsembleForecaster:
    def __init__(self):
        self.short_term = ShortTermForecaster()  # 1-5 days
        self.medium_term = MediumTermForecaster()  # 5-20 days
        self.long_term = LongTermForecaster()  # 20-60 days
    
    def predict(self, ticker, horizon=24):
        # Get predictions from each
        short = self.short_term.predict(ticker, min(horizon, 5))
        medium = self.medium_term.predict(ticker, min(horizon, 20))
        long = self.long_term.predict(ticker, horizon)
        
        # Weighted ensemble
        weights = self._calculate_weights(ticker, horizon)
        return weighted_average([short, medium, long], weights)
```

---

## 📊 Validation Strategy

### Metrics to Track:

1. **Direction Accuracy**
   - Target: 65-70% (currently 57.4%)
   - Measure: % of correct up/down predictions

2. **MAE (Mean Absolute Error)**
   - Target: <$15 (currently $20.54)
   - Measure: Average $ error per prediction

3. **5% Hit Rate**
   - Target: 65-70% (currently 59.2%)
   - Measure: % predictions within 5% of actual

4. **Sharpe Ratio (New)**
   - Target: >1.5
   - Measure: Risk-adjusted returns

5. **Max Drawdown (New)**
   - Target: <15%
   - Measure: Worst peak-to-trough

### Validation Process:

```python
# Out-of-sample validation
# Test on last 3 months of data (not used in training)
validation_period = "2025-09-01 to 2025-12-01"

# Walk-forward validation
# Train on rolling window, test next period
for month in months:
    train_data = get_data(month - 12, month)
    test_data = get_data(month, month + 1)
    
    model.train(train_data)
    results = model.test(test_data)
    track_metrics(results)
```

---

## 🎯 Target Performance

### Current vs Target:

| Metric | Current | Phase 1 | Phase 2 | Phase 3 | Target |
|--------|---------|---------|---------|---------|--------|
| Direction Accuracy | 57.4% | 64-69% | 67-74% | 69-78% | **70%+** |
| MAE | $20.54 | $16-18 | $13-15 | $10-13 | **<$15** |
| 5% Hit Rate | 59.2% | 65-67% | 68-72% | 70-75% | **70%+** |
| Sharpe Ratio | N/A | 1.2 | 1.5 | 1.8+ | **>1.5** |

---

## 💡 Quick Wins (Do First)

### 1. Add Volume Features (1 day)

```python
# In forecast_engine.py
def add_volume_features(df):
    df['volume_ma_10'] = df['Volume'].rolling(10).mean()
    df['volume_ma_20'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma_20']
    df['volume_trend'] = df['volume_ma_10'] / df['volume_ma_20']
    return df
```

**Expected: +2% accuracy**

### 2. Add Volatility Features (1 day)

```python
def add_volatility_features(df):
    df['returns'] = df['Close'].pct_change()
    df['volatility_10'] = df['returns'].rolling(10).std()
    df['volatility_20'] = df['returns'].rolling(20).std()
    df['atr_14'] = calculate_atr(df, 14)
    return df
```

**Expected: +1-2% accuracy**

### 3. Add Market Context (1 day)

```python
def add_market_context(df, ticker):
    # Get SPY data
    spy = yf.download('SPY', period='1y')
    
    # Calculate correlation
    df['spy_correlation'] = df['Close'].rolling(60).corr(spy['Close'])
    
    # Get sector strength
    sector = get_sector(ticker)
    df['sector_strength'] = get_sector_strength(sector)
    
    return df
```

**Expected: +1-2% accuracy**

### Total Quick Wins: +4-6% accuracy in 3 days!

---

## 📝 Next Steps

1. ✅ **Trained ML ensemble** (74.6% accuracy) - DONE
2. ✅ **Portfolio-aware system** - DONE
3. ⏳ **Quick wins: Add volume/volatility/market features** (3 days)
4. ⏳ **Retrain forecaster with new features** (1 day)
5. ⏳ **Validate improvements** (1 day)
6. ⏳ **Temporal CNN-LSTM in Colab Pro** (1 week)
7. ⏳ **Build ensemble forecaster** (3 days)
8. ⏳ **Final validation and deployment** (2 days)

**Total Timeline: 2-3 weeks to 70%+ forecaster accuracy**

---

## 🚀 Ready to Start

**Immediate Action:**
```bash
# 1. Create advanced features module
python create_forecaster_features.py

# 2. Retrain with new features
python train_forecaster_v2.py

# 3. Test on your watchlist
python test_forecaster_on_watchlist.py
```

**Goal:** Get forecaster from 57% → 70%+ accuracy on YOUR tickers!
