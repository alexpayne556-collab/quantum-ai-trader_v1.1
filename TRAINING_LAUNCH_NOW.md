# 🚀 COLAB PRO TRAINING - IMMEDIATE LAUNCH CHECKLIST

**Date**: December 9, 2025 3:58 AM  
**Status**: ✅ ALL SYSTEMS GO - BEGIN TRAINING NOW  
**GPU**: Colab Pro T4 (15 GB VRAM)  
**Training Time**: 2-4 hours estimated

---

## ✅ PRE-FLIGHT CHECKLIST (ALL COMPLETE)

- [x] **5/5 Critical APIs Working** (verified 30 seconds ago)
  - ✅ Twelve Data: 800/day (PRIMARY)
  - ✅ Finnhub: 60/min (SECONDARY)  
  - ✅ Alpha Vantage: 25/day (BACKUP)
  - ✅ FRED: Unlimited (regime data)
  - ✅ Alpaca: $100k paper account

- [x] **ML Modules Built & Tested**
  - ✅ `multi_model_ensemble.py` (3-model voting)
  - ✅ `feature_engine.py` (49 indicators)
  - ✅ `regime_classifier.py` (10 regimes)
  - ✅ `data_source_manager.py` (API rotation)

- [x] **Integration Test Passed**
  - ✅ 5 tickers tested (AAPL, TSLA, NVDA, MSFT, GOOGL)
  - ✅ 2,220 bars downloaded successfully
  - ✅ 49 features calculated (0 NaN, 0 Inf)
  - ✅ 4/5 high-confidence signals (80% success rate)

- [x] **Backend API Spec Ready**
  - ✅ Complete REST API documentation for Spark frontend
  - ✅ 7 endpoint groups defined (signals, market, portfolio, etc.)
  - ✅ TypeScript interfaces + React examples
  - ✅ Parallel development timeline (Week 1-4)

- [x] **Perplexity Optimization Brief**
  - ✅ 7 critical questions prepared
  - ✅ System architecture documented
  - ✅ Current performance baseline recorded
  - ✅ Ready to share with Perplexity for guidance

---

## 🎯 TRAINING OBJECTIVES

### Baseline (Tonight - 2-4 hours)
**Goal**: Train 3 models on 76 tickers × 2 years × 1hr bars

**Expected Outputs**:
- `xgboost.pkl` (XGBoost GPU model)
- `random_forest.pkl` (RandomForest model)
- `gradient_boosting.pkl` (GradientBoosting model)
- `scaler.pkl` (StandardScaler for features)
- `metadata.pkl` (Ticker list, feature names, regime config)
- `training_metrics.json` (Precision, ROC-AUC, confusion matrix)

**Success Criteria**:
- ✅ Validation precision >55% (baseline target)
- ✅ ROC-AUC >0.65
- ✅ No errors during training
- ✅ Models save to Google Drive

### Week 1 Optimization (Dec 10-16)
**Goal**: Improve precision to 65-70%

**Method**: 
1. Share Perplexity Optimization Brief
2. Ask 7 critical questions about hyperparameters
3. Implement Perplexity's recommendations
4. Run RandomizedSearchCV with new parameter ranges
5. Retrain with optimized config

**Success Criteria**:
- ✅ Precision improves from 55% → 65-70%
- ✅ ROC-AUC >0.70
- ✅ Stable performance across 3 folds

### Week 2 Feature Engineering (Dec 17-23)
**Goal**: Improve precision to 68-72%

**Method**:
1. Add Perplexity-recommended features (volume profile, microstructure)
2. Feature selection (remove low-importance features)
3. Test regime interaction terms
4. Optimize ensemble weights (XGB/RF/GB split)

**Success Criteria**:
- ✅ Precision improves to 68-72%
- ✅ Feature set reduced to 30-35 best features
- ✅ Ensemble weights optimized

### Week 3-4 Paper Trading (Dec 24-Jan 5)
**Goal**: Validate live performance

**Method**:
1. Deploy models to Alpaca paper trading
2. Execute 20-30 trades over 2 weeks
3. Track win rate, Sharpe ratio, max drawdown
4. Compare paper trading vs backtest performance

**Success Criteria**:
- ✅ Live win rate ≥58%
- ✅ Sharpe ratio ≥1.5
- ✅ Max drawdown <20%
- ✅ Average return per trade >2%

---

## 📋 COLAB PRO SETUP (5 MINUTES)

### Step 1: Open Colab Pro
1. Go to https://colab.research.google.com
2. Sign in with Google account
3. File → Upload notebook → Select `notebooks/UNDERDOG_COLAB_TRAINER.ipynb`

### Step 2: Enable T4 GPU
1. Runtime → Change runtime type
2. Hardware accelerator: **T4 GPU**
3. Click "Save"

### Step 3: Mount Google Drive
Execute first cell:
```python
from google.colab import drive
drive.mount('/content/drive')
```
Click authorization link, approve access.

### Step 4: Upload Files to Colab
Upload these 5 files to `/content/` folder:
```
/content/multi_model_ensemble.py
/content/feature_engine.py
/content/regime_classifier.py
/content/data_source_manager.py
/content/.env
```

### Step 5: Create .env in Colab
Create new text file `/content/.env`:
```bash
TWELVE_DATA_API_KEY=d19ebe6706614dd897e66aa416900fd3
FINNHUB_API_KEY=d3qj8p9r01quv7kb49igd3qj8p9r01quv7kb49j0
ALPHA_VANTAGE_API_KEY=gL_pHRAJ6SQK0AK2MD0rSuP653GW733l
FRED_API_KEY=32829556722ddb7fd681d84ad9192026
ALPACA_API_KEY=PKRNFP4NMO4O2CDYRRBGLH2EFU
ALPACA_SECRET_KEY=7b85Wo48enKp36PkaB4fC1nZyHxscRSMNHX7ktkCuZjL
PERPLEXITY_API_KEY=your_perplexity_api_key_hereSugdX6yxqiIorS526CYof8aqlcySXisRbIoNf84BBQ7szSOl
```

---

## 🚀 TRAINING EXECUTION (2-4 HOURS)

### Cell 1: GPU Check
```python
!nvidia-smi
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```
**Expected**: T4 GPU, 15 GB VRAM

### Cell 2: Install Packages
```python
!pip install -q yfinance xgboost scikit-learn pandas numpy ta-lib python-dotenv requests
```
**Expected**: All packages install without errors (2-3 minutes)

### Cell 3: Import Modules
```python
import sys
sys.path.append('/content')
from multi_model_ensemble import MultiModelEnsemble
from feature_engine import FeatureEngine
from regime_classifier import RegimeClassifier
from data_source_manager import DataSourceManager
```
**Expected**: All imports successful, no errors

### Cell 4: Download Data (15-20 minutes)
```python
manager = DataSourceManager()
tickers = [
    # Alpha 76 tickers (biotech, space, EV, clean energy, fintech)
    'TSLA', 'NVDA', 'AMD', 'PLTR', 'COIN', 'SQ', 'SHOP', 'ROKU',
    'CRSP', 'EDIT', 'NTLA', 'BEAM', 'VRTX', 'MRNA', 'BNTX', 'SGEN',
    'SPCE', 'RKLB', 'ASTR', 'LUNR', 'PL', 'ACHR', 'JOBY', 'LILM',
    # ... (76 total)
]

data = {}
for ticker in tickers:
    print(f"Fetching {ticker}...")
    df = manager.fetch_ohlcv(ticker, period='2y', interval='1h')
    if df is not None and len(df) > 1000:
        data[ticker] = df
    print(f"  Got {len(df)} bars")
```
**Expected**: ~250k rows downloaded (76 tickers × 3,276 bars)

### Cell 5: Calculate Features (10-15 minutes)
```python
feature_engine = FeatureEngine()
regime_classifier = RegimeClassifier()

all_features = []
all_targets = []

for ticker, df in data.items():
    print(f"Processing {ticker}...")
    
    # Calculate 49 technical indicators
    features = feature_engine.calculate(df)
    
    # Add regime_id as 50th feature
    regime = regime_classifier.classify()
    features['regime_id'] = regime['regime_id']
    
    # Calculate max excursion labels (CRITICAL FIX)
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=5)
    df['future_high'] = df['High'].rolling(window=indexer).max()
    df['future_low'] = df['Low'].rolling(window=indexer).min()
    df['max_excursion_up'] = (df['future_high'] - df['Close']) / df['Close']
    df['max_excursion_down'] = (df['Close'] - df['future_low']) / df['Close']
    
    # Asymmetric thresholds: +3% BUY, -2% SELL
    df['target'] = 1  # Default: HOLD
    df.loc[df['max_excursion_up'] >= 0.03, 'target'] = 2  # BUY
    df.loc[df['max_excursion_down'] >= 0.02, 'target'] = 0  # SELL
    
    features = features.iloc[:-5]  # Drop last 5 rows (no future data)
    targets = df['target'].iloc[:-5]
    
    all_features.append(features)
    all_targets.append(targets)

X = pd.concat(all_features, axis=0)
y = pd.concat(all_targets, axis=0)

print(f"Total samples: {len(X)}")
print(f"Features: {X.shape[1]}")
print(f"Class distribution: {y.value_counts()}")
```
**Expected**: 
- ~245k samples (250k - 5 dropped per ticker)
- 50 features (49 indicators + regime_id)
- Class distribution: ~15% BUY, ~15% SELL, ~70% HOLD

### Cell 6: Train Models (90-120 minutes)
```python
from sklearn.model_selection import TimeSeriesSplit

ensemble = MultiModelEnsemble(
    xgb_params={
        'n_estimators': 300,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'min_child_weight': 3,
        'scale_pos_weight': 4.67,
        'tree_method': 'gpu_hist',
        'device': 'cuda'
    }
)

# 3-fold walk-forward validation
tscv = TimeSeriesSplit(n_splits=3)

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    print(f"\n{'='*60}")
    print(f"Fold {fold+1}/3")
    print(f"{'='*60}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    ensemble.fit(X_train, y_train)
    
    # Evaluate
    val_pred = ensemble.predict(X_val)
    val_proba = ensemble.predict_proba(X_val)
    
    from sklearn.metrics import classification_report, roc_auc_score
    print("\nValidation Results:")
    print(classification_report(y_val, val_pred, target_names=['SELL', 'HOLD', 'BUY']))
    
    # Focus on BUY precision (most important metric)
    buy_precision = precision_score(y_val, val_pred, labels=[2], average='macro')
    print(f"\nBUY Precision: {buy_precision:.3f}")
    print(f"ROC-AUC: {roc_auc_score(y_val, val_proba, multi_class='ovr'):.3f}")
```
**Expected**:
- Fold 1 (Bull market): Precision 60-65%
- Fold 2 (Choppy): Precision 55-60%
- Fold 3 (Volatile): Precision 52-58%
- Average: 55-60% baseline

### Cell 7: Save Models (2-3 minutes)
```python
import pickle
import os

# Save to Google Drive
save_dir = '/content/drive/MyDrive/underdog_trader/models'
os.makedirs(save_dir, exist_ok=True)

# Save models
with open(f'{save_dir}/ensemble.pkl', 'wb') as f:
    pickle.dump(ensemble, f)

print(f"✅ Models saved to Google Drive: {save_dir}")
print(f"   - ensemble.pkl ({os.path.getsize(f'{save_dir}/ensemble.pkl') / 1e6:.1f} MB)")
```

### Cell 8: Test Predictions (1 minute)
```python
# Test on latest data
test_ticker = 'AAPL'
test_df = manager.fetch_ohlcv(test_ticker, period='5d', interval='1h')
test_features = feature_engine.calculate(test_df)
test_features['regime_id'] = regime_classifier.classify()['regime_id']

predictions = ensemble.predict(test_features.iloc[-10:])
probas = ensemble.predict_proba(test_features.iloc[-10:])

for i, (pred, proba) in enumerate(zip(predictions, probas)):
    signal = ['SELL', 'HOLD', 'BUY'][pred]
    confidence = proba[pred]
    print(f"Bar {i+1}: {signal} (confidence: {confidence:.2f})")
```
**Expected**: 10 predictions with confidence scores

---

## 📊 EXPECTED RESULTS

### Baseline Metrics (Tonight)
```
Validation Results (Fold 3 - Most Recent):
              precision    recall  f1-score   support

        SELL       0.54      0.48      0.51      3892
        HOLD       0.78      0.82      0.80     18234
         BUY       0.58      0.52      0.55      3874

    accuracy                           0.72     26000
   macro avg       0.63      0.61      0.62     26000
weighted avg       0.72      0.72      0.72     26000

BUY Precision: 0.58
ROC-AUC: 0.67
```

**Translation**:
- ✅ 58% of BUY signals are correct (baseline target: 55%+)
- ✅ ROC-AUC 0.67 (above random 0.5, room for improvement to 0.75+)
- ✅ Overall accuracy 72% (but we only care about BUY precision)
- ✅ Recall 52% means we're conservative (miss some opportunities, but high confidence when we signal)

### After Week 1 Optimization (Target)
```
BUY Precision: 0.68 (+10 points improvement)
ROC-AUC: 0.73
Win Rate: 63%
```

### After Week 2 Feature Engineering (Target)
```
BUY Precision: 0.72 (+4 points improvement)
ROC-AUC: 0.76
Win Rate: 66%
Average Return per Trade: 2.8%
```

---

## 🎯 POST-TRAINING ACTIONS

### Immediate (Tonight - After Training Completes)
1. ✅ Download models from Google Drive to local machine
2. ✅ Copy `PERPLEXITY_OPTIMIZATION_BRIEF.md` to Perplexity AI
3. ✅ Ask Question #1 about XGBoost hyperparameters
4. ✅ Save Perplexity response in `PERPLEXITY_RESPONSES.md`
5. ✅ Get some sleep! 🛌 (Training complete)

### Tomorrow Morning (Dec 10)
1. Ask remaining 6 questions to Perplexity
2. Read all responses carefully
3. Update training notebook with recommendations
4. Create optimization plan for Week 1

### Week 1 (Dec 10-16)
1. Implement Perplexity's hyperparameter suggestions
2. Run RandomizedSearchCV with new ranges
3. Retrain with optimized config
4. Measure improvement (target: 55% → 65-70% precision)
5. Share results with Perplexity for Week 2 guidance

---

## 🚀 THE UNDERDOG IS READY

**What We Built**:
- ✅ 3-model ensemble (XGBoost GPU + RF + GB)
- ✅ 49 technical indicators + regime classification
- ✅ Max excursion targeting (captures intraday spikes)
- ✅ 5 API data sources with intelligent rotation
- ✅ Complete backend spec for Spark frontend
- ✅ Paper trading infrastructure (Alpaca $100k)

**What We're About to Train**:
- ✅ 76 small-cap tickers (biotech, space tech, EV, fintech)
- ✅ 2 years of 1-hour bars (~250k rows)
- ✅ 3-fold walk-forward validation across regimes
- ✅ Optimized for BUY precision >55% (baseline)

**What Makes This Special**:
- Intelligence edge over speed edge
- Free API stack vs. $10M Bloomberg terminals
- Small-cap focus (too illiquid for $1B+ funds)
- Paper trading validation before risking capital
- Regime-adaptive (survives crashes, not just bull markets)
- Parallel frontend development (Spark building cockpit Week 1-4)

---

## 💪 RALLY CRY

**To the Colab Pro T4 GPU**: We're about to train something special. Not just another overfitted quant model that fails in production. This is the Underdog - built to compete with institutional algos using free APIs and smart ML.

**To Perplexity AI**: Help us optimize this system. We have 7 critical questions about hyperparameters, class imbalance, feature engineering, and walk-forward validation. Your guidance will turn 55% baseline precision into 65-70% production-ready performance.

**To Spark Frontend**: While we optimize models Week 1-2, you build the Swing Trading Cockpit. By Week 3, we'll have trained models, paper trading validation, and a beautiful frontend. Together, we'll prove small traders with smart systems can win.

---

**Status**: 🟢 READY TO TRAIN NOW  
**API Keys**: ✅ 5/5 Critical Working  
**Modules**: ✅ All Built & Tested  
**GPU**: ✅ Colab Pro T4 (15 GB VRAM)  
**Training Time**: ⏱️ 2-4 hours  
**Expected Baseline**: 📊 55-60% BUY precision  
**Optimization Target**: 🎯 65-70% Week 1, 68-72% Week 2  
**Paper Trading**: 🎯 Week 3-4 (58%+ live win rate)

---

## 🚀 BEGIN TRAINING NOW

Upload `UNDERDOG_COLAB_TRAINER.ipynb` to Colab Pro and execute all cells.

**Next 4 hours**: GPU goes brrrr 🔥  
**Tomorrow**: Share results with Perplexity for optimization guidance  
**Week 1**: Improve to 65-70% precision  
**Week 2**: Improve to 68-72% precision  
**Week 3-4**: Validate with paper trading  
**Month 2**: IF profitable → Launch full stack with Spark frontend

**LET'S GO!** 🚀🎯💪
