# 🔧 Optimization Guide - From 60% to 70%+ Accuracy

## Step-by-Step Optimization Process

After initial Colab training, follow these steps to maximize profitability.

---

## 1. Hyperparameter Optimization

### Create Optimization Script
```python
# optimize_hyperparameters.py

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'python'))

from multi_model_ensemble import MultiModelEnsemble
from feature_engine import FeatureEngine
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Load your training data (after Colab training)
# Assuming you have X_train, y_train, X_val, y_val

# XGBoost Hyperparameter Grid
xgb_param_grid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [6, 8, 10, 12],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2]
}

# Randomized search (faster than grid search)
xgb_model = xgb.XGBClassifier(
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    random_state=42,
    n_jobs=-1
)

xgb_search = RandomizedSearchCV(
    xgb_model,
    xgb_param_grid,
    n_iter=50,  # Try 50 random combinations
    scoring='roc_auc',
    cv=3,
    verbose=2,
    n_jobs=1,  # GPU, so 1 job
    random_state=42
)

print("🔍 Optimizing XGBoost...")
xgb_search.fit(X_train, y_train)

print(f"✅ Best XGBoost params: {xgb_search.best_params_}")
print(f"✅ Best ROC-AUC: {xgb_search.best_score_:.4f}")

# Random Forest Hyperparameter Grid
rf_param_grid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [12, 15, 20, None],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [2, 4, 8],
    'max_features': ['sqrt', 'log2', 0.5]
}

rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_search = RandomizedSearchCV(
    rf_model,
    rf_param_grid,
    n_iter=50,
    scoring='roc_auc',
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

print("\n🔍 Optimizing Random Forest...")
rf_search.fit(X_train, y_train)

print(f"✅ Best RF params: {rf_search.best_params_}")
print(f"✅ Best ROC-AUC: {rf_search.best_score_:.4f}")

# Gradient Boosting Hyperparameter Grid
gb_param_grid = {
    'n_estimators': [200, 300, 500],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'max_depth': [4, 6, 8],
    'subsample': [0.8, 0.9, 1.0],
    'min_samples_split': [2, 5, 10]
}

gb_model = GradientBoostingClassifier(random_state=42)
gb_search = RandomizedSearchCV(
    gb_model,
    gb_param_grid,
    n_iter=50,
    scoring='roc_auc',
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

print("\n🔍 Optimizing Gradient Boosting...")
gb_search.fit(X_train, y_train)

print(f"✅ Best GB params: {gb_search.best_params_}")
print(f"✅ Best ROC-AUC: {gb_search.best_score_:.4f}")

# Test on validation set
xgb_val_pred = xgb_search.best_estimator_.predict(X_val)
rf_val_pred = rf_search.best_estimator_.predict(X_val)
gb_val_pred = gb_search.best_estimator_.predict(X_val)

print("\n📊 Validation Performance:")
print(f"XGBoost Accuracy: {accuracy_score(y_val, xgb_val_pred):.4f}")
print(f"Random Forest Accuracy: {accuracy_score(y_val, rf_val_pred):.4f}")
print(f"Gradient Boosting Accuracy: {accuracy_score(y_val, gb_val_pred):.4f}")
```

---

## 2. Feature Selection

### Method 1: Mutual Information
```python
from sklearn.feature_selection import mutual_info_classif

# Calculate mutual information scores
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)

# Create feature importance DataFrame
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print("📊 Top 20 Features by Mutual Information:")
print(feature_importance.head(20))

# Keep top 30 features
top_features = feature_importance.head(30)['feature'].tolist()
X_train_reduced = X_train[top_features]
X_val_reduced = X_val[top_features]
```

### Method 2: SHAP Values
```python
import shap

# Train a model first
xgb_model = xgb.XGBClassifier(tree_method='gpu_hist', **best_xgb_params)
xgb_model.fit(X_train, y_train)

# Calculate SHAP values
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_train[:1000])  # Sample for speed

# Get feature importance
shap_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': np.abs(shap_values).mean(axis=0)
}).sort_values('importance', ascending=False)

print("📊 Top 20 Features by SHAP:")
print(shap_importance.head(20))

# Keep top 30
top_features = shap_importance.head(30)['feature'].tolist()
```

### Method 3: Correlation Filtering
```python
# Remove highly correlated features
correlation_matrix = X_train.corr().abs()

# Find pairs with correlation > 0.95
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if correlation_matrix.iloc[i, j] > 0.95:
            high_corr_pairs.append((
                correlation_matrix.columns[i],
                correlation_matrix.columns[j],
                correlation_matrix.iloc[i, j]
            ))

print(f"Found {len(high_corr_pairs)} highly correlated pairs")

# Remove one from each pair (keep the one with higher MI score)
features_to_remove = set()
for feat1, feat2, corr in high_corr_pairs:
    mi1 = mi_scores[feature_names.index(feat1)]
    mi2 = mi_scores[feature_names.index(feat2)]
    if mi1 < mi2:
        features_to_remove.add(feat1)
    else:
        features_to_remove.add(feat2)

features_to_keep = [f for f in feature_names if f not in features_to_remove]
print(f"Keeping {len(features_to_keep)} features after correlation filtering")
```

---

## 3. Label Optimization

### Test Different Horizons
```python
# Test 3-bar, 5-bar, 10-bar forward returns
def create_labels(df, horizon=5, buy_threshold=0.03, sell_threshold=-0.02):
    """Create labels with different parameters"""
    future_return = (df['close'].shift(-horizon) - df['close']) / df['close']
    
    labels = np.zeros(len(df))
    labels[future_return > buy_threshold] = 2  # BUY
    labels[future_return < sell_threshold] = 0  # SELL
    labels[(future_return >= sell_threshold) & (future_return <= buy_threshold)] = 1  # HOLD
    
    return labels, future_return

# Test different configurations
configs = [
    (3, 0.02, -0.015),  # 3-bar, 2%, -1.5%
    (5, 0.03, -0.02),   # 5-bar, 3%, -2% (current)
    (5, 0.025, -0.02),  # 5-bar, 2.5%, -2%
    (10, 0.05, -0.03),  # 10-bar, 5%, -3%
]

results = []
for horizon, buy_th, sell_th in configs:
    labels, returns = create_labels(df, horizon, buy_th, sell_th)
    
    # Train quick model
    ensemble = MultiModelEnsemble(use_gpu=False)
    # ... train ...
    
    # Evaluate
    val_acc = ensemble.score(X_val, labels_val)
    results.append({
        'horizon': horizon,
        'buy_threshold': buy_th,
        'sell_threshold': sell_th,
        'accuracy': val_acc
    })

results_df = pd.DataFrame(results).sort_values('accuracy', ascending=False)
print("📊 Label Configuration Results:")
print(results_df)
```

---

## 4. Ensemble Weighting Optimization

```python
from scipy.optimize import minimize

# Get predictions from all 3 models
xgb_probs = xgb_model.predict_proba(X_val)
rf_probs = rf_model.predict_proba(X_val)
gb_probs = gb_model.predict_proba(X_val)

def ensemble_accuracy(weights):
    """Calculate ensemble accuracy with given weights"""
    w1, w2, w3 = weights
    
    # Weighted average of probabilities
    ensemble_probs = (w1 * xgb_probs + w2 * rf_probs + w3 * gb_probs)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    
    return -accuracy_score(y_val, ensemble_preds)  # Negative for minimization

# Constraints: weights sum to 1, all non-negative
constraints = ({'type': 'eq', 'fun': lambda w: w.sum() - 1})
bounds = [(0, 1), (0, 1), (0, 1)]

# Start with equal weights
initial_weights = [1/3, 1/3, 1/3]

# Optimize
result = minimize(
    ensemble_accuracy,
    initial_weights,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

optimal_weights = result.x
print(f"✅ Optimal weights: XGBoost={optimal_weights[0]:.3f}, RF={optimal_weights[1]:.3f}, GB={optimal_weights[2]:.3f}")
print(f"✅ Ensemble accuracy: {-result.fun:.4f}")
```

---

## 5. Regime-Adaptive Training

```python
from regime_classifier import RegimeClassifier

# Load regime data
regime_clf = RegimeClassifier()

# Classify each training sample
regimes = []
for i in range(len(df)):
    regime_data = {
        'vix': df['vix'].iloc[i],
        'spy_return': df['spy_return'].iloc[i],
        # ... other regime indicators
    }
    regime = regime_clf.classify_regime_from_data(regime_data)
    regimes.append(regime['name'])

df['regime'] = regimes

# Train separate models for BULL, BEAR, CHOPPY
bull_mask = df['regime'].str.contains('BULL')
bear_mask = df['regime'].str.contains('BEAR')
choppy_mask = df['regime'].str.contains('CHOPPY')

# Train BULL ensemble
print("Training BULL regime models...")
bull_ensemble = MultiModelEnsemble(use_gpu=True)
bull_ensemble.train(X_train[bull_mask], y_train[bull_mask])
bull_ensemble.save('models/underdog_v1/bull_ensemble')

# Train BEAR ensemble
print("Training BEAR regime models...")
bear_ensemble = MultiModelEnsemble(use_gpu=True)
bear_ensemble.train(X_train[bear_mask], y_train[bear_mask])
bear_ensemble.save('models/underdog_v1/bear_ensemble')

# Train CHOPPY ensemble
print("Training CHOPPY regime models...")
choppy_ensemble = MultiModelEnsemble(use_gpu=True)
choppy_ensemble.train(X_train[choppy_mask], y_train[choppy_mask])
choppy_ensemble.save('models/underdog_v1/choppy_ensemble')

# At prediction time, switch models based on current regime
```

---

## 6. Walk-Forward Validation

```python
from sklearn.model_selection import TimeSeriesSplit

# Time-based splits (not random)
tscv = TimeSeriesSplit(n_splits=5)

walk_forward_results = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    print(f"\n📅 Fold {fold+1}/5")
    
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_val_fold = y.iloc[val_idx]
    
    # Train ensemble
    ensemble = MultiModelEnsemble(use_gpu=True)
    ensemble.train(X_train_fold, y_train_fold)
    
    # Evaluate
    val_acc = ensemble.score(X_val_fold, y_val_fold)
    
    walk_forward_results.append({
        'fold': fold + 1,
        'train_size': len(train_idx),
        'val_size': len(val_idx),
        'val_accuracy': val_acc
    })
    
    print(f"Validation Accuracy: {val_acc:.4f}")

# Average across folds
avg_accuracy = np.mean([r['val_accuracy'] for r in walk_forward_results])
print(f"\n✅ Average Walk-Forward Accuracy: {avg_accuracy:.4f}")
```

---

## Optimization Checklist

### Week 1 (After Initial Training)
- [ ] Run hyperparameter optimization for XGBoost (2-3 hours)
- [ ] Run hyperparameter optimization for Random Forest (1-2 hours)
- [ ] Run hyperparameter optimization for Gradient Boosting (1-2 hours)
- [ ] Retrain with optimal hyperparameters
- [ ] **Expected gain**: +3-5% accuracy

### Week 2 (Feature & Label Engineering)
- [ ] Calculate mutual information scores
- [ ] Run SHAP analysis
- [ ] Remove highly correlated features (r > 0.95)
- [ ] Reduce from 49 → 30 best features
- [ ] Test different forward horizons (3, 5, 10 bars)
- [ ] Test different thresholds (2%, 2.5%, 3%)
- [ ] Retrain with optimal features + labels
- [ ] **Expected gain**: +5-8% accuracy

### Week 3 (Advanced Ensemble)
- [ ] Optimize ensemble weights (not just majority vote)
- [ ] Train regime-adaptive models (BULL/BEAR/CHOPPY)
- [ ] Run walk-forward validation
- [ ] **Expected gain**: +3-7% accuracy

### Week 4 (Paper Trading)
- [ ] Deploy optimized models
- [ ] Track live signals for 2 weeks
- [ ] Compare backtest vs live performance
- [ ] Identify failure modes
- [ ] Final adjustments

---

## Performance Tracking

After each optimization step, track:
```python
metrics = {
    'validation_accuracy': 0.0,
    'validation_roc_auc': 0.0,
    'backtest_win_rate': 0.0,
    'backtest_avg_return': 0.0,
    'backtest_sharpe': 0.0,
    'backtest_max_drawdown': 0.0,
}
```

Log everything. Compare before/after for each optimization.

**Goal**: Accuracy 60% → 70%, Win rate 55% → 65%, Sharpe 1.2 → 2.0+

Let's make it profitable! 💰
