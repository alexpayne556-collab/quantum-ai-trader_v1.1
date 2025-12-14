"""
🎯 NUMERICAL AUTO-OPTIMIZER - Colab Pro Edition
Upload this to Google Colab Pro and run with T4 GPU

Automatically optimizes numerical model through:
1. Advanced feature engineering (50+ indicators)
2. Hyperparameter optimization (Optuna - 100 trials per model)
3. Multi-model ensemble (XGBoost + LightGBM + HistGB)
4. Ensemble weight optimization
5. Class balancing with SMOTE

Target: 61.7% → 65-67% (+3-5%)
Time on T4 GPU: 45-60 minutes (vs 3-4 hours on CPU)

SETUP:
1. Upload this file to Google Colab
2. Runtime → Change runtime type → T4 GPU
3. Run all cells
4. Download results from /content/optimization_results.json
"""

# ============================================================================
# CELL 1: Install Dependencies
# ============================================================================
"""
!pip install -q yfinance optuna xgboost lightgbm imbalanced-learn scikit-learn
print("✅ Dependencies installed")
"""

# ============================================================================
# CELL 2: Imports and Configuration
# ============================================================================

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import HistGradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import optuna
import json
import warnings
warnings.filterwarnings('ignore')

CONFIG = {
    'tickers': [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-B',
        'JPM', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'HD', 'DIS',
        'NFLX', 'PYPL', 'ADBE', 'CRM', 'INTC', 'AMD', 'QCOM', 'COST',
        'PEP', 'KO', 'NKE', 'MCD', 'BA', 'GE', 'F', 'GM',
        'C', 'BAC', 'WFC', 'GS', 'XOM', 'CVX', 'SLB', 'COP',
        'UNH', 'PFE', 'ABBV', 'TMO', 'LLY', 'MRK', 'ABT', 'DHR'
    ],
    'window_size': 60,
    'horizon': 5,
    'buy_threshold': 0.03,
    'sell_threshold': -0.03,
    'optuna_trials': 100,
}

print("✅ Configuration loaded")

# ============================================================================
# CELL 3: Feature Engineering Functions
# ============================================================================

def calculate_advanced_features(df, window=60):
    """Generate 50+ advanced numerical features"""
    features = {}
    
    # Handle both Series and DataFrame
    if isinstance(df['Close'], pd.DataFrame):
        close = df['Close'].values.flatten()
        high = df['High'].values.flatten()
        low = df['Low'].values.flatten()
        volume = df['Volume'].values.flatten()
        open_price = df['Open'].values.flatten()
    else:
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values
        open_price = df['Open'].values
    
    if len(close) < window:
        return None
    
    close_window = close[-window:]
    high_window = high[-window:]
    low_window = low[-window:]
    volume_window = volume[-window:]
    
    # Price statistics (8)
    features['price_mean'] = np.mean(close_window)
    features['price_std'] = np.std(close_window)
    features['price_min'] = np.min(close_window)
    features['price_max'] = np.max(close_window)
    features['price_range'] = (np.max(close_window) - np.min(close_window)) / (np.mean(close_window) + 1e-8)
    features['price_return'] = (close_window[-1] - close_window[0]) / (close_window[0] + 1e-8)
    features['price_zscore'] = (close_window[-1] - np.mean(close_window)) / (np.std(close_window) + 1e-8)
    features['high_low_ratio'] = np.mean(high_window / (low_window + 1e-8))
    
    # Moving averages (12)
    for period in [5, 10, 20, 50]:
        if len(close_window) >= period:
            ma = np.mean(close_window[-period:])
            features[f'ma_{period}'] = close_window[-1] / (ma + 1e-8) - 1
            past_start = max(0, len(close_window) - period * 2)
            past_end = max(period, len(close_window) - period)
            ma_past = np.mean(close_window[past_start:past_end]) if past_end > past_start else ma
            features[f'ma_{period}_slope'] = (ma - ma_past) / (ma + 1e-8)
    
    if len(close_window) >= 50:
        features['ma_5_20_cross'] = (np.mean(close_window[-5:]) / (np.mean(close_window[-20:]) + 1e-8)) - 1
        features['ma_10_50_cross'] = (np.mean(close_window[-10:]) / (np.mean(close_window[-50:]) + 1e-8)) - 1
        features['ma_20_50_cross'] = (np.mean(close_window[-20:]) / (np.mean(close_window[-50:]) + 1e-8)) - 1
        features['price_above_ma50'] = 1.0 if close_window[-1] > np.mean(close_window[-50:]) else 0.0
    
    # Momentum (10)
    for period in [3, 5, 10, 20, 30]:
        if len(close_window) >= period:
            features[f'momentum_{period}'] = (close_window[-1] - close_window[-period]) / (close_window[-period] + 1e-8)
    
    if len(close_window) >= 10:
        features['roc_5'] = (close_window[-1] - close_window[-5]) / (close_window[-5] + 1e-8)
        features['roc_10'] = (close_window[-1] - close_window[-10]) / (close_window[-10] + 1e-8)
    
    if len(close_window) >= 6:
        mom_recent = (close_window[-1] - close_window[-3]) / (close_window[-3] + 1e-8)
        mom_past = (close_window[-3] - close_window[-6]) / (close_window[-6] + 1e-8)
        features['momentum_acceleration'] = mom_recent - mom_past
    
    # Volatility (8)
    returns = np.diff(close_window) / (close_window[:-1] + 1e-8)
    features['volatility_10'] = np.std(returns[-10:]) if len(returns) >= 10 else np.std(returns)
    features['volatility_20'] = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
    features['volatility_ratio'] = (np.std(returns[-10:]) / (np.std(returns[-20:]) + 1e-8)) if len(returns) >= 20 else 1.0
    
    if len(close_window) >= 14:
        tr = np.maximum(high_window[-14:] - low_window[-14:], 
                        np.abs(high_window[-14:] - np.roll(close_window[-14:], 1)))
        tr = np.maximum(tr, np.abs(low_window[-14:] - np.roll(close_window[-14:], 1)))
        features['atr_14'] = np.mean(tr) / (close_window[-1] + 1e-8)
    
    features['hist_vol'] = np.std(returns) * np.sqrt(252)
    
    if len(high_window) >= 20:
        park_vol = np.sqrt(np.mean(np.log(high_window[-20:] / (low_window[-20:] + 1e-8))**2) / (4 * np.log(2)))
        features['parkinson_vol'] = park_vol
    
    if len(volume_window) >= 20:
        vol_weights = volume_window[-20:] / (np.sum(volume_window[-20:]) + 1e-8)
        features['vol_weighted_volatility'] = np.sqrt(np.sum(vol_weights * returns[-20:]**2))
    
    # Volume (7)
    features['volume_mean'] = np.mean(volume_window)
    features['volume_std'] = np.std(volume_window)
    features['volume_ratio'] = volume_window[-1] / (np.mean(volume_window) + 1e-8)
    features['volume_trend'] = (np.mean(volume_window[-10:]) - np.mean(volume_window[-20:])) / (np.mean(volume_window[-20:]) + 1e-8) if len(volume_window) >= 20 else 0.0
    
    if len(close_window) >= 20 and len(volume_window) >= 20:
        price_changes = np.diff(close_window[-20:])
        volume_changes = volume_window[-19:]
        if np.std(price_changes) > 0 and np.std(volume_changes) > 0:
            features['volume_price_corr'] = np.corrcoef(price_changes, volume_changes)[0, 1]
        else:
            features['volume_price_corr'] = 0.0
    
    obv = np.zeros(len(close_window))
    for i in range(1, len(close_window)):
        if close_window[i] > close_window[i-1]:
            obv[i] = obv[i-1] + volume_window[i]
        elif close_window[i] < close_window[i-1]:
            obv[i] = obv[i-1] - volume_window[i]
        else:
            obv[i] = obv[i-1]
    features['obv_trend'] = (obv[-1] - obv[-20]) / (abs(obv[-20]) + 1e-8) if len(obv) >= 20 else 0.0
    
    features['volume_spike'] = 1.0 if volume_window[-1] > np.mean(volume_window) + 2*np.std(volume_window) else 0.0
    
    # Patterns (5)
    if len(high_window) >= 20:
        recent_high = np.max(high_window[-10:])
        past_high = np.max(high_window[-20:-10])
        features['higher_highs'] = 1.0 if recent_high > past_high else 0.0
    
    if len(low_window) >= 20:
        recent_low = np.min(low_window[-10:])
        past_low = np.min(low_window[-20:-10])
        features['lower_lows'] = 1.0 if recent_low < past_low else 0.0
    
    features['dist_from_high'] = (np.max(close_window) - close_window[-1]) / (close_window[-1] + 1e-8)
    features['dist_from_low'] = (close_window[-1] - np.min(close_window)) / (close_window[-1] + 1e-8)
    
    if len(close_window) >= 2:
        body = close_window[-1] - open_price[-1]
        range_val = high_window[-1] - low_window[-1]
        features['candle_body_ratio'] = body / (range_val + 1e-8)
    
    return features

print("✅ Feature engineering functions loaded")

# ============================================================================
# CELL 4: Download Data and Create Dataset
# ============================================================================

print("📥 Downloading data...")
data = {}
for i, ticker in enumerate(CONFIG['tickers'], 1):
    try:
        df = yf.download(ticker, period='3y', interval='1d', progress=False)
        if len(df) > 100:
            data[ticker] = df
        print(f"   [{i}/{len(CONFIG['tickers'])}] {ticker}: {len(df)} days", end='\r')
    except:
        pass

print(f"\n✅ Downloaded {len(data)} tickers\n")

print("🔧 Engineering features...")
X_list = []
y_list = []

for ticker_idx, (ticker, df) in enumerate(data.items(), 1):
    print(f"   [{ticker_idx}/{len(data)}] {ticker}...", end='\r')
    
    df = df.copy()
    df['Return'] = df['Close'].pct_change(CONFIG['horizon']).shift(-CONFIG['horizon'])
    
    for i in range(CONFIG['window_size'], len(df) - CONFIG['horizon']):
        window = df.iloc[i-CONFIG['window_size']:i]
        future_return = df['Return'].iloc[i]
        
        if pd.isna(future_return):
            continue
        
        if future_return > CONFIG['buy_threshold']:
            label = 0  # BUY
        elif future_return < CONFIG['sell_threshold']:
            label = 2  # SELL
        else:
            label = 1  # HOLD
        
        features = calculate_advanced_features(window, CONFIG['window_size'])
        if features is None:
            continue
        
        X_list.append(list(features.values()))
        y_list.append(label)

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int32)
X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
feature_names = list(features.keys())

print(f"\n✅ Generated {len(X)} samples with {len(feature_names)} features")
print(f"   BUY: {np.sum(y==0)} ({100*np.mean(y==0):.1f}%)")
print(f"   HOLD: {np.sum(y==1)} ({100*np.mean(y==1):.1f}%)")
print(f"   SELL: {np.sum(y==2)} ({100*np.mean(y==2):.1f}%)\n")

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

print(f"📊 Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}\n")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("⚖️ Applying SMOTE for class balancing...")
smote = SMOTE(random_state=42)
X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
print(f"✅ Resampled to {len(X_train_scaled)} samples\n")

# ============================================================================
# CELL 5: Optimize XGBoost
# ============================================================================

print("🔬 Optimizing XGBoost (100 trials)...")

def objective_xgb(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'tree_method': 'hist',
        'device': 'cuda',
        'random_state': 42,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
    return accuracy_score(y_val, model.predict(X_val_scaled))

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=CONFIG['optuna_trials'], show_progress_bar=True)

model_xgb = xgb.XGBClassifier(**study_xgb.best_params, tree_method='hist', device='cuda', random_state=42)
model_xgb.fit(X_train_scaled, y_train)
print(f"✅ XGBoost optimized: {study_xgb.best_value:.4f}\n")

# ============================================================================
# CELL 6: Optimize LightGBM
# ============================================================================

print("🔬 Optimizing LightGBM (100 trials)...")

def objective_lgb(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'device': 'gpu',
        'random_state': 42,
        'verbose': -1,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)])
    return accuracy_score(y_val, model.predict(X_val_scaled))

study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(objective_lgb, n_trials=CONFIG['optuna_trials'], show_progress_bar=True)

model_lgb = lgb.LGBMClassifier(**study_lgb.best_params, device='gpu', random_state=42, verbose=-1)
model_lgb.fit(X_train_scaled, y_train)
print(f"✅ LightGBM optimized: {study_lgb.best_value:.4f}\n")

# ============================================================================
# CELL 7: Optimize HistGradientBoosting
# ============================================================================

print("🔬 Optimizing HistGradientBoosting (100 trials)...")

def objective_histgb(trial):
    params = {
        'max_iter': trial.suggest_int('max_iter', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 100),
        'l2_regularization': trial.suggest_float('l2_regularization', 0, 10),
        'random_state': 42,
    }
    model = HistGradientBoostingClassifier(**params)
    model.fit(X_train_scaled, y_train)
    return accuracy_score(y_val, model.predict(X_val_scaled))

study_histgb = optuna.create_study(direction='maximize')
study_histgb.optimize(objective_histgb, n_trials=CONFIG['optuna_trials'], show_progress_bar=True)

model_histgb = HistGradientBoostingClassifier(**study_histgb.best_params, random_state=42)
model_histgb.fit(X_train_scaled, y_train)
print(f"✅ HistGB optimized: {study_histgb.best_value:.4f}\n")

# ============================================================================
# CELL 8: Optimize Ensemble Weights
# ============================================================================

print("🔬 Optimizing ensemble weights...")

models = [model_xgb, model_lgb, model_histgb]
model_names = ['XGBoost', 'LightGBM', 'HistGB']
all_probs = [m.predict_proba(X_val_scaled) for m in models]

def objective_ensemble(trial):
    weights = [trial.suggest_float(f'w{i}', 0, 1) for i in range(3)]
    total = sum(weights)
    if total == 0:
        return 0
    weights = [w/total for w in weights]
    
    ensemble_probs = np.zeros_like(all_probs[0])
    for prob, weight in zip(all_probs, weights):
        ensemble_probs += prob * weight
    
    return accuracy_score(y_val, np.argmax(ensemble_probs, axis=1))

study_ensemble = optuna.create_study(direction='maximize')
study_ensemble.optimize(objective_ensemble, n_trials=50, show_progress_bar=True)

weights = [study_ensemble.best_params[f'w{i}'] for i in range(3)]
total = sum(weights)
weights = [w/total for w in weights]

print(f"✅ Optimal weights: {dict(zip(model_names, [f'{w:.3f}' for w in weights]))}")
print(f"✅ Ensemble val accuracy: {study_ensemble.best_value:.4f}\n")

# ============================================================================
# CELL 9: Final Evaluation
# ============================================================================

print("🧪 Evaluating on test set...\n")

print("📊 Individual Model Performance:")
individual_accs = {}
for model, name in zip(models, model_names):
    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    individual_accs[name] = float(acc)
    print(f"   {name}: {acc:.4f} ({100*acc:.2f}%)")

print("\n🎯 Ensemble Performance:")
ensemble_probs = np.zeros((len(X_test_scaled), 3))
for model, weight in zip(models, weights):
    ensemble_probs += model.predict_proba(X_test_scaled) * weight

ensemble_preds = np.argmax(ensemble_probs, axis=1)
ensemble_acc = accuracy_score(y_test, ensemble_preds)

print(f"   Weighted Ensemble: {ensemble_acc:.4f} ({100*ensemble_acc:.2f}%)")
print(f"   Weights: {dict(zip(model_names, [f'{w:.3f}' for w in weights]))}")

print(f"\n{'='*80}")
print("📊 FINAL RESULTS")
print(f"{'='*80}")
print(classification_report(y_test, ensemble_preds, target_names=['BUY', 'HOLD', 'SELL'], digits=4))
print(f"{'='*80}\n")

# Save results
results = {
    'test_accuracy': float(ensemble_acc),
    'validation_accuracy': float(study_ensemble.best_value),
    'individual_models': individual_accs,
    'ensemble_weights': dict(zip(model_names, [float(w) for w in weights])),
    'num_features': len(feature_names),
    'feature_names': feature_names[:20],  # First 20 features
    'xgboost_best_params': study_xgb.best_params,
    'lightgbm_best_params': study_lgb.best_params,
    'histgb_best_params': study_histgb.best_params,
}

with open('/content/optimization_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("💾 Results saved to: /content/optimization_results.json")

if ensemble_acc >= 0.65:
    print(f"\n🎉 SUCCESS! Achieved 65%+ accuracy!")
    print(f"   Improvement: 61.7% → {100*ensemble_acc:.2f}% (+{100*(ensemble_acc-0.617):.2f}%)")
elif ensemble_acc >= 0.63:
    print(f"\n✅ Good progress! Above 63%")
    print(f"   Improvement: 61.7% → {100*ensemble_acc:.2f}% (+{100*(ensemble_acc-0.617):.2f}%)")
else:
    print(f"\n📈 Current: {100*ensemble_acc:.2f}%")

print("\n✅ COMPLETE! Download /content/optimization_results.json\n")
