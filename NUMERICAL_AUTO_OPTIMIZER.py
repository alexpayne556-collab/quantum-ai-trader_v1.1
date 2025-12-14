"""
🎯 NUMERICAL AUTO-OPTIMIZER - Path to 65%+
Automatically optimizes numerical model through:
1. Advanced feature engineering (50+ indicators)
2. Hyperparameter optimization (Optuna - 200 trials)
3. Multi-model ensemble (XGBoost + LightGBM + HistGB)
4. Ensemble weight optimization
5. Threshold tuning
6. Class balancing

Target: 61.7% → 65-67% (+3-5%)
Time: 3-4 hours on GPU
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier
import optuna
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Optional imports (will use if available)
try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False
    print("⚠️ XGBoost not available (pip install xgboost)")

try:
    import lightgbm as lgb
    HAS_LGB = True
except:
    HAS_LGB = False
    print("⚠️ LightGBM not available (pip install lightgbm)")

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except:
    HAS_SMOTE = False
    print("⚠️ SMOTE not available (pip install imbalanced-learn)")

#===============================================================================
# CONFIGURATION
#===============================================================================

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
    'optuna_trials': 100,  # 100 trials per model
    'use_smote': True,
    'ensemble_optimize': True,
}

CHECKPOINT_DIR = Path('numerical_optimizer_results')
CHECKPOINT_DIR.mkdir(exist_ok=True)

#===============================================================================
# ADVANCED FEATURE ENGINEERING (50+ FEATURES)
#===============================================================================

def calculate_advanced_features(df, window=60):
    """Generate 50+ advanced numerical features"""
    features = {}
    
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    volume = df['Volume'].values
    open_price = df['Open'].values
    
    # Ensure we have enough data
    if len(close) < window:
        return None
    
    close_window = close[-window:]
    high_window = high[-window:]
    low_window = low[-window:]
    volume_window = volume[-window:]
    
    # ========== PRICE STATISTICS (8 features) ==========
    features['price_mean'] = np.mean(close_window)
    features['price_std'] = np.std(close_window)
    features['price_min'] = np.min(close_window)
    features['price_max'] = np.max(close_window)
    features['price_range'] = (np.max(close_window) - np.min(close_window)) / (np.mean(close_window) + 1e-8)
    features['price_return'] = (close_window[-1] - close_window[0]) / (close_window[0] + 1e-8)
    features['price_zscore'] = (close_window[-1] - np.mean(close_window)) / (np.std(close_window) + 1e-8)
    features['high_low_ratio'] = np.mean(high_window / (low_window + 1e-8))
    
    # ========== MOVING AVERAGES (12 features) ==========
    for period in [5, 10, 20, 50]:
        if len(close_window) >= period:
            ma = np.mean(close_window[-period:])
            features[f'ma_{period}'] = close_window[-1] / (ma + 1e-8) - 1
            features[f'ma_{period}_slope'] = (ma - np.mean(close_window[-period*2:-period] if len(close_window) >= period*2 else close_window[:period])) / (ma + 1e-8)
    
    # MA Crossovers
    if len(close_window) >= 50:
        features['ma_5_20_cross'] = (np.mean(close_window[-5:]) / (np.mean(close_window[-20:]) + 1e-8)) - 1
        features['ma_10_50_cross'] = (np.mean(close_window[-10:]) / (np.mean(close_window[-50:]) + 1e-8)) - 1
        features['ma_20_50_cross'] = (np.mean(close_window[-20:]) / (np.mean(close_window[-50:]) + 1e-8)) - 1
        features['price_above_ma50'] = 1.0 if close_window[-1] > np.mean(close_window[-50:]) else 0.0
    
    # ========== MOMENTUM (10 features) ==========
    for period in [3, 5, 10, 20, 30]:
        if len(close_window) >= period:
            features[f'momentum_{period}'] = (close_window[-1] - close_window[-period]) / (close_window[-period] + 1e-8)
    
    # Rate of Change
    if len(close_window) >= 10:
        features['roc_5'] = (close_window[-1] - close_window[-5]) / (close_window[-5] + 1e-8)
        features['roc_10'] = (close_window[-1] - close_window[-10]) / (close_window[-10] + 1e-8)
    
    # Acceleration
    if len(close_window) >= 6:
        mom_recent = (close_window[-1] - close_window[-3]) / (close_window[-3] + 1e-8)
        mom_past = (close_window[-3] - close_window[-6]) / (close_window[-6] + 1e-8)
        features['momentum_acceleration'] = mom_recent - mom_past
    
    # ========== VOLATILITY (8 features) ==========
    returns = np.diff(close_window) / (close_window[:-1] + 1e-8)
    features['volatility_10'] = np.std(returns[-10:]) if len(returns) >= 10 else np.std(returns)
    features['volatility_20'] = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
    features['volatility_ratio'] = (np.std(returns[-10:]) / (np.std(returns[-20:]) + 1e-8)) if len(returns) >= 20 else 1.0
    
    # ATR (Average True Range)
    if len(close_window) >= 14:
        tr = np.maximum(high_window[-14:] - low_window[-14:], 
                        np.abs(high_window[-14:] - np.roll(close_window[-14:], 1)))
        tr = np.maximum(tr, np.abs(low_window[-14:] - np.roll(close_window[-14:], 1)))
        features['atr_14'] = np.mean(tr) / (close_window[-1] + 1e-8)
    
    # Historical volatility
    features['hist_vol'] = np.std(returns) * np.sqrt(252)
    
    # Parkinson volatility (high-low)
    if len(high_window) >= 20:
        park_vol = np.sqrt(np.mean(np.log(high_window[-20:] / (low_window[-20:] + 1e-8))**2) / (4 * np.log(2)))
        features['parkinson_vol'] = park_vol
    
    # Volume-weighted volatility
    if len(volume_window) >= 20:
        vol_weights = volume_window[-20:] / (np.sum(volume_window[-20:]) + 1e-8)
        features['vol_weighted_volatility'] = np.sqrt(np.sum(vol_weights * returns[-20:]**2))
    
    # ========== VOLUME (7 features) ==========
    features['volume_mean'] = np.mean(volume_window)
    features['volume_std'] = np.std(volume_window)
    features['volume_ratio'] = volume_window[-1] / (np.mean(volume_window) + 1e-8)
    features['volume_trend'] = (np.mean(volume_window[-10:]) - np.mean(volume_window[-20:])) / (np.mean(volume_window[-20:]) + 1e-8) if len(volume_window) >= 20 else 0.0
    
    # Volume-price correlation
    if len(close_window) >= 20 and len(volume_window) >= 20:
        price_changes = np.diff(close_window[-20:])
        volume_changes = volume_window[-19:]
        if np.std(price_changes) > 0 and np.std(volume_changes) > 0:
            features['volume_price_corr'] = np.corrcoef(price_changes, volume_changes)[0, 1]
        else:
            features['volume_price_corr'] = 0.0
    
    # On-Balance Volume (OBV) trend
    obv = np.zeros(len(close_window))
    for i in range(1, len(close_window)):
        if close_window[i] > close_window[i-1]:
            obv[i] = obv[i-1] + volume_window[i]
        elif close_window[i] < close_window[i-1]:
            obv[i] = obv[i-1] - volume_window[i]
        else:
            obv[i] = obv[i-1]
    features['obv_trend'] = (obv[-1] - obv[-20]) / (abs(obv[-20]) + 1e-8) if len(obv) >= 20 else 0.0
    
    # Volume spike
    features['volume_spike'] = 1.0 if volume_window[-1] > np.mean(volume_window) + 2*np.std(volume_window) else 0.0
    
    # ========== PATTERN FEATURES (5 features) ==========
    # Higher highs, lower lows
    if len(high_window) >= 20:
        recent_high = np.max(high_window[-10:])
        past_high = np.max(high_window[-20:-10])
        features['higher_highs'] = 1.0 if recent_high > past_high else 0.0
    
    if len(low_window) >= 20:
        recent_low = np.min(low_window[-10:])
        past_low = np.min(low_window[-20:-10])
        features['lower_lows'] = 1.0 if recent_low < past_low else 0.0
    
    # Distance from 52-week high/low
    features['dist_from_high'] = (np.max(close_window) - close_window[-1]) / (close_window[-1] + 1e-8)
    features['dist_from_low'] = (close_window[-1] - np.min(close_window)) / (close_window[-1] + 1e-8)
    
    # Candle patterns (simplified)
    if len(close_window) >= 2:
        body = close_window[-1] - open_price[-1]
        range_val = high_window[-1] - low_window[-1]
        features['candle_body_ratio'] = body / (range_val + 1e-8)
    
    return features

def create_numerical_dataset(data):
    """Create dataset with advanced features"""
    print("🔧 Engineering features...")
    
    X_list = []
    y_list = []
    returns_list = []
    
    for ticker_idx, (ticker, df) in enumerate(data.items(), 1):
        print(f"   [{ticker_idx}/{len(data)}] {ticker}...", end='\r')
        
        df = df.copy()
        df['Return'] = df['Close'].pct_change(CONFIG['horizon']).shift(-CONFIG['horizon'])
        
        for i in range(CONFIG['window_size'], len(df) - CONFIG['horizon']):
            window = df.iloc[i-CONFIG['window_size']:i]
            future_return = df['Return'].iloc[i]
            
            if pd.isna(future_return):
                continue
            
            # Label
            if future_return > CONFIG['buy_threshold']:
                label = 0  # BUY
            elif future_return < CONFIG['sell_threshold']:
                label = 2  # SELL
            else:
                label = 1  # HOLD
            
            # Generate features
            features = calculate_advanced_features(window, CONFIG['window_size'])
            if features is None:
                continue
            
            X_list.append(list(features.values()))
            y_list.append(label)
            returns_list.append(future_return)
    
    print(f"\n✅ Generated {len(X_list)} samples with {len(features)} features")
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    
    # Replace inf/nan
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    print(f"   BUY: {np.sum(y==0)} ({100*np.mean(y==0):.1f}%)")
    print(f"   HOLD: {np.sum(y==1)} ({100*np.mean(y==1):.1f}%)")
    print(f"   SELL: {np.sum(y==2)} ({100*np.mean(y==2):.1f}%)\n")
    
    return X, y, list(features.keys())

#===============================================================================
# HYPERPARAMETER OPTIMIZATION
#===============================================================================

def optimize_xgboost(X_train, y_train, X_val, y_val):
    """Optimize XGBoost with Optuna"""
    print("🔬 Optimizing XGBoost...")
    
    def objective(trial):
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
            'random_state': 42,
            'n_jobs': -1,
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)
    
    study = optuna.create_study(direction='maximize', study_name='xgboost')
    study.optimize(objective, n_trials=CONFIG['optuna_trials'], show_progress_bar=True)
    
    print(f"   ✅ Best accuracy: {study.best_value:.4f}")
    return study.best_params, study.best_value

def optimize_lightgbm(X_train, y_train, X_val, y_val):
    """Optimize LightGBM with Optuna"""
    print("🔬 Optimizing LightGBM...")
    
    def objective(trial):
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
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
        }
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)
    
    study = optuna.create_study(direction='maximize', study_name='lightgbm')
    study.optimize(objective, n_trials=CONFIG['optuna_trials'], show_progress_bar=True)
    
    print(f"   ✅ Best accuracy: {study.best_value:.4f}")
    return study.best_params, study.best_value

def optimize_histgb(X_train, y_train, X_val, y_val):
    """Optimize HistGradientBoosting with Optuna"""
    print("🔬 Optimizing HistGradientBoosting...")
    
    def objective(trial):
        params = {
            'max_iter': trial.suggest_int('max_iter', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 100),
            'l2_regularization': trial.suggest_float('l2_regularization', 0, 10),
            'random_state': 42,
        }
        
        model = HistGradientBoostingClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)
    
    study = optuna.create_study(direction='maximize', study_name='histgb')
    study.optimize(objective, n_trials=CONFIG['optuna_trials'], show_progress_bar=True)
    
    print(f"   ✅ Best accuracy: {study.best_value:.4f}")
    return study.best_params, study.best_value

#===============================================================================
# ENSEMBLE OPTIMIZATION
#===============================================================================

def optimize_ensemble_weights(models, X_val, y_val):
    """Optimize ensemble weights with Optuna"""
    print("🔬 Optimizing ensemble weights...")
    
    # Get predictions from each model
    all_probs = []
    for model in models:
        probs = model.predict_proba(X_val)
        all_probs.append(probs)
    
    def objective(trial):
        weights = [trial.suggest_float(f'w{i}', 0, 1) for i in range(len(models))]
        total = sum(weights)
        if total == 0:
            return 0
        weights = [w/total for w in weights]
        
        ensemble_probs = np.zeros_like(all_probs[0])
        for prob, weight in zip(all_probs, weights):
            ensemble_probs += prob * weight
        
        preds = np.argmax(ensemble_probs, axis=1)
        return accuracy_score(y_val, preds)
    
    study = optuna.create_study(direction='maximize', study_name='ensemble')
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    weights = [study.best_params[f'w{i}'] for i in range(len(models))]
    total = sum(weights)
    weights = [w/total for w in weights]
    
    print(f"   ✅ Optimal weights: {[f'{w:.3f}' for w in weights]}")
    print(f"   ✅ Ensemble accuracy: {study.best_value:.4f}")
    
    return weights, study.best_value

#===============================================================================
# MAIN OPTIMIZATION PIPELINE
#===============================================================================

def main():
    print("\n" + "="*80)
    print("🎯 NUMERICAL AUTO-OPTIMIZER")
    print("="*80)
    print("Target: 61.7% → 65-67% through automated optimization")
    print("="*80 + "\n")
    
    # Download data
    print("📥 Downloading data...")
    data = {}
    for ticker in CONFIG['tickers']:
        try:
            df = yf.download(ticker, period='3y', interval='1d', progress=False)
            if len(df) > 100:
                data[ticker] = df
        except:
            pass
    print(f"✅ Downloaded {len(data)} tickers\n")
    
    # Create dataset with advanced features
    X, y, feature_names = create_numerical_dataset(data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"📊 Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}\n")
    
    # Scale features
    print("⚙️ Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    print("✅ Features scaled\n")
    
    # Apply SMOTE if available
    if CONFIG['use_smote'] and HAS_SMOTE:
        print("⚖️ Applying SMOTE for class balancing...")
        smote = SMOTE(random_state=42)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        print(f"✅ Resampled to {len(X_train_scaled)} samples\n")
    
    # Optimize models
    models = []
    model_names = []
    model_accuracies = {}
    
    if HAS_XGB:
        best_params_xgb, best_acc_xgb = optimize_xgboost(X_train_scaled, y_train, X_val_scaled, y_val)
        model_xgb = xgb.XGBClassifier(**best_params_xgb)
        model_xgb.fit(X_train_scaled, y_train)
        models.append(model_xgb)
        model_names.append('XGBoost')
        model_accuracies['XGBoost'] = best_acc_xgb
        print()
    
    if HAS_LGB:
        best_params_lgb, best_acc_lgb = optimize_lightgbm(X_train_scaled, y_train, X_val_scaled, y_val)
        model_lgb = lgb.LGBMClassifier(**best_params_lgb)
        model_lgb.fit(X_train_scaled, y_train)
        models.append(model_lgb)
        model_names.append('LightGBM')
        model_accuracies['LightGBM'] = best_acc_lgb
        print()
    
    best_params_histgb, best_acc_histgb = optimize_histgb(X_train_scaled, y_train, X_val_scaled, y_val)
    model_histgb = HistGradientBoostingClassifier(**best_params_histgb)
    model_histgb.fit(X_train_scaled, y_train)
    models.append(model_histgb)
    model_names.append('HistGB')
    model_accuracies['HistGB'] = best_acc_histgb
    print()
    
    # Optimize ensemble
    if CONFIG['ensemble_optimize'] and len(models) > 1:
        weights, ensemble_acc = optimize_ensemble_weights(models, X_val_scaled, y_val)
    else:
        weights = [1.0 / len(models)] * len(models)
        ensemble_acc = 0
    print()
    
    # Final evaluation on test set
    print("🧪 Evaluating on test set...")
    
    # Individual model performance
    print("\n📊 Individual Model Performance:")
    for model, name in zip(models, model_names):
        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        print(f"   {name}: {acc:.4f} ({100*acc:.2f}%)")
    
    # Ensemble performance
    print("\n🎯 Ensemble Performance:")
    ensemble_probs = np.zeros((len(X_test_scaled), 3))
    for model, weight in zip(models, weights):
        probs = model.predict_proba(X_test_scaled)
        ensemble_probs += probs * weight
    
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    ensemble_test_acc = accuracy_score(y_test, ensemble_preds)
    
    print(f"   Weighted Ensemble: {ensemble_test_acc:.4f} ({100*ensemble_test_acc:.2f}%)")
    print(f"   Weights: {dict(zip(model_names, [f'{w:.3f}' for w in weights]))}")
    
    # Detailed classification report
    print(f"\n{'='*80}")
    print("📊 FINAL RESULTS")
    print(f"{'='*80}")
    print(classification_report(y_test, ensemble_preds, target_names=['BUY', 'HOLD', 'SELL'], digits=4))
    print(f"{'='*80}\n")
    
    # Save results
    results = {
        'test_accuracy': float(ensemble_test_acc),
        'validation_accuracy': float(ensemble_acc) if ensemble_acc else None,
        'individual_models': model_accuracies,
        'ensemble_weights': dict(zip(model_names, [float(w) for w in weights])),
        'num_features': len(feature_names),
        'feature_names': feature_names,
        'config': CONFIG,
    }
    
    with open(CHECKPOINT_DIR / 'optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {CHECKPOINT_DIR / 'optimization_results.json'}")
    
    # Success message
    print(f"\n{'='*80}")
    if ensemble_test_acc >= 0.65:
        print("🎉 SUCCESS! Achieved 65%+ accuracy!")
        print(f"   Improvement: 61.7% → {100*ensemble_test_acc:.2f}% (+{100*(ensemble_test_acc-0.617):.2f}%)")
    elif ensemble_test_acc >= 0.63:
        print("✅ Good progress! Above 63%")
        print(f"   Improvement: 61.7% → {100*ensemble_test_acc:.2f}% (+{100*(ensemble_test_acc-0.617):.2f}%)")
    else:
        print("📈 Needs more tuning")
        print(f"   Current: {100*ensemble_test_acc:.2f}%")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
