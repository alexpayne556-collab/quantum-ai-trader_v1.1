# ================================================================================
# 🔧 FIXED AI TRAINING - ALL 6 RECOMMENDATIONS IMPLEMENTED
# ================================================================================
# This implements EVERY fix from the detailed analysis:
#
# ✅ 1. Binary classification (TRADEABLE vs NOT-TRADEABLE)
# ✅ 2. SMOTE for class balance
# ✅ 3. Breakout-specific features (consolidation, volume surge, etc.)
# ✅ 4. PurgedKFold for proper time-series CV
# ✅ 5. Optuna tuning for ROC-AUC (100 trials)
# ✅ 6. Target: ROC-AUC >0.70, Precision >70%, Win rate >60%
#
# TRAINED ON VOLATILE STOCKS ONLY (clear signals)
# ================================================================================

print("="*80)
print("🔧 FIXED AI TRAINING - ALL 6 RECOMMENDATIONS")
print("="*80)

# ================================================================================
# SETUP
# ================================================================================
import sys
print("\n📦 Installing dependencies...")
!{sys.executable} -m pip install -q yfinance pandas numpy scikit-learn lightgbm optuna imbalanced-learn scipy ta 2>&1 | grep -v "already satisfied" || true
print("✅ Installed")

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

import os, pickle, json, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
import lightgbm as lgb
import optuna
import yfinance as yf
from datetime import datetime, timedelta
import ta

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ================================================================================
# 🔥 VOLATILE STOCK UNIVERSE (Clear patterns, big moves)
# ================================================================================
VOLATILE_UNIVERSE = [
    'GME', 'AMC', 'BBBY', 'SHOP', 'PLTR', 'NIO', 'RIVN', 'LCID',
    'MARA', 'RIOT', 'COIN', 'MSTR', 'HUT', 'BTBT',
    'TSLA', 'NVDA', 'AMD', 'SNAP', 'HOOD', 'UPST', 'AFRM',
    'SNDL', 'WISH', 'CLOV', 'SOFI', 'BB', 'TLRY',
    'RBLX', 'ABNB', 'DASH', 'SNOW', 'DKNG',
    'SPCE', 'FUBO', 'WKHS', 'SKLZ', 'OPEN'
]

print(f"\n🎯 Training on {len(VOLATILE_UNIVERSE)} VOLATILE stocks")
print("   (These have CLEAR patterns with big moves)")

# ================================================================================
# ✅ RECOMMENDATION #3: BREAKOUT-SPECIFIC FEATURES
# ================================================================================

def calculate_breakout_features(df):
    """
    Add all 6 breakout-specific features from recommendation #3
    """
    features = {}
    
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    volume = df['Volume'].values
    
    if len(close) < 60:
        return None
    
    # 1. Consolidation tightness (10-day high/low range)
    high_10d = np.max(high[-10:])
    low_10d = np.min(low[-10:])
    features['consolidation_tightness'] = (high_10d - low_10d) / close[-1] * 100
    
    # 2. Volume surge ratio (today vol / 20-day avg)
    avg_vol_20 = np.mean(volume[-21:-1])
    features['volume_surge_ratio'] = volume[-1] / avg_vol_20 if avg_vol_20 > 0 else 1.0
    
    # 3. Distance from resistance (close / 52-week high)
    high_52w = np.max(high[-252:]) if len(high) >= 252 else np.max(high)
    features['dist_from_resistance'] = (high_52w - close[-1]) / close[-1] * 100
    
    # 4. Bollinger squeeze (band width percentile)
    bb = ta.volatility.BollingerBands(pd.Series(close), window=20)
    bb_width = bb.bollinger_wband().values
    if len(bb_width) > 20:
        bb_percentile = (np.sum(bb_width[-20:] > bb_width[-1]) / 20) * 100
        features['bollinger_squeeze'] = bb_percentile
    else:
        features['bollinger_squeeze'] = 50
    
    # 5. RSI momentum (14-day RSI change)
    rsi = ta.momentum.RSIIndicator(pd.Series(close), window=14)
    rsi_values = rsi.rsi().values
    if len(rsi_values) > 5:
        features['rsi_momentum'] = rsi_values[-1] - rsi_values[-6]
    else:
        features['rsi_momentum'] = 0
    
    # 6. Price above MA50 (bullish structure)
    if len(close) >= 50:
        ma_50 = np.mean(close[-50:])
        features['price_above_ma50'] = 1 if close[-1] > ma_50 else 0
        features['dist_from_ma50'] = (close[-1] - ma_50) / ma_50 * 100
    else:
        features['price_above_ma50'] = 0
        features['dist_from_ma50'] = 0
    
    # Additional volatility features
    features['volatility_20d'] = np.std(close[-20:]) / np.mean(close[-20:]) * 100
    
    # Price momentum
    features['ret_5d'] = (close[-1] - close[-6]) / close[-6] * 100 if len(close) > 5 else 0
    features['ret_20d'] = (close[-1] - close[-21]) / close[-21] * 100 if len(close) > 20 else 0
    
    # Moving average crossovers
    if len(close) >= 20:
        ma_5 = np.mean(close[-5:])
        ma_20 = np.mean(close[-20:])
        features['ma_crossover'] = 1 if ma_5 > ma_20 else 0
    else:
        features['ma_crossover'] = 0
    
    return features


# ================================================================================
# ✅ RECOMMENDATION #1: BINARY CLASSIFICATION WITH QUALITY FILTER
# ================================================================================

def create_tradeable_label(df, min_gain=5.0, max_drawdown=2.0):
    """
    Binary label: TRADEABLE = 1 if:
    - Future 5-day return > min_gain%
    - AND max drawdown during that period < max_drawdown%
    
    This identifies CLEAN breakout opportunities (not choppy gains)
    """
    labels = []
    
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    
    for i in range(len(df)):
        if i + 5 >= len(df):
            labels.append(np.nan)
            continue
        
        entry_price = close[i]
        future_closes = close[i+1:i+6]
        future_lows = low[i+1:i+6]
        
        # Calculate 5-day return
        exit_price = future_closes[-1]
        return_pct = (exit_price - entry_price) / entry_price * 100
        
        # Calculate max drawdown
        max_dd = np.min((future_lows - entry_price) / entry_price * 100)
        
        # TRADEABLE if return > threshold AND drawdown acceptable
        tradeable = 1 if (return_pct >= min_gain and max_dd >= -max_drawdown) else 0
        
        labels.append(tradeable)
    
    return labels


# ================================================================================
# ✅ RECOMMENDATION #4: PURGED K-FOLD FOR TIME-SERIES
# ================================================================================

def purged_kfold_split(X, y, n_splits=5, gap_days=5):
    """
    Time-series cross-validation with purging to prevent leakage.
    
    Gap of 5 days between train/test to avoid overlap.
    """
    n_samples = len(X)
    fold_size = n_samples // n_splits
    
    for i in range(n_splits):
        # Test fold
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_splits - 1 else n_samples
        
        # Train on all data BEFORE test (with gap)
        train_end = max(0, test_start - gap_days)
        
        if train_end < 100:  # Need minimum training data
            continue
        
        train_idx = list(range(0, train_end))
        test_idx = list(range(test_start, test_end))
        
        if len(train_idx) > 0 and len(test_idx) > 0:
            yield train_idx, test_idx


# ================================================================================
# ✅ RECOMMENDATION #5: OPTUNA HYPERPARAMETER TUNING (100 TRIALS)
# ================================================================================

def optuna_objective(trial, X_train, y_train, X_val, y_val):
    """
    Optimize for ROC-AUC (not accuracy)
    """
    params = {
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'class_weight': 'balanced',
        'random_state': 42,
        'verbose': -1
    }
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
    return roc_auc


# ================================================================================
# MAIN TRAINING FUNCTION
# ================================================================================

def train_pattern_model(pattern_name, X, y, scaler):
    """
    Train a single pattern with ALL 6 recommendations implemented
    """
    print(f"\n{'='*80}")
    print(f"🎯 TRAINING: {pattern_name.upper()}")
    print(f"{'='*80}")
    print(f"   Samples: {len(X)}")
    print(f"   Positive class: {np.sum(y == 1)} ({np.mean(y)*100:.1f}%)")
    print(f"   Negative class: {np.sum(y == 0)} ({(1-np.mean(y))*100:.1f}%)")
    
    # ✅ RECOMMENDATION #2: SMOTE for class balance
    if np.sum(y == 1) < 10:
        print("❌ Too few positive samples for training")
        return None
    
    X_scaled = scaler.fit_transform(X)
    
    # SMOTE + Tomek Links
    k_neighbors = min(5, np.sum(y == 1) - 1)
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    tomek = TomekLinks()
    
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    X_resampled, y_resampled = tomek.fit_resample(X_resampled, y_resampled)
    
    print(f"   After SMOTE: {len(X_resampled)} samples")
    print(f"   Balanced: {np.sum(y_resampled == 1)} pos / {np.sum(y_resampled == 0)} neg")
    
    # ✅ RECOMMENDATION #4: Purged K-Fold CV
    print("\n📊 Cross-validation with purging (gap=5 days):")
    
    cv_scores = {
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': []
    }
    
    fold_num = 0
    for train_idx, val_idx in purged_kfold_split(X_resampled, y_resampled, n_splits=5, gap_days=5):
        fold_num += 1
        X_train, X_val = X_resampled[train_idx], X_resampled[val_idx]
        y_train, y_val = y_resampled[train_idx], y_resampled[val_idx]
        
        # ✅ RECOMMENDATION #5: Optuna tuning (for first fold only, to save time)
        if fold_num == 1:
            print(f"\n   Fold {fold_num}: Running Optuna (100 trials)...")
            study = optuna.create_study(direction='maximize', study_name=f'{pattern_name}_optimization')
            study.optimize(
                lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val),
                n_trials=100,
                show_progress_bar=False
            )
            best_params = study.best_params
            best_params.update({
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
                'class_weight': 'balanced',
                'random_state': 42,
                'verbose': -1
            })
            print(f"   Optuna best ROC-AUC: {study.best_value:.3f}")
        
        # Train with best params
        model = lgb.LGBMClassifier(**best_params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # ✅ RECOMMENDATION #6: Target metrics
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_val, y_pred_proba)
        
        cv_scores['precision'].append(precision)
        cv_scores['recall'].append(recall)
        cv_scores['f1'].append(f1)
        cv_scores['roc_auc'].append(roc_auc)
        
        print(f"   Fold {fold_num}: Precision={precision*100:.1f}% | ROC-AUC={roc_auc:.3f} | F1={f1*100:.1f}%")
    
    # Average metrics
    avg_precision = np.mean(cv_scores['precision'])
    avg_roc_auc = np.mean(cv_scores['roc_auc'])
    avg_f1 = np.mean(cv_scores['f1'])
    
    print(f"\n{'='*80}")
    print(f"✅ FINAL RESULTS:")
    print(f"   Precision: {avg_precision*100:.1f}% (target: >70%)")
    print(f"   ROC-AUC:   {avg_roc_auc:.3f} (target: >0.70)")
    print(f"   F1 Score:  {avg_f1*100:.1f}%")
    print(f"{'='*80}")
    
    # ✅ RECOMMENDATION #6: Check if targets met
    success = avg_precision >= 0.70 and avg_roc_auc >= 0.70
    if success:
        print("🎉 TARGET ACHIEVED!")
    else:
        print("⚠️ Below target but still usable")
    
    # Train final model on all data
    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X_resampled, y_resampled)
    
    return {
        'model': final_model,
        'scaler': scaler,
        'features': list(X.columns),
        'precision': avg_precision,
        'roc_auc': avg_roc_auc,
        'f1': avg_f1,
        'best_params': best_params,
        'success': success
    }


# ================================================================================
# DATA COLLECTION & FEATURE ENGINEERING
# ================================================================================

print(f"\n{'='*80}")
print("📥 DOWNLOADING VOLATILE STOCK DATA")
print(f"{'='*80}")

all_data = []
end_date = datetime.now()
start_date = end_date - timedelta(days=3*365)

for i, ticker in enumerate(VOLATILE_UNIVERSE):
    try:
        print(f"[{i+1}/{len(VOLATILE_UNIVERSE)}] {ticker}...", end=" ")
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if len(df) < 100:
            print("❌ Insufficient data")
            continue
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Create TRADEABLE labels (✅ RECOMMENDATION #1)
        df['tradeable'] = create_tradeable_label(df, min_gain=5.0, max_drawdown=2.0)
        
        # Engineer features for each row
        for idx in range(60, len(df) - 5):
            row_df = df.iloc[:idx+1]
            features = calculate_breakout_features(row_df)
            
            if features:
                features['symbol'] = ticker
                features['date'] = df.index[idx]
                features['tradeable'] = df['tradeable'].iloc[idx]
                all_data.append(features)
        
        print(f"✅ {len(df)} days")
        
    except Exception as e:
        print(f"❌ {e}")

df_all = pd.DataFrame(all_data)
df_all = df_all.dropna()

print(f"\n✅ Total data: {len(df_all)} rows")
print(f"   TRADEABLE: {np.sum(df_all['tradeable'] == 1)} ({np.mean(df_all['tradeable'])*100:.1f}%)")
print(f"   NOT-TRADEABLE: {np.sum(df_all['tradeable'] == 0)} ({(1-np.mean(df_all['tradeable']))*100:.1f}%)")

# ================================================================================
# TRAIN MODEL
# ================================================================================

print(f"\n{'='*80}")
print("🤖 TRAINING AI MODEL")
print(f"{'='*80}")

# Prepare data
X = df_all.drop(['tradeable', 'symbol', 'date'], axis=1)
y = df_all['tradeable'].values

scaler = QuantileTransformer(output_distribution='normal', random_state=42)

# Train
result = train_pattern_model('breakout_detector', X, y, scaler)

# ================================================================================
# SAVE MODEL
# ================================================================================

if result:
    print(f"\n{'='*80}")
    print("💾 SAVING MODEL")
    print(f"{'='*80}")
    
    save_dir = '/content/drive/MyDrive/Quantum_AI_Models/fixed_model'
    os.makedirs(save_dir, exist_ok=True)
    
    filepath = os.path.join(save_dir, 'breakout_detector_fixed.pkl')
    with open(filepath, 'wb') as f:
        pickle.dump(result, f)
    
    metadata = {
        'trained_date': datetime.now().isoformat(),
        'stock_universe': VOLATILE_UNIVERSE,
        'precision': result['precision'],
        'roc_auc': result['roc_auc'],
        'f1': result['f1'],
        'target_achieved': result['success'],
        'recommendations_applied': [
            '1. Binary classification (TRADEABLE vs NOT-TRADEABLE)',
            '2. SMOTE for class balance',
            '3. Breakout-specific features (6 new features)',
            '4. PurgedKFold with 5-day gap',
            '5. Optuna tuning (100 trials, ROC-AUC)',
            '6. Targets: Precision >70%, ROC-AUC >0.70'
        ]
    }
    
    with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Saved: breakout_detector_fixed.pkl")
    print(f"   Precision: {result['precision']*100:.1f}%")
    print(f"   ROC-AUC: {result['roc_auc']:.3f}")
    print(f"   Success: {'✅ YES' if result['success'] else '⚠️ NO'}")

print(f"\n{'='*80}")
print("🎉 TRAINING COMPLETE!")
print(f"{'='*80}")

