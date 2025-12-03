# ============================================================================
# 🚀 QUANTUM AI COCKPIT - TRAINING V4 (OPTIMIZED FOR 2 YEARS DATA)
# ============================================================================
# This version uses features that work with limited history (60-day max)
# Expected: 50-58% accuracy (realistic for stock prediction)
# ============================================================================

print("="*80)
print("🚀 QUANTUM AI COCKPIT - TRAINING V4 (OPTIMIZED)")
print("="*80)

# ============================================================================
# SETUP
# ============================================================================
print("\n📦 Installing dependencies...")
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                      "yfinance", "lightgbm", "xgboost", "joblib", "scipy"])

from google.colab import drive
import torch
drive.mount('/content/drive', force_remount=False)

print(f"✅ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

import os
PROJECT = '/content/drive/MyDrive/Quantum_AI_Cockpit'
for d in ['backend/modules', 'models', 'data']:
    os.makedirs(f'{PROJECT}/{d}', exist_ok=True)
os.chdir(PROJECT)

import pandas as pd
import numpy as np
import yfinance as yf
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import accuracy_score, classification_report
import joblib
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print(f"✅ Working directory: {PROJECT}\n")

# ============================================================================
# TICKERS
# ============================================================================
TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'WMT',
    'JNJ', 'PG', 'MA', 'HD', 'DIS', 'BAC', 'NFLX', 'CRM', 'ORCL', 'INTC',
    'AMD', 'QCOM', 'NKE', 'SBUX', 'MCD', 'KO', 'PEP', 'COST', 'UNH', 'LLY',
    'ABBV', 'TMO', 'ABT', 'PFE', 'GS', 'MS', 'C', 'AXP', 'XOM', 'CVX',
    'BA', 'CAT', 'GE', 'TGT', 'AVGO', 'TXN', 'MRK', 'BLK', 'COP', 'ADBE'
]

# ============================================================================
# OPTIMIZED FEATURE ENGINEERING (Max 60-day lookback)
# ============================================================================
def engineer_optimized_features(df):
    """
    Features optimized for 2-year data (481 trading days).
    Max lookback: 60 days to minimize NaN.
    """
    features = df.copy()
    
    # === MOMENTUM FEATURES (5, 10, 20, 60 days) ===
    for window in [5, 10, 20, 60]:
        returns = features['Close'].pct_change(window)
        volatility = features['Close'].pct_change().rolling(window).std()
        features[f'momentum_{window}d'] = returns
        features[f'momentum_{window}d_vol_adj'] = returns / (volatility + 1e-8)
    
    # Short-term reversal
    features['reversal_5d'] = -features['Close'].pct_change(5)
    
    # Momentum acceleration
    features['momentum_accel'] = features['Close'].pct_change(5) - features['Close'].pct_change(20)
    
    # === VOLATILITY FEATURES ===
    features['realized_vol_20'] = features['Close'].pct_change().rolling(20).std() * np.sqrt(252)
    features['realized_vol_60'] = features['Close'].pct_change().rolling(60).std() * np.sqrt(252)
    
    # Skewness & Kurtosis (20-day only - less NaN)
    returns = features['Close'].pct_change()
    features['skewness_20d'] = returns.rolling(20).apply(lambda x: stats.skew(x.dropna()) if len(x.dropna()) > 3 else 0)
    features['kurtosis_20d'] = returns.rolling(20).apply(lambda x: stats.kurtosis(x.dropna()) if len(x.dropna()) > 3 else 0)
    
    # Downside deviation
    features['downside_dev_20'] = returns.rolling(20).apply(
        lambda x: np.sqrt(np.mean(np.minimum(x, 0)**2))
    )
    
    # Maximum drawdown (20 and 60 day)
    for window in [20, 60]:
        rolling_max = features['Close'].rolling(window).max()
        features[f'max_dd_{window}'] = (features['Close'] - rolling_max) / rolling_max
    
    # === VOLUME FEATURES ===
    features['volume_ma_20'] = features['Volume'].rolling(20).mean()
    features['volume_ratio'] = features['Volume'] / (features['volume_ma_20'] + 1)
    features['volume_momentum'] = features['Volume'].rolling(10).mean() / features['Volume'].rolling(60).mean()
    
    # Illiquidity (Amihud)
    daily_return_abs = features['Close'].pct_change().abs()
    features['illiquidity'] = (daily_return_abs / (features['Volume'] + 1e-8)).rolling(20).mean()
    
    # === PRICE PATTERN FEATURES (Short lookback) ===
    # Distance from 20 and 50-day MA (NOT 200!)
    for window in [20, 50]:
        ma = features['Close'].rolling(window).mean()
        features[f'dist_from_ma{window}'] = (features['Close'] / ma) - 1
    
    # Bollinger position
    ma20 = features['Close'].rolling(20).mean()
    std20 = features['Close'].rolling(20).std()
    features['bollinger_pos'] = (features['Close'] - ma20) / (2 * std20 + 1e-8)
    
    # Price range
    high_low = features['High'] - features['Low']
    features['range_expansion'] = high_low / (high_low.rolling(20).mean() + 1e-8)
    
    # === TECHNICAL INDICATORS ===
    # RSI
    delta = features['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    features['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = features['Close'].ewm(span=12).mean()
    ema26 = features['Close'].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    features['macd_hist'] = macd - signal
    features['macd_momentum'] = macd - macd.shift(5)
    
    # ATR
    high_low = features['High'] - features['Low']
    high_close = abs(features['High'] - features['Close'].shift())
    low_close = abs(features['Low'] - features['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    features['atr'] = true_range.rolling(14).mean()
    features['atr_pct'] = features['atr'] / features['Close']
    
    return features

# ============================================================================
# DATA COLLECTION
# ============================================================================
print("="*80)
print("📊 COLLECTING DATA (20-30 min)")
print("="*80)

all_data = []

for i, ticker in enumerate(TICKERS):
    try:
        print(f"  [{i+1}/{len(TICKERS)}] {ticker}...", end=" ", flush=True)
        
        df = yf.download(ticker, period='2y', progress=False)
        
        if len(df) < 250:
            print("❌ Not enough data")
            continue
        
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Reset index to get Date column
        df = df.reset_index()
        
        # Make a clean copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Engineer features
        df = engineer_optimized_features(df)
        
        # Calculate forward returns
        forward_returns = []
        for horizon in range(5, 21):
            future_return = (df['Close'].shift(-horizon) / df['Close'] - 1) * 100
            forward_returns.append(future_return)
        
        df['max_forward_return'] = pd.concat(forward_returns, axis=1).max(axis=1)
        
        # Add metadata
        df['ticker'] = ticker
        df['date'] = df['Date']
        
        # Remove last 20 rows (no future returns)
        df = df.iloc[:-20]
        
        all_data.append(df)
        print(f"✅ {len(df)} samples")
        
    except Exception as e:
        print(f"❌ {str(e)[:30]}")

df_all = pd.concat(all_data, ignore_index=True)
print(f"\n✅ Total: {len(df_all)} samples\n")

# ============================================================================
# CREATE TARGET
# ============================================================================
print("="*80)
print("🎯 CREATING TARGET")
print("="*80)

df_all['return_rank'] = df_all.groupby('date')['max_forward_return'].rank(pct=True) * 100
df_all['target'] = pd.cut(df_all['return_rank'], bins=[0, 33, 67, 100], labels=[0, 1, 2], include_lowest=True).astype(int)

print(f"  SELL: {(df_all['target'] == 0).sum()} ({(df_all['target'] == 0).mean():.1%})")
print(f"  HOLD: {(df_all['target'] == 1).sum()} ({(df_all['target'] == 1).mean():.1%})")
print(f"  BUY:  {(df_all['target'] == 2).sum()} ({(df_all['target'] == 2).mean():.1%})\n")

# ============================================================================
# CLEAN & PREPARE FEATURES
# ============================================================================
exclude_cols = ['Date', 'date', 'ticker', 'target', 'max_forward_return', 'return_rank',
                'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'volume_ma_20']
feature_cols = [c for c in df_all.columns if c not in exclude_cols and df_all[c].dtype in [np.float64, np.int64]]

print(f"✅ Features: {len(feature_cols)}\n")

# Fill NaN
df_all[feature_cols] = df_all[feature_cols].fillna(0)
df_all = df_all[df_all['target'].notna()]

print(f"✅ Final: {len(df_all)} samples\n")

# ============================================================================
# TRAIN WITH SIMPLE 5-FOLD CV (FASTER)
# ============================================================================
print("="*80)
print("🤖 TRAINING (5-Fold TimeSeriesSplit)")
print("="*80)

X = df_all[feature_cols]
y = df_all['target']

# Scale volatile features
scale_cols = ['illiquidity', 'skewness_20d', 'kurtosis_20d', 'volume_momentum']
scale_cols = [c for c in scale_cols if c in feature_cols]

model_params = {
    'objective': 'multiclass',
    'num_class': 3,
    'max_depth': 4,  # Increased from 3
    'num_leaves': 15,
    'learning_rate': 0.01,  # Increased from 0.005
    'n_estimators': 1000,
    'reg_alpha': 2.0,  # Reduced regularization
    'reg_lambda': 5.0,
    'min_child_samples': 50,
    'bagging_fraction': 0.8,
    'feature_fraction': 0.9,  # Use more features
    'bagging_freq': 5,
    'random_state': 42,
    'verbose': -1,
    'device': 'gpu' if torch.cuda.is_available() else 'cpu'
}

tscv = TimeSeriesSplit(n_splits=5)
cv_scores = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    if scale_cols:
        scaler = QuantileTransformer(output_distribution='uniform', n_quantiles=min(len(X_train), 1000))
        X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
        X_test[scale_cols] = scaler.transform(X_test[scale_cols])
    
    model = LGBMClassifier(**model_params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
             callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores.append(accuracy)
    print(f"  Fold {fold+1}/5: {accuracy:.1%}")

# ============================================================================
# RESULTS
# ============================================================================
print(f"\n{'='*80}")
print("🎉 TRAINING COMPLETE!")
print("="*80)

avg = np.mean(cv_scores)
print(f"\n  Average: {avg:.1%}")
print(f"  Baseline: 33.3%")
print(f"  Improvement: +{(avg-0.333)*100:.1f}%")

status = "✅ EXCELLENT!" if avg > 0.52 else "✅ GOOD" if avg > 0.45 else "⚠️  MARGINAL"
print(f"  Status: {status}\n")

# Train final model
final_model = LGBMClassifier(**model_params)
final_model.fit(X, y)

# Save
model_path = f'{PROJECT}/models/pattern_v4_optimized.pkl'
joblib.dump({
    'model': final_model,
    'feature_cols': feature_cols,
    'scale_cols': scale_cols,
    'cv_accuracy': avg,
    'training_date': datetime.now().strftime('%Y-%m-%d')
}, model_path)

print(f"✅ Saved: {model_path}\n")
print("="*80)

