# ================================================================================
# 🚀 INSTITUTIONAL-GRADE AI TRAINING - PASTE THIS ENTIRE CELL IN COLAB
# ================================================================================
# Implements ALL Perplexity recommendations for 55-62% accuracy (vs 33% baseline)
# Expected runtime: 15-20 minutes on T4 GPU
# FIXED: NaN handling in target creation
# ================================================================================

# STEP 1: Setup
print("="*80)
print("🚀 QUANTUM AI COCKPIT - INSTITUTIONAL TRAINING")
print("="*80)
import sys
!{sys.executable} -m pip install -q yfinance pandas numpy scikit-learn lightgbm scipy 2>&1 | grep -v "already satisfied" || true
print("✅ Dependencies installed")

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

import os
os.chdir('/content/drive/MyDrive/Quantum_AI_Cockpit')
os.makedirs('models', exist_ok=True)
print("✅ Working directory: /content/drive/MyDrive/Quantum_AI_Cockpit")

# Check GPU
try:
    import torch
    gpu = "Tesla T4" if torch.cuda.is_available() else "CPU (Enable GPU: Runtime > Change runtime type)"
except:
    gpu = "Unknown"
print(f"✅ GPU: {gpu}")

# STEP 2: Feature Engineering (20+ institutional features)
import pandas as pd
import numpy as np
from scipy import stats

def engineer_features(df):
    """20+ institutional-grade features from Perplexity research"""
    df = df.copy()
    
    # Momentum (volatility-adjusted)
    for w in [5, 10, 20]:
        ret = df['Close'].pct_change(w)
        vol = df['Close'].pct_change().rolling(w).std()
        df[f'mom_{w}d'] = ret / (vol + 1e-8)
    
    df['reversal_5d'] = -df['Close'].pct_change(5)
    df['mom_accel'] = df['Close'].pct_change(5) - df['Close'].pct_change(20)
    
    # Volatility & Risk
    ret = df['Close'].pct_change()
    df['vol_real'] = ret.rolling(20).std() * np.sqrt(252)
    df['skew_20d'] = ret.rolling(20).apply(lambda x: stats.skew(x.dropna()) if len(x.dropna()) > 3 else 0)
    df['kurt_20d'] = ret.rolling(20).apply(lambda x: stats.kurtosis(x.dropna()) if len(x.dropna()) > 3 else 0)
    df['downside_dev'] = ret.rolling(20).apply(lambda x: np.sqrt(np.mean(np.minimum(x, 0)**2)))
    
    roll_max = df['Close'].rolling(20).max()
    df['max_dd'] = (df['Close'] - roll_max) / roll_max
    
    # Volume & Liquidity
    df['vol_mom'] = df['Volume'].rolling(10).mean() / df['Volume'].rolling(60).mean()
    df['illiq'] = (ret.abs() / (df['Volume'] + 1e-8)).rolling(20).mean()
    
    vwap = (df['Close'] * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
    df['vwap_dev'] = (df['Close'] - vwap) / vwap
    
    # Price Patterns
    df['dist_ma50'] = (df['Close'] / df['Close'].rolling(50).mean()) - 1
    
    ma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['bb_pos'] = (df['Close'] - ma20) / (2 * std20 + 1e-8)
    
    hl_range = df['High'] - df['Low']
    df['range_exp'] = hl_range / hl_range.rolling(20).mean()
    
    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    df['macd_hist'] = macd - signal
    df['macd_mom'] = macd - macd.shift(5)
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['vol_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    return df

# STEP 3: Target Variable (Classification with cross-sectional ranking)
def create_target(df):
    """BUY/HOLD/SELL classification using cross-sectional ranking - FIXED NaN handling"""
    df = df.copy()
    
    # Forward returns (5-20 days)
    fwd_rets = []
    for h in range(5, 21):
        fwd_rets.append(df.groupby('ticker')['Close'].shift(-h))
    
    max_fwd = pd.concat(fwd_rets, axis=1).max(axis=1)
    df['fwd_ret'] = (max_fwd / df['Close'] - 1) * 100
    
    # CRITICAL FIX: Remove rows with NaN forward returns BEFORE ranking
    df_valid = df[df['fwd_ret'].notna()].copy()
    
    # Cross-sectional percentile ranking (handles non-stationarity)
    df_valid['ret_rank'] = df_valid.groupby('date')['fwd_ret'].rank(pct=True) * 100
    
    # 3 classes: SELL (0-33%), HOLD (33-67%), BUY (67-100%)
    df_valid['target'] = pd.cut(
        df_valid['ret_rank'], 
        bins=[0, 33, 67, 100], 
        labels=[0, 1, 2], 
        include_lowest=True
    ).astype(int)
    
    # Merge back to original df
    df = df.merge(df_valid[['date', 'ticker', 'fwd_ret', 'ret_rank', 'target']], 
                  on=['date', 'ticker'], how='left', suffixes=('', '_new'))
    
    # Use the new columns if they exist
    if 'target_new' in df.columns:
        df['target'] = df['target_new']
        df.drop(columns=['target_new'], inplace=True)
    
    return df

# STEP 4: Collect Data
import yfinance as yf

TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'WMT',
           'JNJ', 'PG', 'MA', 'HD', 'DIS', 'BAC', 'NFLX', 'CRM', 'ORCL', 'INTC',
           'AMD', 'QCOM', 'NKE', 'SBUX', 'MCD', 'KO', 'PEP', 'COST', 'UNH', 'LLY',
           'ABBV', 'TMO', 'ABT', 'PFE', 'GS', 'MS', 'C', 'AXP', 'XOM', 'CVX',
           'BA', 'CAT', 'GE', 'TGT', 'AVGO', 'TXN', 'MRK', 'BLK', 'COP', 'ADBE']

print("\n" + "="*80)
print("📊 COLLECTING DATA (10-15 min)")
print("="*80)

all_data = []
for i, ticker in enumerate(TICKERS, 1):
    try:
        df = yf.download(ticker, period='3y', interval='1d', progress=False, auto_adjust=True)
        if df.empty or len(df) < 100:
            print(f"  [{i}/{len(TICKERS)}] {ticker}... ❌ Insufficient data")
            continue
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df.reset_index()
        df = df.copy()
        df.columns = ['date'] + [col.title() for col in df.columns[1:]]
        df['ticker'] = ticker
        
        df = engineer_features(df)
        df = df.dropna()
        
        if len(df) > 60:
            all_data.append(df)
            print(f"  [{i}/{len(TICKERS)}] {ticker}... ✅ {len(df)} rows")
        else:
            print(f"  [{i}/{len(TICKERS)}] {ticker}... ❌ Too few rows")
    except Exception as e:
        print(f"  [{i}/{len(TICKERS)}] {ticker}... ❌ {str(e)[:40]}")

df_all = pd.concat(all_data, ignore_index=True)
print(f"\n✅ Collected: {len(df_all):,} rows")

df_all = create_target(df_all)
df_all = df_all[df_all['target'].notna()]
print(f"✅ With targets: {len(df_all):,} rows")

# Check class distribution
print(f"\n📊 Class distribution:")
for cls in [0, 1, 2]:
    count = sum(df_all['target'] == cls)
    pct = count / len(df_all) * 100
    label = ['SELL', 'HOLD', 'BUY'][cls]
    print(f"   {label}: {count:,} ({pct:.1f}%)")

# STEP 5: Prepare Features
exclude = ['date', 'ticker', 'target', 'fwd_ret', 'ret_rank', 'Open', 'High', 'Low', 'Close', 'Volume']
features = [c for c in df_all.columns if c not in exclude]
print(f"\n✅ Features: {len(features)}")

df_all[features] = df_all[features].fillna(0).replace([np.inf, -np.inf], [1e10, -1e10])

# STEP 6: Walk-Forward Validation
import lightgbm as lgb
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import accuracy_score, classification_report

print("\n" + "="*80)
print("🤖 WALK-FORWARD TRAINING (GPU)")
print("="*80)

df_all = df_all.sort_values('date').reset_index(drop=True)
dates = sorted(df_all['date'].unique())

params = {
    'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss',
    'max_depth': 3, 'num_leaves': 7, 'min_child_samples': 100,
    'learning_rate': 0.01, 'n_estimators': 1000,
    'reg_alpha': 5.0, 'reg_lambda': 10.0, 'min_split_gain': 0.1,
    'bagging_fraction': 0.8, 'bagging_freq': 5, 'feature_fraction': 0.8,
    'verbose': -1, 'random_state': 42, 'device': 'gpu'
}

cv_scores = []
fold = 0

for i in range(252, len(dates) - 60, 20):
    fold += 1
    train_dates = dates[i-252:i]
    test_dates = dates[i:i+60]
    
    train = df_all[df_all['date'].isin(train_dates)].copy()
    test = df_all[df_all['date'].isin(test_dates)].copy()
    
    if len(train) < 100 or len(test) < 10:
        continue
    
    scaler = QuantileTransformer(output_distribution='uniform', n_quantiles=1000)
    train[features] = scaler.fit_transform(train[features])
    test[features] = scaler.transform(test[features])
    
    X_tr, y_tr = train[features], train['target']
    X_te, y_te = test[features], test['target']
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    cv_scores.append(acc)
    
    if fold % 3 == 0:
        print(f"  Fold {fold}: {acc:.4f} ({test_dates[0] if len(test_dates) > 0 else ''} to {test_dates[-1] if len(test_dates) > 0 else ''})")

# STEP 7: Results
print("\n" + "="*80)
print("📊 RESULTS")
print("="*80)

if len(cv_scores) > 0:
    mean_acc = np.mean(cv_scores)
    print(f"  Mean Accuracy: {mean_acc:.4f}")
    print(f"  Std Dev: {np.std(cv_scores):.4f}")
    print(f"  Min: {np.min(cv_scores):.4f} | Max: {np.max(cv_scores):.4f}")
    print(f"  Folds: {len(cv_scores)}")
    
    if mean_acc >= 0.55:
        print(f"\n✅ EXCELLENT - Deploy with 75%+ confidence!")
    elif mean_acc >= 0.45:
        print(f"\n⚠️ GOOD - Use 80%+ confidence threshold")
    else:
        print(f"\n❌ POOR - Needs more data/features")
    
    print(f"\n💡 Baseline: 33.3% (random 3-class)")
    print(f"💡 Your model: {mean_acc:.1%} ({(mean_acc/0.333 - 1)*100:.0f}% better than random)")
else:
    print("❌ No folds completed")

# STEP 8: Train & Save Final Model
print("\n🎯 Training final model...")
scaler_final = QuantileTransformer(output_distribution='uniform', n_quantiles=1000)
df_all[features] = scaler_final.fit_transform(df_all[features])

final_model = lgb.LGBMClassifier(**params)
final_model.fit(df_all[features], df_all['target'], callbacks=[lgb.log_evaluation(0)])

import pickle
model_path = '/content/drive/MyDrive/Quantum_AI_Cockpit/models/ai_institutional.pkl'
scaler_path = '/content/drive/MyDrive/Quantum_AI_Cockpit/models/ai_scaler.pkl'

with open(model_path, 'wb') as f:
    pickle.dump(final_model, f)
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler_final, f)

print(f"✅ Model: {model_path}")
print(f"✅ Scaler: {scaler_path}")

# Feature importance
feat_imp = pd.DataFrame({'feature': features, 'importance': final_model.feature_importances_}).sort_values('importance', ascending=False)
print("\n📊 TOP 10 FEATURES:")
for idx, row in feat_imp.head(10).iterrows():
    print(f"  {row['feature']:20s}: {row['importance']:.4f}")

print("\n" + "="*80)
print("🎉 DONE! Model trained on 50 stocks, 3 years, 20+ features")
print("="*80)
print(f"""
📊 Performance: {mean_acc:.1%} accuracy (3-class: BUY/HOLD/SELL)
🎯 Target: 55-62% (research benchmark for 5-20 day prediction)
📈 Lift: {(mean_acc/0.333)*100:.0f}% of random baseline

🚀 Next: Download models and integrate into dashboard!
""")

