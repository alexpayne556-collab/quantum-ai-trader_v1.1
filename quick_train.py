"""
🎯 QUICK TRAINING SCRIPT - Train on your watchlist only
No portfolio analysis yet - just get the ML models trained
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

try:
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_ML = True
except ImportError:
    print("❌ ML libraries not found. Install with: pip install xgboost lightgbm scikit-learn")
    exit(1)

# Load watchlist
with open('MY_WATCHLIST.txt', 'r') as f:
    MY_WATCHLIST = [line.strip() for line in f if line.strip()]

print(f"✅ Loaded {len(MY_WATCHLIST)} tickers from watchlist")

# Training function
def collect_training_data(tickers, period='2y'):
    """Collect training data from tickers"""
    X_list, y_list = [], []
    successful_tickers = []
    
    for i, ticker in enumerate(tickers, 1):
        print(f"\r[{i}/{len(tickers)}] Collecting {ticker:6}...", end='', flush=True)
        
        try:
            df = yf.download(ticker, period=period, interval='1d', progress=False)
            
            # Fix multi-index columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Flatten Close column
            if 'Close' in df.columns:
                close_col = df['Close']
                if isinstance(close_col, pd.DataFrame):
                    close_col = close_col.iloc[:, 0]
                df['Close'] = close_col.values.flatten()
            
            if len(df) < 200:
                continue
            
            # Calculate features
            window_size = 60
            horizon = 5
            
            df = df.copy()
            close = df['Close'].values
            returns = np.concatenate([np.diff(close[i:i+horizon]) / (close[i:i+horizon][:-1] + 1e-8) 
                                     for i in range(len(close) - horizon)])
            
            for j in range(window_size, min(len(close), len(returns) + window_size)):
                if j >= len(returns):
                    break
                    
                future_return = returns[j - window_size] if j - window_size < len(returns) else 0
                
                # Label
                if future_return > 0.03:
                    label = 0  # BUY
                elif future_return < -0.03:
                    label = 2  # SELL
                else:
                    label = 1  # HOLD
                
                # Basic features
                window = close[j-window_size:j]
                if len(window) < window_size:
                    continue
                
                features = [
                    np.mean(window),  # price_mean
                    np.std(window),  # price_std
                    (window[-1] - window[-5]) / (window[-5] + 1e-8) if len(window) >= 5 else 0,  # momentum_5
                    (window[-1] - window[-10]) / (window[-10] + 1e-8) if len(window) >= 10 else 0,  # momentum_10
                    np.std(np.diff(window) / (window[:-1] + 1e-8)),  # volatility
                    np.mean(window[-5:]) if len(window) >= 5 else window[-1],  # ma_5
                    np.mean(window[-20:]) if len(window) >= 20 else window[-1],  # ma_20
                ]
                
                X_list.append(features)
                y_list.append(label)
            
            successful_tickers.append(ticker)
            
        except Exception as e:
            continue
    
    print(f"\n✅ Collected data from {len(successful_tickers)} tickers")
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32), successful_tickers

# Collect training data
print("\n" + "="*100)
print("🎓 COLLECTING TRAINING DATA")
print("="*100 + "\n")

X, y, successful_tickers = collect_training_data(MY_WATCHLIST)

if len(X) < 100:
    print(f"\n❌ Insufficient training data ({len(X)} samples)")
    exit(1)

print(f"\n✅ Total samples: {len(X):,}")
print(f"   Features: {X.shape[1]}")
print(f"   Labels: BUY={np.sum(y==0)}, HOLD={np.sum(y==1)}, SELL={np.sum(y==2)}")

# Clean data
X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
print("\n" + "="*100)
print("🎓 TRAINING ML ENSEMBLE")
print("="*100 + "\n")

models = {
    'lightgbm': lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=7, random_state=42, verbose=-1),
    'xgboost': xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=7, random_state=42, verbosity=0),
    'histgb': HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_depth=7, random_state=42)
}

results = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    results[name] = {'train': train_acc, 'test': test_acc}
    print(f"   ✅ Train: {train_acc*100:.1f}% | Test: {test_acc*100:.1f}%")

# Save models
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

for name, model in models.items():
    filepath = models_dir / f"{name}_watchlist.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

with open(models_dir / "scaler.pkl", 'wb') as f:
    pickle.dump(scaler, f)

print(f"\n✅ Models saved to {models_dir}/")

# Save training metadata
training_meta = {
    'trained_at': datetime.now().isoformat(),
    'watchlist_size': len(MY_WATCHLIST),
    'successful_tickers': successful_tickers,
    'total_samples': len(X),
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'features': 7,
    'models': results,
    'best_model': max(results.items(), key=lambda x: x[1]['test'])[0],
    'best_accuracy': max(results.items(), key=lambda x: x[1]['test'])[1]['test']
}

with open('training_metadata.json', 'w') as f:
    json.dump(training_meta, f, indent=2)

print("\n" + "="*100)
print("✅ TRAINING COMPLETE!")
print("="*100)
print(f"\n📊 Best Model: {training_meta['best_model']}")
print(f"   Accuracy: {training_meta['best_accuracy']*100:.1f}%")
print(f"\n✅ Trained on {len(successful_tickers)} tickers:")
print(f"   {', '.join(successful_tickers[:10])}...")
print(f"\n✅ Training metadata saved to training_metadata.json")
print(f"\nNext: Update MY_PORTFOLIO.json with your actual positions, then run analyze_my_portfolio.py")
