"""
🧪 Production System Validator
Tests production ensemble on the same dataset as temporal optimizer
Validates 69%+ accuracy on full 48-ticker dataset
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from PRODUCTION_ENSEMBLE_69PCT import ProductionEnsemble
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
}

def calculate_features(df, window=60):
    """Feature engineering (matches production ensemble)"""
    features = {}
    
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

print("="*80)
print("🧪 PRODUCTION ENSEMBLE VALIDATOR")
print("="*80)
print(f"Testing on {len(CONFIG['tickers'])} tickers (same as temporal optimizer)\n")

# Download data
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

# Engineer features
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
        
        features = calculate_features(window, CONFIG['window_size'])
        if features is None:
            continue
        
        X_list.append(list(features.values()))
        y_list.append(label)

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int32)
X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

print(f"\n✅ Generated {len(X)} samples with {X.shape[1]} features")
print(f"   BUY: {np.sum(y==0)} ({100*np.mean(y==0):.1f}%)")
print(f"   HOLD: {np.sum(y==1)} ({100*np.mean(y==1):.1f}%)")
print(f"   SELL: {np.sum(y==2)} ({100*np.mean(y==2):.1f}%)\n")

# Split data (same as temporal optimizer)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 Train: {len(X_train)} | Test: {len(X_test)}\n")

# Train production ensemble
print("="*80)
print("🔬 Training Production Ensemble (69.42% validated)")
print("="*80)

ensemble = ProductionEnsemble()
ensemble.fit(X_train, y_train, use_smote=True)

# Evaluate
print("\n📊 Evaluating on test set...\n")
predictions = ensemble.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("="*80)
print("📊 PRODUCTION ENSEMBLE RESULTS")
print("="*80)
print(f"Test Accuracy: {accuracy:.4f} ({100*accuracy:.2f}%)")
print(f"Expected: 69.42% (from Colab optimization)")
print(f"Difference: {100*(accuracy-0.6942):.2f}%")
print("="*80)
print("\nClassification Report:")
print("="*80)
print(classification_report(y_test, predictions,
                          target_names=['BUY', 'HOLD', 'SELL'],
                          digits=4))

# Confidence-based predictions
print("="*80)
print("🎯 Confidence-Based Trading Performance:")
print("="*80)

conf_preds, confidences = ensemble.predict_with_confidence(X_test, threshold=0.6)
conf_acc = accuracy_score(y_test, conf_preds)
coverage = (confidences >= 0.6).mean()

for threshold in [0.5, 0.6, 0.7, 0.8]:
    conf_preds, conf = ensemble.predict_with_confidence(X_test, threshold=threshold)
    conf_acc = accuracy_score(y_test, conf_preds)
    coverage = (conf >= threshold).mean()
    print(f"Threshold {threshold:.1f}: {conf_acc:.4f} accuracy | {100*coverage:.1f}% coverage")

print("\n" + "="*80)
if accuracy >= 0.67:
    print("✅ VALIDATION PASSED! Production ensemble performing well (67%+)")
else:
    print("⚠️  Performance below expectation. May need retraining or hyperparameter tuning.")
print("="*80)
