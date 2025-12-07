"""
🚀 TEMPORAL ENHANCED OPTIMIZER - Target: 72-75% Accuracy
Builds on 69.42% baseline with advanced temporal modeling

ENHANCEMENTS:
1. Temporal CNN-LSTM with attention (expected +2-3%)
2. Enhanced feature engineering with regime detection (+1-2%)
3. Feature selection with mutual information (+0.5-1%)
4. Stacking ensemble (+1-2%)
5. Confidence-based filtering (+1-2%)

Expected final accuracy: 72-75%
Time on T4 GPU: ~90 minutes

SETUP:
1. Upload to Google Colab Pro
2. Runtime → T4 GPU
3. Run all cells
"""

# ============================================================================
# CELL 1: Install Dependencies
# ============================================================================
"""
!pip install -q yfinance optuna xgboost lightgbm imbalanced-learn scikit-learn torch
print("✅ Dependencies installed")
"""

# ============================================================================
# CELL 2: Imports and Configuration
# ============================================================================

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import optuna
import json
import warnings
warnings.filterwarnings('ignore')

# PyTorch for temporal modeling
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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
    'optuna_trials': 150,  # Increased from 100
    'sequence_length': 10,  # For temporal modeling
}

print("✅ Configuration loaded")

# ============================================================================
# CELL 3: Enhanced Feature Engineering
# ============================================================================

def calculate_enhanced_features(df, window=60):
    """Generate 60+ features with regime detection and interactions"""
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
    
    # ========================================================================
    # ORIGINAL FEATURES (Price statistics, MAs, Momentum, Volatility, Volume)
    # ========================================================================
    
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
    
    # ========================================================================
    # NEW ENHANCED FEATURES
    # ========================================================================
    
    # Regime Detection (5)
    if len(returns) >= 20:
        volatility = np.std(returns[-20:])
        vol_percentile = np.percentile(np.std([returns[max(0, i-20):i] for i in range(20, len(returns))], axis=1), [33, 67])
        features['vol_regime_low'] = 1.0 if volatility < vol_percentile[0] else 0.0
        features['vol_regime_high'] = 1.0 if volatility > vol_percentile[1] else 0.0
    
    # Trend strength (ADX-like)
    if len(high_window) >= 28:
        plus_dm = np.maximum(np.diff(high_window), 0)
        minus_dm = np.maximum(-np.diff(low_window), 0)
        tr = np.maximum(high_window[1:] - low_window[1:],
                       np.maximum(np.abs(high_window[1:] - close_window[:-1]),
                                 np.abs(low_window[1:] - close_window[:-1])))
        
        tr_smooth = np.convolve(tr, np.ones(14)/14, mode='valid')
        plus_dm_smooth = np.convolve(plus_dm, np.ones(14)/14, mode='valid')
        minus_dm_smooth = np.convolve(minus_dm, np.ones(14)/14, mode='valid')
        
        if len(tr_smooth) > 0:
            plus_di = 100 * (plus_dm_smooth / (tr_smooth + 1e-8))
            minus_di = 100 * (minus_dm_smooth / (tr_smooth + 1e-8))
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
            adx = np.mean(dx[-14:]) if len(dx) >= 14 else 50
            
            features['trend_strength'] = adx / 100.0
            features['trend_direction'] = 1.0 if plus_di[-1] > minus_di[-1] else 0.0
    
    # Interaction Features (10)
    features['price_x_volatility'] = features['price_zscore'] * features['volatility_20']
    features['momentum_x_volume'] = features['momentum_10'] * features['volume_ratio']
    features['rsi_proxy'] = (features['price_zscore'] + 3) / 6  # Normalized to [0,1]
    features['volume_x_volatility'] = features['volume_ratio'] * features['volatility_20']
    
    if 'ma_5' in features and 'ma_20' in features and 'ma_50' in features:
        features['ma_convergence'] = abs(features['ma_5'] - features['ma_20']) / (abs(features['ma_20'] - features['ma_50']) + 1e-8)
        features['trend_acceleration'] = (features['ma_5_slope'] - features['ma_20_slope']) / (abs(features['ma_20_slope']) + 1e-8)
    
    # Volume regime
    if len(volume_window) >= 40:
        hist_vol_avg = np.mean(volume_window[-40:-20])
        recent_vol_avg = np.mean(volume_window[-20:])
        features['volume_regime_change'] = recent_vol_avg / (hist_vol_avg + 1e-8)
    
    # Autocorrelation Features (5)
    if len(returns) >= 30:
        autocorr_1 = np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) > 1 else 0
        autocorr_5 = np.corrcoef(returns[:-5], returns[5:])[0, 1] if len(returns) > 5 else 0
        features['autocorr_1'] = autocorr_1 if not np.isnan(autocorr_1) else 0.0
        features['autocorr_5'] = autocorr_5 if not np.isnan(autocorr_5) else 0.0
        features['mean_reversion_strength'] = -autocorr_1
    
    # Price patterns (keep existing 5)
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

print("✅ Enhanced feature engineering loaded")

# ============================================================================
# CELL 4: Temporal CNN-LSTM Model
# ============================================================================

class TemporalCNNLSTM(nn.Module):
    """CNN-LSTM with attention for temporal pattern recognition"""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        
        # 1D CNN for local pattern extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout)
        )
        
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            32, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)
        )
    
    def forward(self, x):
        # x shape: [batch, seq_len, features]
        batch_size, seq_len, features = x.shape
        
        # CNN
        x = x.permute(0, 2, 1)  # [batch, features, seq_len]
        x = self.cnn(x)         # [batch, 32, seq_len]
        x = x.permute(0, 2, 1)  # [batch, seq_len, 32]
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_dim]
        
        # Attention
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # [batch, hidden_dim]
        
        # Classification
        return self.fc(context)

def create_sequences(X, y, seq_len=10):
    """Convert data to sequences for temporal modeling"""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])
    return np.array(X_seq), np.array(y_seq)

print("✅ Temporal model defined")

# ============================================================================
# CELL 5: Download Data and Create Dataset
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

print("🔧 Engineering enhanced features...")
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
        
        features = calculate_enhanced_features(window, CONFIG['window_size'])
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

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

print(f"📊 Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}\n")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# CELL 6: Feature Selection with Mutual Information
# ============================================================================

print("🔬 Selecting features with mutual information...")
mi_scores = mutual_info_classif(X_train_scaled, y_train, random_state=42)
mi_threshold = np.percentile(mi_scores, 30)  # Keep top 70%
selected_features = mi_scores > mi_threshold

print(f"✅ Selected {np.sum(selected_features)} / {len(selected_features)} features")
print(f"   Removed {np.sum(~selected_features)} noisy features\n")

# Apply selection
X_train_selected = X_train_scaled[:, selected_features]
X_val_selected = X_val_scaled[:, selected_features]
X_test_selected = X_test_scaled[:, selected_features]

# Apply SMOTE
print("⚖️ Applying SMOTE for class balancing...")
smote = SMOTE(random_state=42)
X_train_selected, y_train = smote.fit_resample(X_train_selected, y_train)
print(f"✅ Resampled to {len(X_train_selected)} samples\n")

# ============================================================================
# CELL 7: Train Base Models (XGBoost, LightGBM, HistGB)
# ============================================================================

print("🔬 Training base models with optimized hyperparameters...\n")

# Use optimized params from 69.42% run
xgb_params = {
    'max_depth': 9,
    'learning_rate': 0.22975529672912376,
    'n_estimators': 308,
    'subsample': 0.6818680891178277,
    'colsample_bytree': 0.9755172622676036,
    'min_child_weight': 5,
    'gamma': 0.1741229332454554,
    'reg_alpha': 2.6256661239908117,
    'reg_lambda': 5.601071337321665,
    'tree_method': 'hist',
    'device': 'cuda',
    'random_state': 42
}

lgb_params = {
    'num_leaves': 187,
    'max_depth': 12,
    'learning_rate': 0.13636384853167902,
    'n_estimators': 300,
    'subsample': 0.7414206358162381,
    'colsample_bytree': 0.8881981645023311,
    'min_child_samples': 21,
    'reg_alpha': 1.3595268415034327,
    'reg_lambda': 0.004122799441053829,
    'device': 'gpu',
    'random_state': 42,
    'verbose': -1
}

histgb_params = {
    'max_iter': 492,
    'max_depth': 9,
    'learning_rate': 0.2747825638707255,
    'min_samples_leaf': 13,
    'l2_regularization': 2.008590502593976,
    'random_state': 42
}

print("Training XGBoost...")
model_xgb = xgb.XGBClassifier(**xgb_params)
model_xgb.fit(X_train_selected, y_train)
xgb_acc = accuracy_score(y_val, model_xgb.predict(X_val_selected))
print(f"✅ XGBoost: {xgb_acc:.4f}")

print("Training LightGBM...")
model_lgb = lgb.LGBMClassifier(**lgb_params)
model_lgb.fit(X_train_selected, y_train)
lgb_acc = accuracy_score(y_val, model_lgb.predict(X_val_selected))
print(f"✅ LightGBM: {lgb_acc:.4f}")

print("Training HistGradientBoosting...")
model_histgb = HistGradientBoostingClassifier(**histgb_params)
model_histgb.fit(X_train_selected, y_train)
histgb_acc = accuracy_score(y_val, model_histgb.predict(X_val_selected))
print(f"✅ HistGB: {histgb_acc:.4f}\n")

# ============================================================================
# CELL 8: Train Temporal CNN-LSTM
# ============================================================================

print("🔬 Training Temporal CNN-LSTM...")

# Create sequences
seq_len = CONFIG['sequence_length']
X_train_seq, y_train_seq = create_sequences(X_train_selected, y_train, seq_len)
X_val_seq, y_val_seq = create_sequences(X_val_selected, y_val, seq_len)
X_test_seq, y_test_seq = create_sequences(X_test_selected, y_test, seq_len)

print(f"   Sequence shapes: Train={X_train_seq.shape}, Val={X_val_seq.shape}, Test={X_test_seq.shape}")

# Setup device and dataloaders
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Using device: {device}")

train_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_train_seq, dtype=torch.float32),
        torch.tensor(y_train_seq, dtype=torch.long)
    ),
    batch_size=64, shuffle=True
)

val_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_val_seq, dtype=torch.float32),
        torch.tensor(y_val_seq, dtype=torch.long)
    ),
    batch_size=64
)

# Initialize model
temporal_model = TemporalCNNLSTM(
    input_dim=X_train_seq.shape[2],
    hidden_dim=128,
    num_layers=2,
    dropout=0.3
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(temporal_model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# Training loop
best_val_acc = 0
patience_counter = 0
max_patience = 10

for epoch in range(50):
    temporal_model.train()
    train_loss = 0
    train_correct = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = temporal_model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(temporal_model.parameters(), 1.0)
        optimizer.step()
        
        train_loss += loss.item()
        train_correct += (outputs.argmax(1) == y_batch).sum().item()
    
    # Validation
    temporal_model.eval()
    val_correct = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = temporal_model(X_batch)
            val_correct += (outputs.argmax(1) == y_batch).sum().item()
    
    train_acc = train_correct / len(y_train_seq)
    val_acc = val_correct / len(y_val_seq)
    scheduler.step(val_acc)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(temporal_model.state_dict(), '/content/temporal_model_best.pth')
        patience_counter = 0
    else:
        patience_counter += 1
    
    if epoch % 5 == 0 or patience_counter >= max_patience:
        print(f"   Epoch {epoch+1}/50 | Train: {train_acc:.4f} | Val: {val_acc:.4f} | Best: {best_val_acc:.4f}")
    
    if patience_counter >= max_patience:
        print(f"   Early stopping at epoch {epoch+1}")
        break

print(f"✅ Temporal model best validation: {best_val_acc:.4f}\n")

# Load best model
temporal_model.load_state_dict(torch.load('/content/temporal_model_best.pth'))
temporal_model.eval()

# ============================================================================
# CELL 9: Create Stacking Ensemble with Temporal Model
# ============================================================================

print("🔬 Creating stacking ensemble with all models...")

# Wrapper for temporal model
class TemporalWrapper:
    """Wrapper to make temporal model compatible with sklearn"""
    def __init__(self, model, device, seq_len):
        self.model = model
        self.device = device
        self.seq_len = seq_len
    
    def fit(self, X, y):
        return self
    
    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
    
    def predict_proba(self, X):
        # Create sequences
        if len(X) <= self.seq_len:
            # Pad if too short
            X_seq = np.repeat(X[np.newaxis, :, :], self.seq_len, axis=0)
            X_seq = X_seq.transpose(1, 0, 2)
        else:
            X_seq, _ = create_sequences(X, np.zeros(len(X)), self.seq_len)
        
        dataset = TensorDataset(torch.tensor(X_seq, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=64)
        
        probs = []
        self.model.eval()
        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                probs.append(torch.softmax(outputs, dim=1).cpu().numpy())
        
        return np.vstack(probs)

temporal_wrapper = TemporalWrapper(temporal_model, device, seq_len)

# Create stacking classifier
estimators = [
    ('xgb', model_xgb),
    ('lgb', model_lgb),
    ('histgb', model_histgb),
]

stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(
        multi_class='multinomial',
        max_iter=1000,
        random_state=42,
        C=1.0
    ),
    cv=5,
    passthrough=False
)

print("   Fitting stacking ensemble...")
stacking_clf.fit(X_train_selected, y_train)
stacking_acc = accuracy_score(y_val, stacking_clf.predict(X_val_selected))
print(f"✅ Stacking ensemble: {stacking_acc:.4f}\n")

# ============================================================================
# CELL 10: Final Ensemble with Temporal + Stacking
# ============================================================================

print("🎯 Creating final hybrid ensemble...")

# Get predictions from both ensembles
stacking_probs = stacking_clf.predict_proba(X_test_selected)
temporal_probs = temporal_wrapper.predict_proba(X_test_selected[:len(X_test_seq)])

# Weighted combination (tune on validation set)
def optimize_final_weights(val_X, val_y):
    """Find optimal weights for final ensemble"""
    best_acc = 0
    best_weight = 0.5
    
    stacking_val_probs = stacking_clf.predict_proba(val_X)
    temporal_val_probs = temporal_wrapper.predict_proba(val_X[:len(X_val_seq)])
    
    for weight in np.linspace(0, 1, 21):
        # Combine predictions
        combined = weight * stacking_val_probs[:len(temporal_val_probs)] + (1-weight) * temporal_val_probs
        preds = combined.argmax(axis=1)
        acc = accuracy_score(val_y[:len(temporal_val_probs)], preds)
        
        if acc > best_acc:
            best_acc = acc
            best_weight = weight
    
    return best_weight, best_acc

optimal_weight, final_val_acc = optimize_final_weights(X_val_selected, y_val)
print(f"   Optimal stacking weight: {optimal_weight:.3f}")
print(f"✅ Final ensemble validation: {final_val_acc:.4f}\n")

# ============================================================================
# CELL 11: Final Evaluation
# ============================================================================

print("🧪 Evaluating on test set...\n")

# Combine predictions with optimal weight
final_probs = optimal_weight * stacking_probs[:len(temporal_probs)] + (1-optimal_weight) * temporal_probs
final_preds = final_probs.argmax(axis=1)
final_acc = accuracy_score(y_test[:len(temporal_probs)], final_preds)

print("="*80)
print("📊 FINAL RESULTS - Temporal Enhanced Ensemble")
print("="*80)
print(f"Test Accuracy: {final_acc:.4f} ({100*final_acc:.2f}%)")
print(f"Improvement from baseline: {100*(final_acc-0.6942):.2f}%")
print(f"Total improvement from 61.7%: +{100*(final_acc-0.617):.2f}%")
print("="*80)
print("\nClassification Report:")
print("="*80)
print(classification_report(y_test[:len(temporal_probs)], final_preds,
                          target_names=['BUY', 'HOLD', 'SELL'],
                          digits=4))
print("="*80)

# Component performance
print("\n📊 Component Model Performance:")
print(f"   XGBoost: {xgb_acc:.4f}")
print(f"   LightGBM: {lgb_acc:.4f}")
print(f"   HistGB: {histgb_acc:.4f}")
print(f"   Temporal CNN-LSTM: {best_val_acc:.4f}")
print(f"   Stacking Ensemble: {stacking_acc:.4f}")
print(f"   Final Hybrid: {final_acc:.4f}")

# ============================================================================
# CELL 12: Confidence-Based Predictions
# ============================================================================

print("\n" + "="*80)
print("🎯 Confidence-Based Trading Performance:")
print("="*80)

for threshold in [0.5, 0.6, 0.7, 0.8]:
    confidences = final_probs.max(axis=1)
    conf_preds = final_preds.copy()
    conf_preds[confidences < threshold] = 1  # Set to HOLD
    
    conf_acc = accuracy_score(y_test[:len(temporal_probs)], conf_preds)
    coverage = (confidences >= threshold).mean()
    
    print(f"Threshold {threshold:.1f}: {conf_acc:.4f} accuracy | {100*coverage:.1f}% coverage")

# Save results
results = {
    'final_accuracy': float(final_acc),
    'baseline_accuracy': 0.6942,
    'improvement': float(final_acc - 0.6942),
    'total_improvement_from_617': float(final_acc - 0.617),
    'component_accuracies': {
        'xgboost': float(xgb_acc),
        'lightgbm': float(lgb_acc),
        'histgb': float(histgb_acc),
        'temporal_cnn_lstm': float(best_val_acc),
        'stacking_ensemble': float(stacking_acc)
    },
    'optimal_stacking_weight': float(optimal_weight),
    'num_features': int(np.sum(selected_features)),
    'total_features_engineered': len(feature_names)
}

with open('/content/temporal_enhanced_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n💾 Results saved to: /content/temporal_enhanced_results.json")

if final_acc >= 0.72:
    print(f"\n🎉🎉🎉 SUCCESS! Achieved 72%+ accuracy target!")
    print(f"   Target: 72-75% | Achieved: {100*final_acc:.2f}%")
elif final_acc >= 0.70:
    print(f"\n✅ Excellent progress! Above 70%")
    print(f"   Current: {100*final_acc:.2f}% | Target: 72-75%")
else:
    print(f"\n📈 Current: {100*final_acc:.2f}% | Target: 72-75%")

print("\n✅ COMPLETE! Download /content/temporal_enhanced_results.json\n")
