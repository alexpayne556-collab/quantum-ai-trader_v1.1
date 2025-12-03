"""
AINVEST-STYLE PATTERN TRAINING - PHASE 2
Train 7 additional patterns to reach 10 total trained patterns

Copy this entire file to Google Colab and run it!
"""

import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Mount Drive (run this first in Colab)
"""
from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.insert(0, '/content/drive/MyDrive/Quantum_AI_Cockpit')
"""

print("="*80)
print("🚀 PHASE 2: TRAINING 7 MORE PATTERNS")
print("="*80)
print("\nTarget patterns:")
print("  1. Golden Cross (50/200 MA)")
print("  2. Bullish Flag")
print("  3. Double Bottom")
print("  4. RSI Divergence")
print("  5. Breakout Above 20-Day High")
print("  6. EMA Ribbon Bullish")
print("  7. Head & Shoulders (Inverse)")
print("\n" + "="*80)

# ========================================
# STOCK UNIVERSE (86 tickers)
# ========================================

TICKERS = [
    # Mega caps
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'BRK-B',
    # Tech
    'ADBE', 'NFLX', 'CRM', 'ORCL', 'INTC', 'AMD', 'QCOM', 'AVGO', 'TXN',
    # Finance
    'JPM', 'BAC', 'GS', 'MS', 'C', 'AXP', 'V', 'MA',
    # Consumer
    'WMT', 'HD', 'DIS', 'NKE', 'SBUX', 'MCD', 'KO', 'PEP', 'COST', 'TGT',
    # Healthcare
    'JNJ', 'UNH', 'LLY', 'ABBV', 'TMO', 'ABT', 'PFE', 'MRK',
    # Energy
    'XOM', 'CVX', 'COP',
    # Industrial
    'BA', 'CAT', 'GE',
    # Growth/Cloud
    'UBER', 'LYFT', 'PYPL', 'SHOP', 'ROKU', 'ZM', 'DOCU', 'SNOW',
    'DDOG', 'NET', 'CRWD', 'ZS', 'OKTA', 'TWLO', 'DKNG', 'COIN',
    # Newer growth
    'RIVN', 'PLTR', 'SOFI', 'LCID', 'SPCE', 'HOOD', 'UPST', 'MSTR',
    # ETFs
    'QQQ', 'SPY', 'IWM', 'DIA', 'VTI', 'VOO', 'SOXX', 'XLK', 'SMH',
    # Crypto-related
    'MARA', 'RIOT',
    # ARK funds
    'ARKK', 'ARKF', 'ARKG', 'ARKW', 'PG'
]

# ========================================
# PATTERN DETECTION FUNCTIONS
# ========================================

def add_technical_indicators(df):
    """Add all technical indicators needed for pattern detection"""
    
    # Price features
    df['returns'] = df['Close'].pct_change()
    df['high_low_range'] = (df['High'] - df['Low']) / df['Low']
    
    # Moving Averages (many periods for different patterns)
    for period in [5, 8, 10, 13, 20, 21, 50, 100, 200]:
        df[f'sma_{period}'] = df['Close'].rolling(period).mean()
        df[f'ema_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
    
    # Volume
    df['volume_sma_20'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26'] if 'ema_12' in df.columns else df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['macd_signal'] = df['macd'].rolling(9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_mid'] = df['Close'].rolling(20).mean()
    df['bb_std'] = df['Close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * 2)
    df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # ATR
    df['tr'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(14).mean()
    
    # Momentum
    df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
    df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
    df['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
    
    # Support/Resistance levels
    df['high_20'] = df['High'].rolling(20).max()
    df['low_20'] = df['Low'].rolling(20).min()
    df['distance_to_high'] = (df['high_20'] - df['Close']) / df['Close']
    df['distance_to_low'] = (df['Close'] - df['low_20']) / df['Close']
    
    return df


def detect_golden_cross(df):
    """Golden Cross: 50 MA crosses above 200 MA"""
    df['golden_cross'] = 0
    
    if 'sma_50' in df.columns and 'sma_200' in df.columns:
        # Current: 50 > 200
        cond1 = df['sma_50'] > df['sma_200']
        # Previous: 50 <= 200 (crossover just happened)
        cond2 = df['sma_50'].shift(1) <= df['sma_200'].shift(1)
        # Within last 5 days
        for i in range(5):
            if i == 0:
                cross_recent = cond1 & cond2
            else:
                cross_recent = cross_recent | (
                    (df['sma_50'].shift(i) > df['sma_200'].shift(i)) &
                    (df['sma_50'].shift(i+1) <= df['sma_200'].shift(i+1))
                )
        
        df.loc[cross_recent, 'golden_cross'] = 1
    
    return df


def detect_bullish_flag(df):
    """Bullish Flag: Strong move up + parallel consolidation"""
    df['bullish_flag'] = 0
    
    # Strong upward move (10%+ in 5-10 days)
    df['strong_move'] = df['Close'] / df['Close'].shift(10) > 1.10
    
    # Followed by consolidation (low volatility)
    df['volatility_5'] = df['returns'].rolling(5).std()
    df['low_volatility'] = df['volatility_5'] < df['volatility_5'].rolling(20).mean()
    
    # Price staying in tight range
    df['tight_range'] = (df['High'].rolling(5).max() / df['Low'].rolling(5).min() - 1) < 0.05
    
    # Flag detected
    df.loc[df['strong_move'].shift(5) & df['low_volatility'] & df['tight_range'], 'bullish_flag'] = 1
    
    return df


def detect_double_bottom(df):
    """Double Bottom: Two lows at similar price + bounce"""
    df['double_bottom'] = 0
    
    # Find local minima
    df['local_min'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'] < df['Low'].shift(-1))
    
    # Find pairs of lows within 10-30 days, at similar price (within 3%)
    for i in range(len(df) - 30):
        if df['local_min'].iloc[i]:
            # Look for second low in next 10-30 days
            for j in range(i + 10, min(i + 30, len(df))):
                if df['local_min'].iloc[j]:
                    low1 = df['Low'].iloc[i]
                    low2 = df['Low'].iloc[j]
                    # Check if lows are within 3% of each other
                    if abs(low1 - low2) / low1 < 0.03:
                        # Check if price bounced (5%+ from second low)
                        if j + 5 < len(df):
                            future_high = df['High'].iloc[j:j+5].max()
                            if future_high / low2 > 1.05:
                                df.iloc[j, df.columns.get_loc('double_bottom')] = 1
                                break
    
    return df


def detect_rsi_divergence(df):
    """RSI Bullish Divergence: Price makes lower low, RSI makes higher low"""
    df['rsi_divergence'] = 0
    
    if 'rsi' not in df.columns:
        return df
    
    # Find local minima in price and RSI
    df['price_local_min'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'] < df['Low'].shift(-1))
    df['rsi_local_min'] = (df['rsi'] < df['rsi'].shift(1)) & (df['rsi'] < df['rsi'].shift(-1))
    
    # Find divergences
    for i in range(len(df) - 20):
        if df['price_local_min'].iloc[i] and df['rsi_local_min'].iloc[i]:
            # Look for next low in 5-20 days
            for j in range(i + 5, min(i + 20, len(df))):
                if df['price_local_min'].iloc[j] and df['rsi_local_min'].iloc[j]:
                    # Bullish divergence: price lower, RSI higher
                    if df['Low'].iloc[j] < df['Low'].iloc[i] and df['rsi'].iloc[j] > df['rsi'].iloc[i]:
                        df.iloc[j, df.columns.get_loc('rsi_divergence')] = 1
                        break
    
    return df


def detect_breakout_20d_high(df):
    """Breakout: Price breaks above 20-day high with volume"""
    df['breakout_20d'] = 0
    
    # 20-day high
    df['high_20d'] = df['High'].rolling(20).max()
    
    # Breakout conditions
    breakout = (
        (df['Close'] > df['high_20d'].shift(1)) &  # Breaks above previous 20d high
        (df['volume_ratio'] > 1.2) &  # Volume 1.2x average
        (df['Close'] > df['Open'])  # Closes higher than open (bullish day)
    )
    
    df.loc[breakout, 'breakout_20d'] = 1
    
    return df


def detect_ema_ribbon_bullish(df):
    """EMA Ribbon: 8/13/21 EMAs aligned bullish + price above all"""
    df['ema_ribbon'] = 0
    
    if 'ema_8' in df.columns and 'ema_13' in df.columns and 'ema_21' in df.columns:
        # Bullish alignment: 8 > 13 > 21
        aligned = (df['ema_8'] > df['ema_13']) & (df['ema_13'] > df['ema_21'])
        
        # Price above all EMAs
        price_above = (df['Close'] > df['ema_8']) & (df['Close'] > df['ema_13']) & (df['Close'] > df['ema_21'])
        
        # Recently aligned (within last 5 days)
        recently_aligned = False
        for i in range(5):
            if i == 0:
                recently_aligned = aligned
            else:
                recently_aligned = recently_aligned | aligned.shift(i)
        
        df.loc[recently_aligned & price_above, 'ema_ribbon'] = 1
    
    return df


def detect_inverse_head_shoulders(df):
    """Inverse Head & Shoulders: Three lows (middle one lowest) + breakout"""
    df['inverse_hs'] = 0
    
    # Find three consecutive local minima
    df['local_min'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'] < df['Low'].shift(-1))
    
    for i in range(len(df) - 40):
        if df['local_min'].iloc[i]:
            # Look for second low (head) - should be lower
            for j in range(i + 5, min(i + 20, len(df))):
                if df['local_min'].iloc[j] and df['Low'].iloc[j] < df['Low'].iloc[i]:
                    # Look for third low (right shoulder) - similar to first
                    for k in range(j + 5, min(j + 20, len(df))):
                        if df['local_min'].iloc[k]:
                            left_shoulder = df['Low'].iloc[i]
                            head = df['Low'].iloc[j]
                            right_shoulder = df['Low'].iloc[k]
                            
                            # Check proportions
                            if abs(left_shoulder - right_shoulder) / left_shoulder < 0.05:  # Shoulders similar
                                if head < left_shoulder * 0.95:  # Head significantly lower
                                    # Check for breakout above neckline
                                    neckline = max(df['High'].iloc[i:j].max(), df['High'].iloc[j:k].max())
                                    if k + 5 < len(df):
                                        if df['Close'].iloc[k:k+5].max() > neckline:
                                            df.iloc[k, df.columns.get_loc('inverse_hs')] = 1
                                            break
    
    return df


# ========================================
# DATA COLLECTION
# ========================================

print("\n📊 COLLECTING 2 YEARS OF DATA...")
print(f"Tickers: {len(TICKERS)}")

all_data = []
failed = []

for idx, ticker in enumerate(TICKERS, 1):
    try:
        df = yf.download(ticker, period='2y', interval='1d', progress=False)
        
        if df.empty:
            failed.append(ticker)
            continue
        
        # Handle MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df.reset_index()
        df['ticker'] = ticker
        
        # Add technical indicators
        df = add_technical_indicators(df)
        
        # Detect all 7 new patterns
        df = detect_golden_cross(df)
        df = detect_bullish_flag(df)
        df = detect_double_bottom(df)
        df = detect_rsi_divergence(df)
        df = detect_breakout_20d_high(df)
        df = detect_ema_ribbon_bullish(df)
        df = detect_inverse_head_shoulders(df)
        
        all_data.append(df)
        
        # Count patterns
        gc = df['golden_cross'].sum()
        bf = df['bullish_flag'].sum()
        db = df['double_bottom'].sum()
        rd = df['rsi_divergence'].sum()
        br = df['breakout_20d'].sum()
        er = df['ema_ribbon'].sum()
        hs = df['inverse_hs'].sum()
        
        print(f"  [{idx}/{len(TICKERS)}] {ticker:6s} | GC:{gc:2d} Flag:{bf:2d} DB:{db:2d} RSI:{rd:2d} Break:{br:2d} EMA:{er:2d} H&S:{hs:2d}")
        
    except Exception as e:
        print(f"  [{idx}/{len(TICKERS)}] {ticker:6s} | ERROR: {str(e)[:40]}")
        failed.append(ticker)

if not all_data:
    print("\n❌ No data collected!")
    exit()

df_all = pd.concat(all_data, ignore_index=True)
print(f"\n✅ Total samples: {len(df_all):,}")
print(f"❌ Failed tickers: {len(failed)} - {failed[:5]}")

# ========================================
# CREATE TARGETS
# ========================================

print("\n🎯 CREATING TARGETS (5% gain in 5 days)...")

df_all['future_return'] = df_all.groupby('ticker')['Close'].transform(
    lambda x: x.shift(-5) / x - 1
)
df_all['target'] = (df_all['future_return'] > 0.05).astype(int)

# Remove NaN targets
df_all = df_all[df_all['target'].notna()].copy()

print(f"✅ Final dataset: {len(df_all):,} samples")

# ========================================
# TRAIN EACH PATTERN
# ========================================

PATTERNS = [
    'golden_cross',
    'bullish_flag',
    'double_bottom',
    'rsi_divergence',
    'breakout_20d',
    'ema_ribbon',
    'inverse_hs'
]

results = {}

print("\n" + "="*80)
print("🤖 TRAINING 7 PATTERN DETECTORS")
print("="*80)

for pattern in PATTERNS:
    print(f"\n{'='*80}")
    print(f"Training {pattern}...")
    print(f"{'='*80}")
    
    # Filter to pattern detections
    df_pattern = df_all[df_all[pattern] == 1].copy()
    
    total = len(df_pattern)
    print(f"   Total examples: {total}")
    
    if total < 50:
        print(f"   ❌ SKIP: Not enough examples (need 50+)")
        results[pattern] = {'precision': 0, 'recall': 0, 'status': 'INSUFFICIENT_DATA'}
        continue
    
    # Feature columns (all technical indicators)
    feature_cols = [
        'returns', 'volume_ratio', 'rsi', 'macd', 'macd_histogram',
        'bb_position', 'momentum_5', 'momentum_10', 'momentum_20',
        'distance_to_high', 'distance_to_low', 'atr', 'high_low_range'
    ]
    
    # Add MA features
    for period in [5, 8, 10, 13, 20, 21, 50]:
        if f'sma_{period}' in df_pattern.columns:
            feature_cols.append(f'sma_{period}')
        if f'ema_{period}' in df_pattern.columns:
            feature_cols.append(f'ema_{period}')
    
    # Keep only existing columns
    feature_cols = [f for f in feature_cols if f in df_pattern.columns]
    
    X = df_pattern[feature_cols].fillna(0)
    y = df_pattern['target']
    
    # Time series split
    tscv = TimeSeriesSplit(n_splits=5)
    
    precisions = []
    recalls = []
    accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train LightGBM
        model = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            min_child_samples=20,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        acc = accuracy_score(y_test, y_pred)
        
        precisions.append(prec)
        recalls.append(rec)
        accuracies.append(acc)
    
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_accuracy = np.mean(accuracies)
    
    # Success rate
    success_rate = df_pattern['target'].mean()
    
    # Status
    if avg_precision >= 0.70:
        status = "EXCELLENT"
    elif avg_precision >= 0.60:
        status = "GOOD"
    else:
        status = "MARGINAL"
    
    print(f"   Successful: {int(success_rate * total)} ({success_rate*100:.1f}%)")
    print(f"   Accuracy: {avg_accuracy*100:.1f}%")
    print(f"   Precision: {avg_precision*100:.1f}%")
    print(f"   Recall: {avg_recall*100:.1f}%")
    print(f"   Status: {status}")
    
    results[pattern] = {
        'precision': avg_precision,
        'recall': avg_recall,
        'accuracy': avg_accuracy,
        'total_examples': total,
        'status': status
    }
    
    # Train final model on all data and save
    final_model = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        min_child_samples=20,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1
    )
    final_model.fit(X, y)
    
    # Save model
    import joblib
    save_path = f'/content/drive/MyDrive/Quantum_AI_Cockpit/models/patterns_ainvest/pattern_{pattern}_v2.pkl'
    joblib.dump(final_model, save_path)
    print(f"   ✅ Saved: {save_path}")

# ========================================
# SUMMARY
# ========================================

print("\n" + "="*80)
print("🎉 PHASE 2 TRAINING COMPLETE!")
print("="*80)
print(f"\nTrained {len([r for r in results.values() if r['status'] != 'INSUFFICIENT_DATA'])} patterns:\n")

for pattern, data in results.items():
    if data['status'] == 'INSUFFICIENT_DATA':
        print(f"  {pattern:25s}: SKIPPED (not enough examples)")
    else:
        emoji = "✅" if data['precision'] >= 0.60 else "⚠️"
        print(f"  {emoji} {pattern:25s}: {data['precision']*100:5.1f}% precision ({data['total_examples']} examples)")

print("\n" + "="*80)
print("📊 TOTAL PATTERNS TRAINED: 3 (yesterday) + 7 (today) = 10 PATTERNS!")
print("="*80)
print("\n✅ Ready to build dashboard!")

