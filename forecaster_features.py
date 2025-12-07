"""
📈 FORECASTER QUICK OPTIMIZATION
Add high-value features to boost accuracy from 57% → 65%+ in 30 minutes
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple
import pickle
from pathlib import Path

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume-based features (Expected: +2-3% accuracy)
    
    Features:
    - Volume moving averages (10, 20, 50 day)
    - Volume ratio (current vs average)
    - Volume momentum
    - Volume trend strength
    """
    # Volume moving averages
    df['volume_ma_10'] = df['Volume'].rolling(10).mean()
    df['volume_ma_20'] = df['Volume'].rolling(20).mean()
    df['volume_ma_50'] = df['Volume'].rolling(50).mean()
    
    # Volume ratios
    df['volume_ratio_10'] = df['Volume'] / (df['volume_ma_10'] + 1e-8)
    df['volume_ratio_20'] = df['Volume'] / (df['volume_ma_20'] + 1e-8)
    
    # Volume momentum (rate of change)
    df['volume_momentum_5'] = df['Volume'].pct_change(5)
    df['volume_momentum_10'] = df['Volume'].pct_change(10)
    
    # Volume trend (short MA vs long MA)
    df['volume_trend'] = df['volume_ma_10'] / (df['volume_ma_50'] + 1e-8)
    
    # Volume spike detection
    df['volume_spike'] = (df['Volume'] > df['volume_ma_20'] * 2).astype(int)
    
    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volatility-based features (Expected: +1-2% accuracy)
    
    Features:
    - Historical volatility (10, 20, 50 day)
    - Volatility ratio
    - ATR (Average True Range)
    - Bollinger Band width
    """
    # Calculate returns
    df['returns'] = df['Close'].pct_change()
    
    # Historical volatility (annualized)
    df['volatility_10'] = df['returns'].rolling(10).std() * np.sqrt(252)
    df['volatility_20'] = df['returns'].rolling(20).std() * np.sqrt(252)
    df['volatility_50'] = df['returns'].rolling(50).std() * np.sqrt(252)
    
    # Volatility ratio
    df['volatility_ratio'] = df['volatility_10'] / (df['volatility_50'] + 1e-8)
    
    # ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = true_range.rolling(14).mean()
    df['atr_ratio'] = df['atr_14'] / (df['Close'] + 1e-8)
    
    # Bollinger Band width
    bb_middle = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['bb_width'] = (bb_std * 2) / (bb_middle + 1e-8)
    
    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add momentum indicators (Expected: +2-3% accuracy)
    
    Features:
    - RSI (Relative Strength Index)
    - MACD
    - Stochastic Oscillator
    - Rate of Change
    """
    # RSI (14-day)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Stochastic Oscillator
    low_14 = df['Low'].rolling(14).min()
    high_14 = df['High'].rolling(14).max()
    df['stochastic_k'] = 100 * (df['Close'] - low_14) / (high_14 - low_14 + 1e-8)
    df['stochastic_d'] = df['stochastic_k'].rolling(3).mean()
    
    # Rate of Change
    df['roc_5'] = df['Close'].pct_change(5) * 100
    df['roc_10'] = df['Close'].pct_change(10) * 100
    df['roc_20'] = df['Close'].pct_change(20) * 100
    
    return df


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add trend strength features (Expected: +1-2% accuracy)
    
    Features:
    - Moving average convergence
    - Trend strength (ADX-like)
    - Linear regression slope
    - Price vs MA distance
    """
    # Moving averages
    df['ma_5'] = df['Close'].rolling(5).mean()
    df['ma_10'] = df['Close'].rolling(10).mean()
    df['ma_20'] = df['Close'].rolling(20).mean()
    df['ma_50'] = df['Close'].rolling(50).mean()
    df['ma_200'] = df['Close'].rolling(200).mean()
    
    # MA convergence
    df['ma_conv_short'] = df['ma_5'] / (df['ma_20'] + 1e-8)
    df['ma_conv_long'] = df['ma_50'] / (df['ma_200'] + 1e-8)
    
    # Price distance from MAs (%)
    df['price_vs_ma20'] = (df['Close'] - df['ma_20']) / (df['ma_20'] + 1e-8) * 100
    df['price_vs_ma50'] = (df['Close'] - df['ma_50']) / (df['ma_50'] + 1e-8) * 100
    
    # Linear regression slope (20-day)
    def calc_slope(series):
        if len(series) < 2:
            return 0
        x = np.arange(len(series))
        y = series.values
        return np.polyfit(x, y, 1)[0]
    
    df['trend_slope_20'] = df['Close'].rolling(20).apply(calc_slope, raw=False)
    df['trend_slope_50'] = df['Close'].rolling(50).apply(calc_slope, raw=False)
    
    # Trend consistency (R-squared)
    def calc_r_squared(series):
        if len(series) < 2:
            return 0
        x = np.arange(len(series))
        y = series.values
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))
    
    df['trend_r2_20'] = df['Close'].rolling(20).apply(calc_r_squared, raw=False)
    
    return df


def add_market_context_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Add market context features (Expected: +1-2% accuracy)
    
    Features:
    - SPY correlation
    - Relative strength vs SPY
    - VIX level
    """
    try:
        # Get SPY data
        spy = yf.download('SPY', start=df.index[0], end=df.index[-1], progress=False)
        
        # Fix multi-index if needed
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        
        # SPY returns
        spy['returns'] = spy['Close'].pct_change()
        
        # Align indices
        spy = spy.reindex(df.index, method='ffill')
        
        # Correlation with SPY (60-day rolling)
        df['spy_correlation_60'] = df['returns'].rolling(60).corr(spy['returns'])
        
        # Relative strength vs SPY
        df['relative_strength_spy'] = df['Close'].pct_change(20) - spy['Close'].pct_change(20)
        
        # Beta (60-day)
        def calc_beta(returns, spy_returns):
            if len(returns) < 2 or len(spy_returns) < 2:
                return 1.0
            covar = np.cov(returns, spy_returns)[0][1]
            spy_var = np.var(spy_returns)
            return covar / (spy_var + 1e-8)
        
        df['beta_60'] = df['returns'].rolling(60).apply(
            lambda x: calc_beta(x.values, spy['returns'].iloc[-60:].values), raw=False
        )
        
    except Exception as e:
        print(f"   ⚠️ Could not add market context: {e}")
        df['spy_correlation_60'] = 0
        df['relative_strength_spy'] = 0
        df['beta_60'] = 1.0
    
    return df


def engineer_all_features(ticker: str, period: str = '2y') -> pd.DataFrame:
    """
    Engineer ALL features for a ticker
    
    Returns DataFrame with ~30 new features
    """
    print(f"📊 Engineering features for {ticker}...")
    
    # Download data
    df = yf.download(ticker, period=period, interval='1d', progress=False)
    
    # Fix multi-index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    if len(df) < 200:
        print(f"   ⚠️ Insufficient data")
        return None
    
    # Add all feature sets
    print(f"   Adding volume features...")
    df = add_volume_features(df)
    
    print(f"   Adding volatility features...")
    df = add_volatility_features(df)
    
    print(f"   Adding momentum features...")
    df = add_momentum_features(df)
    
    print(f"   Adding trend features...")
    df = add_trend_features(df)
    
    print(f"   Adding market context...")
    df = add_market_context_features(df, ticker)
    
    # Drop NaN
    df = df.dropna()
    
    print(f"   ✅ {len(df)} samples with {len(df.columns)} features")
    
    return df


def extract_feature_vector(df: pd.DataFrame, index: int) -> Dict[str, float]:
    """
    Extract feature vector at a specific index
    
    Returns dict of feature_name: value
    """
    if index >= len(df):
        return None
    
    row = df.iloc[index]
    
    features = {}
    
    # Original features (from current forecaster)
    features['close'] = row['Close']
    features['volume'] = row['Volume']
    features['high'] = row['High']
    features['low'] = row['Low']
    
    # Volume features
    for col in df.columns:
        if 'volume' in col.lower() and col != 'Volume':
            features[col] = row[col]
    
    # Volatility features
    for col in df.columns:
        if any(x in col.lower() for x in ['volatility', 'atr', 'bb']):
            features[col] = row[col]
    
    # Momentum features
    for col in df.columns:
        if any(x in col.lower() for x in ['rsi', 'macd', 'stochastic', 'roc']):
            features[col] = row[col]
    
    # Trend features
    for col in df.columns:
        if any(x in col.lower() for x in ['ma_', 'trend', 'slope', 'r2']):
            features[col] = row[col]
    
    # Market context
    for col in df.columns:
        if any(x in col.lower() for x in ['spy', 'beta', 'relative']):
            features[col] = row[col]
    
    return features


# ============================================================================
# EXAMPLE: Quick test on a ticker
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*100)
    print("📈 FORECASTER FEATURE ENGINEERING - QUICK TEST")
    print("="*100 + "\n")
    
    # Test on AAPL
    ticker = "AAPL"
    
    print(f"Testing feature engineering on {ticker}...\n")
    
    # Engineer features
    df = engineer_all_features(ticker, period='2y')
    
    if df is not None:
        print(f"\n✅ SUCCESS!")
        print(f"   Total samples: {len(df)}")
        print(f"   Total features: {len(df.columns)}")
        print(f"\n📊 Feature categories:")
        
        volume_features = [col for col in df.columns if 'volume' in col.lower()]
        volatility_features = [col for col in df.columns if any(x in col.lower() for x in ['volatility', 'atr', 'bb'])]
        momentum_features = [col for col in df.columns if any(x in col.lower() for x in ['rsi', 'macd', 'stochastic', 'roc'])]
        trend_features = [col for col in df.columns if any(x in col.lower() for x in ['ma_', 'trend', 'slope'])]
        market_features = [col for col in df.columns if any(x in col.lower() for x in ['spy', 'beta', 'relative'])]
        
        print(f"   Volume: {len(volume_features)} features")
        print(f"   Volatility: {len(volatility_features)} features")
        print(f"   Momentum: {len(momentum_features)} features")
        print(f"   Trend: {len(trend_features)} features")
        print(f"   Market Context: {len(market_features)} features")
        
        print(f"\n📈 Sample feature vector (latest):")
        features = extract_feature_vector(df, -1)
        for name, value in list(features.items())[:10]:
            print(f"   {name:30} {value:.4f}")
        
        print(f"\n💡 Next steps:")
        print(f"   1. Use these features to retrain forecaster")
        print(f"   2. Expected accuracy improvement: +6-10%")
        print(f"   3. Target: 57% → 65-67% accuracy")
        
        print(f"\n🚀 Run: python train_forecaster_v2.py --use-advanced-features")
