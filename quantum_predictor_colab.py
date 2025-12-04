#!/usr/bin/env python3
"""
🔮 QUANTUM PREDICTOR - Uses Colab-trained model
==============================================

This script loads the model trained in Colab Pro (T4 GPU + High RAM)
and generates predictions locally.

Usage:
1. Train model in Colab using COLAB_QUANTUM_TRAINER.ipynb
2. Download quantum_models.zip and extract here
3. Run: python quantum_predictor_colab.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
from datetime import datetime
from pathlib import Path

# Check for model files
MODEL_DIR = Path('quantum_models')
MODEL_FILE = MODEL_DIR / 'unrestricted_model.txt'
FEATURES_FILE = MODEL_DIR / 'feature_cols.json'

def calculate_features(df):
    """Calculate all features - must match Colab training exactly"""
    df = df.copy()
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    open_price = df['Open']
    
    # MACD (TOP FEATURE!)
    ema5 = close.ewm(span=5).mean()
    ema13 = close.ewm(span=13).mean()
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    
    df['MACD_5_13'] = ema5 - ema13
    df['MACD_Signal_5_13'] = df['MACD_5_13'].ewm(span=9).mean()
    df['MACD_Hist_5_13'] = df['MACD_5_13'] - df['MACD_Signal_5_13']
    
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Gap
    df['Gap'] = (open_price - close.shift(1)) / close.shift(1)
    
    # Returns
    df['Return_1d'] = close.pct_change(1)
    df['Return_2d'] = close.pct_change(2)
    df['Return_3d'] = close.pct_change(3)
    df['Return_5d'] = close.pct_change(5)
    
    # ATR and Range
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(14).mean()
    df['ATR_7'] = tr.rolling(7).mean()
    df['Range_vs_ATR'] = (high - low) / df['ATR_14']
    df['ATR_Ratio'] = df['ATR_7'] / df['ATR_14']
    
    # CMF
    mfm = ((close - low) - (high - close)) / (high - low + 1e-8)
    mfv = mfm * volume
    df['CMF'] = mfv.rolling(20).sum() / volume.rolling(20).sum()
    
    # MFI
    typical_price = (high + low + close) / 3
    raw_mf = typical_price * volume
    mf_positive = raw_mf.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
    mf_negative = raw_mf.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
    df['MFI'] = 100 - (100 / (1 + mf_positive / (mf_negative + 1e-8)))
    
    # Volume Ratios
    df['Vol_Ratio_50'] = volume / volume.rolling(50).mean()
    df['Vol_Ratio_20'] = volume / volume.rolling(20).mean()
    df['Vol_Ratio_10'] = volume / volume.rolling(10).mean()
    
    # OBV
    obv = (np.sign(close.diff()) * volume).cumsum()
    df['OBV'] = obv
    df['OBV_Slope'] = obv.diff(5) / 5
    
    # ADX
    plus_dm = high.diff().where((high.diff() > low.diff().abs()) & (high.diff() > 0), 0)
    minus_dm = low.diff().abs().where((low.diff().abs() > high.diff()) & (low.diff() < 0), 0)
    atr_14 = tr.rolling(14).mean()
    df['PLUS_DI'] = 100 * (plus_dm.rolling(14).mean() / atr_14)
    df['MINUS_DI'] = 100 * (minus_dm.rolling(14).mean() / atr_14)
    dx = 100 * abs(df['PLUS_DI'] - df['MINUS_DI']) / (df['PLUS_DI'] + df['MINUS_DI'] + 1e-8)
    df['ADX'] = dx.rolling(14).mean()
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI_Momentum'] = df['RSI'] - df['RSI'].shift(5)
    
    # Candlestick patterns
    body = abs(close - open_price)
    total_range = high - low + 1e-8
    df['body_to_range'] = body / total_range
    df['Upper_Wick'] = (high - pd.concat([close, open_price], axis=1).max(axis=1)) / total_range
    df['Lower_Wick'] = (pd.concat([close, open_price], axis=1).min(axis=1) - low) / total_range
    df['lower_shadow_ratio'] = df['Lower_Wick'] / (df['body_to_range'] + 1e-8)
    df['Wick_Ratio'] = df['Upper_Wick'] / (df['Lower_Wick'] + 1e-8)
    
    # Volume-Price
    df['Vol_Price_Trend'] = (volume * close.pct_change()).cumsum()
    df['AD'] = ((close - low) - (high - close)) / (high - low + 1e-8) * volume
    df['AD'] = df['AD'].cumsum()
    
    # Relative Strength
    df['rs_vs_sector_20d'] = close.pct_change(20)
    
    # Trend slope
    df['trend_slope_20'] = (close - close.shift(20)) / close.shift(20)
    df['trend_slope_10'] = (close - close.shift(10)) / close.shift(10)
    
    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    sma50 = close.rolling(50).mean()
    std50 = close.rolling(50).std()
    
    df['BB_Upper'] = sma20 + 2 * std20
    df['BB_Lower'] = sma20 - 2 * std20
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / sma20
    df['BB_Width_50'] = (4 * std50) / sma50
    df['BB_Position'] = (close - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-8)
    
    # Stochastic
    lowest_14 = low.rolling(14).min()
    highest_14 = high.rolling(14).max()
    df['Stoch_K'] = 100 * (close - lowest_14) / (highest_14 - lowest_14 + 1e-8)
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
    
    # SMAs and EMAs
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'SMA_{period}'] = close.rolling(period).mean()
        df[f'EMA_{period}'] = close.ewm(span=period).mean()
        df[f'Close_vs_SMA_{period}'] = close / df[f'SMA_{period}']
        df[f'Close_vs_EMA_{period}'] = close / df[f'EMA_{period}']
    
    df['EMA_Ribbon_Spread'] = (df['EMA_10'] - df['EMA_50']) / close
    
    # Momentum
    df['ROC_10'] = close.pct_change(10) * 100
    df['ROC_20'] = close.pct_change(20) * 100
    
    # Volatility
    df['Volatility_20'] = close.pct_change().rolling(20).std() * np.sqrt(252)
    df['Volatility_10'] = close.pct_change().rolling(10).std() * np.sqrt(252)
    
    return df


class QuantumPredictorColab:
    """Uses the Colab-trained model for predictions"""
    
    def __init__(self):
        self.model = None
        self.feature_cols = None
        self.load_model()
    
    def load_model(self):
        """Load the Colab-trained model"""
        if not MODEL_FILE.exists():
            print(f"⚠️ Model not found at {MODEL_FILE}")
            print("   Train in Colab first using COLAB_QUANTUM_TRAINER.ipynb")
            print("   Then download and extract quantum_models.zip here")
            return False
        
        try:
            import lightgbm as lgb
            self.model = lgb.Booster(model_file=str(MODEL_FILE))
            print(f"✅ Model loaded: {MODEL_FILE}")
            
            with open(FEATURES_FILE, 'r') as f:
                self.feature_cols = json.load(f)
            print(f"✅ Features loaded: {len(self.feature_cols)} features")
            
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict(self, ticker: str) -> dict:
        """Get prediction for a single ticker"""
        if self.model is None:
            return {'error': 'Model not loaded'}
        
        try:
            # Download data
            df = yf.download(ticker, period='6mo', progress=False)
            if len(df) < 50:
                return {'error': 'Insufficient data'}
            
            # Calculate features
            df = calculate_features(df)
            
            # Get latest features
            latest = df[self.feature_cols].iloc[-1:].values
            latest = np.nan_to_num(latest, nan=0.0)
            
            # Predict
            confidence = float(self.model.predict(latest)[0])
            
            # Determine signal
            if confidence >= 0.85:
                signal = 'STRONG BUY'
            elif confidence >= 0.70:
                signal = 'BUY'
            elif confidence >= 0.50:
                signal = 'HOLD'
            else:
                signal = 'AVOID'
            
            return {
                'ticker': ticker,
                'confidence': confidence,
                'signal': signal,
                'price': float(df['Close'].iloc[-1]),
                'rsi': float(df['RSI'].iloc[-1]) if 'RSI' in df.columns else None,
                'macd_hist': float(df['MACD_Hist_5_13'].iloc[-1]) if 'MACD_Hist_5_13' in df.columns else None,
                'vol_ratio': float(df['Vol_Ratio_20'].iloc[-1]) if 'Vol_Ratio_20' in df.columns else None,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'ticker': ticker, 'error': str(e)}
    
    def scan_watchlist(self, tickers: list) -> list:
        """Scan entire watchlist and return sorted predictions"""
        predictions = []
        
        for ticker in tickers:
            result = self.predict(ticker)
            if 'error' not in result:
                predictions.append(result)
        
        # Sort by confidence (highest first)
        predictions.sort(key=lambda x: -x['confidence'])
        return predictions


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("🔮 QUANTUM PREDICTOR - Colab Model Edition")
    print("=" * 70)
    
    # Your watchlist
    TICKERS = [
        'APLD', 'SERV', 'MRVL', 'HOOD', 'LUNR', 'BAC', 'QCOM', 'UUUU',
        'TSLA', 'AMD', 'NOW', 'NVDA', 'MU', 'PG', 'DLB', 'XME',
        'KRYS', 'LEU', 'QTUM', 'SPY', 'UNH', 'WMT', 'OKLO', 'RXRX',
        'MTZ', 'SNOW', 'GRRR', 'BSX', 'LLY', 'VOO', 'GEO', 'CXW',
        'LYFT', 'MNDY', 'BA', 'LAC', 'INTC', 'ALK', 'LMT', 'CRDO',
        'ANET', 'META', 'RIVN', 'GOOGL', 'HL', 'TEM', 'TDOC', 'KMTS'
    ]
    
    predictor = QuantumPredictorColab()
    
    if predictor.model is not None:
        print(f"\n📊 Scanning {len(TICKERS)} tickers...")
        print("-" * 70)
        
        predictions = predictor.scan_watchlist(TICKERS)
        
        # Display results
        print("\n🟢 STRONG BUY (>85% confidence):")
        for p in predictions:
            if p['confidence'] >= 0.85:
                print(f"   🔥 {p['ticker']:5} | {p['confidence']*100:5.1f}% | ${p['price']:.2f}")
        
        print("\n✅ BUY (70-85% confidence):")
        for p in predictions:
            if 0.70 <= p['confidence'] < 0.85:
                print(f"   ✅ {p['ticker']:5} | {p['confidence']*100:5.1f}% | ${p['price']:.2f}")
        
        print("\n⚪ HOLD (50-70% confidence):")
        count = 0
        for p in predictions:
            if 0.50 <= p['confidence'] < 0.70:
                count += 1
                if count <= 10:  # Limit display
                    print(f"      {p['ticker']:5} | {p['confidence']*100:5.1f}%")
        if count > 10:
            print(f"      ... and {count - 10} more")
        
        print("\n" + "=" * 70)
        print("📊 SUMMARY:")
        print(f"   Strong Buy: {sum(1 for p in predictions if p['confidence'] >= 0.85)}")
        print(f"   Buy:        {sum(1 for p in predictions if 0.70 <= p['confidence'] < 0.85)}")
        print(f"   Hold:       {sum(1 for p in predictions if 0.50 <= p['confidence'] < 0.70)}")
        print(f"   Avoid:      {sum(1 for p in predictions if p['confidence'] < 0.50)}")
        print("=" * 70)
        
        # Save predictions
        output = {
            'generated_at': datetime.now().isoformat(),
            'predictions': predictions
        }
        with open('latest_predictions.json', 'w') as f:
            json.dump(output, f, indent=2)
        print("\n✅ Predictions saved to latest_predictions.json")
    
    else:
        print("\n" + "=" * 70)
        print("📋 TO GET STARTED:")
        print("=" * 70)
        print("1. Open COLAB_QUANTUM_TRAINER.ipynb in Google Colab")
        print("2. Enable GPU: Runtime → Change runtime type → T4 GPU")
        print("3. Run all cells to train the model")
        print("4. Download quantum_models.zip when prompted")
        print("5. Extract to this directory")
        print("6. Run this script again!")
        print("=" * 70)
