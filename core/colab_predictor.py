"""
COLAB-TRAINED PRODUCTION PREDICTOR
==================================
Uses the XGBoost + LightGBM ensemble trained in Google Colab
with SHAP-selected features, Focal Loss, and Purged CV.

This module loads the trained models and provides predictions
that match EXACTLY what was trained.
"""

import json
import joblib
import numpy as np
import pandas as pd
import talib
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
import warnings

# Suppress sklearn version warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

logger = logging.getLogger(__name__)

# Model paths
MODEL_DIR = Path(__file__).parent.parent / 'trained_models' / 'colab'


class ColabPredictor:
    """
    Production predictor using Colab-trained models.
    
    Features:
    - XGBoost + LightGBM ensemble
    - SHAP-selected top 50 features
    - Same feature engineering as training
    - Confidence-weighted predictions
    """
    
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.xgb_model = None
        self.lgb_model = None
        self.scaler = None
        self.top_features = None
        self.is_loaded = False
        
        self._load_models()
    
    def _load_models(self):
        """Load trained models from disk."""
        try:
            # Load top features
            with open(self.model_dir / 'top_features.json', 'r') as f:
                self.top_features = json.load(f)
            logger.info(f"✅ Loaded {len(self.top_features)} top features")
            
            # Load XGBoost model
            with open(self.model_dir / 'xgboost_model.pkl', 'rb') as f:
                import pickle
                self.xgb_model = pickle.load(f)
            logger.info("✅ Loaded XGBoost model")
            
            # Load LightGBM model (needs joblib)
            self.lgb_model = joblib.load(self.model_dir / 'lightgbm_model.pkl')
            logger.info("✅ Loaded LightGBM model")
            
            # Load scaler (needs joblib)
            self.scaler = joblib.load(self.model_dir / 'scaler.pkl')
            logger.info("✅ Loaded StandardScaler")
            
            self.is_loaded = True
            logger.info("✅ All Colab models loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            self.is_loaded = False
    
    def engineer_features(self, df: pd.DataFrame, spy_data: pd.DataFrame = None, 
                         vix_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Engineer features EXACTLY as done in Colab training.
        Must match the training feature engineering precisely.
        """
        df = df.copy()
        
        # Ensure float64 for TA-Lib
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = df[col].astype('float64')
        
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values
        open_price = df['Open'].values
        
        features = pd.DataFrame(index=df.index)
        
        # === BASIC FEATURES ===
        features['Returns'] = df['Close'].pct_change()
        features['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        features['Range'] = (df['High'] - df['Low']) / df['Close']
        features['Body'] = abs(df['Close'] - df['Open']) / df['Close']
        features['Upper_Shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Close']
        features['Lower_Shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Close']
        
        # === MOMENTUM ===
        for period in [7, 14, 21, 50]:
            features[f'RSI_{period}'] = talib.RSI(close, timeperiod=period)
        
        macd, signal, hist = talib.MACD(close)
        features['MACD'] = macd
        features['MACD_Signal'] = signal
        features['MACD_Hist'] = hist
        
        slowk, slowd = talib.STOCH(high, low, close)
        features['Stoch_K'] = slowk
        features['Stoch_D'] = slowd
        
        features['ADX'] = talib.ADX(high, low, close, timeperiod=14)
        features['Plus_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)
        features['Minus_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)
        
        # === VOLATILITY ===
        features['ATR'] = talib.ATR(high, low, close, timeperiod=14)
        features['ATR_Percentile'] = features['ATR'].rolling(90).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        
        upper, middle, lower = talib.BBANDS(close, timeperiod=20)
        features['BB_Upper'] = upper
        features['BB_Middle'] = middle
        features['BB_Lower'] = lower
        features['BB_Width'] = (upper - lower) / middle
        features['BB_Position'] = (close - lower) / (upper - lower + 1e-10)
        
        # === EMA RIBBON ===
        ema_periods = [8, 13, 21, 34, 55, 89, 144, 233]
        for period in ema_periods:
            features[f'EMA_{period}'] = talib.EMA(close, timeperiod=period)
            features[f'Price_vs_EMA_{period}'] = (close - features[f'EMA_{period}'].values) / features[f'EMA_{period}'].values
            features[f'EMA_{period}_Slope'] = features[f'EMA_{period}'].diff(3) / features[f'EMA_{period}'].shift(3)
        
        features['EMA_Ribbon_Width'] = (features['EMA_8'] - features['EMA_233']) / close
        features['EMA_Ribbon_Compression'] = features['EMA_Ribbon_Width'].rolling(20).std()
        
        features['EMA_8_21_Cross'] = np.where(features['EMA_8'] > features['EMA_21'], 1, -1)
        features['EMA_21_55_Cross'] = np.where(features['EMA_21'] > features['EMA_55'], 1, -1)
        
        features['EMA_Ribbon_Bullish'] = (
            (features['EMA_8'] > features['EMA_13']).astype(int) +
            (features['EMA_13'] > features['EMA_21']).astype(int) +
            (features['EMA_21'] > features['EMA_34']).astype(int) +
            (features['EMA_34'] > features['EMA_55']).astype(int) +
            (features['EMA_55'] > features['EMA_89']).astype(int) +
            (features['EMA_89'] > features['EMA_144']).astype(int) +
            (features['EMA_144'] > features['EMA_233']).astype(int)
        ) / 7.0
        
        features['Golden_Cross_Strength'] = features['EMA_8_21_Cross'] * features['ADX'] / 100
        
        # === FIBONACCI ===
        swing_high = df['High'].rolling(20, center=True).max()
        swing_low = df['Low'].rolling(20, center=True).min()
        fib_range = swing_high - swing_low
        
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        for level in fib_levels:
            level_name = str(level).replace('.', '_')
            features[f'Fib_Retrace_{level_name}'] = swing_high - (fib_range * level)
            features[f'Dist_to_Fib_{level_name}'] = (close - features[f'Fib_Retrace_{level_name}'].values) / close
        
        fib_extensions = [1.272, 1.618, 2.0, 2.618]
        for ext in fib_extensions:
            ext_name = str(ext).replace('.', '_')
            features[f'Fib_Ext_{ext_name}'] = swing_low + (fib_range * ext)
            features[f'Dist_to_FibExt_{ext_name}'] = (close - features[f'Fib_Ext_{ext_name}'].values) / close
        
        features['Near_Fib_0_618'] = (abs(features['Dist_to_Fib_0_618']) < 0.01).astype(int)
        features['Near_Fib_0_382'] = (abs(features['Dist_to_Fib_0_382']) < 0.01).astype(int)
        features['Near_Fib_0_5'] = (abs(features['Dist_to_Fib_0_5']) < 0.01).astype(int)
        
        features['Golden_Zone_Bullish'] = ((features['Near_Fib_0_618'] == 1) & 
                                           (features['EMA_Ribbon_Bullish'] > 0.5)).astype(int)
        
        # === VOLUME ===
        features['Volume_MA_20'] = df['Volume'].rolling(20).mean()
        features['Volume_Ratio'] = df['Volume'] / features['Volume_MA_20']
        
        features['OBV'] = talib.OBV(close, volume)
        features['OBV_Change'] = features['OBV'].pct_change(5)
        
        features['CMF'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
        
        # === CANDLESTICK PATTERNS ===
        features['CDLLONGLINE'] = talib.CDLLONGLINE(open_price, high, low, close)
        features['CDLHIKKAKE'] = talib.CDLHIKKAKE(open_price, high, low, close)
        
        # === REGIME ===
        features['Trend_Regime'] = np.where(close > features['EMA_55'].values, 1, -1)
        features['Vol_Regime'] = np.where(features['ATR_Percentile'] > 0.7, 1, 
                                          np.where(features['ATR_Percentile'] < 0.3, -1, 0))
        
        # === PERCENTILES ===
        features['RSI_14_Percentile_90d'] = features['RSI_14'].rolling(90).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        features['MACD_Percentile_90d'] = features['MACD'].rolling(90).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        features['Volume_Ratio_Percentile_90d'] = features['Volume_Ratio'].rolling(90).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        
        # === CROSS-ASSET ===
        if spy_data is not None and len(spy_data) > 0:
            try:
                spy_aligned = spy_data.reindex(df.index, method='ffill')
                spy_returns = spy_aligned['Close'].pct_change()
                stock_returns = df['Close'].pct_change()
                features['Correlation_SPY'] = stock_returns.rolling(20).corr(spy_returns)
                features['Beta_SPY'] = stock_returns.rolling(60).cov(spy_returns) / spy_returns.rolling(60).var()
            except:
                features['Correlation_SPY'] = 0
                features['Beta_SPY'] = 1
        else:
            features['Correlation_SPY'] = 0
            features['Beta_SPY'] = 1
        
        if vix_data is not None and len(vix_data) > 0:
            try:
                vix_aligned = vix_data.reindex(df.index, method='ffill')
                features['VIX_Level'] = vix_aligned['Close']
                features['VIX_Change'] = vix_aligned['Close'].pct_change()
            except:
                features['VIX_Level'] = 20
                features['VIX_Change'] = 0
        else:
            features['VIX_Level'] = 20
            features['VIX_Change'] = 0
        
        # === INTERACTION FEATURES ===
        features['RSI_x_Volume'] = features['RSI_14'] * features['Volume_Ratio']
        features['Trend_x_Vol'] = features['Trend_Regime'] * features['Vol_Regime']
        
        # Cleanup
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill().bfill()
        
        return features
    
    def _flatten_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flatten multi-level columns from yfinance downloads."""
        if isinstance(df.columns, pd.MultiIndex):
            # Take first level of column names
            df = df.copy()
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        return df
    
    def predict(self, df: pd.DataFrame, spy_data: pd.DataFrame = None,
                vix_data: pd.DataFrame = None) -> Dict:
        """
        Generate prediction using the Colab-trained ensemble.
        
        Returns:
            Dict with signal, confidence, probabilities
        """
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Check model files exist.")
        
        # Flatten multi-level columns from yfinance
        df = self._flatten_columns(df)
        if spy_data is not None:
            spy_data = self._flatten_columns(spy_data)
        if vix_data is not None:
            vix_data = self._flatten_columns(vix_data)
        
        # Engineer features
        features = self.engineer_features(df, spy_data, vix_data)
        
        # Select only the top features used in training
        missing_features = [f for f in self.top_features if f not in features.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features[:5]}...")
            # Fill missing with 0
            for f in missing_features:
                features[f] = 0
        
        X = features[self.top_features].iloc[-1:].values
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from both models
        xgb_proba = self.xgb_model.predict_proba(X_scaled)[0]
        lgb_proba = self.lgb_model.predict_proba(X_scaled)[0]
        
        # Ensemble (weighted average - XGBoost slightly higher weight)
        ensemble_proba = 0.55 * xgb_proba + 0.45 * lgb_proba
        
        # Get prediction
        pred_class = np.argmax(ensemble_proba)
        confidence = ensemble_proba[pred_class]
        
        # Map to signal
        signal_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        signal = signal_map[pred_class]
        
        return {
            'signal': signal,
            'confidence': float(confidence),
            'probabilities': {
                'HOLD': float(ensemble_proba[0]),
                'BUY': float(ensemble_proba[1]),
                'SELL': float(ensemble_proba[2])
            },
            'xgb_proba': xgb_proba.tolist(),
            'lgb_proba': lgb_proba.tolist(),
            'features_used': len(self.top_features)
        }
    
    def predict_batch(self, df: pd.DataFrame, spy_data: pd.DataFrame = None,
                     vix_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Generate predictions for all rows in dataframe.
        """
        if not self.is_loaded:
            raise RuntimeError("Models not loaded.")
        
        features = self.engineer_features(df, spy_data, vix_data)
        
        # Handle missing features
        for f in self.top_features:
            if f not in features.columns:
                features[f] = 0
        
        X = features[self.top_features].values
        
        # Remove NaN rows
        valid_mask = ~np.isnan(X).any(axis=1)
        X_valid = X[valid_mask]
        
        if len(X_valid) == 0:
            return pd.DataFrame()
        
        X_scaled = self.scaler.transform(X_valid)
        
        # Ensemble predictions
        xgb_proba = self.xgb_model.predict_proba(X_scaled)
        lgb_proba = self.lgb_model.predict_proba(X_scaled)
        ensemble_proba = 0.55 * xgb_proba + 0.45 * lgb_proba
        
        predictions = np.argmax(ensemble_proba, axis=1)
        confidences = ensemble_proba.max(axis=1)
        
        # Create result dataframe
        result = pd.DataFrame(index=df.index[valid_mask])
        result['signal'] = [['HOLD', 'BUY', 'SELL'][p] for p in predictions]
        result['confidence'] = confidences
        result['prob_hold'] = ensemble_proba[:, 0]
        result['prob_buy'] = ensemble_proba[:, 1]
        result['prob_sell'] = ensemble_proba[:, 2]
        
        return result


# Singleton instance for easy import
_predictor = None

def get_predictor() -> ColabPredictor:
    """Get or create the singleton predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = ColabPredictor()
    return _predictor


if __name__ == '__main__':
    # Test the predictor
    import yfinance as yf
    
    print("🧪 Testing ColabPredictor...")
    
    # Download test data
    ticker = 'AAPL'
    df = yf.download(ticker, period='6mo', progress=False)
    spy = yf.download('SPY', period='6mo', progress=False)
    vix = yf.download('^VIX', period='6mo', progress=False)
    
    if len(df) > 0:
        predictor = ColabPredictor()
        
        if predictor.is_loaded:
            result = predictor.predict(df, spy, vix)
            
            print(f"\n📊 Prediction for {ticker}:")
            print(f"   Signal: {result['signal']}")
            print(f"   Confidence: {result['confidence']:.2%}")
            print(f"   Probabilities:")
            print(f"      HOLD: {result['probabilities']['HOLD']:.2%}")
            print(f"      BUY:  {result['probabilities']['BUY']:.2%}")
            print(f"      SELL: {result['probabilities']['SELL']:.2%}")
            print(f"   Features used: {result['features_used']}")
        else:
            print("❌ Models not loaded")
    else:
        print("❌ Failed to download test data")
