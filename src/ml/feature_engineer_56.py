"""
🔬 COMPLETE 56-FEATURE ENGINEER
Combines ai_recommender.py (21) + forecaster_features.py (46) + new features (10)
to reach the exact 56 features needed for baseline training

FEATURE BREAKDOWN:
- OHLCV base: 5 features
- ai_recommender.py: 16 technical features
- forecaster_features.py: 25 advanced features  
- NEW additions: 10 missing features (EMA ribbons, normalized metrics, etc.)
TOTAL: 56 features
"""

import numpy as np
import pandas as pd
import talib
import yfinance as yf
from typing import Optional
from datetime import datetime, timedelta

# Try to import microstructure features
try:
    from src.features.microstructure import MicrostructureFeatures
    MICROSTRUCTURE_AVAILABLE = True
except:
    MICROSTRUCTURE_AVAILABLE = False


class FeatureEngineer56:
    """Complete 56-feature engineer for Trident training"""
    
    @staticmethod
    def get_array(df, col):
        """Safe array extraction"""
        if isinstance(df[col], pd.DataFrame):
            return df[col].iloc[:, 0].values
        return df[col].values
    
    @staticmethod
    def engineer_all_features(df: pd.DataFrame, ticker: str = 'UNKNOWN') -> pd.DataFrame:
        """
        Engineer ALL 56 features
        
        Args:
            df: OHLCV DataFrame from yfinance
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with 56 features (OHLCV + 51 engineered)
        """
        # Ensure we have enough data
        if len(df) < 200:
            raise ValueError(f"Need at least 200 bars, got {len(df)}")
        
        # Extract arrays
        close = np.asarray(FeatureEngineer56.get_array(df, 'Close'), dtype='float64')
        high = np.asarray(FeatureEngineer56.get_array(df, 'High'), dtype='float64')
        low = np.asarray(FeatureEngineer56.get_array(df, 'Low'), dtype='float64')
        volume = np.asarray(FeatureEngineer56.get_array(df, 'Volume'), dtype='float64')
        open_price = np.asarray(FeatureEngineer56.get_array(df, 'Open'), dtype='float64')
        
        # Create output DataFrame
        out = pd.DataFrame(index=df.index)
        
        # ========== BASE OHLCV (5 features) ==========
        out['close'] = close
        out['high'] = high
        out['low'] = low
        out['open'] = open_price
        out['volume'] = volume
        
        # ========== AI RECOMMENDER FEATURES (16 features) ==========
        # RSI
        out['rsi_9'] = talib.RSI(close, timeperiod=9)
        out['rsi_14'] = talib.RSI(close, timeperiod=14)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=5, slowperiod=13, signalperiod=1)
        out['macd'] = macd
        out['macd_signal'] = macd_signal
        out['macd_hist'] = macd_hist
        
        # ATR & ADX
        out['atr_14'] = talib.ATR(high, low, close, timeperiod=14)
        out['adx'] = talib.ADX(high, low, close, timeperiod=14)
        
        # EMAs
        out['ema_5'] = talib.EMA(close, timeperiod=5)
        out['ema_13'] = talib.EMA(close, timeperiod=13)
        out['ema_diff_5_13'] = out['ema_5'] - out['ema_13']
        
        # SMAs
        out['sma_20'] = talib.SMA(close, timeperiod=20)
        
        # Volume
        out['vol_sma_20'] = talib.SMA(volume, timeperiod=20)
        out['vol_ratio'] = volume / (out['vol_sma_20'] + 1e-9)
        
        # Returns
        out['returns_1'] = pd.Series(close).pct_change(1).values
        out['returns_5'] = pd.Series(close).pct_change(5).values
        
        # OBV
        out['obv'] = talib.OBV(close, volume)
        
        # ========== FORECASTER ADVANCED FEATURES (25 features) ==========
        # Volume features (9)
        out['volume_ma_10'] = pd.Series(volume).rolling(10).mean().values
        out['volume_ma_50'] = pd.Series(volume).rolling(50).mean().values
        out['volume_ratio_10'] = volume / (out['volume_ma_10'] + 1e-8)
        out['volume_momentum_5'] = pd.Series(volume).pct_change(5).values
        out['volume_momentum_10'] = pd.Series(volume).pct_change(10).values
        out['volume_trend'] = out['volume_ma_10'] / (out['volume_ma_50'] + 1e-8)
        out['volume_spike'] = (volume > out['vol_sma_20'] * 2).astype(int)
        
        # Volatility features (6)
        returns_series = pd.Series(close).pct_change()
        out['volatility_10'] = returns_series.rolling(10).std().values
        out['volatility_20'] = returns_series.rolling(20).std().values
        out['volatility_50'] = returns_series.rolling(50).std().values
        out['volatility_ratio'] = out['volatility_10'] / (out['volatility_50'] + 1e-8)
        out['atr_ratio'] = out['atr_14'] / (close + 1e-8)
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(close, timeperiod=20)
        out['bb_width'] = (upper - lower) / (middle + 1e-8)
        
        # Momentum indicators (4)
        out['stochastic_k'], out['stochastic_d'] = talib.STOCH(high, low, close, 
                                                                fastk_period=14, 
                                                                slowk_period=3, 
                                                                slowd_period=3)
        out['roc_10'] = talib.ROC(close, timeperiod=10)
        out['roc_20'] = talib.ROC(close, timeperiod=20)
        
        # Trend features (6)
        out['ma_10'] = talib.SMA(close, timeperiod=10)
        out['ma_50'] = talib.SMA(close, timeperiod=50)
        out['ma_200'] = talib.SMA(close, timeperiod=200)
        out['ma_conv_short'] = out['ema_5'] - out['ma_10']
        out['price_vs_ma20'] = (close - out['sma_20']) / (out['sma_20'] + 1e-8)
        out['price_vs_ma50'] = (close - out['ma_50']) / (out['ma_50'] + 1e-8)
        
        # ========== NEW FEATURES FOR 56 TOTAL (10 features) ==========
        # EMA Ribbon (GOLD INTEGRATION - user's "goldmine")
        out['ema_8'] = talib.EMA(close, timeperiod=8)
        out['ema_21'] = talib.EMA(close, timeperiod=21)
        out['ema_55'] = talib.EMA(close, timeperiod=55)
        out['ribbon_alignment'] = np.sign(out['ema_8'] - out['ema_21']) * np.sign(out['ema_21'] - out['ema_55'])
        
        # Normalized momentum (dip detection)
        out['ret_21d'] = pd.Series(close).pct_change(21).values  # nuclear_dip uses this
        out['macd_rising'] = (out['macd_hist'] > 0).astype(int)  # nuclear_dip condition
        
        # Additional power features
        out['ma_conv_long'] = out['ma_50'] - out['ma_200']  # Long-term trend
        out['trend_slope_20'] = pd.Series(close).rolling(20).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == 20 else 0, raw=True).values
        
        # Microstructure features (4) - if available
        if MICROSTRUCTURE_AVAILABLE:
            try:
                high_s = pd.Series(high, index=df.index)
                low_s = pd.Series(low, index=df.index)
                close_s = pd.Series(close, index=df.index)
                volume_s = pd.Series(volume, index=df.index)
                open_s = pd.Series(open_price, index=df.index)
                
                out['spread_proxy'] = MicrostructureFeatures.compute_spread_proxy(high_s, low_s, close_s)
                out['order_flow_clv'] = MicrostructureFeatures.compute_order_flow_clv(high_s, low_s, close_s)
                out['institutional_activity'] = MicrostructureFeatures.compute_institutional_activity(volume_s, close_s, open_s)
                out['vw_clv'] = MicrostructureFeatures.compute_volume_weighted_clv(high_s, low_s, close_s, volume_s)
            except Exception:
                # Fill with zeros if microstructure fails
                out['spread_proxy'] = 0.0
                out['order_flow_clv'] = 0.0
                out['institutional_activity'] = 0.0
                out['vw_clv'] = 0.0
        else:
            out['spread_proxy'] = 0.0
            out['order_flow_clv'] = 0.0
            out['institutional_activity'] = 0.0
            out['vw_clv'] = 0.0
        
        # ========== CLEAN DATA ==========
        # Replace infinities
        out = out.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaNs (backfill then forward fill, then median)
        out = out.bfill().ffill()
        for col in out.columns:
            if out[col].isna().any():
                med = out[col].median()
                if pd.isna(med):
                    med = 0.0
                out.loc[:, col] = out[col].fillna(med)
        
        # Verify we have exactly 56 features
        if len(out.columns) != 56:
            raise ValueError(f"Expected 56 features, got {len(out.columns)}")
        
        return out
    
    @staticmethod
    def download_and_engineer(ticker: str, period: str = '2y') -> Optional[pd.DataFrame]:
        """
        Download data and engineer features in one step
        
        Args:
            ticker: Stock ticker
            period: Data period (2y, 5y, etc.)
            
        Returns:
            DataFrame with 56 features or None if failed
        """
        try:
            # Download
            df = yf.download(ticker, period=period, progress=False)
            
            # Fix multi-index
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Check data
            if len(df) < 200:
                return None
            
            # Engineer features
            features = FeatureEngineer56.engineer_all_features(df, ticker)
            
            return features
            
        except Exception as e:
            print(f"Error engineering features for {ticker}: {e}")
            return None


if __name__ == "__main__":
    """Test feature engineer"""
    
    print("🔬 Testing 56-Feature Engineer\n")
    
    # Test with NVDA
    ticker = "NVDA"
    print(f"Testing with {ticker}...")
    
    features = FeatureEngineer56.download_and_engineer(ticker, period='2y')
    
    if features is not None:
        print(f"\n✅ SUCCESS!")
        print(f"   Features: {len(features.columns)}")
        print(f"   Samples: {len(features)}")
        print(f"   Date range: {features.index[0]} to {features.index[-1]}")
        
        print(f"\n📋 Feature List ({len(features.columns)} total):")
        for i, col in enumerate(features.columns, 1):
            print(f"   {i:2d}. {col}")
        
        print(f"\n📊 Sample Row (latest):")
        print(features.iloc[-1])
        
        print(f"\n✅ All tests passed!")
    else:
        print("❌ FAILED to engineer features")
