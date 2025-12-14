"""
Feature Engine - Technical Indicators + Microstructure Metrics

30+ features for Alpha 76 watchlist:
- Momentum indicators (RSI, MACD, Stochastic)
- Trend indicators (EMA, Bollinger Bands, ADX)
- Volume indicators (OBV, VWAP, Volume MA)
- Microstructure proxies (spread, imbalance, price impact)

Designed for 1-hour bars on small/mid-cap stocks.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngine:
    """
    Calculate technical indicators and microstructure features
    
    All features normalized to [0, 1] or z-score for ML compatibility
    """
    
    def __init__(self):
        self.feature_names = []
        
    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all features for a single ticker
        
        Args:
            df: OHLCV dataframe with columns [open, high, low, close, volume]
            
        Returns:
            df_features: Original df + all features
        """
        df = df.copy()
        
        # Momentum indicators
        df = self._add_momentum_features(df)
        
        # Trend indicators
        df = self._add_trend_features(df)
        
        # Volume indicators
        df = self._add_volume_features(df)
        
        # Microstructure proxies
        df = self._add_microstructure_features(df)
        
        # Price action patterns
        df = self._add_price_patterns(df)
        
        # Store feature names (exclude OHLCV and metadata)
        base_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'ticker', 'datetime', 'date', 'time']
        self.feature_names = [c for c in df.columns if c not in base_cols]
        
        logger.info(f"Calculated {len(self.feature_names)} features")
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum oscillators"""
        
        # RSI (14-period)
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        
        # Stochastic (14,3,3)
        df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(df, 14, 3, 3)
        
        # MACD (12,26,9)
        df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(df['close'])
        
        # ROC (Rate of Change) - 5, 10, 20 periods
        df['roc_5'] = df['close'].pct_change(5)
        df['roc_10'] = df['close'].pct_change(10)
        df['roc_20'] = df['close'].pct_change(20)
        
        # Williams %R (14)
        df['williams_r'] = self._calculate_williams_r(df, 14)
        
        return df
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trend following indicators"""
        
        # EMAs (8, 21, 50, 200)
        for period in [8, 21, 50, 200]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            # Distance from EMA (normalized)
            df[f'dist_ema_{period}'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}']
        
        # Bollinger Bands (20, 2std)
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (2 * bb_std)
        df['bb_lower'] = df['bb_middle'] - (2 * bb_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ADX (Average Directional Index) - 14 period
        df['adx'] = self._calculate_adx(df, 14)
        
        # ATR (Average True Range) - 14 period
        df['atr'] = self._calculate_atr(df, 14)
        df['atr_pct'] = df['atr'] / df['close']  # Normalized ATR
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume-based indicators"""
        
        # OBV (On Balance Volume)
        df['obv'] = self._calculate_obv(df)
        df['obv_ema_10'] = df['obv'].ewm(span=10, adjust=False).mean()
        df['obv_trend'] = (df['obv'] - df['obv_ema_10']) / df['obv_ema_10']
        
        # VWAP (Volume Weighted Average Price)
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['dist_vwap'] = (df['close'] - df['vwap']) / df['vwap']
        
        # Volume MA and relative volume
        df['vol_ma_20'] = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_ma_20']
        
        # Volume trend
        df['vol_ema_10'] = df['volume'].ewm(span=10, adjust=False).mean()
        df['vol_trend'] = (df['volume'] - df['vol_ema_10']) / df['vol_ema_10']
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Microstructure proxies (no level 2 data needed)"""
        
        # Spread proxy (high-low as % of close)
        df['spread_proxy'] = (df['high'] - df['low']) / df['close']
        
        # Amihud illiquidity (price impact per $1M volume)
        df['amihud'] = np.abs(df['close'].pct_change()) / (df['volume'] * df['close'] / 1e6)
        df['amihud'] = df['amihud'].replace([np.inf, -np.inf], np.nan)
        
        # Order imbalance proxy (close position in bar)
        df['bar_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['bar_position'] = df['bar_position'].fillna(0.5)
        
        # Price pressure (volume-weighted bar position)
        df['price_pressure'] = df['bar_position'] * df['vol_ratio']
        
        # Roll spread estimator (Hasbrouck 2009)
        df['roll_spread'] = 2 * np.sqrt(np.abs(df['close'].diff().rolling(20).cov(df['close'].diff().shift(1))))
        df['roll_spread_pct'] = df['roll_spread'] / df['close']
        
        return df
    
    def _add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price action patterns"""
        
        # Candle body size
        df['body_size'] = np.abs(df['close'] - df['open']) / df['open']
        
        # Upper/lower shadows
        df['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['open']
        df['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['open']
        
        # Bullish/bearish bar
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        
        # Consecutive bull/bear bars
        df['consec_bull'] = (df['is_bullish'] == 1).astype(int).groupby((df['is_bullish'] != 1).cumsum()).cumsum()
        df['consec_bear'] = (df['is_bullish'] == 0).astype(int).groupby((df['is_bullish'] != 0).cumsum()).cumsum()
        
        # Higher highs / lower lows
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        
        return df
    
    # Helper calculation methods
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI calculation"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int, d_period: int, smooth: int) -> tuple:
        """Stochastic Oscillator"""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        k = 100 * (df['close'] - low_min) / (high_max - low_min)
        k = k.rolling(window=smooth).mean()  # Smooth %K
        d = k.rolling(window=d_period).mean()  # %D
        
        return k, d
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """MACD calculation"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        return macd, signal_line, histogram
    
    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Williams %R"""
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        
        williams_r = -100 * (high_max - df['close']) / (high_max - low_min)
        return williams_r
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """ADX (simplified version)"""
        # True Range
        tr1 = df['high'] - df['low']
        tr2 = np.abs(df['high'] - df['close'].shift(1))
        tr3 = np.abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = df['high'] - df['high'].shift(1)
        down_move = df['low'].shift(1) - df['low']
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed TR and DM
        tr_smooth = pd.Series(tr).rolling(window=period).sum()
        plus_dm_smooth = pd.Series(plus_dm).rolling(window=period).sum()
        minus_dm_smooth = pd.Series(minus_dm).rolling(window=period).sum()
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        # ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = df['high'] - df['low']
        tr2 = np.abs(df['high'] - df['close'].shift(1))
        tr3 = np.abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        return atr
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """On Balance Volume"""
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return obv
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names"""
        return self.feature_names
    
    def fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN values with safe defaults"""
        df = df.copy()
        
        # Forward fill first (carry last known value)
        df = df.ffill()
        
        # Then backward fill (for leading NaNs)
        df = df.bfill()
        
        # Finally fill any remaining with 0
        df = df.fillna(0)
        
        # Replace inf with large numbers
        df = df.replace([np.inf, -np.inf], [1e10, -1e10])
        
        return df


def quick_test():
    """Quick test of FeatureEngine"""
    print("Testing FeatureEngine...")
    
    # Generate synthetic OHLCV data
    np.random.seed(42)
    n_bars = 500
    
    dates = pd.date_range('2024-01-01', periods=n_bars, freq='1h')
    
    # Random walk price
    returns = np.random.randn(n_bars) * 0.02
    close = 100 * (1 + returns).cumprod()
    
    # OHLV based on close
    high = close * (1 + np.abs(np.random.randn(n_bars) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n_bars) * 0.01))
    open_price = close * (1 + np.random.randn(n_bars) * 0.005)
    volume = np.random.randint(1000000, 5000000, n_bars)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    # Calculate features
    engine = FeatureEngine()
    df_features = engine.calculate_all_features(df)
    
    print(f"\nOriginal columns: {len(df.columns)}")
    print(f"With features: {len(df_features.columns)}")
    print(f"Features added: {len(engine.get_feature_names())}")
    
    print("\nSample features:")
    feature_sample = engine.get_feature_names()[:10]
    print(df_features[feature_sample].tail())
    
    # Check for NaN/Inf
    nan_count = df_features.isna().sum().sum()
    inf_count = np.isinf(df_features.select_dtypes(include=[np.number])).sum().sum()
    
    print(f"\nData quality:")
    print(f"NaN values: {nan_count}")
    print(f"Inf values: {inf_count}")
    
    # Fill missing
    df_clean = engine.fill_missing_values(df_features)
    nan_after = df_clean.isna().sum().sum()
    inf_after = np.isinf(df_clean.select_dtypes(include=[np.number])).sum().sum()
    
    print(f"\nAfter cleaning:")
    print(f"NaN values: {nan_after}")
    print(f"Inf values: {inf_after}")
    
    print("\n✅ FeatureEngine test complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    quick_test()
