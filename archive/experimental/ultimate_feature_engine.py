"""
UltimateFeatureEngine: Generate 50+ trading indicators
Used for AI pattern recognition matching human visual analysis

Optimized for Colab T4 High-RAM training
Uses TA-Lib for institutional-grade indicators
"""

import talib
import pandas as pd
import numpy as np
from typing import Optional, List, Dict


class UltimateFeatureEngine:
    """
    Generate 50+ technical indicators for universal AI model training.
    
    Categories:
    - Moving Averages (SMA, EMA - 5,10,20,50,100,200 periods)
    - EMA Ribbon Analysis (alignment, width, stack detection)
    - Momentum (RSI multiple periods, StochRSI, MACD, Williams %R)
    - Volatility (ATR, Bollinger Bands, Keltner Channels)
    - Volume Analysis (OBV, VROC, Volume Ratio, MFI)
    - Trend Strength (ADX, DI+/DI-, Aroon)
    - Price Action (candle body, wicks, gaps)
    - Support/Resistance (pivots, levels)
    - Returns (1d, 5d, 20d, cumulative)
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with OHLCV DataFrame.
        
        Args:
            df: DataFrame with Open, High, Low, Close, Volume columns
        """
        self.df = df.copy()
        self.features = pd.DataFrame(index=df.index)
        
        # Handle multi-index columns from yfinance
        if isinstance(self.df.columns, pd.MultiIndex):
            self.df.columns = self.df.columns.get_level_values(0)
        
        # Ensure numeric types
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
    
    def compute_all_indicators(self) -> pd.DataFrame:
        """
        Compute all 50+ indicators and return features DataFrame.
        Returns cleaned DataFrame (NaN rows dropped).
        """
        self._compute_moving_averages()
        self._compute_ema_ribbon()
        self._compute_momentum()
        self._compute_volatility()
        self._compute_volume()
        self._compute_trend()
        self._compute_price_action()
        self._compute_support_resistance()
        self._compute_returns()
        self._compute_advanced_patterns()
        
        return self.features.dropna()
    
    def _compute_moving_averages(self):
        """SMAs and EMAs for 5, 10, 20, 50, 100, 200 periods"""
        close = self.df['Close'].values
        
        for period in [5, 10, 20, 50, 100, 200]:
            self.features[f'SMA{period}'] = talib.SMA(close, period)
            self.features[f'EMA{period}'] = talib.EMA(close, period)
            
            # Price relative to moving average (normalized)
            sma = talib.SMA(close, period)
            self.features[f'Close_vs_SMA{period}'] = (close - sma) / (close + 1e-8)
    
    def _compute_ema_ribbon(self):
        """EMA ribbon alignment and width analysis"""
        close = self.df['Close'].values
        
        # Calculate core EMAs
        ema5 = talib.EMA(close, 5)
        ema10 = talib.EMA(close, 10)
        ema20 = talib.EMA(close, 20)
        ema50 = talib.EMA(close, 50)
        
        # Stack detection (all aligned)
        bullish_stack = (
            (ema5 > ema10) & (ema10 > ema20) & (ema20 > ema50)
        ).astype(float)
        bearish_stack = (
            (ema5 < ema10) & (ema10 < ema20) & (ema20 < ema50)
        ).astype(float)
        
        # Handle NaN safely
        bullish_stack = np.nan_to_num(bullish_stack, nan=0)
        bearish_stack = np.nan_to_num(bearish_stack, nan=0)
        
        self.features['EMA_Bullish_Stack'] = bullish_stack
        self.features['EMA_Bearish_Stack'] = bearish_stack
        
        # Ribbon width (spread between fast and slow)
        self.features['EMA_Ribbon_Width'] = (ema5 - ema50) / (close + 1e-8)
        
        # EMA slope (momentum of trend)
        self.features['EMA20_Slope'] = (ema20 - np.roll(ema20, 5)) / (close + 1e-8)
        self.features['EMA50_Slope'] = (ema50 - np.roll(ema50, 5)) / (close + 1e-8)
    
    def _compute_momentum(self):
        """RSI, MACD, Stochastic, Williams %R"""
        close = self.df['Close'].values
        high = self.df['High'].values
        low = self.df['Low'].values
        
        # RSI multiple periods
        self.features['RSI_7'] = talib.RSI(close, 7)
        self.features['RSI_14'] = talib.RSI(close, 14)
        self.features['RSI_21'] = talib.RSI(close, 21)
        
        # RSI zones (normalized 0-1)
        rsi14 = talib.RSI(close, 14)
        self.features['RSI_Oversold'] = (rsi14 < 30).astype(float)
        self.features['RSI_Overbought'] = (rsi14 > 70).astype(float)
        
        # Stochastic RSI
        slowk, slowd = talib.STOCH(high, low, close, 14, 3, 0, 3, 0)
        self.features['StochK'] = slowk
        self.features['StochD'] = slowd
        self.features['StochRSI'] = slowk  # Simplified
        
        # MACD
        macd, signal, hist = talib.MACD(close, 12, 26, 9)
        self.features['MACD'] = macd
        self.features['MACD_Signal'] = signal
        self.features['MACD_Histogram'] = hist
        self.features['MACD_Positive'] = (hist > 0).astype(float)
        self.features['MACD_Crossover'] = (
            (hist > 0) & (np.roll(hist, 1) <= 0)
        ).astype(float)
        
        # Williams %R
        self.features['Williams_R'] = talib.WILLR(high, low, close, 14)
        
        # Rate of Change
        self.features['ROC_10'] = talib.ROC(close, 10)
        self.features['ROC_20'] = talib.ROC(close, 20)
        
        # Momentum
        self.features['MOM_10'] = talib.MOM(close, 10)
    
    def _compute_volatility(self):
        """ATR, Bollinger Bands, Keltner Channels"""
        close = self.df['Close'].values
        high = self.df['High'].values
        low = self.df['Low'].values
        
        # ATR
        atr = talib.ATR(high, low, close, 14)
        self.features['ATR'] = atr
        self.features['ATR_Ratio'] = atr / (close + 1e-8)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, 20, 2, 2)
        self.features['BB_Upper'] = bb_upper
        self.features['BB_Middle'] = bb_middle
        self.features['BB_Lower'] = bb_lower
        self.features['BB_Width'] = (bb_upper - bb_lower) / (bb_middle + 1e-8)
        self.features['BB_Position'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-8)
        
        # Keltner Channel (EMA + ATR-based bands)
        ema20 = talib.EMA(close, 20)
        kelt_upper = ema20 + 2 * atr
        kelt_lower = ema20 - 2 * atr
        self.features['Keltner_Position'] = (close - kelt_lower) / (kelt_upper - kelt_lower + 1e-8)
        
        # Volatility squeeze (BB inside Keltner)
        squeeze = (bb_lower > kelt_lower) & (bb_upper < kelt_upper)
        self.features['Squeeze'] = squeeze.astype(float)
    
    def _compute_volume(self):
        """OBV, Volume Ratio, MFI, VROC"""
        close = self.df['Close'].values
        high = self.df['High'].values
        low = self.df['Low'].values
        volume = self.df['Volume'].values.astype(float)
        
        # Volume MA and ratio
        vol_ma = talib.SMA(volume, 20)
        self.features['Volume_MA'] = vol_ma
        self.features['Volume_Ratio'] = volume / (vol_ma + 1e-8)
        
        # OBV
        self.features['OBV'] = talib.OBV(close, volume)
        
        # Volume ROC
        self.features['VROC'] = talib.ROC(volume, 10)
        
        # Money Flow Index
        self.features['MFI'] = talib.MFI(high, low, close, volume, 14)
        
        # Accumulation/Distribution
        self.features['AD'] = talib.AD(high, low, close, volume)
        
        # Chaikin Money Flow (approximated)
        self.features['CMF'] = talib.ADOSC(high, low, close, volume, 3, 10)
    
    def _compute_trend(self):
        """ADX, DI+/DI-, Aroon, CCI"""
        close = self.df['Close'].values
        high = self.df['High'].values
        low = self.df['Low'].values
        
        # ADX (trend strength)
        self.features['ADX'] = talib.ADX(high, low, close, 14)
        
        # Directional Indicators
        plus_di = talib.PLUS_DI(high, low, close, 14)
        minus_di = talib.MINUS_DI(high, low, close, 14)
        self.features['PLUS_DI'] = plus_di
        self.features['MINUS_DI'] = minus_di
        self.features['DI_Difference'] = plus_di - minus_di
        
        # Aroon
        aroon_down, aroon_up = talib.AROON(high, low, 14)
        self.features['Aroon_Up'] = aroon_up
        self.features['Aroon_Down'] = aroon_down
        self.features['Aroon_Osc'] = aroon_up - aroon_down
        
        # CCI
        self.features['CCI'] = talib.CCI(high, low, close, 14)
    
    def _compute_price_action(self):
        """Candlestick body, wicks, gaps"""
        open_price = self.df['Open'].values
        high = self.df['High'].values
        low = self.df['Low'].values
        close = self.df['Close'].values
        
        # Candle direction
        self.features['Close_Above_Open'] = (close > open_price).astype(float)
        
        # Body size (normalized)
        self.features['Body_Size'] = np.abs(close - open_price) / (close + 1e-8)
        
        # Wick sizes (normalized)
        upper_wick = high - np.maximum(open_price, close)
        lower_wick = np.minimum(open_price, close) - low
        self.features['Upper_Wick'] = upper_wick / (close + 1e-8)
        self.features['Lower_Wick'] = lower_wick / (close + 1e-8)
        
        # Wick ratio (rejection signals)
        total_range = high - low + 1e-8
        self.features['Upper_Wick_Ratio'] = upper_wick / total_range
        self.features['Lower_Wick_Ratio'] = lower_wick / total_range
        
        # Gap analysis
        prev_close = np.roll(close, 1)
        self.features['Gap'] = (open_price - prev_close) / (prev_close + 1e-8)
        self.features['Gap_Up'] = (open_price > prev_close).astype(float)
        self.features['Gap_Down'] = (open_price < prev_close).astype(float)
        
        # High-Low range
        self.features['HL_Range'] = (high - low) / (close + 1e-8)
    
    def _compute_support_resistance(self):
        """Pivot points and levels"""
        high = self.df['High'].values
        low = self.df['Low'].values
        close = self.df['Close'].values
        
        # Classic Pivot Points (shifted for previous day)
        prev_high = np.roll(high, 1)
        prev_low = np.roll(low, 1)
        prev_close = np.roll(close, 1)
        
        pivot = (prev_high + prev_low + prev_close) / 3
        self.features['Pivot'] = pivot
        self.features['R1'] = 2 * pivot - prev_low
        self.features['S1'] = 2 * pivot - prev_high
        self.features['R2'] = pivot + (prev_high - prev_low)
        self.features['S2'] = pivot - (prev_high - prev_low)
        
        # Distance from pivot (normalized)
        self.features['Distance_From_Pivot'] = (close - pivot) / (close + 1e-8)
    
    def _compute_returns(self):
        """Various return calculations"""
        close = self.df['Close'].values
        
        # Simple returns
        self.features['Return_1d'] = np.concatenate([[0], np.diff(close) / (close[:-1] + 1e-8)])
        
        # Multi-period returns
        for period in [5, 10, 20]:
            pct_change = (close - np.roll(close, period)) / (np.roll(close, period) + 1e-8)
            pct_change[:period] = 0  # Avoid using rolled values
            self.features[f'Return_{period}d'] = pct_change
        
        # Cumulative returns (normalized)
        cumret = close / (close[0] + 1e-8) - 1
        self.features['Cumulative_Return'] = cumret
    
    def _compute_advanced_patterns(self):
        """Human trading patterns like Golden Cross, Death Cross"""
        close = self.df['Close'].values
        
        sma50 = talib.SMA(close, 50)
        sma200 = talib.SMA(close, 200)
        
        # Golden Cross (SMA50 crosses above SMA200)
        golden_cross = (
            (sma50 > sma200) & (np.roll(sma50, 1) <= np.roll(sma200, 1))
        )
        self.features['Golden_Cross'] = np.nan_to_num(golden_cross.astype(float), nan=0)
        
        # Death Cross (SMA50 crosses below SMA200)
        death_cross = (
            (sma50 < sma200) & (np.roll(sma50, 1) >= np.roll(sma200, 1))
        )
        self.features['Death_Cross'] = np.nan_to_num(death_cross.astype(float), nan=0)
        
        # Price above/below key MAs
        self.features['Above_SMA200'] = (close > sma200).astype(float)
        self.features['Above_SMA50'] = (close > sma50).astype(float)
        
        # RSI + MA combo (popular human strategy)
        rsi14 = talib.RSI(close, 14)
        self.features['RSI_Bullish_Divergence'] = (
            (rsi14 < 30) & (close > sma50)
        ).astype(float)
        self.features['RSI_Bearish_Divergence'] = (
            (rsi14 > 70) & (close < sma50)
        ).astype(float)


def get_feature_names() -> List[str]:
    """Return list of all feature names generated by UltimateFeatureEngine"""
    return [
        # Moving Averages (18)
        'SMA5', 'SMA10', 'SMA20', 'SMA50', 'SMA100', 'SMA200',
        'EMA5', 'EMA10', 'EMA20', 'EMA50', 'EMA100', 'EMA200',
        'Close_vs_SMA5', 'Close_vs_SMA10', 'Close_vs_SMA20', 
        'Close_vs_SMA50', 'Close_vs_SMA100', 'Close_vs_SMA200',
        # EMA Ribbon (5)
        'EMA_Bullish_Stack', 'EMA_Bearish_Stack', 'EMA_Ribbon_Width',
        'EMA20_Slope', 'EMA50_Slope',
        # Momentum (15)
        'RSI_7', 'RSI_14', 'RSI_21', 'RSI_Oversold', 'RSI_Overbought',
        'StochK', 'StochD', 'StochRSI',
        'MACD', 'MACD_Signal', 'MACD_Histogram', 'MACD_Positive', 'MACD_Crossover',
        'Williams_R', 'ROC_10', 'ROC_20', 'MOM_10',
        # Volatility (9)
        'ATR', 'ATR_Ratio', 'BB_Upper', 'BB_Middle', 'BB_Lower',
        'BB_Width', 'BB_Position', 'Keltner_Position', 'Squeeze',
        # Volume (8)
        'Volume_MA', 'Volume_Ratio', 'OBV', 'VROC', 'MFI', 'AD', 'CMF',
        # Trend (9)
        'ADX', 'PLUS_DI', 'MINUS_DI', 'DI_Difference',
        'Aroon_Up', 'Aroon_Down', 'Aroon_Osc', 'CCI',
        # Price Action (11)
        'Close_Above_Open', 'Body_Size', 'Upper_Wick', 'Lower_Wick',
        'Upper_Wick_Ratio', 'Lower_Wick_Ratio', 'Gap', 'Gap_Up', 'Gap_Down', 'HL_Range',
        # Support/Resistance (6)
        'Pivot', 'R1', 'S1', 'R2', 'S2', 'Distance_From_Pivot',
        # Returns (5)
        'Return_1d', 'Return_5d', 'Return_10d', 'Return_20d', 'Cumulative_Return',
        # Advanced Patterns (6)
        'Golden_Cross', 'Death_Cross', 'Above_SMA200', 'Above_SMA50',
        'RSI_Bullish_Divergence', 'RSI_Bearish_Divergence'
    ]


# Quick test
if __name__ == "__main__":
    import yfinance as yf
    
    print("🚀 UltimateFeatureEngine Test")
    print("=" * 60)
    
    # Download sample data
    df = yf.download("SPY", start="2023-01-01", end="2024-01-01", progress=False)
    
    # Generate features
    engine = UltimateFeatureEngine(df)
    features = engine.compute_all_indicators()
    
    print(f"✅ Generated {features.shape[1]} features from {features.shape[0]} samples")
    print(f"\nFeature columns:")
    for i, col in enumerate(features.columns, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\n📊 Sample values (last row):")
    print(features.iloc[-1].to_string())
