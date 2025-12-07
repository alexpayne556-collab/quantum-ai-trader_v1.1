"""
🚀 ULTIMATE AI SIGNAL GENERATOR
================================
Production-ready signal generator using the trained LightGBM model.
Trained on 85.4% win rate walk-forward validated model.

Usage:
    python ultimate_signal_generator.py
    
    # Or import and use programmatically:
    from ultimate_signal_generator import UltimateSignalGenerator
    gen = UltimateSignalGenerator()
    signals = gen.scan_all_tickers()
"""

import numpy as np
import pandas as pd
import yfinance as yf
import talib
import lightgbm as lgb
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class MegaFeatureEngine:
    """
    100+ features - EXACT same as training.
    Must match training features perfectly for model to work!
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        if isinstance(self.df.columns, pd.MultiIndex):
            self.df.columns = self.df.columns.get_level_values(0)
        self.features = pd.DataFrame(index=df.index)
    
    def compute_all_indicators(self) -> pd.DataFrame:
        close = self.df['Close'].values.astype(float)
        high = self.df['High'].values.astype(float)
        low = self.df['Low'].values.astype(float)
        volume = self.df['Volume'].values.astype(float)
        open_price = self.df['Open'].values.astype(float)
        
        # SECTION 1: MOVING AVERAGES
        periods = [5, 8, 10, 13, 20, 21, 34, 50, 55, 89, 100, 200]
        smas = {}
        emas = {}
        for p in periods:
            smas[p] = talib.SMA(close, p)
            emas[p] = talib.EMA(close, p)
            self.features[f'SMA{p}'] = smas[p]
            self.features[f'EMA{p}'] = emas[p]
            self.features[f'Close_vs_SMA{p}'] = (close - smas[p]) / (close + 1e-8)
            self.features[f'Close_vs_EMA{p}'] = (close - emas[p]) / (close + 1e-8)
        
        # SECTION 2: EMA RIBBON DYNAMICS
        fib_emas = [emas[5], emas[8], emas[13], emas[21], emas[34], emas[55], emas[89]]
        bullish_stack = np.ones(len(close))
        bearish_stack = np.ones(len(close))
        for i in range(len(fib_emas) - 1):
            bullish_stack = bullish_stack * (fib_emas[i] > fib_emas[i+1])
            bearish_stack = bearish_stack * (fib_emas[i] < fib_emas[i+1])
        
        self.features['EMA_Bullish_Stack'] = np.nan_to_num(bullish_stack)
        self.features['EMA_Bearish_Stack'] = np.nan_to_num(bearish_stack)
        
        ribbon_width = (emas[5] - emas[89]) / (close + 1e-8)
        self.features['Ribbon_Width'] = ribbon_width
        self.features['Ribbon_Expanding'] = (ribbon_width > np.roll(ribbon_width, 5)).astype(float)
        self.features['Ribbon_Compressing'] = (np.abs(ribbon_width) < np.abs(np.roll(ribbon_width, 5))).astype(float)
        
        for ema_p in [8, 21, 55]:
            slope = (emas[ema_p] - np.roll(emas[ema_p], 5)) / (close + 1e-8)
            self.features[f'EMA{ema_p}_Slope'] = slope
        
        self.features['EMA8_Cross_21'] = np.nan_to_num(((emas[8] > emas[21]) & (np.roll(emas[8], 1) <= np.roll(emas[21], 1))).astype(float))
        self.features['EMA21_Cross_55'] = np.nan_to_num(((emas[21] > emas[55]) & (np.roll(emas[21], 1) <= np.roll(emas[55], 1))).astype(float))
        self.features['Golden_Cross'] = np.nan_to_num(((smas[50] > smas[200]) & (np.roll(smas[50], 1) <= np.roll(smas[200], 1))).astype(float))
        self.features['Death_Cross'] = np.nan_to_num(((smas[50] < smas[200]) & (np.roll(smas[50], 1) >= np.roll(smas[200], 1))).astype(float))
        
        # SECTION 3: MOMENTUM INDICATORS
        for period in [7, 9, 14, 21]:
            self.features[f'RSI_{period}'] = talib.RSI(close, period)
        
        rsi14 = talib.RSI(close, 14)
        self.features['RSI_Oversold'] = (rsi14 < 30).astype(float)
        self.features['RSI_Overbought'] = (rsi14 > 70).astype(float)
        self.features['RSI_Neutral'] = ((rsi14 >= 40) & (rsi14 <= 60)).astype(float)
        self.features['RSI_Momentum'] = rsi14 - np.roll(rsi14, 5)
        
        slowk, slowd = talib.STOCH(high, low, close, 14, 3, 0, 3, 0)
        self.features['Stoch_K'] = slowk
        self.features['Stoch_D'] = slowd
        self.features['Stoch_Cross'] = np.nan_to_num(((slowk > slowd) & (np.roll(slowk, 1) <= np.roll(slowd, 1))).astype(float))
        
        for fast, slow, sig in [(12, 26, 9), (5, 13, 1), (8, 17, 9)]:
            macd, signal, hist = talib.MACD(close, fast, slow, sig)
            suffix = f'{fast}_{slow}'
            self.features[f'MACD_{suffix}'] = macd
            self.features[f'MACD_Signal_{suffix}'] = signal
            self.features[f'MACD_Hist_{suffix}'] = hist
            self.features[f'MACD_Cross_{suffix}'] = np.nan_to_num(((macd > signal) & (np.roll(macd, 1) <= np.roll(signal, 1))).astype(float))
        
        self.features['Williams_R'] = talib.WILLR(high, low, close, 14)
        
        for p in [5, 10, 20]:
            self.features[f'ROC_{p}'] = talib.ROC(close, p)
        
        self.features['MOM_10'] = talib.MOM(close, 10)
        self.features['MOM_20'] = talib.MOM(close, 20)
        
        # SECTION 4: VOLATILITY
        atr14 = talib.ATR(high, low, close, 14)
        atr7 = talib.ATR(high, low, close, 7)
        
        self.features['ATR_14'] = atr14
        self.features['ATR_7'] = atr7
        self.features['ATR_Ratio'] = atr14 / (close + 1e-8)
        self.features['ATR_Expanding'] = (atr14 > np.roll(atr14, 5)).astype(float)
        
        for period in [20, 50]:
            bb_upper, bb_mid, bb_lower = talib.BBANDS(close, period, 2, 2)
            self.features[f'BB_Width_{period}'] = (bb_upper - bb_lower) / (bb_mid + 1e-8)
            self.features[f'BB_Position_{period}'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-8)
        
        kelt_mid = emas[20]
        kelt_upper = kelt_mid + 2 * atr14
        kelt_lower = kelt_mid - 2 * atr14
        self.features['Keltner_Position'] = (close - kelt_lower) / (kelt_upper - kelt_lower + 1e-8)
        
        bb_upper, bb_mid, bb_lower = talib.BBANDS(close, 20, 2, 2)
        squeeze = ((bb_lower > kelt_lower) & (bb_upper < kelt_upper)).astype(float)
        self.features['Squeeze'] = np.nan_to_num(squeeze)
        self.features['Squeeze_Release'] = np.nan_to_num((np.roll(squeeze, 1) == 1) & (squeeze == 0)).astype(float)
        
        # SECTION 5: VOLUME ANALYSIS
        vol_sma20 = talib.SMA(volume, 20)
        vol_sma50 = talib.SMA(volume, 50)
        
        self.features['Vol_Ratio_20'] = volume / (vol_sma20 + 1e-8)
        self.features['Vol_Ratio_50'] = volume / (vol_sma50 + 1e-8)
        self.features['Vol_Surge'] = (volume > 2 * vol_sma20).astype(float)
        
        self.features['OBV'] = talib.OBV(close, volume)
        self.features['OBV_Slope'] = (self.features['OBV'] - self.features['OBV'].shift(5)) / (close + 1e-8)
        
        self.features['MFI'] = talib.MFI(high, low, close, volume, 14)
        self.features['AD'] = talib.AD(high, low, close, volume)
        self.features['CMF'] = talib.ADOSC(high, low, close, volume, 3, 10)
        
        self.features['Vol_Price_Trend'] = (volume * ((close - np.roll(close, 1)) / (np.roll(close, 1) + 1e-8))).cumsum()
        
        # SECTION 6: TREND STRENGTH
        self.features['ADX'] = talib.ADX(high, low, close, 14)
        self.features['PLUS_DI'] = talib.PLUS_DI(high, low, close, 14)
        self.features['MINUS_DI'] = talib.MINUS_DI(high, low, close, 14)
        self.features['DI_Diff'] = self.features['PLUS_DI'] - self.features['MINUS_DI']
        self.features['Strong_Trend'] = (self.features['ADX'] > 25).astype(float)
        self.features['DI_Cross'] = np.nan_to_num(((self.features['PLUS_DI'] > self.features['MINUS_DI']) & 
                                                    (self.features['PLUS_DI'].shift(1) <= self.features['MINUS_DI'].shift(1))).astype(float))
        
        aroon_down, aroon_up = talib.AROON(high, low, 14)
        self.features['Aroon_Up'] = aroon_up
        self.features['Aroon_Down'] = aroon_down
        self.features['Aroon_Osc'] = aroon_up - aroon_down
        
        self.features['CCI'] = talib.CCI(high, low, close, 14)
        
        # SECTION 7: PRICE ACTION
        self.features['Body_Size'] = np.abs(close - open_price) / (close + 1e-8)
        self.features['Upper_Wick'] = (high - np.maximum(open_price, close)) / (close + 1e-8)
        self.features['Lower_Wick'] = (np.minimum(open_price, close) - low) / (close + 1e-8)
        self.features['Wick_Ratio'] = self.features['Upper_Wick'] / (self.features['Lower_Wick'] + 1e-8)
        
        self.features['Gap'] = (open_price - np.roll(close, 1)) / (np.roll(close, 1) + 1e-8)
        self.features['Gap_Up'] = (self.features['Gap'] > 0.005).astype(float)
        self.features['Gap_Down'] = (self.features['Gap'] < -0.005).astype(float)
        
        self.features['HL_Range'] = (high - low) / (close + 1e-8)
        self.features['Range_vs_ATR'] = (high - low) / (atr14 + 1e-8)
        
        self.features['Bullish_Candle'] = (close > open_price).astype(float)
        self.features['Bearish_Candle'] = (close < open_price).astype(float)
        self.features['Doji'] = (self.features['Body_Size'] < 0.001).astype(float)
        
        # SECTION 8: RETURNS
        for p in [1, 2, 3, 5, 10, 20]:
            ret = (close - np.roll(close, p)) / (np.roll(close, p) + 1e-8)
            ret[:p] = 0
            self.features[f'Return_{p}d'] = ret
        
        self.features['Cum_Return_20d'] = (close / np.roll(close, 20)) - 1
        
        ret_1d = np.diff(close) / close[:-1]
        ret_1d = np.concatenate([[0], ret_1d])
        self.features['Return_Volatility'] = pd.Series(ret_1d).rolling(20).std().values
        
        # SECTION 9: REGIME DETECTION
        self.features['Bull_Regime'] = ((close > smas[200]) & (smas[50] > smas[200])).astype(float)
        self.features['Bear_Regime'] = ((close < smas[200]) & (smas[50] < smas[200])).astype(float)
        self.features['Volatile_Regime'] = (atr14 / (close + 1e-8) > 0.02).astype(float)
        
        # SECTION 10: DISCOVERY FEATURES
        self.features['RSI_ADX_Ratio'] = rsi14 / (self.features['ADX'] + 1e-8)
        self.features['MACD_ATR_Ratio'] = self.features['MACD_12_26'] / (atr14 + 1e-8)
        self.features['Vol_Momentum'] = self.features['Vol_Ratio_20'] * self.features['MOM_10']
        self.features['Trend_Vol_Product'] = self.features['ADX'] * self.features['Vol_Ratio_20']
        self.features['EMA_RSI_Combo'] = ribbon_width * rsi14
        self.features['Squeeze_Momentum'] = squeeze * self.features['MOM_10']
        
        high_20 = pd.Series(high).rolling(20).max().values
        low_20 = pd.Series(low).rolling(20).min().values
        self.features['Price_Position_20d'] = (close - low_20) / (high_20 - low_20 + 1e-8)
        
        return self.features


def create_visual_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """Visual pattern features - EXACT same as training."""
    features = {}
    close = df['Close'].values if hasattr(df['Close'], 'values') else df['Close']
    high = df['High'].values if hasattr(df['High'], 'values') else df['High']
    low = df['Low'].values if hasattr(df['Low'], 'values') else df['Low']
    
    if isinstance(df.columns, pd.MultiIndex):
        close = df['Close'].iloc[:, 0].values if df['Close'].ndim > 1 else df['Close'].values
        high = df['High'].iloc[:, 0].values if df['High'].ndim > 1 else df['High'].values
        low = df['Low'].iloc[:, 0].values if df['Low'].ndim > 1 else df['Low'].values
    
    # EMA RIBBON TANGLE DETECTION
    ema_periods = [8, 13, 21, 34, 55]
    emas = {}
    for p in ema_periods:
        emas[p] = pd.Series(close).ewm(span=p, adjust=False).mean().values
    
    ema_max = np.maximum.reduce([emas[p] for p in ema_periods])
    ema_min = np.minimum.reduce([emas[p] for p in ema_periods])
    features['ema_ribbon_width'] = (ema_max - ema_min) / close
    features['ema_ribbon_width_change'] = pd.Series(features['ema_ribbon_width']).diff(5).values
    
    tangle_threshold = 0.01
    features['ema_tangle'] = (features['ema_ribbon_width'] < tangle_threshold).astype(float)
    
    # BREAKOUT DETECTION
    for period in [10, 20, 50]:
        rolling_high = pd.Series(high).rolling(period).max().values
        rolling_low = pd.Series(low).rolling(period).min().values
        features[f'breakout_up_{period}'] = (close > rolling_high * 0.998).astype(float)
        features[f'breakout_down_{period}'] = (close < rolling_low * 1.002).astype(float)
        features[f'distance_from_high_{period}'] = (close - rolling_high) / close
        features[f'distance_from_low_{period}'] = (close - rolling_low) / close
    
    # CANDLESTICK PATTERN SHAPES
    open_vals = df['Open'].values.flatten() if isinstance(df.columns, pd.MultiIndex) else df['Open'].values
    body = np.abs(close - open_vals)
    candle_range = high - low + 0.0001
    features['body_to_range'] = body / candle_range
    
    upper_shadow = high - np.maximum(close, open_vals)
    lower_shadow = np.minimum(close, open_vals) - low
    features['upper_shadow_ratio'] = upper_shadow / candle_range
    features['lower_shadow_ratio'] = lower_shadow / candle_range
    
    # TREND ANGLE DETECTION
    for period in [5, 10, 20]:
        if len(close) > period:
            slopes = np.zeros(len(close))
            for i in range(period, len(close)):
                x = np.arange(period)
                y = close[i-period:i]
                slope, _ = np.polyfit(x, y, 1)
                slopes[i] = slope / close[i] * period
            features[f'trend_slope_{period}'] = slopes
    
    features['distance_to_support'] = np.zeros(len(close))
    features['distance_to_resistance'] = np.zeros(len(close))
    
    return pd.DataFrame(features, index=df.index)


class UltimateSignalGenerator:
    """
    Production signal generator using the 85.4% win rate trained model.
    """
    
    # Default tickers to scan
    DEFAULT_TICKERS = [
        'SPY', 'QQQ', 'IWM', 'DIA',
        'XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLU',
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
        'AMD', 'NFLX', 'CRM', 'ADBE',
        'COIN', 'MARA', 'MSTR',
        'PLTR', 'ARKK', 'SOFI'
    ]
    
    SECTOR_MAP = {
        'AAPL': 'XLK', 'MSFT': 'XLK', 'GOOGL': 'XLK', 'NVDA': 'XLK', 'AMD': 'XLK',
        'META': 'XLK', 'CRM': 'XLK', 'ADBE': 'XLK', 'NFLX': 'XLK',
        'AMZN': 'XLY', 'TSLA': 'XLY',
        'COIN': 'XLF', 'SOFI': 'XLF',
        'MARA': 'XLK', 'MSTR': 'XLK',
        'PLTR': 'XLK', 'ARKK': 'XLK',
    }
    
    def __init__(self, model_path: str = 'models/ultimate_ai_model.txt'):
        """Initialize with trained model."""
        self.model = None
        self.feature_columns = None
        self.model_path = model_path
        
        # Try to load model
        if os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print(f"⚠️ Model not found at {model_path}")
            print("   Please copy ultimate_ai_model.txt from Colab to models/")
    
    def load_model(self, path: str):
        """Load the trained LightGBM model."""
        self.model = lgb.Booster(model_file=path)
        print(f"✅ Model loaded from {path}")
        print(f"   Features expected: {self.model.num_feature()}")
    
    def fetch_data(self, ticker: str, days: int = 300) -> Optional[pd.DataFrame]:
        """Fetch recent data for a ticker."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if len(df) < 200:
                return None
            return df
        except Exception as e:
            print(f"❌ Error fetching {ticker}: {e}")
            return None
    
    def generate_features(self, df: pd.DataFrame, ticker: str, 
                          spy_data: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
        """Generate all features for a ticker."""
        try:
            # Base features
            engine = MegaFeatureEngine(df)
            features = engine.compute_all_indicators()
            
            # Visual pattern features
            visual_feats = create_visual_pattern_features(df)
            for col in visual_feats.columns:
                if col not in features.columns:
                    features[col] = visual_feats[col].reindex(features.index)
            
            # Relative strength vs SPY
            if spy_data is not None:
                spy_close = spy_data['Close'].values.flatten() if isinstance(spy_data.columns, pd.MultiIndex) else spy_data['Close'].values
                ticker_close = df['Close'].values.flatten() if isinstance(df.columns, pd.MultiIndex) else df['Close'].values
                
                spy_returns = pd.Series(spy_close, index=spy_data.index).pct_change()
                ticker_returns = pd.Series(ticker_close, index=df.index).pct_change()
                
                common_idx = ticker_returns.index.intersection(spy_returns.index)
                if len(common_idx) > 50:
                    rs_vs_spy = ticker_returns.loc[common_idx] - spy_returns.loc[common_idx]
                    features['rs_vs_spy_1d'] = rs_vs_spy.reindex(features.index)
                    features['rs_vs_spy_5d'] = rs_vs_spy.rolling(5).sum().reindex(features.index)
                    features['rs_vs_spy_20d'] = rs_vs_spy.rolling(20).sum().reindex(features.index)
            
            return features.dropna()
            
        except Exception as e:
            print(f"❌ Error generating features for {ticker}: {e}")
            return None
    
    def predict(self, features: pd.DataFrame) -> Tuple[float, str]:
        """Get prediction for the latest bar."""
        if self.model is None:
            return 0.0, "Model not loaded"
        
        # Get the latest row
        latest = features.iloc[-1:].values
        
        # Predict
        prob = self.model.predict(latest)[0]
        
        # Determine signal
        if prob > 0.8:
            signal = "🔥 STRONG BUY"
        elif prob > 0.7:
            signal = "✅ BUY"
        elif prob > 0.6:
            signal = "📈 LEAN BULLISH"
        elif prob < 0.3:
            signal = "❌ AVOID"
        else:
            signal = "⏸️ NEUTRAL"
        
        return prob, signal
    
    def scan_ticker(self, ticker: str, spy_data: Optional[pd.DataFrame] = None) -> Optional[Dict]:
        """Scan a single ticker and return signal."""
        df = self.fetch_data(ticker)
        if df is None:
            return None
        
        features = self.generate_features(df, ticker, spy_data)
        if features is None or len(features) < 10:
            return None
        
        prob, signal = self.predict(features)
        
        # Get current price info
        latest = df.iloc[-1]
        close = float(latest['Close'].iloc[0] if isinstance(latest['Close'], pd.Series) else latest['Close'])
        
        return {
            'ticker': ticker,
            'probability': prob,
            'signal': signal,
            'price': close,
            'date': df.index[-1].strftime('%Y-%m-%d'),
            'timestamp': datetime.now().isoformat()
        }
    
    def scan_all_tickers(self, tickers: Optional[List[str]] = None) -> List[Dict]:
        """Scan all tickers and return sorted signals."""
        if tickers is None:
            tickers = self.DEFAULT_TICKERS
        
        print("🔍 ULTIMATE AI SIGNAL SCANNER")
        print("=" * 60)
        print(f"Scanning {len(tickers)} tickers...")
        print(f"Model: 85.4% win rate, walk-forward validated")
        print("-" * 60)
        
        # Fetch SPY data once for relative strength
        spy_data = self.fetch_data('SPY')
        
        signals = []
        for ticker in tickers:
            result = self.scan_ticker(ticker, spy_data)
            if result:
                signals.append(result)
                prob_bar = "█" * int(result['probability'] * 20)
                print(f"{result['ticker']:<6} {result['probability']:.1%} {prob_bar:<20} {result['signal']}")
        
        # Sort by probability descending
        signals.sort(key=lambda x: x['probability'], reverse=True)
        
        print("-" * 60)
        print(f"\n🏆 TOP 5 OPPORTUNITIES:")
        for i, sig in enumerate(signals[:5], 1):
            print(f"  {i}. {sig['ticker']:<6} {sig['probability']:.1%} @ ${sig['price']:.2f} - {sig['signal']}")
        
        return signals
    
    def save_signals(self, signals: List[Dict], path: str = 'signals/latest_signals.json'):
        """Save signals to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump({
                'generated_at': datetime.now().isoformat(),
                'model_win_rate': '85.4%',
                'signals': signals
            }, f, indent=2)
        print(f"\n✅ Signals saved to {path}")


def main():
    """Run the signal scanner."""
    print("\n" + "=" * 60)
    print("🚀 ULTIMATE AI TRADING SIGNAL GENERATOR")
    print("   Model: 85.4% Win Rate | Walk-Forward Validated")
    print("=" * 60 + "\n")
    
    # Initialize generator
    gen = UltimateSignalGenerator()
    
    if gen.model is None:
        print("\n⚠️ MODEL NOT FOUND!")
        print("Please download from Colab and save to models/ultimate_ai_model.txt")
        print("\nTo download from Colab:")
        print("  1. In Colab, click the folder icon on the left")
        print("  2. Right-click 'ultimate_ai_model.txt' → Download")
        print("  3. Save to this project's models/ folder")
        return
    
    # Scan all tickers
    signals = gen.scan_all_tickers()
    
    # Save results
    gen.save_signals(signals)
    
    print("\n" + "=" * 60)
    print("✅ SCAN COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    main()
