"""
ADVANCED FEATURE ENGINEERING FOR QUANTUM FORECASTER
===================================================
Implements institutional-grade feature extraction including:
- Market microstructure features
- Alternative data proxies
- Sentiment indicators
- Quantum-inspired transformations
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Use built-in TA indicators if pandas_ta not available
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    print("⚠ pandas_ta not available - using built-in indicators")


# ==================== BUILT-IN TECHNICAL INDICATORS ====================

class BuiltInTA:
    """Built-in technical indicators when pandas_ta is not available"""
    
    @staticmethod
    def rsi(series: pd.Series, length: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def sma(series: pd.Series, length: int = 20) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(window=length).mean()
    
    @staticmethod
    def ema(series: pd.Series, length: int = 20) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=length, adjust=False).mean()
    
    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """MACD"""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return pd.DataFrame({
            'MACD_12_26_9': macd_line,
            'MACDs_12_26_9': signal_line,
            'MACDh_12_26_9': histogram
        }, index=series.index)
    
    @staticmethod
    def bbands(series: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
        """Bollinger Bands"""
        middle = series.rolling(window=length).mean()
        std_dev = series.rolling(window=length).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return pd.DataFrame({
            'BBL_20_2.0': lower,
            'BBM_20_2.0': middle,
            'BBU_20_2.0': upper
        }, index=series.index)
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On Balance Volume"""
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
    
    @staticmethod
    def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int = 14) -> pd.Series:
        """Money Flow Index"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price.diff() > 0, 0)
        negative_flow = money_flow.where(typical_price.diff() < 0, 0)
        
        positive_mf = positive_flow.rolling(window=length).sum()
        negative_mf = negative_flow.rolling(window=length).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-8)))
        return mfi
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=length).mean()
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.DataFrame:
        """Average Directional Index"""
        # True Range
        tr = BuiltInTA.atr(high, low, close, length)
        
        # Directional Movement
        up_move = high.diff()
        down_move = -low.diff()
        
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        # Smooth
        plus_di = 100 * (plus_dm.rolling(window=length).mean() / (tr + 1e-8))
        minus_di = 100 * (minus_dm.rolling(window=length).mean() / (tr + 1e-8))
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        adx = dx.rolling(window=length).mean()
        
        return pd.DataFrame({
            'ADX_14': adx,
            'DMP_14': plus_di,
            'DMN_14': minus_di
        }, index=close.index)
    
    @staticmethod
    def kc(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 20, mult: float = 2.0) -> pd.DataFrame:
        """Keltner Channels"""
        middle = close.ewm(span=length, adjust=False).mean()
        atr = BuiltInTA.atr(high, low, close, length)
        upper = middle + (mult * atr)
        lower = middle - (mult * atr)
        return pd.DataFrame({
            'KCLe_20_2': lower,
            'KCBe_20_2': middle,
            'KCUe_20_2': upper
        }, index=close.index)


# If pandas_ta not available, use built-in
if not HAS_PANDAS_TA:
    ta = BuiltInTA


# ==================== MARKET MICROSTRUCTURE FEATURES ====================

class MicrostructureFeatures:
    """
    Extract order flow and microstructure signals
    Simulated from price/volume data when Level 2 unavailable
    """
    
    @staticmethod
    def order_flow_imbalance(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Estimate order flow imbalance from price/volume
        
        Args:
            df: DataFrame with OHLCV data
            window: Rolling window for calculations
            
        Returns:
            DataFrame with OFI features
        """
        features = pd.DataFrame(index=df.index)
        
        # Buy/Sell volume estimation (Lee-Ready algorithm approximation)
        price_change = df['Close'].diff()
        features['aggressive_buy_volume'] = np.where(
            price_change > 0,
            df['Volume'],
            0
        )
        features['aggressive_sell_volume'] = np.where(
            price_change < 0,
            df['Volume'],
            0
        )
        
        # Order imbalance ratio
        buy_vol_sum = features['aggressive_buy_volume'].rolling(window).sum()
        sell_vol_sum = features['aggressive_sell_volume'].rolling(window).sum()
        features['order_imbalance'] = (buy_vol_sum - sell_vol_sum) / (buy_vol_sum + sell_vol_sum + 1e-8)
        
        # Volume-weighted spread proxy
        high_low_range = (df['High'] - df['Low']) / df['Close']
        features['spread_proxy'] = high_low_range.rolling(window).mean() * 10000  # bps
        
        # Trade velocity (frequency proxy)
        features['trade_velocity'] = df['Volume'].rolling(window).count() / window
        
        # Hidden liquidity detection (std of volume)
        features['liquidity_clustering'] = (
            df['Volume'].rolling(window).std() / 
            (df['Volume'].rolling(window).mean() + 1e-8)
        )
        
        return features
    
    @staticmethod
    def dark_pool_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Proxy for dark pool/block trade activity
        Uses large volume spikes as indicator
        """
        features = pd.DataFrame(index=df.index)
        
        # Volume Z-score (unusual volume detection)
        vol_mean = df['Volume'].rolling(50).mean()
        vol_std = df['Volume'].rolling(50).std()
        features['volume_zscore'] = (df['Volume'] - vol_mean) / (vol_std + 1e-8)
        
        # Block trade indicator (volume > 2 std)
        features['block_trade_flag'] = (features['volume_zscore'] > 2).astype(int)
        
        # Institutional accumulation (sustained volume + price increase)
        price_momentum = df['Close'].pct_change(20)
        volume_momentum = df['Volume'].rolling(20).mean() / df['Volume'].rolling(60).mean()
        features['institutional_accumulation'] = (
            (price_momentum > 0.02) & (volume_momentum > 1.2)
        ).astype(float)
        
        # Distribution pattern (opposite)
        features['institutional_distribution'] = (
            (price_momentum < -0.02) & (volume_momentum > 1.2)
        ).astype(float)
        
        return features
    
    @staticmethod
    def market_maker_positioning(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """
        Estimate market maker inventory and positioning
        """
        features = pd.DataFrame(index=df.index)
        
        # Intraday volatility (High-Low range)
        features['intraday_volatility'] = (df['High'] - df['Low']) / df['Open']
        
        # Opening gap (overnight positioning)
        features['opening_gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        
        # Closing strength (MM positioning into close)
        features['closing_strength'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-8)
        
        # Volume at price (concentration indicator)
        features['volume_concentration'] = (
            df['Volume'] / (df['High'] - df['Low'] + 1e-8)
        ).rolling(window).mean()
        
        return features


# ==================== ALTERNATIVE DATA FEATURES ====================

class AlternativeDataFeatures:
    """
    Proxy alternative data signals from public sources
    """
    
    @staticmethod
    def supply_chain_proxy(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Supply chain indicators (momentum-based proxies)
        """
        features = pd.DataFrame(index=df.index)
        
        # Production momentum (volume trend as proxy)
        features['production_momentum'] = (
            df['Volume'].rolling(20).mean() / 
            df['Volume'].rolling(60).mean() - 1
        )
        
        # Demand signal (price momentum + volume)
        price_mom = df['Close'].pct_change(30)
        vol_mom = df['Volume'].rolling(30).mean() / df['Volume'].rolling(90).mean()
        features['demand_signal'] = price_mom * vol_mom
        
        # Inventory cycle (volatility-based proxy)
        returns = df['Close'].pct_change()
        features['inventory_cycle'] = returns.rolling(60).std() / returns.rolling(180).std()
        
        return features
    
    @staticmethod
    def cloud_infrastructure_proxy(df: pd.DataFrame) -> pd.DataFrame:
        """
        Cloud/AI infrastructure demand proxies
        """
        features = pd.DataFrame(index=df.index)
        
        # GPU demand proxy (tech sector momentum)
        features['gpu_demand_proxy'] = df['Close'].pct_change(20).rolling(5).mean()
        
        # AI workload proxy (sustained volume growth)
        features['ai_workload_proxy'] = (
            df['Volume'].rolling(30).mean() / 
            df['Volume'].rolling(90).mean()
        ).rolling(10).mean()
        
        # Data center activity (volatility + volume)
        returns_vol = df['Close'].pct_change().rolling(20).std()
        volume_growth = df['Volume'].pct_change(20)
        features['datacenter_activity'] = returns_vol * volume_growth
        
        return features
    
    @staticmethod
    def social_sentiment_proxy(df: pd.DataFrame) -> pd.DataFrame:
        """
        Social sentiment proxies from price action
        """
        features = pd.DataFrame(index=df.index)
        
        # FOMO indicator (rapid price increase + volume spike)
        price_accel = df['Close'].pct_change(5) - df['Close'].pct_change(20)
        volume_spike = df['Volume'] / df['Volume'].rolling(20).mean()
        features['fomo_indicator'] = (price_accel > 0.05) & (volume_spike > 1.5)
        
        # Panic indicator (rapid decline + volume)
        features['panic_indicator'] = (price_accel < -0.05) & (volume_spike > 1.5)
        
        # Retail momentum (small lot proxy - using volatility)
        features['retail_momentum'] = df['Close'].pct_change().rolling(5).std()
        
        return features


# ==================== QUANTUM-INSPIRED FEATURES ====================

class QuantumInspiredFeatures:
    """
    Quantum-inspired transformations for feature engineering
    """
    
    @staticmethod
    def wavefunction_encoding(prices: pd.Series, window: int = 20) -> pd.DataFrame:
        """
        Encode price series as quantum wavefunction amplitudes
        """
        features = pd.DataFrame(index=prices.index)
        
        # Normalize prices to [-1, 1] range (valid wavefunction)
        normalized_prices = (
            2 * (prices - prices.rolling(window).min()) / 
            (prices.rolling(window).max() - prices.rolling(window).min() + 1e-8) - 1
        )
        
        # Quantum phase encoding
        features['quantum_phase'] = np.arctan2(
            normalized_prices,
            normalized_prices.shift(1).fillna(0)
        )
        
        # Amplitude encoding (probability amplitude)
        features['quantum_amplitude'] = np.sqrt(
            (normalized_prices ** 2 + normalized_prices.shift(1).fillna(0) ** 2) / 2
        )
        
        # Superposition state (multiple price regimes)
        regime_fast = (prices > prices.rolling(10).mean()).astype(float)
        regime_slow = (prices > prices.rolling(50).mean()).astype(float)
        features['superposition_state'] = (regime_fast + regime_slow) / 2
        
        return features
    
    @staticmethod
    def entanglement_correlation(df: pd.DataFrame, corr_window: int = 30) -> pd.DataFrame:
        """
        Model cross-asset correlations as quantum entanglement
        """
        features = pd.DataFrame(index=df.index)
        
        # Return correlation strength (entanglement degree)
        returns = df['Close'].pct_change()
        features['entanglement_strength'] = returns.rolling(corr_window).corr(
            returns.shift(1)
        ).abs()
        
        # Phase coherence (momentum alignment)
        mom_5 = df['Close'].pct_change(5)
        mom_20 = df['Close'].pct_change(20)
        features['phase_coherence'] = np.sign(mom_5) == np.sign(mom_20)
        
        # Quantum interference pattern (oscillation detection)
        detrended = returns - returns.rolling(corr_window).mean()
        features['interference_pattern'] = np.sin(
            2 * np.pi * detrended.rolling(10).sum()
        )
        
        return features


# ==================== TECHNICAL INDICATORS (ENHANCED) ====================

class EnhancedTechnicalIndicators:
    """
    Advanced technical indicators beyond standard TA
    """
    
    @staticmethod
    def generate_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive technical indicator set
        """
        features = pd.DataFrame(index=df.index)
        
        # Momentum indicators
        features['rsi_9'] = ta.rsi(df['Close'], length=9)
        features['rsi_14'] = ta.rsi(df['Close'], length=14)
        features['rsi_divergence'] = features['rsi_9'] - features['rsi_14']
        
        # MACD variations
        macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        features['macd'] = macd['MACD_12_26_9']
        features['macd_signal'] = macd['MACDs_12_26_9']
        features['macd_hist'] = macd['MACDh_12_26_9']
        
        # Bollinger Bands
        bbands = ta.bbands(df['Close'], length=20, std=2)
        features['bb_upper'] = bbands['BBU_20_2.0']
        features['bb_middle'] = bbands['BBM_20_2.0']
        features['bb_lower'] = bbands['BBL_20_2.0']
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']
        features['bb_position'] = (df['Close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'] + 1e-8)
        
        # Volume indicators
        features['obv'] = ta.obv(df['Close'], df['Volume'])
        features['mfi'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
        features['volume_sma_ratio'] = df['Volume'] / ta.sma(df['Volume'], length=20)
        
        # Trend indicators
        adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        features['adx'] = adx['ADX_14']
        features['plus_di'] = adx['DMP_14']
        features['minus_di'] = adx['DMN_14']
        
        # Volatility indicators
        features['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        features['atr_pct'] = features['atr'] / df['Close']
        features['kc_upper'] = ta.kc(df['High'], df['Low'], df['Close'], length=20)['KCUe_20_2']
        features['kc_lower'] = ta.kc(df['High'], df['Low'], df['Close'], length=20)['KCLe_20_2']
        
        # Price action
        features['returns_1d'] = df['Close'].pct_change(1)
        features['returns_5d'] = df['Close'].pct_change(5)
        features['returns_20d'] = df['Close'].pct_change(20)
        features['volatility_20d'] = features['returns_1d'].rolling(20).std()
        
        return features


# ==================== COMPLETE FEATURE PIPELINE ====================

class QuantumFeatureEngineer:
    """
    Complete feature engineering pipeline for Quantum Forecaster
    """
    
    def __init__(self):
        self.microstructure = MicrostructureFeatures()
        self.alternative = AlternativeDataFeatures()
        self.quantum = QuantumInspiredFeatures()
        self.technical = EnhancedTechnicalIndicators()
        
    def engineer_all_features(self, ticker: str, period: str = '5y') -> Dict[str, pd.DataFrame]:
        """
        Generate all feature categories for a ticker
        
        Args:
            ticker: Stock symbol
            period: Historical period
            
        Returns:
            Dictionary with categorized features
        """
        print(f"📊 Fetching data for {ticker}...")
        df = yf.download(ticker, period=period, progress=False)
        
        if df.empty:
            raise ValueError(f"No data retrieved for {ticker}")
        
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        print(f"🔧 Engineering features for {ticker}...")
        
        # Base technical indicators
        technical_features = self.technical.generate_features(df)
        
        # Microstructure features
        microstructure_features = pd.concat([
            self.microstructure.order_flow_imbalance(df),
            self.microstructure.dark_pool_indicators(df),
            self.microstructure.market_maker_positioning(df)
        ], axis=1)
        
        # Alternative data proxies
        alternative_features = pd.concat([
            self.alternative.supply_chain_proxy(ticker, df),
            self.alternative.cloud_infrastructure_proxy(df),
            self.alternative.social_sentiment_proxy(df)
        ], axis=1)
        
        # Quantum-inspired features
        quantum_features = pd.concat([
            self.quantum.wavefunction_encoding(df['Close'].squeeze()),
            self.quantum.entanglement_correlation(df)
        ], axis=1)
        
        # Combine OHLCV with engineered features
        all_features = pd.concat([
            df[['Open', 'High', 'Low', 'Close', 'Volume']],
            technical_features,
            microstructure_features,
            alternative_features,
            quantum_features
        ], axis=1)
        
        # Drop NaN rows
        all_features = all_features.dropna()
        
        print(f"✅ Generated {len(all_features.columns)} features for {ticker}")
        print(f"   - Technical: {len(technical_features.columns)}")
        print(f"   - Microstructure: {len(microstructure_features.columns)}")
        print(f"   - Alternative: {len(alternative_features.columns)}")
        print(f"   - Quantum: {len(quantum_features.columns)}")
        
        return {
            'all_features': all_features,
            'technical': technical_features,
            'microstructure': microstructure_features,
            'alternative': alternative_features,
            'quantum': quantum_features,
            'ohlcv': df
        }
    
    def prepare_model_inputs(self, 
                            features_dict: Dict[str, pd.DataFrame],
                            sequence_length: int = 60,
                            prediction_horizon: int = 14) -> Tuple[np.ndarray, ...]:
        """
        Prepare features for model input
        
        Returns:
            Tuple of (price_features, microstructure, alternative, sentiment, labels)
        """
        all_features = features_dict['all_features']
        
        # Select feature subsets (adjust indices based on actual features)
        price_cols = [col for col in all_features.columns if any(
            x in col.lower() for x in ['close', 'rsi', 'macd', 'returns', 'bb', 'atr']
        )]
        micro_cols = [col for col in all_features.columns if any(
            x in col.lower() for x in ['imbalance', 'spread', 'velocity', 'block', 'institutional']
        )]
        alt_cols = [col for col in all_features.columns if any(
            x in col.lower() for x in ['supply', 'demand', 'cloud', 'gpu', 'datacenter']
        )]
        sent_cols = [col for col in all_features.columns if any(
            x in col.lower() for x in ['fomo', 'panic', 'retail']
        )]
        
        # Pad with zeros if not enough features
        min_features = {'price': 32, 'micro': 16, 'alt': 12, 'sent': 8}
        
        price_features = all_features[price_cols].values
        if price_features.shape[1] < min_features['price']:
            padding = np.zeros((price_features.shape[0], min_features['price'] - price_features.shape[1]))
            price_features = np.hstack([price_features, padding])
        else:
            price_features = price_features[:, :min_features['price']]
        
        micro_features = all_features[micro_cols].values if micro_cols else np.zeros((len(all_features), min_features['micro']))
        alt_features = all_features[alt_cols].values if alt_cols else np.zeros((len(all_features), min_features['alt']))
        sent_features = all_features[sent_cols].values if sent_cols else np.zeros((len(all_features), min_features['sent']))
        
        # Pad to minimum dimensions
        if micro_features.shape[1] < min_features['micro']:
            padding = np.zeros((micro_features.shape[0], min_features['micro'] - micro_features.shape[1]))
            micro_features = np.hstack([micro_features, padding])
        else:
            micro_features = micro_features[:, :min_features['micro']]
            
        if alt_features.shape[1] < min_features['alt']:
            padding = np.zeros((alt_features.shape[0], min_features['alt'] - alt_features.shape[1]))
            alt_features = np.hstack([alt_features, padding])
        else:
            alt_features = alt_features[:, :min_features['alt']]
            
        if sent_features.shape[1] < min_features['sent']:
            padding = np.zeros((sent_features.shape[0], min_features['sent'] - sent_features.shape[1]))
            sent_features = np.hstack([sent_features, padding])
        else:
            sent_features = sent_features[:, :min_features['sent']]
        
        # Create sequences
        X_price, X_micro, X_alt, X_sent, y = [], [], [], [], []
        
        for i in range(sequence_length, len(all_features) - prediction_horizon):
            X_price.append(price_features[i-sequence_length:i])
            X_micro.append(micro_features[i-sequence_length:i])
            X_alt.append(alt_features[i-sequence_length:i])
            X_sent.append(sent_features[i-sequence_length:i])
            
            # Label: 14-day forward return
            future_return = (
                all_features['Close'].iloc[i+prediction_horizon] / 
                all_features['Close'].iloc[i] - 1
            )
            y.append(future_return)
        
        return (
            np.array(X_price, dtype=np.float32),
            np.array(X_micro, dtype=np.float32),
            np.array(X_alt, dtype=np.float32),
            np.array(X_sent, dtype=np.float32),
            np.array(y, dtype=np.float32)
        )


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("=" * 70)
    print("QUANTUM FORECASTER - FEATURE ENGINEERING MODULE")
    print("=" * 70)
    
    # Initialize engineer
    engineer = QuantumFeatureEngineer()
    
    # Test on a single ticker
    ticker = "NVDA"
    print(f"\n🧪 Testing feature engineering on {ticker}...")
    
    try:
        features = engineer.engineer_all_features(ticker, period='2y')
        
        print(f"\n📈 Feature Summary:")
        print(f"  Total features: {len(features['all_features'].columns)}")
        print(f"  Data points: {len(features['all_features'])}")
        print(f"  Date range: {features['all_features'].index[0]} to {features['all_features'].index[-1]}")
        
        # Prepare model inputs
        print(f"\n🔄 Preparing model inputs...")
        X_price, X_micro, X_alt, X_sent, y = engineer.prepare_model_inputs(features)
        
        print(f"\n✅ Model input shapes:")
        print(f"  - Price features: {X_price.shape}")
        print(f"  - Microstructure: {X_micro.shape}")
        print(f"  - Alternative data: {X_alt.shape}")
        print(f"  - Sentiment: {X_sent.shape}")
        print(f"  - Labels (14-day returns): {y.shape}")
        
        print(f"\n📊 Label statistics:")
        print(f"  - Mean return: {y.mean()*100:.2f}%")
        print(f"  - Std return: {y.std()*100:.2f}%")
        print(f"  - Min return: {y.min()*100:.2f}%")
        print(f"  - Max return: {y.max()*100:.2f}%")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
    
    print("\n" + "=" * 70)
