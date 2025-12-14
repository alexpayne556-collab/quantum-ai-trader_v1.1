"""
Market Microstructure Features (Perplexity Q8)
===============================================
Extract institutional flow and order book proxies from free OHLCV data.
No Level 2 data required - all features computed from High/Low/Close/Volume.

Features:
1. Spread Proxy: (High - Low) / Close
   - Corwin-Schultz simplified estimator
   - Wider spreads = less liquidity / higher institutional activity
   
2. Order Flow (CLV): ((Close - Low) - (High - Close)) / (High - Low)
   - Close Location Value (-1 to +1)
   - +1 = close at high (strong buying pressure)
   - -1 = close at low (strong selling pressure)
   
3. Institutional Activity: Volume / abs(Close - Open)
   - High volume + small candle body = dark pool activity
   - Large volume with small price movement = institutional accumulation

Author: Quantum AI Trader
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MicrostructureFeatures:
    """
    Compute market microstructure proxies from OHLCV data.
    
    All features use free yfinance data only.
    No tick data or Level 2 order book required.
    """
    
    @staticmethod
    def compute_spread_proxy(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> pd.Series:
        """
        Spread proxy using Corwin-Schultz simplified estimator.
        
        Formula: (High - Low) / Close
        
        Interpretation:
        - Higher values = wider bid-ask spread
        - Wider spreads typically seen during:
          * Low liquidity periods (pre-market, after-hours)
          * High volatility / uncertainty
          * Institutional block trades (temporarily widens spread)
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            
        Returns:
            Spread proxy (0 to ~0.1 typically, higher = wider spread)
        """
        spread = (high - low) / close
        
        # Replace inf/nan with 0
        spread = spread.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Clip extreme values (>20% spread is unrealistic for liquid stocks)
        spread = spread.clip(upper=0.2)
        
        return spread
    
    @staticmethod
    def compute_order_flow_clv(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> pd.Series:
        """
        Close Location Value (CLV) - order flow pressure indicator.
        
        Formula: ((Close - Low) - (High - Close)) / (High - Low)
        
        Interpretation:
        - +1.0 = close at high (strong buying pressure)
        -  0.0 = close at midpoint (neutral)
        - -1.0 = close at low (strong selling pressure)
        
        When combined with volume, shows buying/selling pressure strength.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            
        Returns:
            CLV values (-1 to +1)
        """
        # Avoid division by zero
        high_low_range = high - low
        high_low_range = high_low_range.replace(0, np.nan)
        
        clv = ((close - low) - (high - close)) / high_low_range
        
        # Fill NaN (doji candles with no range) with 0 (neutral)
        clv = clv.fillna(0)
        
        # Clip to valid range
        clv = clv.clip(-1, 1)
        
        return clv
    
    @staticmethod
    def compute_institutional_activity(
        open_price: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """
        Institutional activity proxy: Volume / abs(Close - Open).
        
        Formula: Volume / abs(Close - Open)
        
        Interpretation:
        - High values = large volume with small price movement
        - Indicates institutional accumulation/distribution via dark pools
        - Institutions trade large size without moving price (stealth mode)
        
        Example:
        - Volume = 10M shares, Close-Open = $0.10 → activity = 100M
        - Volume = 10M shares, Close-Open = $2.00 → activity = 5M
        
        Args:
            open_price: Open prices
            close: Close prices
            volume: Volume
            
        Returns:
            Institutional activity proxy (higher = more dark pool activity)
        """
        # Compute candle body size
        body_size = (close - open_price).abs()
        
        # Avoid division by zero (use small epsilon)
        body_size = body_size.replace(0, 1e-6)
        
        # Compute activity ratio
        activity = volume / body_size
        
        # Replace inf with max value
        max_activity = activity.replace([np.inf, -np.inf], np.nan).max()
        if pd.isna(max_activity):
            max_activity = 0
        
        activity = activity.replace([np.inf, -np.inf], max_activity * 1.1)
        
        # Normalize by rolling median (makes cross-ticker comparable)
        activity_normalized = activity / activity.rolling(window=20, min_periods=5).median()
        activity_normalized = activity_normalized.fillna(1.0)
        
        return activity_normalized
    
    @staticmethod
    def compute_volume_weighted_clv(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """
        Volume-weighted CLV (combines order flow direction with size).
        
        Formula: CLV * Volume
        
        Interpretation:
        - Positive = buying pressure with high volume
        - Negative = selling pressure with high volume
        - Large absolute values = strong institutional directional flow
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume
            
        Returns:
            Volume-weighted CLV
        """
        clv = MicrostructureFeatures.compute_order_flow_clv(high, low, close)
        vw_clv = clv * volume
        
        # Normalize by rolling std (makes cross-ticker comparable)
        vw_clv_normalized = vw_clv / vw_clv.rolling(window=20, min_periods=5).std()
        vw_clv_normalized = vw_clv_normalized.fillna(0)
        
        return vw_clv_normalized
    
    @staticmethod
    def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all microstructure features from OHLCV DataFrame.
        
        Args:
            df: DataFrame with columns [Open, High, Low, Close, Volume]
            
        Returns:
            DataFrame with original data + microstructure features
        """
        result = df.copy()
        
        # Validate required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Compute features
        result['spread_proxy'] = MicrostructureFeatures.compute_spread_proxy(
            df['High'], df['Low'], df['Close']
        )
        
        result['order_flow_clv'] = MicrostructureFeatures.compute_order_flow_clv(
            df['High'], df['Low'], df['Close']
        )
        
        result['institutional_activity'] = MicrostructureFeatures.compute_institutional_activity(
            df['Open'], df['Close'], df['Volume']
        )
        
        result['volume_weighted_clv'] = MicrostructureFeatures.compute_volume_weighted_clv(
            df['High'], df['Low'], df['Close'], df['Volume']
        )
        
        # Add rolling statistics (capture microstructure regime changes)
        result['spread_ma5'] = result['spread_proxy'].rolling(5).mean()
        result['spread_volatility'] = result['spread_proxy'].rolling(10).std()
        
        result['clv_ma5'] = result['order_flow_clv'].rolling(5).mean()
        result['clv_trend'] = result['order_flow_clv'] - result['clv_ma5']
        
        result['institutional_ma10'] = result['institutional_activity'].rolling(10).mean()
        result['institutional_spike'] = (
            result['institutional_activity'] > result['institutional_ma10'] * 1.5
        ).astype(int)
        
        return result
    
    @staticmethod
    def get_feature_names() -> list:
        """Get list of all microstructure feature names."""
        return [
            'spread_proxy',
            'order_flow_clv',
            'institutional_activity',
            'volume_weighted_clv',
            'spread_ma5',
            'spread_volatility',
            'clv_ma5',
            'clv_trend',
            'institutional_ma10',
            'institutional_spike'
        ]
    
    @staticmethod
    def get_feature_descriptions() -> Dict[str, str]:
        """Get descriptions of all microstructure features."""
        return {
            'spread_proxy': 'Bid-ask spread proxy (High-Low)/Close, wider = less liquid',
            'order_flow_clv': 'Close Location Value, +1=buying pressure, -1=selling',
            'institutional_activity': 'Volume/BodySize, high=dark pool accumulation',
            'volume_weighted_clv': 'CLV*Volume, large values = strong institutional flow',
            'spread_ma5': '5-day MA of spread (regime filter)',
            'spread_volatility': '10-day std of spread (volatility regime)',
            'clv_ma5': '5-day MA of order flow',
            'clv_trend': 'CLV deviation from MA (order flow acceleration)',
            'institutional_ma10': '10-day MA of institutional activity',
            'institutional_spike': 'Binary flag: activity > 1.5x MA (surge detection)'
        }


# ============================================================================
# TEST HARNESS
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MICROSTRUCTURE FEATURES - TEST (Perplexity Q8)")
    print("=" * 80)
    
    # Generate synthetic OHLCV data
    np.random.seed(42)
    n_days = 100
    
    # Simulate price with trend + noise
    base_price = 100
    trend = np.linspace(0, 10, n_days)
    noise = np.random.normal(0, 2, n_days)
    close_prices = base_price + trend + noise
    
    # Generate OHLC with realistic spreads
    df = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=n_days, freq='D'),
        'Open': close_prices + np.random.normal(0, 0.5, n_days),
        'High': close_prices + np.random.uniform(0.5, 2.0, n_days),
        'Low': close_prices - np.random.uniform(0.5, 2.0, n_days),
        'Close': close_prices,
        'Volume': np.random.lognormal(15, 0.5, n_days).astype(int)
    })
    
    # Inject institutional activity patterns (day 50-55: dark pool accumulation)
    # High volume + small body = institutional activity
    for i in range(50, 56):
        df.loc[i, 'Volume'] *= 3.0  # 3x volume
        df.loc[i, 'Open'] = df.loc[i, 'Close'] - 0.2  # Small body
    
    # Inject strong buying pressure (day 70-75: CLV spike)
    for i in range(70, 76):
        df.loc[i, 'Close'] = df.loc[i, 'High'] - 0.1  # Close near high
    
    print(f"\n📊 Dataset: {len(df)} days of OHLCV data")
    print(f"   Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    print(f"   Volume range: {df['Volume'].min():,.0f} - {df['Volume'].max():,.0f}")
    
    # Compute features
    print("\n" + "=" * 80)
    print("COMPUTING MICROSTRUCTURE FEATURES")
    print("=" * 80)
    
    df_features = MicrostructureFeatures.compute_all_features(df)
    
    feature_names = MicrostructureFeatures.get_feature_names()
    print(f"\n✅ Computed {len(feature_names)} microstructure features")
    
    # Show feature descriptions
    print("\n" + "=" * 80)
    print("FEATURE DESCRIPTIONS")
    print("=" * 80)
    
    descriptions = MicrostructureFeatures.get_feature_descriptions()
    for feat, desc in descriptions.items():
        print(f"  {feat:<30} {desc}")
    
    # Analyze specific periods
    print("\n" + "=" * 80)
    print("INSTITUTIONAL ACTIVITY ANALYSIS")
    print("=" * 80)
    
    # Day 50-55: Injected dark pool activity
    dark_pool_period = df_features.loc[48:57]
    print("\nDays 50-55 (Injected Dark Pool Accumulation):")
    print(f"  Avg Institutional Activity: {dark_pool_period['institutional_activity'].mean():.2f}")
    print(f"  Institutional Spikes: {dark_pool_period['institutional_spike'].sum()} days")
    print(f"  Avg Spread: {dark_pool_period['spread_proxy'].mean():.4f}")
    
    # Baseline period for comparison
    baseline_period = df_features.loc[20:30]
    print("\nDays 20-30 (Baseline):")
    print(f"  Avg Institutional Activity: {baseline_period['institutional_activity'].mean():.2f}")
    print(f"  Institutional Spikes: {baseline_period['institutional_spike'].sum()} days")
    print(f"  Avg Spread: {baseline_period['spread_proxy'].mean():.4f}")
    
    # Order flow analysis
    print("\n" + "=" * 80)
    print("ORDER FLOW ANALYSIS")
    print("=" * 80)
    
    # Day 70-75: Injected buying pressure
    buying_period = df_features.loc[68:77]
    print("\nDays 70-75 (Injected Buying Pressure):")
    print(f"  Avg CLV: {buying_period['order_flow_clv'].mean():.3f} (target: +1.0)")
    print(f"  Avg Volume-Weighted CLV: {buying_period['volume_weighted_clv'].mean():.2f}")
    print(f"  CLV Trend: {buying_period['clv_trend'].mean():.3f}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    summary = df_features[feature_names].describe()
    print(summary.T[['mean', 'std', 'min', 'max']].round(3))
    
    # Validate feature ranges
    print("\n" + "=" * 80)
    print("VALIDATION CHECKS")
    print("=" * 80)
    
    checks_passed = 0
    checks_total = 0
    
    # CLV should be in [-1, 1]
    checks_total += 1
    clv_min = df_features['order_flow_clv'].min()
    clv_max = df_features['order_flow_clv'].max()
    if -1 <= clv_min and clv_max <= 1:
        print(f"✅ CLV range check: [{clv_min:.3f}, {clv_max:.3f}] (valid: [-1, 1])")
        checks_passed += 1
    else:
        print(f"❌ CLV range check: [{clv_min:.3f}, {clv_max:.3f}] (expected: [-1, 1])")
    
    # Spread should be positive
    checks_total += 1
    spread_min = df_features['spread_proxy'].min()
    if spread_min >= 0:
        print(f"✅ Spread proxy non-negative: min={spread_min:.4f}")
        checks_passed += 1
    else:
        print(f"❌ Spread proxy has negative values: min={spread_min:.4f}")
    
    # Institutional activity should be positive
    checks_total += 1
    inst_min = df_features['institutional_activity'].min()
    if inst_min >= 0:
        print(f"✅ Institutional activity non-negative: min={inst_min:.4f}")
        checks_passed += 1
    else:
        print(f"❌ Institutional activity has negative values: min={inst_min:.4f}")
    
    # No NaN values
    checks_total += 1
    nan_count = df_features[feature_names].isnull().sum().sum()
    if nan_count == 0:
        print(f"✅ No NaN values in features")
        checks_passed += 1
    else:
        print(f"❌ Found {nan_count} NaN values in features")
    
    # Detection accuracy for injected patterns
    checks_total += 1
    detected_dark_pool = df_features.loc[50:55, 'institutional_spike'].sum() >= 3
    if detected_dark_pool:
        print(f"✅ Detected dark pool accumulation (days 50-55)")
        checks_passed += 1
    else:
        print(f"❌ Failed to detect dark pool accumulation (days 50-55)")
    
    checks_total += 1
    detected_buying = df_features.loc[70:75, 'order_flow_clv'].mean() > 0.5
    if detected_buying:
        print(f"✅ Detected strong buying pressure (days 70-75)")
        checks_passed += 1
    else:
        print(f"❌ Failed to detect buying pressure (days 70-75)")
    
    print(f"\n{'=' * 80}")
    print(f"✅ Microstructure Features: {checks_passed}/{checks_total} checks passed")
    print("=" * 80)
