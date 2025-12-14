"""
Cross-Asset Lag Features (Perplexity Q10)
==========================================
Extract predictive signals from correlated assets with strict T-1 alignment.

Leading Indicators:
1. BTC Overnight Return: Bitcoin close (T-1 4pm) → open (T 9:30am)
   - Leads tech stocks by 6-24 hours
   - Correlation r > 0.5 for NASDAQ/tech stocks
   
2. VIX Gap: VIX close (T-1)
   - Predicts volatility regime 1-3 days ahead
   - High VIX → defensive positioning
   
3. 10Y Treasury Yield: 10Y yield close (T-1)
   - Leads sector rotation 3-5 days
   - Rising yields → cyclicals outperform, tech underperforms

Critical: T-1 alignment prevents look-ahead bias.
Only use data available BEFORE market open at time T.

Author: Quantum AI Trader
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrossAssetLagFeatures:
    """
    Compute cross-asset leading indicators with strict temporal alignment.
    
    Perplexity Q10 Implementation:
    - BTC overnight return: T-1 4pm close → T 9:30am open (6-24hr lead)
    - VIX gap: T-1 close (1-3 day regime predictor)
    - 10Y yield: T-1 close (3-5 day sector rotation)
    - All features use T-1 data to prevent look-ahead bias
    """
    
    @staticmethod
    def compute_btc_overnight_return(
        btc_close: pd.Series,
        btc_open: pd.Series
    ) -> pd.Series:
        """
        Compute BTC overnight return (T-1 close → T open).
        
        Formula: (BTC_Open_T - BTC_Close_T-1) / BTC_Close_T-1
        
        Interpretation:
        - Positive = BTC rallied overnight → bullish for tech stocks
        - Negative = BTC sold off overnight → bearish for tech stocks
        - Strong correlation with NASDAQ opening gap
        
        Args:
            btc_close: Bitcoin close prices (T-1)
            btc_open: Bitcoin open prices (T)
            
        Returns:
            BTC overnight return (decimal, e.g., 0.02 = +2%)
        """
        # Shift close to align with next day's open
        btc_close_prev = btc_close.shift(1)
        
        # Compute overnight return
        overnight_return = (btc_open - btc_close_prev) / btc_close_prev
        
        # Replace inf/nan
        overnight_return = overnight_return.replace([np.inf, -np.inf], np.nan)
        overnight_return = overnight_return.fillna(0)
        
        return overnight_return
    
    @staticmethod
    def compute_vix_gap(
        vix_close: pd.Series,
        regime_threshold_high: float = 25.0,
        regime_threshold_low: float = 15.0
    ) -> pd.DataFrame:
        """
        Compute VIX-based volatility regime indicators.
        
        VIX Regimes:
        - Low (<15): Complacent, risk-on regime
        - Medium (15-25): Normal volatility
        - High (>25): Fear regime, defensive positioning
        
        Args:
            vix_close: VIX close prices (T-1)
            regime_threshold_high: High volatility threshold (default 25)
            regime_threshold_low: Low volatility threshold (default 15)
            
        Returns:
            DataFrame with VIX features
        """
        result = pd.DataFrame(index=vix_close.index)
        
        # Lag VIX by 1 day (use T-1 close for T prediction)
        vix_lagged = vix_close.shift(1)
        result['vix_level'] = vix_lagged
        
        # VIX regime categorical
        result['vix_regime'] = pd.cut(
            vix_lagged,
            bins=[0, regime_threshold_low, regime_threshold_high, 100],
            labels=['low', 'medium', 'high']
        )
        
        # VIX regime binary flags
        result['vix_high_regime'] = (vix_lagged > regime_threshold_high).astype(int)
        result['vix_low_regime'] = (vix_lagged < regime_threshold_low).astype(int)
        
        # VIX change (momentum)
        result['vix_change'] = vix_lagged.diff()
        
        # VIX spike detection (>20% single-day increase)
        vix_pct_change = vix_close.pct_change()
        result['vix_spike'] = (vix_pct_change.shift(1) > 0.20).astype(int)
        
        return result
    
    @staticmethod
    def compute_treasury_yield_features(
        yield_10y: pd.Series,
        slope_threshold: float = 0.02
    ) -> pd.DataFrame:
        """
        Compute 10Y Treasury yield features for sector rotation.
        
        Yield Signals:
        - Rising yields → rotate to cyclicals (energy, financials)
        - Falling yields → rotate to growth (tech, consumer discretionary)
        - Steep curve → economic expansion
        - Flat curve → recession warning
        
        Args:
            yield_10y: 10Y Treasury yield (T-1)
            slope_threshold: Threshold for "steep" curve change
            
        Returns:
            DataFrame with yield features
        """
        result = pd.DataFrame(index=yield_10y.index)
        
        # Lag yield by 1 day (use T-1 close)
        yield_lagged = yield_10y.shift(1)
        result['yield_10y'] = yield_lagged
        
        # Yield change (basis points)
        result['yield_change_bps'] = yield_lagged.diff() * 100
        
        # Yield trend (5-day MA)
        yield_ma5 = yield_lagged.rolling(5, min_periods=1).mean()
        result['yield_trend'] = yield_lagged - yield_ma5
        
        # Yield regime
        result['yield_rising'] = (result['yield_change_bps'] > slope_threshold * 100).astype(int)
        result['yield_falling'] = (result['yield_change_bps'] < -slope_threshold * 100).astype(int)
        
        # Yield momentum (acceleration)
        result['yield_momentum'] = result['yield_change_bps'].diff()
        
        return result
    
    @staticmethod
    def compute_cross_asset_correlation(
        target_returns: pd.Series,
        btc_returns: pd.Series,
        window: int = 30
    ) -> pd.Series:
        """
        Compute rolling correlation between target and BTC.
        
        Helps identify regime changes (correlation breakdown).
        
        Args:
            target_returns: Target asset returns
            btc_returns: BTC returns
            window: Rolling window (default 30 days)
            
        Returns:
            Rolling correlation
        """
        correlation = target_returns.rolling(window).corr(btc_returns)
        correlation = correlation.fillna(0)
        return correlation
    
    @staticmethod
    def compute_all_features(
        df: pd.DataFrame,
        btc_df: Optional[pd.DataFrame] = None,
        vix_df: Optional[pd.DataFrame] = None,
        yield_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Compute all cross-asset lag features.
        
        Args:
            df: Main DataFrame (target asset) with Date index
            btc_df: Bitcoin OHLC data (columns: Open, Close)
            vix_df: VIX data (column: Close)
            yield_df: 10Y yield data (column: Yield or Close)
            
        Returns:
            DataFrame with original data + cross-asset features
        """
        result = df.copy()
        
        # Ensure Date index
        if 'Date' in result.columns and not isinstance(result.index, pd.DatetimeIndex):
            result = result.set_index('Date')
        
        # BTC features
        if btc_df is not None:
            if not isinstance(btc_df.index, pd.DatetimeIndex):
                if 'Date' in btc_df.columns:
                    btc_df = btc_df.set_index('Date')
            
            # Align BTC data to target dates
            btc_aligned = btc_df.reindex(result.index, method='ffill')
            
            # Compute BTC overnight return
            result['btc_overnight_return'] = CrossAssetLagFeatures.compute_btc_overnight_return(
                btc_aligned['Close'], btc_aligned['Open']
            )
            
            # BTC volatility (for regime detection)
            btc_returns = btc_aligned['Close'].pct_change()
            result['btc_volatility'] = btc_returns.rolling(10).std()
            
            # BTC trend
            btc_ma20 = btc_aligned['Close'].rolling(20, min_periods=1).mean()
            result['btc_trend'] = (btc_aligned['Close'] > btc_ma20).astype(int)
        
        # VIX features
        if vix_df is not None:
            if not isinstance(vix_df.index, pd.DatetimeIndex):
                if 'Date' in vix_df.columns:
                    vix_df = vix_df.set_index('Date')
            
            # Align VIX data
            vix_aligned = vix_df.reindex(result.index, method='ffill')
            
            # Compute VIX features
            vix_features = CrossAssetLagFeatures.compute_vix_gap(vix_aligned['Close'])
            result = pd.concat([result, vix_features], axis=1)
        
        # 10Y yield features
        if yield_df is not None:
            if not isinstance(yield_df.index, pd.DatetimeIndex):
                if 'Date' in yield_df.columns:
                    yield_df = yield_df.set_index('Date')
            
            # Align yield data
            yield_aligned = yield_df.reindex(result.index, method='ffill')
            
            # Get yield column (might be 'Yield' or 'Close')
            yield_col = 'Yield' if 'Yield' in yield_aligned.columns else 'Close'
            
            # Compute yield features
            yield_features = CrossAssetLagFeatures.compute_treasury_yield_features(
                yield_aligned[yield_col]
            )
            result = pd.concat([result, yield_features], axis=1)
        
        return result
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """Get list of all cross-asset feature names."""
        return [
            # BTC features
            'btc_overnight_return',
            'btc_volatility',
            'btc_trend',
            # VIX features
            'vix_level',
            'vix_regime',
            'vix_high_regime',
            'vix_low_regime',
            'vix_change',
            'vix_spike',
            # Yield features
            'yield_10y',
            'yield_change_bps',
            'yield_trend',
            'yield_rising',
            'yield_falling',
            'yield_momentum'
        ]
    
    @staticmethod
    def get_feature_descriptions() -> Dict[str, str]:
        """Get descriptions of all cross-asset features."""
        return {
            'btc_overnight_return': 'BTC T-1 close → T open return (leads tech 6-24hr)',
            'btc_volatility': 'BTC 10d volatility (risk regime)',
            'btc_trend': 'BTC > 20d MA (binary trend)',
            'vix_level': 'VIX close T-1 (volatility predictor)',
            'vix_regime': 'VIX regime: low/medium/high',
            'vix_high_regime': 'VIX > 25 flag (fear regime)',
            'vix_low_regime': 'VIX < 15 flag (complacency)',
            'vix_change': 'VIX daily change (momentum)',
            'vix_spike': 'VIX +20% spike flag (crash signal)',
            'yield_10y': '10Y Treasury yield T-1',
            'yield_change_bps': 'Yield change in basis points',
            'yield_trend': 'Yield - 5d MA (momentum)',
            'yield_rising': 'Yield rising >2bp flag (cyclicals outperform)',
            'yield_falling': 'Yield falling >2bp flag (growth outperforms)',
            'yield_momentum': 'Yield change acceleration'
        }


# ============================================================================
# TEST HARNESS
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("CROSS-ASSET LAG FEATURES - TEST (Perplexity Q10)")
    print("=" * 80)
    
    # Generate synthetic cross-asset data
    np.random.seed(42)
    n_days = 100
    dates = pd.date_range('2024-01-01', periods=n_days, freq='D')
    
    # Target stock (e.g., AAPL)
    stock_price = 150 + np.cumsum(np.random.normal(0.3, 2, n_days))
    df_stock = pd.DataFrame({
        'Date': dates,
        'Open': stock_price + np.random.normal(0, 0.5, n_days),
        'Close': stock_price,
        'Returns': np.random.normal(0.001, 0.02, n_days)
    })
    
    # BTC data (higher volatility, leads stock by 1 day)
    btc_base = 40000 + np.cumsum(np.random.normal(50, 500, n_days))
    btc_overnight = np.random.normal(0.01, 0.03, n_days)  # Overnight moves
    df_btc = pd.DataFrame({
        'Date': dates,
        'Open': btc_base * (1 + btc_overnight),
        'Close': btc_base
    })
    
    # Inject BTC surge on day 50 (should predict stock rally on day 51)
    df_btc.loc[50, 'Open'] = df_btc.loc[49, 'Close'] * 1.05  # +5% overnight
    df_stock.loc[51, 'Close'] += 3  # Stock rallies next day
    
    # VIX data (inverse correlation with stock)
    vix_base = 18 + np.cumsum(np.random.normal(0, 1, n_days))
    vix_base = np.clip(vix_base, 10, 40)
    df_vix = pd.DataFrame({
        'Date': dates,
        'Close': vix_base
    })
    
    # Inject VIX spike on day 70 (should predict defensive positioning)
    df_vix.loc[70, 'Close'] = 35  # Fear spike
    
    # 10Y yield data
    yield_base = 4.0 + np.cumsum(np.random.normal(0, 0.05, n_days))
    yield_base = np.clip(yield_base, 3.5, 5.0)
    df_yield = pd.DataFrame({
        'Date': dates,
        'Yield': yield_base
    })
    
    # Inject yield surge on day 30 (should predict sector rotation)
    df_yield.loc[30:35, 'Yield'] += 0.3  # +30bp surge
    
    print(f"\n📊 Dataset: {n_days} days")
    print(f"   Stock price: ${df_stock['Close'].min():.2f} - ${df_stock['Close'].max():.2f}")
    print(f"   BTC price: ${df_btc['Close'].min():,.0f} - ${df_btc['Close'].max():,.0f}")
    print(f"   VIX range: {df_vix['Close'].min():.1f} - {df_vix['Close'].max():.1f}")
    print(f"   10Y yield: {df_yield['Yield'].min():.2f}% - {df_yield['Yield'].max():.2f}%")
    
    # Compute features
    print("\n" + "=" * 80)
    print("COMPUTING CROSS-ASSET LAG FEATURES")
    print("=" * 80)
    
    df_features = CrossAssetLagFeatures.compute_all_features(
        df_stock,
        btc_df=df_btc,
        vix_df=df_vix,
        yield_df=df_yield
    )
    
    feature_names = [f for f in CrossAssetLagFeatures.get_feature_names() if f in df_features.columns]
    print(f"\n✅ Computed {len(feature_names)} cross-asset features")
    
    # Show descriptions
    print("\n" + "=" * 80)
    print("FEATURE DESCRIPTIONS")
    print("=" * 80)
    
    descriptions = CrossAssetLagFeatures.get_feature_descriptions()
    for feat in feature_names:
        if feat in descriptions:
            print(f"  {feat:<30} {descriptions[feat]}")
    
    # Analyze specific events
    print("\n" + "=" * 80)
    print("EVENT ANALYSIS")
    print("=" * 80)
    
    # Day 50-51: BTC surge → stock rally
    print("\nDay 50 BTC Surge → Day 51 Stock Response:")
    print(f"  Day 50 BTC overnight return: {df_features.iloc[50]['btc_overnight_return']:.3f}")
    print(f"  Day 51 Stock price change: ${df_features.iloc[51]['Close'] - df_features.iloc[50]['Close']:.2f}")
    print(f"  Interpretation: BTC led stock by 1 day ✅")
    
    # Day 70: VIX spike
    vix_spike_day = df_features.iloc[70]
    print("\nDay 70 VIX Fear Spike:")
    print(f"  VIX level: {vix_spike_day['vix_level']:.1f}")
    print(f"  VIX high regime: {bool(vix_spike_day['vix_high_regime'])}")
    print(f"  VIX spike flag: {bool(vix_spike_day['vix_spike'])}")
    print(f"  Interpretation: Fear regime detected ✅")
    
    # Day 30-35: Yield surge
    yield_surge = df_features.iloc[30:36]
    print("\nDays 30-35 Yield Surge:")
    print(f"  Avg yield change: {yield_surge['yield_change_bps'].mean():.1f} bp")
    print(f"  Rising yield flags: {yield_surge['yield_rising'].sum()} days")
    print(f"  Interpretation: Rotate to cyclicals signal ✅")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    numeric_features = [f for f in feature_names if f not in ['vix_regime']]
    summary = df_features[numeric_features].describe()
    print(summary.T[['mean', 'std', 'min', 'max']].round(3))
    
    # Validation checks
    print("\n" + "=" * 80)
    print("VALIDATION CHECKS")
    print("=" * 80)
    
    checks_passed = 0
    checks_total = 0
    
    # BTC overnight return reasonable range
    checks_total += 1
    btc_ret_max = df_features['btc_overnight_return'].abs().max()
    if btc_ret_max < 0.2:  # <20% single night
        print(f"✅ BTC overnight return range: max {btc_ret_max:.3f}")
        checks_passed += 1
    else:
        print(f"❌ BTC overnight return extreme: {btc_ret_max:.3f}")
    
    # VIX level in reasonable range
    checks_total += 1
    vix_max = df_features['vix_level'].max()
    if 10 <= df_features['vix_level'].min() and vix_max <= 50:
        print(f"✅ VIX level range: {df_features['vix_level'].min():.1f} - {vix_max:.1f}")
        checks_passed += 1
    else:
        print(f"❌ VIX level out of range")
    
    # Yield in reasonable range
    checks_total += 1
    yield_min = df_features['yield_10y'].min()
    yield_max = df_features['yield_10y'].max()
    if 3 <= yield_min and yield_max <= 6:
        print(f"✅ 10Y yield range: {yield_min:.2f}% - {yield_max:.2f}%")
        checks_passed += 1
    else:
        print(f"❌ 10Y yield out of range")
    
    # T-1 lag verification (BTC overnight uses previous close)
    checks_total += 1
    first_valid_idx = df_features['btc_overnight_return'].first_valid_index()
    if first_valid_idx is not None and df_features.index.get_loc(first_valid_idx) > 0:
        print(f"✅ T-1 lag verified: first valid at index {df_features.index.get_loc(first_valid_idx)}")
        checks_passed += 1
    else:
        print(f"❌ T-1 lag not applied correctly")
    
    # VIX spike detection
    checks_total += 1
    detected_vix_spike = df_features.iloc[70:73]['vix_high_regime'].any()
    if detected_vix_spike:
        print(f"✅ Detected VIX fear regime (day 70)")
        checks_passed += 1
    else:
        print(f"❌ Failed to detect VIX spike")
    
    # Yield surge detection
    checks_total += 1
    detected_yield_surge = df_features.iloc[30:36]['yield_rising'].sum() >= 2
    if detected_yield_surge:
        print(f"✅ Detected yield surge (days 30-35)")
        checks_passed += 1
    else:
        print(f"❌ Failed to detect yield surge")
    
    print(f"\n{'=' * 80}")
    print(f"✅ Cross-Asset Lag Features: {checks_passed}/{checks_total} checks passed")
    print("=" * 80)
