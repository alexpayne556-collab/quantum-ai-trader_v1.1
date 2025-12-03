"""
Safe Technical Indicators with Numeric Stability
Uses ta library with robust error handling, NaN guards, and epsilon protection.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    import ta
    HAS_TA = True
except ImportError:
    HAS_TA = False
    print("⚠️  'ta' library not found. Install with: pip install ta")


def safe_rsi(close_series: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate RSI with Wilder's smoothing and safety checks.
    
    Args:
        close_series: Price series
        window: RSI period (default 14)
    
    Returns:
        RSI series (0-100), NaN for insufficient data
    """
    if len(close_series) < window:
        return pd.Series(index=close_series.index, data=np.nan)
    
    if not HAS_TA:
        # Fallback: manual RSI with epsilon protection
        delta = close_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-10)  # Epsilon to avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        return rsi.clip(0, 100)
    
    try:
        # Use ta library (Wilder's smoothing by default)
        rsi_indicator = ta.momentum.RSIIndicator(close=close_series, window=window)
        rsi = rsi_indicator.rsi()
        return rsi.clip(0, 100)
    except Exception as e:
        print(f"⚠️  RSI calculation error: {e}")
        return pd.Series(index=close_series.index, data=np.nan)


def safe_atr(high_series: pd.Series, low_series: pd.Series, close_series: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate Average True Range with safety checks.
    
    Args:
        high_series: High prices
        low_series: Low prices
        close_series: Close prices
        window: ATR period (default 14)
    
    Returns:
        ATR series, NaN for insufficient data
    """
    if len(close_series) < window or len(high_series) < window or len(low_series) < window:
        return pd.Series(index=close_series.index, data=np.nan)
    
    if not HAS_TA:
        # Fallback: manual ATR
        tr1 = high_series - low_series
        tr2 = abs(high_series - close_series.shift())
        tr3 = abs(low_series - close_series.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr
    
    try:
        # Use ta library with additional safety checks
        # Ensure all series have the same index
        high_clean = high_series.dropna()
        low_clean = low_series.dropna()
        close_clean = close_series.dropna()
        
        if len(high_clean) < window or len(low_clean) < window or len(close_clean) < window:
            return pd.Series(index=close_series.index, data=np.nan)
        
        # Use only the common index
        common_index = high_clean.index.intersection(low_clean.index).intersection(close_clean.index)
        if len(common_index) < window:
            return pd.Series(index=close_series.index, data=np.nan)
        
        high_aligned = high_clean.loc[common_index]
        low_aligned = low_clean.loc[common_index]
        close_aligned = close_clean.loc[common_index]
        
        atr_indicator = ta.volatility.AverageTrueRange(
            high=high_aligned,
            low=low_aligned,
            close=close_aligned,
            window=window
        )
        return atr_indicator.average_true_range()
    except Exception as e:
        print(f"⚠️  ATR calculation error: {e}")
        # Fallback to manual calculation
        try:
            tr1 = high_series - low_series
            tr2 = abs(high_series - close_series.shift())
            tr3 = abs(low_series - close_series.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=window).mean()
            return atr
        except Exception as e2:
            print(f"⚠️  Fallback ATR calculation error: {e2}")
            return pd.Series(index=close_series.index, data=np.nan)


def safe_macd(close_series: pd.Series, window_slow: int = 26, window_fast: int = 12, window_sign: int = 9) -> dict:
    """
    Calculate MACD with safety checks.
    
    Args:
        close_series: Price series
        window_slow: Slow EMA period (default 26)
        window_fast: Fast EMA period (default 12)
        window_sign: Signal EMA period (default 9)
    
    Returns:
        Dict with 'macd', 'signal', 'histogram' series
    """
    min_required = window_slow + window_sign
    if len(close_series) < min_required:
        empty = pd.Series(index=close_series.index, data=np.nan)
        return {'macd': empty, 'signal': empty, 'histogram': empty}
    
    if not HAS_TA:
        # Fallback: manual MACD
        exp1 = close_series.ewm(span=window_fast, adjust=False).mean()
        exp2 = close_series.ewm(span=window_slow, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=window_sign, adjust=False).mean()
        histogram = macd - signal
        return {'macd': macd, 'signal': signal, 'histogram': histogram}
    
    try:
        # Use ta library
        macd_indicator = ta.trend.MACD(
            close=close_series,
            window_slow=window_slow,
            window_fast=window_fast,
            window_sign=window_sign
        )
        return {
            'macd': macd_indicator.macd(),
            'signal': macd_indicator.macd_signal(),
            'histogram': macd_indicator.macd_diff()
        }
    except Exception as e:
        print(f"⚠️  MACD calculation error: {e}")
        empty = pd.Series(index=close_series.index, data=np.nan)
        return {'macd': empty, 'signal': empty, 'histogram': empty}


def safe_ema(close_series: pd.Series, span: int) -> pd.Series:
    """
    Calculate EMA with safety checks.
    
    Args:
        close_series: Price series
        span: EMA period
    
    Returns:
        EMA series, NaN for insufficient data
    """
    if len(close_series) < span:
        return pd.Series(index=close_series.index, data=np.nan)
    
    try:
        return close_series.ewm(span=span, adjust=False).mean()
    except Exception as e:
        print(f"⚠️  EMA calculation error: {e}")
        return pd.Series(index=close_series.index, data=np.nan)


def calculate_atr_percent(high_series: pd.Series, low_series: pd.Series, close_series: pd.Series, window: int = 14) -> float:
    """
    Calculate ATR as percentage of current price.
    
    Args:
        high_series: High prices
        low_series: Low prices
        close_series: Close prices
        window: ATR period
    
    Returns:
        ATR percentage (e.g., 2.5 for 2.5% volatility)
    """
    atr = safe_atr(high_series, low_series, close_series, window)
    if atr is None or len(atr) == 0 or pd.isna(atr.iloc[-1]):
        return 1.0  # Default 1% volatility
    
    current_price = close_series.iloc[-1]
    if current_price <= 0:
        return 1.0
    
    atr_pct = (atr.iloc[-1] / current_price) * 100
    return float(atr_pct)


# Validation function
def validate_indicators(df: pd.DataFrame) -> dict:
    """
    Validate that required columns exist and data is sufficient.
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        Dict with validation results
    """
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        return {
            'valid': False,
            'error': f"Missing columns: {missing}",
            'min_rows': 0
        }
    
    min_rows_rsi = 14
    min_rows_macd = 26 + 9  # 35
    min_rows_atr = 14
    min_required = max(min_rows_rsi, min_rows_macd, min_rows_atr)
    
    if len(df) < min_required:
        return {
            'valid': False,
            'error': f"Insufficient data: {len(df)} rows, need {min_required}",
            'min_rows': min_required
        }
    
    return {
        'valid': True,
        'error': None,
        'min_rows': min_required
    }


if __name__ == '__main__':
    # Simple test
    print("Testing safe indicators...")
    
    # Create synthetic data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    base_price = 100
    returns = np.random.randn(100) * 2
    prices = base_price + np.cumsum(returns)
    
    df_test = pd.DataFrame({
        'Open': prices + np.random.randn(100) * 0.5,
        'High': prices + abs(np.random.randn(100) * 1.5),
        'Low': prices - abs(np.random.randn(100) * 1.5),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    # Validate
    validation = validate_indicators(df_test)
    print(f"✓ Validation: {validation}")
    
    # Calculate indicators
    rsi = safe_rsi(df_test['Close'])
    print(f"✓ RSI: Last value = {rsi.iloc[-1]:.2f}")
    
    atr = safe_atr(df_test['High'], df_test['Low'], df_test['Close'])
    print(f"✓ ATR: Last value = {atr.iloc[-1]:.2f}")
    
    atr_pct = calculate_atr_percent(df_test['High'], df_test['Low'], df_test['Close'])
    print(f"✓ ATR%: {atr_pct:.2f}%")
    
    macd_data = safe_macd(df_test['Close'])
    print(f"✓ MACD: Last value = {macd_data['macd'].iloc[-1]:.2f}")
    
    print("\n✅ All indicators calculated successfully!")
