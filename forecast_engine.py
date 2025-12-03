"""
ATR-Based Forecast Engine with Realistic Volatility
Generates 24-day price paths using model signals, confidence, and market volatility.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from safe_indicators import safe_atr, calculate_atr_percent


class ForecastEngine:
    """
    Generate realistic price forecasts using ATR-based volatility.
    
    Key Features:
    - Direction from model prediction (bullish/bearish/neutral)
    - Move size scaled by confidence and ATR
    - Random shocks for realism
    - Decay to neutral after 10 days (prevents runaway forecasts)
    """
    
    def __init__(self, decay_start_day: int = 10, max_forecast_days: int = 24):
        """
        Initialize forecast engine.
        
        Args:
            decay_start_day: Day when forecast starts decaying to neutral (default 10)
            max_forecast_days: Maximum forecast horizon (default 24)
        """
        self.decay_start_day = decay_start_day
        self.max_forecast_days = max_forecast_days
    
    def generate_forecast(
        self,
        df: pd.DataFrame,
        model,
        feature_engineer,
        ticker: str = 'UNKNOWN'
    ) -> pd.DataFrame:
        """
        Generate 24-day forecast with ATR-based volatility.
        
        Args:
            df: Historical OHLCV data (normalized columns)
            model: Trained model with predict() and predict_proba()
            feature_engineer: Feature engineering instance
            ticker: Ticker symbol (for logging)
        
        Returns:
            DataFrame with forecast (date, price, confidence, signal, atr_pct)
        """
        if len(df) < 60:
            print(f"⚠️  {ticker}: Insufficient data for forecast ({len(df)} rows)")
            return pd.DataFrame()
        
        # Calculate current ATR
        try:
            atr_series = safe_atr(df['High'], df['Low'], df['Close'], window=14)
            if atr_series is None or len(atr_series) == 0 or pd.isna(atr_series.iloc[-1]):
                atr_current = df['Close'].iloc[-1] * 0.01  # Default 1% volatility
            else:
                atr_current = float(atr_series.iloc[-1])
        except Exception as e:
            print(f"⚠️  ATR calculation error: {e}")
            atr_current = df['Close'].iloc[-1] * 0.01  # Default 1% volatility
        
        atr_pct = (atr_current / df['Close'].iloc[-1]) * 100
        
        forecasts = []
        current_df = df.copy()
        last_date = df.index[-1]
        
        for day in range(1, self.max_forecast_days + 1):
            try:
                # Engineer features for current state
                features_df = feature_engineer.engineer(current_df.copy())
                if features_df is None or len(features_df) == 0:
                    break
                
                latest_features = features_df.iloc[-1:].values
                if len(latest_features) == 0:
                    break
                
                # Ensure latest_features is a 2D array
                if isinstance(latest_features, list):
                    latest_features = np.array(latest_features)
                if latest_features.ndim == 1:
                    latest_features = latest_features.reshape(1, -1)
                
                # Get model prediction
                try:
                    pred_result = model.predict(latest_features)
                    pred = pred_result[0] if isinstance(pred_result, (list, np.ndarray)) else pred_result
                    
                    proba_result = model.predict_proba(latest_features)
                    if isinstance(proba_result, list):
                        proba_result = np.array(proba_result)
                    conf = float(proba_result[0].max())
                except Exception as e:
                    print(f"  ⚠️  Model prediction error: {e}")
                    break
                
                # Map prediction to direction
                if pred == 2:  # BULLISH
                    direction = 1
                    signal = 'BULLISH'
                elif pred == 0:  # BEARISH
                    direction = -1
                    signal = 'BEARISH'
                else:  # NEUTRAL
                    direction = 0
                    signal = 'NEUTRAL'
                
                # Calculate decay factor (gradually blend to neutral after day 10)
                if day <= self.decay_start_day:
                    decay_factor = 1.0
                else:
                    # Linear decay from 1.0 to 0.0 over remaining days
                    days_past_decay = day - self.decay_start_day
                    max_decay_days = self.max_forecast_days - self.decay_start_day
                    decay_factor = 1.0 - (days_past_decay / max_decay_days)
                    decay_factor = max(0.0, decay_factor)  # Clamp to [0, 1]
                
                # Calculate base move (direction * confidence * ATR * decay)
                base_move = direction * conf * atr_current * 0.5 * decay_factor
                
                # Add random shock (scaled by ATR)
                random_shock = np.random.normal(0, atr_current * 0.2)
                
                # Total daily return
                daily_return = base_move + random_shock
                
                # Apply boundary checks to prevent unrealistic moves
                # Limit to smaller of: ATR*2 OR 5% of price
                last_close = float(current_df['Close'].iloc[-1])
                max_move = min(atr_current * 2, last_close * 0.05)
                daily_return = np.clip(daily_return, -max_move, max_move)
                
                # Calculate forecast price with Kalman-style smoothing
                forecast_price = last_close + daily_return
                
                # Additional boundary: never more than 20% daily move
                forecast_price = np.clip(
                    forecast_price,
                    last_close * 0.80,  # Max 20% down
                    last_close * 1.20   # Max 20% up
                )
                
                # Ensure price stays positive
                forecast_price = max(forecast_price, last_close * 0.5)
                
                forecast_date = last_date + timedelta(days=day)
                
                forecasts.append({
                    'date': forecast_date,
                    'price': forecast_price,
                    'confidence': float(conf),
                    'signal': signal,
                    'atr_pct': atr_pct,
                    'decay_factor': decay_factor,
                    'day': day
                })
                
                # Update current_df for next iteration
                new_row = pd.DataFrame({
                    'Open': [forecast_price],
                    'High': [forecast_price * (1 + atr_pct/200)],  # +half ATR
                    'Low': [forecast_price * (1 - atr_pct/200)],   # -half ATR
                    'Close': [forecast_price],
                    'Volume': [float(current_df['Volume'].mean())]
                }, index=[forecast_date])
                
                current_df = pd.concat([current_df, new_row])
                
                # Update ATR for next step (recalculate with new data)
                try:
                    atr_series = safe_atr(
                        current_df['High'],
                        current_df['Low'],
                        current_df['Close'],
                        window=14
                    )
                    if atr_series is None or len(atr_series) == 0 or pd.isna(atr_series.iloc[-1]):
                        atr_current = forecast_price * 0.01
                    else:
                        atr_current = float(atr_series.iloc[-1])
                except Exception as e:
                    print(f"  ⚠️  ATR update error: {e}")
                    atr_current = forecast_price * 0.01
                
            except Exception as e:
                print(f"  ⚠️  {ticker} forecast day {day} error: {e}")
                break
        
        return pd.DataFrame(forecasts) if forecasts else pd.DataFrame()
    
    def simple_trend_projection(self, df: pd.DataFrame, days: int = 24) -> pd.DataFrame:
        """
        Simple linear trend projection (fallback when no model available).
        
        Args:
            df: Historical OHLCV data
            days: Number of days to forecast
        
        Returns:
            DataFrame with simple trend forecast
        """
        if len(df) < 20:
            return pd.DataFrame()
        
        last_close = float(df['Close'].iloc[-1])
        last_date = df.index[-1]
        
        # Calculate trend from last 20 days
        trend = (df['Close'].iloc[-1] - df['Close'].iloc[-20]) / 20
        trend = float(trend)
        
        # Get ATR for volatility
        atr = safe_atr(df['High'], df['Low'], df['Close'], window=14).iloc[-1]
        if pd.isna(atr):
            atr = last_close * 0.01
        
        forecasts = []
        for day in range(1, days + 1):
            # Simple trend + random noise
            forecast_price = last_close + (trend * day) + np.random.normal(0, atr * 0.2)
            forecast_date = last_date + timedelta(days=day)
            
            forecasts.append({
                'date': forecast_date,
                'price': max(forecast_price, last_close * 0.5),  # Stay positive
                'confidence': 0.5,
                'signal': 'NEUTRAL',
                'atr_pct': (atr / last_close) * 100,
                'decay_factor': 1.0,
                'day': day
            })
        
        return pd.DataFrame(forecasts)


def map_confidence_to_move(confidence: float, atr: float, prediction: int) -> float:
    """
    Map model confidence to expected price move.
    
    Args:
        confidence: Model confidence (0-1)
        atr: Current ATR value
        prediction: 0=BEARISH, 1=NEUTRAL, 2=BULLISH
    
    Returns:
        Expected move in price units
    """
    # Direction
    if prediction == 2:  # BULLISH
        direction = 1
    elif prediction == 0:  # BEARISH
        direction = -1
    else:  # NEUTRAL
        direction = 0
    
    # Scale move by confidence
    # High confidence (>0.7): 0.3 * ATR
    # Medium confidence (0.5-0.7): 0.1-0.3 * ATR
    # Low confidence (<0.5): 0-0.1 * ATR
    
    if confidence > 0.7:
        scale = 0.3
    elif confidence > 0.5:
        scale = 0.1 + (confidence - 0.5) * 1.0  # 0.1 to 0.3
    else:
        scale = confidence * 0.2  # 0 to 0.1
    
    return direction * scale * atr


if __name__ == '__main__':
    # Test forecast engine
    print("Testing forecast engine...")
    
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
    
    # Mock model
    class MockModel:
        def predict(self, X):
            return np.array([2])  # BULLISH
        
        def predict_proba(self, X):
            return np.array([[0.1, 0.2, 0.7]])  # 70% confidence
    
    # Mock feature engineer
    class MockFE:
        def engineer(self, df):
            return pd.DataFrame([[1, 2, 3, 4, 5]])  # Dummy features
    
    engine = ForecastEngine()
    forecast = engine.generate_forecast(
        df_test,
        MockModel(),
        MockFE(),
        'TEST'
    )
    
    print(f"\n✓ Generated {len(forecast)} day forecast")
    print(f"✓ Last historical price: ${df_test['Close'].iloc[-1]:.2f}")
    print(f"✓ Forecast day 1: ${forecast['price'].iloc[0]:.2f} ({forecast['signal'].iloc[0]})")
    print(f"✓ Forecast day 24: ${forecast['price'].iloc[-1]:.2f} (decay={forecast['decay_factor'].iloc[-1]:.2f})")
    print(f"✓ ATR%: {forecast['atr_pct'].iloc[0]:.2f}%")
    
    # Check forecast is reasonable (no >50% daily jumps)
    max_daily_move = forecast['price'].pct_change().abs().max() * 100
    print(f"✓ Max daily move: {max_daily_move:.2f}%")
    
    if max_daily_move < 50:
        print("\n✅ Forecast looks realistic!")
    else:
        print(f"\n⚠️  Warning: Large daily moves detected ({max_daily_move:.2f}%)")
