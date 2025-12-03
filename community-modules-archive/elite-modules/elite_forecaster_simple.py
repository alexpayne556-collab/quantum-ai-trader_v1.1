"""
Simple Elite Forecaster - Works in Colab
Uses Prophet + basic models for 21-day predictions
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EliteForecaster:
    """Simple 21-day forecaster using Prophet"""
    
    def __init__(self):
        self.horizon = 21
        self.models_loaded = []
        
        # Try to load Prophet
        try:
            from prophet import Prophet
            self.prophet_available = True
            self.models_loaded.append('Prophet')
        except:
            self.prophet_available = False
    
    def predict(self, symbol, horizon=21):
        """
        Predict stock price for next N days
        
        Args:
            symbol: Stock ticker
            horizon: Days to forecast (default 21)
        
        Returns:
            dict with predictions
        """
        try:
            # Download data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if len(df) < 60:
                return None
            
            # Prepare data
            df_prophet = pd.DataFrame({
                'ds': df.index,
                'y': df['Close'].values
            })
            
            current_price = df['Close'].iloc[-1]
            
            # Use Prophet if available
            if self.prophet_available:
                from prophet import Prophet
                
                model = Prophet(daily_seasonality=True, weekly_seasonality=True)
                model.fit(df_prophet)
                
                future = model.make_future_dataframe(periods=horizon)
                forecast = model.predict(future)
                
                forecast_values = forecast['yhat'].tail(horizon).values
                
            else:
                # Simple linear trend + noise
                returns = df['Close'].pct_change().dropna()
                avg_return = returns.mean()
                volatility = returns.std()
                
                forecast_values = []
                last_price = current_price
                
                for i in range(horizon):
                    # Trend + noise
                    change = avg_return + np.random.normal(0, volatility)
                    last_price = last_price * (1 + change)
                    forecast_values.append(last_price)
            
            final_price = forecast_values[-1]
            predicted_return = (final_price - current_price) / current_price
            
            # Confidence based on volatility
            volatility_score = min(returns.std() * 10, 1.0)
            confidence = 'High' if volatility_score < 0.3 else ('Medium' if volatility_score < 0.6 else 'Low')
            
            return {
                'symbol': symbol,
                'current_price': float(current_price),
                'predicted_return': float(predicted_return),
                'final_price': float(final_price),
                'forecast_path': [float(x) for x in forecast_values],
                'models_used': self.models_loaded,
                'confidence': confidence,
                'horizon': horizon
            }
            
        except Exception as e:
            print(f"Error predicting {symbol}: {e}")
            return None

