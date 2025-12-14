
import pandas as pd
import numpy as np
from forecast_engine import ForecastEngine
from local_dashboard.app import ProxyModel, ProxyFeatureEngineer

# Create dummy data
dates = pd.date_range('2024-01-01', periods=100, freq='D')
df = pd.DataFrame({
    'Open': np.random.rand(100) * 100,
    'High': np.random.rand(100) * 100,
    'Low': np.random.rand(100) * 100,
    'Close': np.random.rand(100) * 100,
    'Volume': np.random.rand(100) * 1000000
}, index=dates)

engine = ForecastEngine()
model = ProxyModel()
fe = ProxyFeatureEngineer()

print("Running forecast...")
try:
    forecast = engine.generate_forecast(df, model, fe, 'TEST')
    print("Forecast generated successfully")
except Exception as e:
    print(f"Caught exception: {e}")
    import traceback
    traceback.print_exc()
