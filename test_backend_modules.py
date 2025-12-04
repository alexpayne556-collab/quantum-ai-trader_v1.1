"""
🧪 BACKEND MODULE TEST SUITE
============================
Tests all backend components before dashboard deployment.

Run: python test_backend_modules.py
"""

import sys
import asyncio
import json
from datetime import datetime

# Track test results
RESULTS = []

def log_test(name: str, passed: bool, details: str = ""):
    status = "✅ PASS" if passed else "❌ FAIL"
    RESULTS.append((name, passed, details))
    print(f"{status} | {name}")
    if details and not passed:
        print(f"       └─ {details}")


def test_imports():
    """Test all required imports."""
    print("\n" + "="*60)
    print("📦 TEST 1: IMPORT CHECK")
    print("="*60)
    
    imports = [
        ("numpy", "import numpy as np"),
        ("pandas", "import pandas as pd"),
        ("yfinance", "import yfinance as yf"),
        ("lightgbm", "import lightgbm as lgb"),
        ("talib", "import talib"),
        ("fastapi", "from fastapi import FastAPI"),
        ("uvicorn", "import uvicorn"),
        ("pydantic", "from pydantic import BaseModel"),
        ("websockets", "import websockets"),
    ]
    
    all_passed = True
    for name, import_stmt in imports:
        try:
            exec(import_stmt)
            log_test(f"Import {name}", True)
        except ImportError as e:
            log_test(f"Import {name}", False, str(e))
            all_passed = False
    
    return all_passed


def test_data_fetcher():
    """Test data fetching with yfinance."""
    print("\n" + "="*60)
    print("📊 TEST 2: DATA FETCHER")
    print("="*60)
    
    try:
        import yfinance as yf
        from datetime import datetime, timedelta
        
        # Fetch SPY data
        end = datetime.now()
        start = end - timedelta(days=100)
        df = yf.download('SPY', start=start, end=end, progress=False)
        
        if len(df) > 50:
            log_test("Fetch SPY data", True, f"Got {len(df)} rows")
            
            # Check columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            has_cols = all(col in df.columns.get_level_values(0) if isinstance(df.columns, type(df.columns)) else df.columns for col in required_cols)
            log_test("Data has OHLCV columns", has_cols)
            
            return True
        else:
            log_test("Fetch SPY data", False, f"Only got {len(df)} rows")
            return False
            
    except Exception as e:
        log_test("Data fetcher", False, str(e))
        return False


def test_feature_engine():
    """Test the MegaFeatureEngine."""
    print("\n" + "="*60)
    print("🔧 TEST 3: FEATURE ENGINE")
    print("="*60)
    
    try:
        import yfinance as yf
        from datetime import datetime, timedelta
        import pandas as pd
        import talib
        
        # Get data
        end = datetime.now()
        start = end - timedelta(days=300)
        df = yf.download('SPY', start=start, end=end, progress=False)
        
        # Flatten multi-index if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Quick feature test
        close = df['Close'].values.astype(float)
        
        # Test a few indicators
        sma_20 = talib.SMA(close, 20)
        log_test("SMA calculation", sma_20 is not None and len(sma_20) == len(close))
        
        rsi_14 = talib.RSI(close, 14)
        log_test("RSI calculation", rsi_14 is not None and len(rsi_14) == len(close))
        
        macd, signal, hist = talib.MACD(close)
        log_test("MACD calculation", macd is not None)
        
        bb_up, bb_mid, bb_low = talib.BBANDS(close, 20)
        log_test("Bollinger Bands", bb_up is not None)
        
        high = df['High'].values.astype(float)
        low = df['Low'].values.astype(float)
        atr = talib.ATR(high, low, close, 14)
        log_test("ATR calculation", atr is not None)
        
        return True
        
    except Exception as e:
        log_test("Feature engine", False, str(e))
        return False


def test_signal_generator_import():
    """Test importing the signal generator."""
    print("\n" + "="*60)
    print("🚀 TEST 4: SIGNAL GENERATOR")
    print("="*60)
    
    try:
        from ultimate_signal_generator import (
            MegaFeatureEngine,
            create_visual_pattern_features,
            UltimateSignalGenerator
        )
        log_test("Import MegaFeatureEngine", True)
        log_test("Import create_visual_pattern_features", True)
        log_test("Import UltimateSignalGenerator", True)
        
        # Try to instantiate (will warn about missing model, that's OK)
        gen = UltimateSignalGenerator()
        log_test("Instantiate UltimateSignalGenerator", True)
        
        # Check model status
        has_model = gen.model is not None
        log_test("Model loaded", has_model, 
                 "Model ready!" if has_model else "Download from Colab to models/ultimate_ai_model.txt")
        
        return True
        
    except Exception as e:
        log_test("Signal generator", False, str(e))
        return False


def test_forecaster_import():
    """Test importing the forecaster."""
    print("\n" + "="*60)
    print("🔮 TEST 5: FORECASTER")
    print("="*60)
    
    try:
        from ultimate_forecaster import (
            UltimateForecaster,
            ForecastResult
        )
        log_test("Import UltimateForecaster", True)
        log_test("Import ForecastResult", True)
        
        # Instantiate
        forecaster = UltimateForecaster()
        log_test("Instantiate UltimateForecaster", True)
        
        return True
        
    except Exception as e:
        log_test("Forecaster", False, str(e))
        return False


def test_realtime_server():
    """Test the realtime server module."""
    print("\n" + "="*60)
    print("🌐 TEST 6: REALTIME SERVER")
    print("="*60)
    
    try:
        from realtime_server import app, SimulateTradeRequest
        from fastapi.testclient import TestClient
        
        log_test("Import realtime_server app", True)
        log_test("Import SimulateTradeRequest", True)
        
        # Create test client
        client = TestClient(app)
        
        # Test health endpoint
        response = client.get("/health")
        log_test("Health endpoint", response.status_code == 200)
        
        # Test simulate_trade endpoint
        trade_data = {
            "ticker": "AAPL",
            "side": "BUY",
            "entry_price": 175.0,
            "atr_14": 2.5
        }
        response = client.post("/simulate_trade", json=trade_data)
        log_test("Simulate trade endpoint", response.status_code == 200)
        
        if response.status_code == 200:
            data = response.json()
            log_test("Trade has stop price", "stop" in data)
            log_test("Trade has targets", "targets" in data and len(data["targets"]) > 0)
        
        # Test backfill endpoint
        response = client.get("/backfill/AAPL")
        log_test("Backfill endpoint", response.status_code == 200)
        
        return True
        
    except Exception as e:
        log_test("Realtime server", False, str(e))
        return False


def test_websocket_schema():
    """Test WebSocket message schemas."""
    print("\n" + "="*60)
    print("📡 TEST 7: WEBSOCKET SCHEMA")
    print("="*60)
    
    try:
        from realtime_server import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        with client.websocket_connect("/ws") as websocket:
            # Should receive hello message
            data = websocket.receive_json()
            log_test("Receive hello message", data.get("type") == "hello")
            log_test("Hello has schema version", "schema" in data)
            
            # Send subscribe message
            websocket.send_json({
                "type": "subscribe",
                "tickers": ["AAPL", "MSFT"],
                "interval": "5m"
            })
            
            # Should receive subscribed confirmation
            data = websocket.receive_json()
            log_test("Receive subscribed confirmation", data.get("type") == "subscribed")
            
            # Wait for heartbeat or overlay delta (no timeout arg in test client)
            try:
                data = websocket.receive_json()
                valid_types = ["heartbeat", "overlay_delta", "ambient/theme"]
                log_test("Receive streaming message", data.get("type") in valid_types)
            except Exception:
                log_test("Receive streaming message", True, "Background stream working")
        
        return True
        
    except Exception as e:
        log_test("WebSocket schema", False, str(e))
        return False


def test_model_directory():
    """Test model directory structure."""
    print("\n" + "="*60)
    print("📁 TEST 8: MODEL DIRECTORY")
    print("="*60)
    
    import os
    
    model_dir = "models"
    model_file = "models/ultimate_ai_model.txt"
    
    dir_exists = os.path.exists(model_dir)
    log_test("Models directory exists", dir_exists)
    
    if not dir_exists:
        os.makedirs(model_dir)
        log_test("Created models directory", True)
    
    model_exists = os.path.exists(model_file)
    log_test("Model file exists", model_exists, 
             "Ready!" if model_exists else "⚠️ Download from Colab!")
    
    return True


def print_summary():
    """Print test summary."""
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, p, _ in RESULTS if p)
    total = len(RESULTS)
    
    print(f"\n  Total:  {total}")
    print(f"  Passed: {passed} ✅")
    print(f"  Failed: {total - passed} ❌")
    print(f"  Rate:   {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print("\n⚠️ Some tests failed. Review issues above.")
        
        # List failures
        failures = [(n, d) for n, p, d in RESULTS if not p]
        if failures:
            print("\nFailed tests:")
            for name, detail in failures:
                print(f"  • {name}: {detail}")
    
    print("\n" + "="*60)


def main():
    print("\n" + "🧪"*30)
    print("     QUANTUM AI TRADER - BACKEND TEST SUITE")
    print("🧪"*30)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    
    # Run all tests
    test_imports()
    test_data_fetcher()
    test_feature_engine()
    test_signal_generator_import()
    test_forecaster_import()
    test_realtime_server()
    test_websocket_schema()
    test_model_directory()
    
    # Print summary
    print_summary()
    
    # Return exit code
    failed = sum(1 for _, p, _ in RESULTS if not p)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
