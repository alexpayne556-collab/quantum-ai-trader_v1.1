"""
⚡ QUICK BACKEND TEST
=====================
Fast validation of core modules (< 30 seconds)
"""

import sys
import os

# Ensure we're in the right directory
WORKSPACE = "/workspaces/quantum-ai-trader_v1.1"
os.chdir(WORKSPACE)
sys.path.insert(0, WORKSPACE)

print("="*60)
print("⚡ QUICK BACKEND MODULE TEST")
print(f"📁 Working directory: {os.getcwd()}")
print("="*60)

results = []

def test(name, func):
    try:
        func()
        print(f"✅ {name}")
        results.append((name, True))
    except Exception as e:
        print(f"❌ {name}: {e}")
        results.append((name, False))

# TEST 1: Core imports
print("\n📦 IMPORTS:")
test("numpy", lambda: __import__("numpy"))
test("pandas", lambda: __import__("pandas"))
test("yfinance", lambda: __import__("yfinance"))
test("lightgbm", lambda: __import__("lightgbm"))
test("talib", lambda: __import__("talib"))
test("fastapi", lambda: __import__("fastapi"))

# TEST 2: Our modules
print("\n🔧 OUR MODULES:")

def test_signal_gen():
    from ultimate_signal_generator import UltimateSignalGenerator
    return UltimateSignalGenerator

def test_forecaster():
    from ultimate_forecaster import UltimateForecaster
    return UltimateForecaster()

def test_config():
    # config is a folder with __init__.py
    from config import settings
    return True
    
def test_data_fetcher():
    from data_fetcher import DataFetcher
    return DataFetcher()

test("MegaFeatureEngine", test_signal_gen)
test("UltimateForecaster", test_forecaster)
test("Config", test_config)
test("DataFetcher", test_data_fetcher)

# TEST 3: Quick data fetch
print("\n📊 DATA FETCH (SPY):")
def test_yf_fetch():
    import yfinance as yf
    data = yf.download("SPY", period="5d", progress=False)
    assert len(data) >= 3, f"Only got {len(data)} rows"
    return data

test("yfinance fetch", test_yf_fetch)

# TEST 4: Indicator calculation
print("\n📈 INDICATORS:")
def test_indicators():
    import yfinance as yf
    import talib
    import numpy as np
    
    data = yf.download("SPY", period="60d", progress=False)
    close = data['Close'].values.flatten().astype(float)
    
    # Test core indicators
    rsi = talib.RSI(close, 14)
    macd, signal, hist = talib.MACD(close)
    sma = talib.SMA(close, 20)
    
    assert not np.isnan(rsi[-1]), "RSI failed"
    assert not np.isnan(macd[-1]), "MACD failed"
    return True

test("TA-Lib indicators", test_indicators)

# TEST 5: Model file check
print("\n🤖 MODEL:")
model_path = "models/ultimate_ai_model.txt"
if os.path.exists(model_path):
    print(f"✅ Model found: {model_path}")
    results.append(("Model file", True))
else:
    print(f"⚠️  Model NOT found: {model_path}")
    print("   → Download from Colab: trainer.save_model('ultimate_ai_model.txt')")
    results.append(("Model file", False))

# TEST 6: FastAPI app creation
print("\n🌐 API:")
def test_api():
    from fastapi import FastAPI
    from pydantic import BaseModel
    
    app = FastAPI()
    
    @app.get("/health")
    def health():
        return {"status": "ok"}
    
    return True

test("FastAPI app", test_api)

# SUMMARY
print("\n" + "="*60)
passed = sum(1 for _, p in results if p)
total = len(results)
pct = passed/total*100

if pct >= 90:
    print(f"🎉 EXCELLENT: {passed}/{total} tests passed ({pct:.0f}%)")
    print("   Backend is READY for dashboard!")
elif pct >= 70:
    print(f"✅ GOOD: {passed}/{total} tests passed ({pct:.0f}%)")
    print("   Minor issues - see above")
else:
    print(f"⚠️  NEEDS WORK: {passed}/{total} tests passed ({pct:.0f}%)")

# Show what's missing
failed = [n for n, p in results if not p]
if failed:
    print(f"\n🔧 TO FIX: {', '.join(failed)}")
    
print("="*60)
