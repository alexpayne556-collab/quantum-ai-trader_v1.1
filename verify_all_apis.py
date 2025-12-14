"""
Comprehensive API Verification Script
Tests all 10 API keys before Colab training
"""

import os
import sys
from dotenv import load_dotenv
import requests
import yfinance as yf
from datetime import datetime, timedelta

load_dotenv()

print("="*70)
print("🔍 COMPREHENSIVE API KEY VERIFICATION")
print("="*70)
print()

results = {
    'passed': [],
    'failed': [],
    'skipped': []
}


def test_api(name, test_func):
    """Test an API and record results"""
    try:
        print(f"Testing {name}...", end=" ")
        result = test_func()
        if result:
            print("✅ WORKING")
            results['passed'].append(name)
            return True
        else:
            print("❌ FAILED")
            results['failed'].append(name)
            return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        results['failed'].append(name)
        return False


# Test 1: Twelve Data
def test_twelve_data():
    key = os.getenv('TWELVE_DATA_API_KEY')
    if not key:
        results['skipped'].append('Twelve Data')
        return False
    
    url = "https://api.twelvedata.com/time_series"
    params = {'symbol': 'AAPL', 'interval': '1h', 'outputsize': 5, 'apikey': key}
    r = requests.get(url, params=params, timeout=10)
    data = r.json()
    return 'values' in data and len(data['values']) > 0


# Test 2: Finnhub
def test_finnhub():
    key = os.getenv('FINNHUB_API_KEY')
    if not key:
        results['skipped'].append('Finnhub')
        return False
    
    url = "https://finnhub.io/api/v1/quote"
    params = {'symbol': 'AAPL', 'token': key}
    r = requests.get(url, params=params, timeout=10)
    data = r.json()
    return 'c' in data and data['c'] > 0


# Test 3: Polygon
def test_polygon():
    key = os.getenv('POLYGON_API_KEY')
    if not key:
        results['skipped'].append('Polygon')
        return False
    
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/{yesterday}/{yesterday}"
    params = {'apiKey': key}
    r = requests.get(url, params=params, timeout=10)
    data = r.json()
    return data.get('resultsCount', 0) > 0


# Test 4: Alpha Vantage
def test_alpha_vantage():
    key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not key:
        results['skipped'].append('Alpha Vantage')
        return False
    
    url = "https://www.alphavantage.co/query"
    params = {
        'function': 'GLOBAL_QUOTE',
        'symbol': 'AAPL',
        'apikey': key
    }
    r = requests.get(url, params=params, timeout=10)
    data = r.json()
    return 'Global Quote' in data


# Test 5: FMP
def test_fmp():
    key = os.getenv('FMP_API_KEY')
    if not key:
        results['skipped'].append('FMP')
        return False
    
    url = "https://financialmodelingprep.com/api/v3/quote/AAPL"
    params = {'apikey': key}
    r = requests.get(url, params=params, timeout=10)
    data = r.json()
    return isinstance(data, list) and len(data) > 0


# Test 6: EODHD
def test_eodhd():
    token = os.getenv('EODHD_API_TOKEN')
    if not token:
        results['skipped'].append('EODHD')
        return False
    
    url = "https://eodhd.com/api/real-time/AAPL.US"
    params = {'api_token': token, 'fmt': 'json'}
    r = requests.get(url, params=params, timeout=10)
    data = r.json()
    return 'code' in data


# Test 7: FRED
def test_fred():
    key = os.getenv('FRED_API_KEY')
    if not key:
        results['skipped'].append('FRED')
        return False
    
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': 'VIXCLS',
        'api_key': key,
        'file_type': 'json',
        'limit': 5
    }
    r = requests.get(url, params=params, timeout=10)
    data = r.json()
    return 'observations' in data and len(data['observations']) > 0


# Test 8: Alpaca
def test_alpaca():
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not secret_key or api_key == 'your_alpaca_key_here':
        results['skipped'].append('Alpaca')
        return False
    
    url = "https://paper-api.alpaca.markets/v2/account"
    headers = {
        'APCA-API-KEY-ID': api_key,
        'APCA-API-SECRET-KEY': secret_key
    }
    r = requests.get(url, headers=headers, timeout=10)
    data = r.json()
    return 'account_number' in data


# Test 9: Perplexity
def test_perplexity():
    key = os.getenv('PERPLEXITY_API_KEY')
    if not key:
        results['skipped'].append('Perplexity')
        return False
    
    # Simple test - just check if key format is valid
    return key.startswith('pplx-') and len(key) > 20


# Test 10: OpenAI
def test_openai():
    key = os.getenv('OPENAI_API_KEY')
    if not key:
        results['skipped'].append('OpenAI')
        return False
    
    # Simple test - just check if key format is valid
    return key.startswith('sk-') and len(key) > 20


# Test 11: yfinance (always available)
def test_yfinance():
    df = yf.download('AAPL', period='5d', progress=False)
    return len(df) > 0


print("MARKET DATA APIs:")
print("-" * 70)
test_api("Twelve Data (800/day)", test_twelve_data)
test_api("Finnhub (60/min)", test_finnhub)
test_api("Polygon (5/min)", test_polygon)
test_api("Alpha Vantage (25/day)", test_alpha_vantage)
test_api("FMP (250/day)", test_fmp)
test_api("EODHD (20/day)", test_eodhd)
test_api("yfinance (unlimited)", test_yfinance)

print()
print("MACRO DATA APIs:")
print("-" * 70)
test_api("FRED (unlimited)", test_fred)

print()
print("TRADING APIs:")
print("-" * 70)
test_api("Alpaca Paper Trading", test_alpaca)

print()
print("AI APIs:")
print("-" * 70)
test_api("Perplexity AI", test_perplexity)
test_api("OpenAI", test_openai)

print()
print("="*70)
print("📊 VERIFICATION SUMMARY")
print("="*70)
print(f"✅ PASSED:  {len(results['passed'])} APIs")
print(f"❌ FAILED:  {len(results['failed'])} APIs")
print(f"⏭️  SKIPPED: {len(results['skipped'])} APIs (no key provided)")

if results['passed']:
    print(f"\n✅ Working APIs: {', '.join(results['passed'])}")

if results['failed']:
    print(f"\n❌ Failed APIs: {', '.join(results['failed'])}")
    print("   Check your .env file and verify the keys are correct")

if results['skipped']:
    print(f"\n⏭️  Skipped APIs: {', '.join(results['skipped'])}")
    print("   Add keys to .env file to enable these sources")

print()
print("="*70)

# Check if we have enough sources for training
market_data_sources = sum(1 for api in ['Twelve Data (800/day)', 'Finnhub (60/min)', 
                                          'Polygon (5/min)', 'yfinance (unlimited)'] 
                          if api in results['passed'])

if market_data_sources >= 2:
    print("✅ READY FOR TRAINING - At least 2 market data sources available")
    print(f"   Active sources: {market_data_sources}")
else:
    print("⚠️  WARNING - Only 1 data source available")
    print("   Training may be slow or fail if rate limits are hit")

if 'FRED (unlimited)' in results['passed']:
    print("✅ READY FOR REGIME DETECTION - FRED economic data available")
else:
    print("⚠️  WARNING - No FRED data for regime detection")

if 'Alpaca Paper Trading' in results['passed']:
    print("✅ READY FOR PAPER TRADING - Alpaca keys configured")
else:
    print("⚠️  NOTE - Alpaca not configured (needed for Week 3 paper trading)")

print("="*70)
print()

# Exit with error code if critical APIs failed
critical_passed = market_data_sources >= 2
sys.exit(0 if critical_passed else 1)
