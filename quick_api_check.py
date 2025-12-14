"""
Quick API Status Check - Run Before Training
Shows which APIs are working and rate limit status
"""

import os
from dotenv import load_dotenv
import requests
from datetime import datetime

load_dotenv()

print("="*60)
print(f"🔍 QUICK API CHECK - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)
print()

# Quick checks (faster than full verification)
checks = []

# 1. Twelve Data (PRIMARY)
try:
    r = requests.get('https://api.twelvedata.com/time_series', 
                     params={'symbol': 'AAPL', 'interval': '1h', 'outputsize': 1, 
                            'apikey': os.getenv('TWELVE_DATA_API_KEY')}, timeout=5)
    checks.append(('Twelve Data (800/day)', '✅' if 'values' in r.json() else '❌'))
except:
    checks.append(('Twelve Data (800/day)', '❌'))

# 2. Finnhub (SECONDARY)
try:
    r = requests.get('https://finnhub.io/api/v1/quote',
                     params={'symbol': 'AAPL', 'token': os.getenv('FINNHUB_API_KEY')}, timeout=5)
    checks.append(('Finnhub (60/min)', '✅' if 'c' in r.json() else '❌'))
except:
    checks.append(('Finnhub (60/min)', '❌'))

# 3. Alpha Vantage (BACKUP)
try:
    r = requests.get('https://www.alphavantage.co/query',
                     params={'function': 'GLOBAL_QUOTE', 'symbol': 'AAPL', 
                            'apikey': os.getenv('ALPHA_VANTAGE_API_KEY')}, timeout=5)
    checks.append(('Alpha Vantage (25/day)', '✅' if 'Global Quote' in r.json() else '❌'))
except:
    checks.append(('Alpha Vantage (25/day)', '❌'))

# 4. FRED (REGIME)
try:
    r = requests.get('https://api.stlouisfed.org/fred/series/observations',
                     params={'series_id': 'VIXCLS', 'api_key': os.getenv('FRED_API_KEY'),
                            'file_type': 'json', 'limit': 1}, timeout=5)
    checks.append(('FRED (regime data)', '✅' if 'observations' in r.json() else '❌'))
except:
    checks.append(('FRED (regime data)', '❌'))

# 5. Alpaca (PAPER TRADING)
try:
    r = requests.get('https://paper-api.alpaca.markets/v2/account',
                     headers={'APCA-API-KEY-ID': os.getenv('ALPACA_API_KEY'),
                             'APCA-API-SECRET-KEY': os.getenv('ALPACA_SECRET_KEY')}, timeout=5)
    checks.append(('Alpaca (paper trading)', '✅' if 'account_number' in r.json() else '❌'))
except:
    checks.append(('Alpaca (paper trading)', '❌'))

# Print results
print("CRITICAL APIs:")
for name, status in checks:
    print(f"  {status} {name}")

working = sum(1 for _, s in checks if s == '✅')
print()
print("="*60)
if working >= 3:
    print(f"✅ {working}/5 CRITICAL APIs WORKING - READY TO TRAIN")
else:
    print(f"⚠️  {working}/5 APIs working - May have issues")
print("="*60)
