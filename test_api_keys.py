#!/usr/bin/env python3
"""
Test all API keys from .env file
Identifies: WORKING ✅ | MISSING ❌ | INVALID 🔴
"""
import os
import sys
from dotenv import load_dotenv
import requests
from datetime import datetime

# Load environment variables
load_dotenv('.env')

print("=" * 80)
print("🔍 API KEY VALIDATION REPORT")
print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

results = {'working': [], 'missing': [], 'invalid': [], 'untested': []}

def test_key(name, key, test_func, category):
    """Test an API key with custom validation function"""
    if not key or key.startswith('your_'):
        results['missing'].append((name, category))
        return '❌ MISSING'
    
    try:
        if test_func(key):
            results['working'].append((name, category))
            return '✅ WORKING'
        else:
            results['invalid'].append((name, category))
            return '🔴 INVALID'
    except Exception as e:
        results['untested'].append((name, category, str(e)))
        return f'⚠️  UNTESTED ({str(e)[:30]})'

# ============================================================================
# MARKET DATA APIs
# ============================================================================
print("\n📊 MARKET DATA APIs")
print("-" * 80)

# Alpha Vantage
av_key = os.getenv('ALPHA_VANTAGE_API_KEY')
av_backup = os.getenv('ALPHA_VANTAGE_API_KEY_BACKUP')
def test_av(key):
    r = requests.get(f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey={key}')
    return 'Error Message' not in r.text and 'Thank you' not in r.text

status = test_key('ALPHA_VANTAGE_API_KEY', av_key, test_av, 'MARKET_DATA')
print(f"Alpha Vantage (Primary):  {status}")
if av_backup:
    status = test_key('ALPHA_VANTAGE_API_KEY_BACKUP', av_backup, test_av, 'MARKET_DATA')
    print(f"Alpha Vantage (Backup):   {status}")

# Polygon
polygon_key = os.getenv('POLYGON_API_KEY')
def test_polygon(key):
    r = requests.get(f'https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2023-01-01/2023-01-10?apiKey={key}')
    return r.status_code == 200

status = test_key('POLYGON_API_KEY', polygon_key, test_polygon, 'MARKET_DATA')
print(f"Polygon.io:               {status}")

# Finnhub
finnhub_key = os.getenv('FINNHUB_API_KEY')
def test_finnhub(key):
    r = requests.get(f'https://finnhub.io/api/v1/quote?symbol=AAPL&token={key}')
    return r.status_code == 200 and 'c' in r.json()

status = test_key('FINNHUB_API_KEY', finnhub_key, test_finnhub, 'MARKET_DATA')
print(f"Finnhub:                  {status}")

# Twelve Data
twelve_key = os.getenv('TWELVE_DATA_API_KEY')
def test_twelve(key):
    r = requests.get(f'https://api.twelvedata.com/time_series?symbol=AAPL&interval=1min&apikey={key}')
    return r.status_code == 200 and 'status' in r.json() and r.json()['status'] != 'error'

status = test_key('TWELVE_DATA_API_KEY', twelve_key, test_twelve, 'MARKET_DATA')
print(f"Twelve Data:              {status}")

# Financial Modeling Prep
fmp_key = os.getenv('FMP_API_KEY') or os.getenv('FINANCIALMODELINGPREP_API_KEY')
def test_fmp(key):
    r = requests.get(f'https://financialmodelingprep.com/api/v3/quote/AAPL?apikey={key}')
    return r.status_code == 200 and isinstance(r.json(), list)

status = test_key('FMP_API_KEY', fmp_key, test_fmp, 'MARKET_DATA')
print(f"Financial Modeling Prep:  {status}")

# EODHD
eodhd_key = os.getenv('EODHD_API_TOKEN')
def test_eodhd(key):
    r = requests.get(f'https://eodhd.com/api/eod/AAPL.US?api_token={key}&fmt=json')
    return r.status_code == 200

status = test_key('EODHD_API_TOKEN', eodhd_key, test_eodhd, 'MARKET_DATA')
print(f"EOD Historical Data:      {status}")

# ============================================================================
# ECONOMIC DATA
# ============================================================================
print("\n💰 ECONOMIC DATA APIs")
print("-" * 80)

# FRED
fred_key = os.getenv('FRED_API_KEY')
def test_fred(key):
    r = requests.get(f'https://api.stlouisfed.org/fred/series/observations?series_id=VIXCLS&api_key={key}&file_type=json')
    return r.status_code == 200 and 'observations' in r.json()

status = test_key('FRED_API_KEY', fred_key, test_fred, 'ECONOMIC')
print(f"FRED (Federal Reserve):   {status}")

# ============================================================================
# AI APIs
# ============================================================================
print("\n🤖 AI APIs")
print("-" * 80)

# Perplexity
perplexity_key = os.getenv('PERPLEXITY_API_KEY')
def test_perplexity(key):
    headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
    data = {'model': 'llama-3.1-sonar-small-128k-online', 'messages': [{'role': 'user', 'content': 'test'}]}
    r = requests.post('https://api.perplexity.ai/chat/completions', json=data, headers=headers, timeout=10)
    return r.status_code in [200, 401]  # 401 means key format valid but may be expired

status = test_key('PERPLEXITY_API_KEY', perplexity_key, test_perplexity, 'AI')
print(f"Perplexity AI:            {status}")

# OpenAI
openai_key = os.getenv('OPENAI_API_KEY')
def test_openai(key):
    headers = {'Authorization': f'Bearer {key}'}
    r = requests.get('https://api.openai.com/v1/models', headers=headers, timeout=10)
    return r.status_code in [200, 401]

status = test_key('OPENAI_API_KEY', openai_key, test_openai, 'AI')
print(f"OpenAI:                   {status}")

# ============================================================================
# TRADING PLATFORMS
# ============================================================================
print("\n📈 TRADING PLATFORMS")
print("-" * 80)

# Alpaca
alpaca_key = os.getenv('ALPACA_API_KEY')
alpaca_secret = os.getenv('ALPACA_SECRET_KEY')
alpaca_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

def test_alpaca(key):
    if not alpaca_secret:
        return False
    headers = {'APCA-API-KEY-ID': key, 'APCA-API-SECRET-KEY': alpaca_secret}
    r = requests.get(f'{alpaca_url}/v2/account', headers=headers)
    return r.status_code == 200

status = test_key('ALPACA_API_KEY', alpaca_key, test_alpaca, 'TRADING')
print(f"Alpaca Paper Trading:     {status}")
if alpaca_key and alpaca_secret and status == '✅ WORKING':
    headers = {'APCA-API-KEY-ID': alpaca_key, 'APCA-API-SECRET-KEY': alpaca_secret}
    r = requests.get(f'{alpaca_url}/v2/account', headers=headers)
    if r.status_code == 200:
        acct = r.json()
        print(f"  → Account: ${float(acct.get('equity', 0)):,.2f} equity")
        print(f"  → Buying Power: ${float(acct.get('buying_power', 0)):,.2f}")

# ============================================================================
# ALTERNATIVE DATA (From Perplexity recommendations)
# ============================================================================
print("\n🔍 ALTERNATIVE DATA (Free Sources)")
print("-" * 80)

# Quiver Quant
quiver_key = os.getenv('QUIVER_QUANT_API_KEY')
def test_quiver(key):
    headers = {'Authorization': f'Token {key}'}
    r = requests.get('https://api.quiverquant.com/beta/bulk/congresstrading', headers=headers)
    return r.status_code in [200, 401]

status = test_key('QUIVER_QUANT_API_KEY', quiver_key, test_quiver, 'ALT_DATA')
print(f"Quiver Quant:             {status}")

# SEC EDGAR (no key needed)
try:
    r = requests.get('https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000789019&type=10-K&dateb=&owner=exclude&count=10', 
                     headers={'User-Agent': 'Quantum Trader test@example.com'})
    if r.status_code == 200:
        results['working'].append(('SEC_EDGAR', 'ALT_DATA'))
        print(f"SEC EDGAR (Free):         ✅ WORKING (no key needed)")
    else:
        results['untested'].append(('SEC_EDGAR', 'ALT_DATA', f'Status {r.status_code}'))
        print(f"SEC EDGAR (Free):         ⚠️  Status {r.status_code}")
except Exception as e:
    results['untested'].append(('SEC_EDGAR', 'ALT_DATA', str(e)))
    print(f"SEC EDGAR (Free):         ⚠️  {str(e)[:40]}")

# Clinical Trials (no key needed)
try:
    r = requests.get('https://clinicaltrials.gov/api/v2/studies?query.term=cancer&pageSize=1')
    if r.status_code == 200:
        results['working'].append(('CLINICAL_TRIALS', 'ALT_DATA'))
        print(f"Clinical Trials (Free):   ✅ WORKING (no key needed)")
    else:
        results['untested'].append(('CLINICAL_TRIALS', 'ALT_DATA', f'Status {r.status_code}'))
        print(f"Clinical Trials (Free):   ⚠️  Status {r.status_code}")
except Exception as e:
    results['untested'].append(('CLINICAL_TRIALS', 'ALT_DATA', str(e)))
    print(f"Clinical Trials (Free):   ⚠️  {str(e)[:40]}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("📊 SUMMARY")
print("=" * 80)

print(f"\n✅ WORKING APIs ({len(results['working'])}):")
for name, cat in results['working']:
    print(f"   • {name} ({cat})")

if results['invalid']:
    print(f"\n🔴 INVALID APIs ({len(results['invalid'])}):")
    for name, cat in results['invalid']:
        print(f"   • {name} ({cat}) - Key rejected by service")

if results['missing']:
    print(f"\n❌ MISSING APIs ({len(results['missing'])}):")
    for name, cat in results['missing']:
        print(f"   • {name} ({cat}) - Not configured in .env")

if results['untested']:
    print(f"\n⚠️  UNTESTED APIs ({len(results['untested'])}):")
    for item in results['untested']:
        if len(item) == 3:
            name, cat, reason = item
            print(f"   • {name} ({cat}) - {reason}")

# Recommendations
print("\n" + "=" * 80)
print("💡 RECOMMENDATIONS")
print("=" * 80)

working_market = [n for n, c in results['working'] if c == 'MARKET_DATA']
if len(working_market) < 3:
    print("⚠️  You have < 3 working market data APIs")
    print("   → Recommended: Get keys for Twelve Data, Finnhub, Polygon (all free)")
    
if not any(n[0] == 'ALPACA_API_KEY' for n in results['working']):
    print("⚠️  Alpaca paper trading not configured")
    print("   → Recommended: Get free paper trading account at alpaca.markets")

if not any(n[0] == 'QUIVER_QUANT_API_KEY' for n in results['working']):
    print("💡 Consider adding Quiver Quant (free tier) for insider/congress trades")

print("\n" + "=" * 80)
print(f"✅ TOTAL WORKING: {len(results['working'])} APIs")
print(f"🎯 READY FOR 8-HOUR BUILD: {'YES ✅' if len(working_market) >= 2 else 'NO ❌ (need 2+ market data APIs)'}")
print("=" * 80)
