"""
Configuration Management
Loads settings from .env file with fallback defaults
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', '')
ALPHAVANTAGE_API_KEY = os.getenv('ALPHAVANTAGE_API_KEY', '')
IEX_API_KEY = os.getenv('IEX_API_KEY', '')
FMP_API_KEY = os.getenv('FMP_API_KEY', '')
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY', '')
TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY', '')
EODHD_API_TOKEN = os.getenv('EODHD_API_TOKEN', '')
MARKETAUX_API_KEY = os.getenv('MARKETAUX_API_KEY', '')
INTRINIO_API_KEY = os.getenv('INTRINIO_API_KEY', '')
NEWSDATA_API_KEY = os.getenv('NEWSDATA_API_KEY', '')
NEWSAPI_API_KEY = os.getenv('NEWSAPI_API_KEY', '')
TIINGO_API_KEY = os.getenv('TIINGO_API_KEY', '')
FRED_API_KEY = os.getenv('FRED_API_KEY', '')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

# System Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
MAX_CACHE_SIZE = int(os.getenv('MAX_CACHE_SIZE', '10000'))
REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '30'))
SECRET_KEY = os.getenv('SECRET_KEY', 'quantum_ai_cockpit_secret_key_change_in_production')
ENCRYPTION_SALT = os.getenv('ENCRYPTION_SALT', 'quantum_ai_cockpit_salt_change_in_production')

# Symbol Universe
SYMBOL_UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
    'MU', 'APLD', 'IONQ', 'ANNX', 'PLTR', 'COIN', 'SQ', 'SHOP',
    'AMD', 'INTC', 'QCOM', 'CRM'
]

# Validate critical configuration
missing_keys = []
if not FINNHUB_API_KEY:
    missing_keys.append('FINNHUB_API_KEY')
if not ALPHAVANTAGE_API_KEY:
    missing_keys.append('ALPHAVANTAGE_API_KEY')
if not FMP_API_KEY:
    missing_keys.append('FMP_API_KEY')

if missing_keys:
    print("🚨 CRITICAL: Missing API keys required for data fetching:")
    for key in missing_keys:
        print(f"   - {key}")
    print("   Please copy .env.template to .env and fill in your API keys")
    print("   Get keys from: https://finnhub.io, https://alphavantage.co, https://financialmodelingprep.com")
else:
    print("✅ All critical API keys configured")

# Check for at least one news API key
news_keys = [MARKETAUX_API_KEY, NEWSAPI_API_KEY, NEWSDATA_API_KEY]
if not any(news_keys):
    print("⚠️ WARNING: No news API keys configured - news features will be limited")

print(f"✅ Configuration loaded - Log Level: {LOG_LEVEL}, Cache Size: {MAX_CACHE_SIZE}")