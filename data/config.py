"""
Data Configuration

All settings for data ingestion, storage, and validation.
Modify these to control behavior without changing code.
"""

from pathlib import Path
from datetime import datetime, timedelta

# === PATHS ===
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / 'data_storage'

# Data directories
RAW_DATA_DIR = DATA_ROOT / 'raw'
PROCESSED_DATA_DIR = DATA_ROOT / 'processed'
MODELS_DATA_DIR = DATA_ROOT / 'models'
REFERENCE_DATA_DIR = DATA_ROOT / 'reference'

# Create directories
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DATA_DIR, REFERENCE_DATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# === DATA FETCHING ===
# Yahoo Finance rate limiting
REQUESTS_PER_SECOND = 2  # Conservative to avoid bans
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5
TIMEOUT_SECONDS = 30

# Date ranges
DEFAULT_LOOKBACK_YEARS = 3
DEFAULT_START_DATE = (datetime.now() - timedelta(days=365 * DEFAULT_LOOKBACK_YEARS)).strftime('%Y-%m-%d')
DEFAULT_END_DATE = datetime.now().strftime('%Y-%m-%d')

# === TICKER UNIVERSE ===
# Primary trading universe
TRADING_TICKERS = [
    # Tech Giants & AI
    'AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'TSLA', 'AVGO', 'ORCL', 'ADBE',
    'PLTR', 'SNOW', 'DDOG', 'NET', 'CRWD', 'ZS', 'PANW', 'FTNT', 'CRM', 'NOW',
    
    # Quantum & Space
    'IONQ', 'RGTI', 'QUBT', 'RKLB', 'ASTS', 'LUNR', 'JOBY', 'ACHR', 'SPIR', 'PL',
    
    # Biotech
    'VKTX', 'NTLA', 'BEAM', 'CRSP', 'EDIT', 'VERV', 'BLUE', 'MRNA', 'BNTX', 'GILD',
    
    # Clean Energy
    'FLNC', 'BE', 'ENPH', 'QS', 'PLUG', 'FCEL', 'NEE', 'VST', 'AES', 'NOVA',
    
    # Fintech
    'COIN', 'HOOD', 'SOFI', 'UPST', 'AFRM', 'SQ', 'PYPL', 'MARA', 'RIOT', 'MSTR',
    
    # Semiconductors
    'INTC', 'QCOM', 'MU', 'AMAT', 'LRCX', 'KLAC', 'ASML', 'TSM', 'MRVL', 'MPWR',
    
    # Autonomy & Robotics
    'SYM', 'AMBA', 'LAZR', 'OUST', 'AEVA', 'INVZ', 'LIDR', 'VLDR', 'BLDE', 'PATH',
    
    # Consumer
    'CELH', 'ONON', 'DUOL', 'FOUR', 'RBLX', 'U', 'DASH', 'ABNB', 'LYFT', 'UBER',
    
    # Healthcare
    'TDOC', 'DOCS', 'VEEV', 'DXCM', 'ISRG', 'PODD', 'ALGN', 'ZBH', 'SYK', 'TMO'
]

# Reference market data
REFERENCE_TICKERS = {
    'SPY': 'S&P 500 ETF',
    'QQQ': 'Nasdaq 100 ETF',
    'IWM': 'Russell 2000 ETF',
    '^VIX': 'CBOE Volatility Index',
    'TLT': '20+ Year Treasury ETF',
    'GLD': 'Gold ETF',
    'DXY': 'US Dollar Index'
}

# Sector ETFs for correlation analysis
SECTOR_ETFS = {
    'XLK': 'Technology',
    'XLV': 'Healthcare',
    'XLF': 'Financials',
    'XLE': 'Energy',
    'XLI': 'Industrials',
    'XLC': 'Communication Services',
    'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples',
    'XLRE': 'Real Estate',
    'XLB': 'Materials',
    'XLU': 'Utilities'
}

# Ticker to sector mapping (manual for now, can be automated later)
TICKER_SECTOR_MAP = {
    # Tech
    'AAPL': 'XLK', 'MSFT': 'XLK', 'GOOGL': 'XLK', 'META': 'XLK', 'NVDA': 'XLK',
    'AMD': 'XLK', 'AVGO': 'XLK', 'ORCL': 'XLK', 'ADBE': 'XLK', 'CRM': 'XLK',
    'PLTR': 'XLK', 'SNOW': 'XLK', 'DDOG': 'XLK', 'NET': 'XLK', 'CRWD': 'XLK',
    
    # Semiconductors
    'INTC': 'XLK', 'QCOM': 'XLK', 'MU': 'XLK', 'AMAT': 'XLK', 'LRCX': 'XLK',
    
    # Healthcare/Biotech
    'VKTX': 'XLV', 'NTLA': 'XLV', 'BEAM': 'XLV', 'CRSP': 'XLV', 'EDIT': 'XLV',
    'MRNA': 'XLV', 'BNTX': 'XLV', 'GILD': 'XLV', 'TDOC': 'XLV', 'DOCS': 'XLV',
    'VEEV': 'XLV', 'DXCM': 'XLV', 'ISRG': 'XLV', 'PODD': 'XLV',
    
    # Energy
    'FLNC': 'XLE', 'BE': 'XLE', 'ENPH': 'XLE', 'PLUG': 'XLE', 'FCEL': 'XLE',
    'NEE': 'XLU',
    
    # Financials
    'COIN': 'XLF', 'HOOD': 'XLF', 'SOFI': 'XLF', 'UPST': 'XLF', 'AFRM': 'XLF',
    'SQ': 'XLF', 'PYPL': 'XLF',
    
    # Consumer Discretionary
    'TSLA': 'XLY', 'CELH': 'XLY', 'ONON': 'XLY', 'DUOL': 'XLY', 'RBLX': 'XLY',
    'DASH': 'XLY', 'ABNB': 'XLY', 'LYFT': 'XLY', 'UBER': 'XLY',
    
    # Default to XLK for unclassified
}

# === DATA VALIDATION ===
# Quality thresholds
MIN_TRADING_DAYS = 252  # At least 1 year of data
MAX_MISSING_DAYS_PCT = 5  # Max 5% missing days
MAX_CONSECUTIVE_MISSING = 5  # Max 5 consecutive missing days
MIN_VOLUME_THRESHOLD = 100_000  # Minimum average daily volume

# Outlier detection
MAX_DAILY_RETURN_PCT = 50  # Flag returns > 50% as potential errors
MIN_PRICE = 0.01  # Flag prices below $0.01

# === FILE MANAGEMENT ===
# Parquet settings
PARQUET_COMPRESSION = 'snappy'  # Fast compression
PARQUET_ENGINE = 'pyarrow'

# File naming
DATE_FORMAT = '%Y%m%d'
TIMESTAMP_FORMAT = '%Y%m%d_%H%M%S'

# Retention
MAX_RAW_FILE_DAYS = 90  # Keep raw data for 90 days
MAX_PROCESSED_FILE_DAYS = 365  # Keep processed data for 1 year

# === LOGGING ===
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = PROJECT_ROOT / 'logs' / 'data_pipeline.log'
LOG_FILE.parent.mkdir(exist_ok=True)
