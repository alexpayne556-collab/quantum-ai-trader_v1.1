import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class Config:
    # API Keys
    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "d3qj8p9r01quv7kb49igd3qj8p9r01quv7kb49j0")
    NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY", "e6f793dfd61f473786f69466f9313fe8")
    FMP_API_KEY = os.getenv("FMP_API_KEY", "15zYYtksuJnQsTBODSNs3MrfEedOSd3i")
    ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "9OS7LP4D495FW43S")
    
    # Trading Parameters
    INITIAL_CAPITAL = 500.0  # Your $500 account
    RISK_PER_TRADE = 0.02    # 2% risk per trade
    MAX_POSITION_SIZE = 0.1  # Max 10% of capital per trade
    
    # Database
    DB_PATH = Path("data/trading.db")
    
    # API Rate Limits (requests per minute)
    FINNHUB_RATE_LIMIT = 60
    NEWSAPI_RATE_LIMIT = 100
    FMP_RATE_LIMIT = 250
    
    # Symbols to monitor
    DEFAULT_SYMBOLS = ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'AMD', 'SPY', 'QQQ']
    
    # Timeframes
    TIMEFRAMES = ['1min', '5min', '15min', '1h', '1d']
    
    @classmethod
    def validate(cls):
        """Validate all required configurations are present"""
        required_keys = ["FINNHUB_API_KEY", "FMP_API_KEY", "NEWSAPI_API_KEY"]
        missing = [key for key in required_keys if not getattr(cls, key)]
        
        if missing:
            raise ValueError(f"❌ Missing API keys: {missing}")
        
        # Create necessary directories
        cls.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        print("✅ Configuration validated successfully")
        return True

# Global config instance
config = Config()
