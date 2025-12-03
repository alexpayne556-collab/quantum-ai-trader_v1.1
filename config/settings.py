import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class Config:
    # API Keys
    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "d3qj8p9r01quv7kb49igd3qj8p9r01quv7kb49j0")
    NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY", "e6f793dfd61f473786f69466f9313fe8")
    FMP_API_KEY = os.getenv("FMP_API_KEY", "15zYYtksuJnQsTBODSNs3MrfEedOSd3i")
    ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "9OS7LP4D495FW43S")
    
    # Trading Parameters (research-optimized for $500 account)
    INITIAL_CAPITAL = 500.0
    RISK_PER_TRADE = 0.005
    DAILY_LOSS_CAP = 0.03
    MAX_POSITION_SIZE = 0.1
    
    # Technical Indicator Parameters
    RSI_PARAMS = {
        '5min': {'period': 9, 'overbought': 75, 'oversold': 25},
        '15min': {'period': 14, 'overbought': 70, 'oversold': 30},
        '1h': {'period': 14, 'overbought': 70, 'oversold': 30},
        '4h': {'period': 14, 'overbought': 65, 'oversold': 35}
    }
    
    MACD_PARAMS = {
        '5min': {'fast': 5, 'slow': 13, 'signal': 1},
        '15min': {'fast': 5, 'slow': 15, 'signal': 9},
        '1h': {'fast': 8, 'slow': 17, 'signal': 9},
        '4h': {'fast': 12, 'slow': 26, 'signal': 9}
    }
    
    BB_PARAMS = {
        '5min': {'period': 10, 'std': 1.5},
        '15min': {'period': 15, 'std': 2.0},
        '1h': {'period': 20, 'std': 2.0}
    }
    
    # Strategy Focus
    HOLDING_PERIODS = {
        'short': (2, 5),
        'medium': (5, 21),  
        'long': (21, 63)
    }
    
    # Asset Universe - 20 Institutional Grade Tickers
    SYMBOL_UNIVERSE = [
        # Mega Cap Tech (7)
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META',
        # Growth Tech (6)
        'SNOW', 'DDOG', 'NET', 'CRWD', 'PLTR', 'AMD',
        # High Growth (4)
        'SOFI', 'UPST', 'RIVN', 'LCID',
        # Market Indices (3)
        'SPY', 'QQQ', 'IWM'
    ]
    
    # Categorized for strategy routing
    TICKER_CATEGORIES = {
        'large_cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META'],
        'mid_cap': ['SNOW', 'DDOG', 'NET', 'CRWD', 'PLTR', 'AMD'],
        'small_cap': ['SOFI', 'UPST', 'RIVN', 'LCID'],
        'etf': ['SPY', 'QQQ', 'IWM']
    }
    
    @classmethod
    def get_all_symbols(cls) -> list:
        """Get flat list of all symbols"""
        return cls.SYMBOL_UNIVERSE
    
    @classmethod
    def get_symbols_by_category(cls, category: str) -> list:
        """Get symbols by category"""
        return cls.TICKER_CATEGORIES.get(category, [])
    
    # Database
    DB_PATH = Path("data/trading.db")
    
    @classmethod
    def validate(cls):
        required_keys = ["FINNHUB_API_KEY", "NEWSAPI_API_KEY"]
        missing = [key for key in required_keys if not getattr(cls, key)]
        if missing:
            raise ValueError(f"Missing API keys: {missing}")
        cls.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        return True

config = Config()