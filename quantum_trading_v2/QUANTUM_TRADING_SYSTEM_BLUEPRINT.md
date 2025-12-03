# 🎯 QUANTUM TRADING SYSTEM - MASTER BLUEPRINT

## PROJECT STRUCTURE
```
quantum_trading_v2/
├── .env
├── requirements.txt
├── main.py
├── config/
│   ├── __init__.py
│   └── settings.py
├── data/
│   ├── __init__.py
│   ├── database_manager.py
│   ├── price_collector.py
│   └── news_collector.py
├── analysis/
│   ├── __init__.py
│   ├── pattern_detector.py
│   └── signal_generator.py
├── trading/
│   ├── __init__.py
│   ├── portfolio_manager.py
│   └── risk_manager.py
└── utils/
    ├── __init__.py
    └── logger.py
```

## API KEYS (.env)
```
FINNHUB_API_KEY=d3qj8p9r01quv7kb49igd3qj8p9r01quv7kb49j0
FMP_API_KEY=15zYYtksuJnQsTBODSNs3MrfEedOSd3i
NEWSAPI_API_KEY=e6f793dfd61f473786f69466f9313fe8
ALPHAVANTAGE_API_KEY=9OS7LP4D495FW43S
OPENAI_API_KEY=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
```

## DEPENDENCIES (requirements.txt)
```
pandas>=2.0.0
numpy>=1.24.0
requests>=2.28.0
python-dotenv>=1.0.0
plotly>=5.15.0
streamlit>=1.28.0
sqlite3
ta-lib>=0.4.24
scikit-learn>=1.3.0
```

## 🏗️ COMPLETE MODULE SPECIFICATIONS

### 1. CONFIG/SETTINGS.PY
```python
import os
from dotenv import load_dotenv
from pathlib import Path

class Config:
    # API Keys
    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
    NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")
    FMP_API_KEY = os.getenv("FMP_API_KEY")
    ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
    
    # Database
    DB_PATH = Path("data/trading.db")
    
    # Validation
    @classmethod
    def validate(cls):
        required = ["FINNHUB_API_KEY", "FMP_API_KEY"]
        missing = [key for key in required if not getattr(cls, key)]
        if missing:
            raise ValueError(f"Missing API keys: {missing}")
```

### 2. DATA/DATABASE_MANAGER.PY
```python
import sqlite3
from pathlib import Path

class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Price data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_data (
                id INTEGER PRIMARY KEY,
                symbol TEXT, date TEXT, open REAL, high REAL, 
                low REAL, close REAL, volume INTEGER, timeframe TEXT,
                UNIQUE(symbol, date, timeframe)
            )
        ''')
        
        # News sentiment table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_sentiment (
                id INTEGER PRIMARY KEY,
                symbol TEXT, headline TEXT, sentiment REAL,
                published_date TEXT, source TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
```

### 3. DATA/PRICE_COLLECTOR.PY
```python
import requests
import pandas as pd
from datetime import datetime, timedelta

class PriceDataCollector:
    def __init__(self, api_key, db_manager):
        self.api_key = api_key
        self.db_manager = db_manager
        self.base_url = "https://finnhub.io/api/v1"
    
    def get_historical_data(self, symbol, days=30):
        end = datetime.now()
        start = end - timedelta(days=days)
        
        url = f"{self.base_url}/stock/candle"
        params = {
            'symbol': symbol, 'resolution': 'D',
            'from': int(start.timestamp()),
            'to': int(end.timestamp()),
            'token': self.api_key
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data.get('s') == 'ok':
                return self._process_price_data(symbol, data)
        return None
```

### 4. DATA/NEWS_COLLECTOR.PY
```python
import requests
from textblob import TextBlob

class NewsDataCollector:
    def __init__(self, api_key, db_manager):
        self.api_key = api_key
        self.db_manager = db_manager
    
    def get_news_sentiment(self, symbol):
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': symbol, 'apiKey': self.api_key,
            'pageSize': 10, 'sortBy': 'publishedAt'
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            articles = response.json().get('articles', [])
            return self._analyze_sentiment(symbol, articles)
        return []
    
    def _analyze_sentiment(self, symbol, articles):
        sentiments = []
        for article in articles:
            text = f"{article['title']} {article.get('description', '')}"
            sentiment = TextBlob(text).sentiment.polarity
            sentiments.append({
                'symbol': symbol,
                'headline': article['title'],
                'sentiment': sentiment,
                'published_date': article['publishedAt'],
                'source': article['source']['name']
            })
        return sentiments
```

### 5. ANALYSIS/PATTERN_DETECTOR.PY
```python
import pandas as pd
import numpy as np

class PatternDetector:
    def detect_rsi_signals(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        signals = []
        for i, val in enumerate(rsi):
            if val < 30:
                signals.append(('BUY', 'OVERSOLD', val))
            elif val > 70:
                signals.append(('SELL', 'OVERBOUGHT', val))
        return signals
    
    def detect_macd_signals(self, prices, fast=12, slow=26, signal=9):
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal).mean()
        
        signals = []
        for i in range(1, len(macd)):
            if macd[i] > macd_signal[i] and macd[i-1] <= macd_signal[i-1]:
                signals.append(('BUY', 'MACD_CROSSOVER', macd[i]))
            elif macd[i] < macd_signal[i] and macd[i-1] >= macd_signal[i-1]:
                signals.append(('SELL', 'MACD_CROSSUNDER', macd[i]))
        return signals
```

### 6. ANALYSIS/SIGNAL_GENERATOR.PY
```python
class SignalGenerator:
    def __init__(self, pattern_detector):
        self.pattern_detector = pattern_detector
    
    def generate_signals(self, symbol, price_data, news_sentiment):
        # Technical signals
        rsi_signals = self.pattern_detector.detect_rsi_signals(price_data['close'])
        macd_signals = self.pattern_detector.detect_macd_signals(price_data['close'])
        
        # Sentiment signals
        avg_sentiment = sum([n['sentiment'] for n in news_sentiment]) / len(news_sentiment) if news_sentiment else 0
        
        # Combine signals
        signals = []
        if rsi_signals and macd_signals:
            signals.append({
                'symbol': symbol,
                'signal': 'STRONG_BUY' if avg_sentiment > 0.3 else 'BUY',
                'confidence': min(0.9, (len(rsi_signals) + len(macd_signals)) / 10),
                'reasoning': f"RSI and MACD confluence with {avg_sentiment:.2f} sentiment"
            })
        
        return signals
```

### 7. TRADING/PORTFOLIO_MANAGER.PY
```python
class PortfolioManager:
    def __init__(self, initial_capital=10000):
        self.capital = initial_capital
        self.positions = {}
        self.performance = []
    
    def calculate_position_size(self, signal_confidence, risk_per_trade=0.02):
        max_risk = self.capital * risk_per_trade
        position_size = max_risk * signal_confidence
        return min(position_size, self.capital * 0.1)  # Max 10% per trade
    
    def execute_trade(self, symbol, signal, price, confidence):
        position_size = self.calculate_position_size(confidence)
        shares = position_size / price
        
        if signal in ['BUY', 'STRONG_BUY'] and position_size > 0:
            self.positions[symbol] = {
                'shares': shares,
                'entry_price': price,
                'position_size': position_size
            }
            self.capital -= position_size
            return f"Bought {shares:.2f} shares of {symbol} at ${price:.2f}"
        
        return "No trade executed"
```

### 8. MAIN.PY
```python
from config.settings import Config
from data.database_manager import DatabaseManager
from data.price_collector import PriceDataCollector
from data.news_collector import NewsDataCollector
from analysis.pattern_detector import PatternDetector
from analysis.signal_generator import SignalGenerator
from trading.portfolio_manager import PortfolioManager

def main():
    # Initialize configuration
    Config.validate()
    
    # Initialize components
    db_manager = DatabaseManager(Config.DB_PATH)
    price_collector = PriceDataCollector(Config.FINNHUB_API_KEY, db_manager)
    news_collector = NewsDataCollector(Config.NEWSAPI_API_KEY, db_manager)
    pattern_detector = PatternDetector()
    signal_generator = SignalGenerator(pattern_detector)
    portfolio_manager = PortfolioManager()
    
    # Test symbols
    symbols = ['AAPL', 'MSFT', 'TSLA']
    
    print("🚀 Quantum Trading System Started")
    print("=" * 50)
    
    for symbol in symbols:
        print(f"\nAnalyzing {symbol}...")
        
        # Collect data
        price_data = price_collector.get_historical_data(symbol)
        news_data = news_collector.get_news_sentiment(symbol)
        
        if price_data and news_data:
            # Generate signals
            signals = signal_generator.generate_signals(symbol, price_data, news_data)
            
            for signal in signals:
                print(f"  Signal: {signal['signal']} (Confidence: {signal['confidence']:.2f})")
                print(f"  Reasoning: {signal['reasoning']}")

if __name__ == "__main__":
    main()
```
