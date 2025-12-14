import sqlite3
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                timeframe TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp, timeframe)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_sentiment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                headline TEXT NOT NULL,
                sentiment REAL NOT NULL,
                published_date TEXT NOT NULL,
                source TEXT,
                confidence REAL DEFAULT 0.0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                action TEXT NOT NULL,
                confidence REAL NOT NULL,
                price REAL NOT NULL,
                reasoning TEXT,
                timestamp TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                shares REAL NOT NULL,
                price REAL NOT NULL,
                total_value REAL NOT NULL,
                timestamp TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _get_connection(self):
        return sqlite3.connect(self.db_path)
    
    def save_price_data(self, symbol: str, data: pd.DataFrame, timeframe: str = '1d'):
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            for index, row in data.iterrows():
                cursor.execute('''
                    INSERT OR REPLACE INTO price_data 
                    (symbol, timestamp, open, high, low, close, volume, timeframe)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol, index.isoformat() if hasattr(index, 'isoformat') else str(index),
                    row['open'], row['high'], row['low'], row['close'], 
                    row.get('volume', 0), timeframe
                ))
            conn.commit()
        finally:
            conn.close()
    
    def save_news_sentiment(self, symbol: str, sentiment_data: List[Dict]):
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            for news_item in sentiment_data:
                cursor.execute('''
                    INSERT INTO news_sentiment 
                    (symbol, headline, sentiment, published_date, source, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    news_item['headline'],
                    news_item['sentiment'],
                    news_item['published_date'],
                    news_item.get('source', 'unknown'),
                    news_item.get('confidence', 0.0)
                ))
            conn.commit()
        finally:
            conn.close()
    
    def save_portfolio_transaction(self, transaction: Dict):
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO portfolio_transactions 
                (symbol, action, shares, price, total_value, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                transaction['symbol'],
                transaction['action'],
                transaction['shares'],
                transaction['price'],
                transaction['total_value'],
                transaction['timestamp'].isoformat()
            ))
            conn.commit()
        finally:
            conn.close()

db_manager = None

def get_db_manager(db_path: str = "data/trading.db"):
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager(db_path)
    return db_manager