"""SQLite manager for price_data and news_sentiment tables with CRUD and validation."""
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        self.conn: Optional[sqlite3.Connection] = None
        self._ensure_tables()

    def connect(self):
        if not self.conn:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def _ensure_tables(self):
        self.connect()
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS price_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                timeframe TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS news_sentiment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                headline TEXT,
                sentiment TEXT,
                published_date TEXT,
                source TEXT
            )
        """)
        self.conn.commit()

    def insert_price(self, data: Dict[str, Any]):
        self.connect()
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO price_data (symbol, date, open, high, low, close, volume, timeframe)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data["symbol"], data["date"], data["open"], data["high"], data["low"], data["close"], data["volume"], data["timeframe"]
        ))
        self.conn.commit()

    def insert_news(self, data: Dict[str, Any]):
        self.connect()
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO news_sentiment (symbol, headline, sentiment, published_date, source)
            VALUES (?, ?, ?, ?, ?)
        """, (
            data["symbol"], data["headline"], data["sentiment"], data["published_date"], data["source"]
        ))
        self.conn.commit()

    def get_prices(self, symbol: str) -> List[Dict[str, Any]]:
        self.connect()
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM price_data WHERE symbol = ? ORDER BY date DESC", (symbol,))
        return [dict(row) for row in cur.fetchall()]

    def get_news(self, symbol: str) -> List[Dict[str, Any]]:
        self.connect()
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM news_sentiment WHERE symbol = ? ORDER BY published_date DESC", (symbol,))
        return [dict(row) for row in cur.fetchall()]

    def validate(self):
        self.connect()
        cur = self.conn.cursor()
        cur.execute("PRAGMA integrity_check;")
        result = cur.fetchone()
        return result[0] == "ok"
