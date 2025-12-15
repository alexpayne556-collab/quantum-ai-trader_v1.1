"""
DAY 1: TRADING SYSTEM FOUNDATION - DATABASE LAYER
==================================================

Purpose: Production-grade SQLite database for 353-ticker universe
No demos. No shortcuts. Real foundation for real money.

Schema Design:
1. tickers - Master list of 353 tickers with metadata
2. ohlcv_daily - Price/volume data from yfinance
3. fundamentals - P/E, market cap, short interest, etc.
4. sec_filings - Form 4 (insider) and 8-K (events)
5. news_sentiment - Headlines + FinBERT scores
6. volume_anomalies - Scanner alerts (vol/avg > 3.0)
7. trading_signals - All strategy signals in one place

Author: Human-AI Collective
Date: December 15, 2025
"""

import sqlite3
import pandas as pd
import os
from datetime import datetime
from pathlib import Path

# Database location
DB_PATH = '/workspaces/quantum-ai-trader_v1.1/data/trading_system.db'

# Ensure data directory exists
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

class TradingDatabase:
    """Production database for trading system. Thread-safe, transaction-based."""
    
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.conn = None
        
    def connect(self):
        """Connect to database with optimizations for bulk inserts."""
        self.conn = sqlite3.connect(self.db_path, timeout=30.0)
        self.conn.execute("PRAGMA journal_mode=WAL")  # Write-ahead logging
        self.conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
        self.conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        return self.conn
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def create_schema(self):
        """Create all tables with proper indexes."""
        cursor = self.conn.cursor()
        
        # 1. TICKERS TABLE - Master list
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tickers (
                ticker TEXT PRIMARY KEY,
                sector TEXT,
                industry TEXT,
                market_cap_category TEXT,
                notes TEXT,
                active BOOLEAN DEFAULT 1,
                added_date TEXT DEFAULT CURRENT_TIMESTAMP,
                last_updated TEXT
            )
        """)
        
        # 2. OHLCV DAILY - Price and volume data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_daily (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                adj_close REAL,
                data_source TEXT DEFAULT 'yfinance',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, date)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_ticker_date ON ohlcv_daily(ticker, date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_date ON ohlcv_daily(date)")
        
        # 3. FUNDAMENTALS - Company metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fundamentals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                market_cap REAL,
                trailing_pe REAL,
                forward_pe REAL,
                peg_ratio REAL,
                price_to_book REAL,
                price_to_sales REAL,
                short_pct REAL,
                insider_pct REAL,
                institution_pct REAL,
                beta REAL,
                data_source TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, date)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fund_ticker ON fundamentals(ticker)")
        
        # 4. SEC FILINGS - Insider trades and major events
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sec_filings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                cik TEXT,
                filing_date TEXT NOT NULL,
                form_type TEXT NOT NULL,
                description TEXT,
                accession_number TEXT,
                filing_url TEXT,
                keywords TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, filing_date, form_type, accession_number)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sec_ticker_date ON sec_filings(ticker, filing_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sec_form ON sec_filings(form_type)")
        
        # 5. NEWS SENTIMENT - Headlines + FinBERT scores
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_sentiment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                published_date TEXT NOT NULL,
                headline TEXT,
                source TEXT DEFAULT 'google_news',
                url TEXT,
                sentiment_label TEXT,
                sentiment_score REAL,
                sentiment_numeric REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_news_ticker_date ON news_sentiment(ticker, published_date)")
        
        # 6. VOLUME ANOMALIES - Scanner alerts
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS volume_anomalies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                alert_date TEXT NOT NULL,
                current_volume INTEGER,
                avg_volume_20d INTEGER,
                volume_ratio REAL,
                price_change_pct REAL,
                close_price REAL,
                alert_score REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, alert_date)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_vol_date ON volume_anomalies(alert_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_vol_score ON volume_anomalies(alert_score DESC)")
        
        # 7. TRADING SIGNALS - All strategy signals
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trading_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                signal_date TEXT NOT NULL,
                strategy TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                entry_price REAL,
                confidence_score REAL,
                metadata TEXT,
                status TEXT DEFAULT 'active',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_ticker_date ON trading_signals(ticker, signal_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_strategy ON trading_signals(strategy)")
        
        # 8. BACKTEST RESULTS - Strategy validation
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                backtest_period_start TEXT,
                backtest_period_end TEXT,
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                win_rate REAL,
                total_return_pct REAL,
                sharpe_ratio REAL,
                max_drawdown_pct REAL,
                avg_trade_return_pct REAL,
                config_json TEXT,
                passed_validation BOOLEAN,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.commit()
        print("✅ Database schema created successfully")
        print(f"📁 Location: {self.db_path}")
        
        # Show table summary
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = cursor.fetchall()
        print(f"\n📊 Tables created: {len(tables)}")
        for table in tables:
            print(f"   - {table[0]}")
    
    def load_ticker_universe(self, csv_path):
        """Load 353 tickers from CSV into database."""
        if not os.path.exists(csv_path):
            print(f"⚠️ Ticker universe file not found: {csv_path}")
            return 0
        
        df = pd.read_csv(csv_path)
        df['added_date'] = datetime.now().isoformat()
        df['last_updated'] = datetime.now().isoformat()
        df['active'] = 1
        
        # Ensure required columns exist
        required_cols = ['ticker', 'sector', 'industry', 'market_cap_category', 'notes']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 'Unknown' if col != 'notes' else ''
        
        # Insert into database
        df[required_cols + ['active', 'added_date', 'last_updated']].to_sql(
            'tickers', 
            self.conn, 
            if_exists='replace', 
            index=False
        )
        
        self.conn.commit()
        print(f"✅ Loaded {len(df)} tickers into database")
        return len(df)
    
    def get_active_tickers(self):
        """Get list of all active tickers."""
        query = "SELECT ticker FROM tickers WHERE active = 1 ORDER BY ticker"
        return pd.read_sql_query(query, self.conn)['ticker'].tolist()
    
    def insert_ohlcv_batch(self, ticker, ohlcv_df):
        """Insert OHLCV data for a ticker (handles duplicates)."""
        if ohlcv_df.empty:
            return 0
        
        # Prepare dataframe
        ohlcv_df = ohlcv_df.reset_index()
        ohlcv_df['ticker'] = ticker
        ohlcv_df['date'] = pd.to_datetime(ohlcv_df['Date']).dt.strftime('%Y-%m-%d')
        ohlcv_df['created_at'] = datetime.now().isoformat()
        
        # Rename columns to match database
        col_mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close'
        }
        ohlcv_df = ohlcv_df.rename(columns=col_mapping)
        
        # Select only needed columns
        cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close', 'created_at']
        insert_df = ohlcv_df[cols].copy()
        
        # Use executemany for better control (avoid pandas SQL column issues)
        cursor = self.conn.cursor()
        for _, row in insert_df.iterrows():
            cursor.execute("""
                INSERT OR REPLACE INTO ohlcv_daily 
                (ticker, date, open, high, low, close, volume, adj_close, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, tuple(row))
        
        self.conn.commit()
        
        return len(insert_df)
    
    def get_ohlcv(self, ticker, start_date=None, end_date=None):
        """Retrieve OHLCV data for a ticker."""
        query = "SELECT * FROM ohlcv_daily WHERE ticker = ?"
        params = [ticker]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date ASC"
        
        return pd.read_sql_query(query, self.conn, params=params)
    
    def get_latest_ohlcv_date(self, ticker):
        """Get the most recent date we have OHLCV data for a ticker."""
        query = "SELECT MAX(date) as max_date FROM ohlcv_daily WHERE ticker = ?"
        result = pd.read_sql_query(query, self.conn, params=[ticker])
        return result['max_date'].iloc[0] if not result.empty else None
    
    def insert_volume_anomaly(self, ticker, alert_date, current_vol, avg_vol_20d, 
                             volume_ratio, price_change_pct, close_price, alert_score):
        """Insert volume anomaly alert."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO volume_anomalies 
            (ticker, alert_date, current_volume, avg_volume_20d, volume_ratio, 
             price_change_pct, close_price, alert_score, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ticker, alert_date, current_vol, avg_vol_20d, volume_ratio, 
              price_change_pct, close_price, alert_score, datetime.now().isoformat()))
        self.conn.commit()
    
    def get_volume_anomalies(self, days_back=30, min_score=5.0):
        """Get recent volume anomalies above threshold score."""
        query = """
            SELECT * FROM volume_anomalies 
            WHERE alert_date >= date('now', '-{} days')
            AND alert_score >= ?
            ORDER BY alert_score DESC, alert_date DESC
        """.format(days_back)
        return pd.read_sql_query(query, self.conn, params=[min_score])
    
    def get_database_stats(self):
        """Get summary statistics for database health check."""
        cursor = self.conn.cursor()
        
        stats = {}
        
        # Ticker count
        cursor.execute("SELECT COUNT(*) FROM tickers WHERE active = 1")
        stats['active_tickers'] = cursor.fetchone()[0]
        
        # OHLCV coverage
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT ticker) as tickers_with_data,
                MIN(date) as earliest_date,
                MAX(date) as latest_date,
                COUNT(*) as total_bars
            FROM ohlcv_daily
        """)
        row = cursor.fetchone()
        stats['ohlcv_tickers'] = row[0]
        stats['ohlcv_earliest'] = row[1]
        stats['ohlcv_latest'] = row[2]
        stats['ohlcv_total_bars'] = row[3]
        
        # Volume anomalies
        cursor.execute("SELECT COUNT(*) FROM volume_anomalies")
        stats['volume_anomaly_count'] = cursor.fetchone()[0]
        
        # Recent anomalies (last 7 days)
        cursor.execute("""
            SELECT COUNT(*) FROM volume_anomalies 
            WHERE alert_date >= date('now', '-7 days')
        """)
        stats['recent_anomalies_7d'] = cursor.fetchone()[0]
        
        return stats


# ==================== INITIALIZATION SCRIPT ====================

def initialize_database(ticker_universe_path='/workspaces/quantum-ai-trader_v1.1/data/ticker_universe_300.csv'):
    """Initialize database and load ticker universe."""
    print("=" * 60)
    print("DATABASE INITIALIZATION")
    print("=" * 60)
    
    db = TradingDatabase()
    db.connect()
    
    # Create schema
    print("\n1️⃣ Creating database schema...")
    db.create_schema()
    
    # Load tickers
    print("\n2️⃣ Loading ticker universe...")
    ticker_count = db.load_ticker_universe(ticker_universe_path)
    
    # Verify
    print("\n3️⃣ Verification...")
    tickers = db.get_active_tickers()
    print(f"✅ {len(tickers)} active tickers ready for data collection")
    print(f"\nSample tickers: {tickers[:10]}")
    
    # Get stats
    stats = db.get_database_stats()
    print("\n📊 Database Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    db.close()
    print("\n✅ Database initialization complete")
    print(f"📁 Database location: {DB_PATH}")
    print("=" * 60)
    
    return DB_PATH


if __name__ == "__main__":
    # Run initialization
    db_path = initialize_database()
    print(f"\n🚀 Database ready for Day 1 data collection")
    print(f"   Next step: Run OHLCV collector on all tickers")
