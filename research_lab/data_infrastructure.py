"""
Data Infrastructure for Quantitative Research
============================================

Purpose: Download, clean, and store historical market data for rigorous research

Features:
- Downloads 10 years of daily OHLCV data
- Handles missing data, bad prices, corporate actions
- Stores in SQLite for fast queries
- Implements data quality checks
- Tracks survivorship bias
- Provides clean API for research

Author: Built Dec 18, 2025
"""

import pandas as pd
import numpy as np
import yfinance as yf
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
import logging
from dataclasses import dataclass
import json
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Report on data quality issues"""
    ticker: str
    total_days_expected: int
    total_days_received: int
    missing_days: int
    zero_volume_days: int
    price_gaps_over_50pct: int
    negative_prices: int
    start_date: str
    end_date: str
    passed_quality_check: bool
    issues: List[str]


class MarketDataStorage:
    """
    Handles storage and retrieval of market data using SQLite
    
    Schema:
    - daily_bars: ticker, date, open, high, low, close, volume, adj_close
    - data_quality: ticker, download_date, quality_report_json
    - tickers: ticker, name, sector, market_cap, active (for survivorship tracking)
    """
    
    def __init__(self, db_path: str = "data/market_data.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Create tables if they don't exist"""
        self.conn = sqlite3.connect(str(self.db_path))
        cursor = self.conn.cursor()
        
        # Daily bars table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_bars (
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                adj_close REAL,
                PRIMARY KEY (ticker, date)
            )
        """)
        
        # Data quality table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_quality (
                ticker TEXT PRIMARY KEY,
                download_date TEXT,
                quality_report TEXT,
                passed_check INTEGER
            )
        """)
        
        # Tickers table (for survivorship tracking)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tickers (
                ticker TEXT PRIMARY KEY,
                name TEXT,
                sector TEXT,
                market_cap REAL,
                active INTEGER DEFAULT 1,
                delisted_date TEXT,
                notes TEXT
            )
        """)
        
        # Create indexes for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_daily_bars_ticker 
            ON daily_bars(ticker)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_daily_bars_date 
            ON daily_bars(date)
        """)
        
        self.conn.commit()
        logger.info(f"Database initialized at {self.db_path}")
    
    def store_daily_bars(self, ticker: str, df: pd.DataFrame):
        """Store daily bars for a ticker"""
        if df.empty:
            logger.warning(f"Empty dataframe for {ticker}, skipping storage")
            return
        
        # Prepare data
        df = df.copy()
        df['ticker'] = ticker
        df = df.reset_index()
        df['date'] = df['Date'].astype(str) if 'Date' in df.columns else df.index.astype(str)
        
        # Select columns in correct order
        columns = ['ticker', 'date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        df_to_store = df[columns].copy()
        df_to_store.columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
        
        # Store in database (replace if exists)
        df_to_store.to_sql('daily_bars', self.conn, if_exists='append', index=False)
        self.conn.commit()
        
        logger.info(f"Stored {len(df_to_store)} bars for {ticker}")
    
    def store_quality_report(self, report: DataQualityReport):
        """Store data quality report"""
        cursor = self.conn.cursor()
        
        # Convert numpy int64 to Python int for JSON serialization
        report_json = json.dumps({
            'total_days_expected': int(report.total_days_expected),
            'total_days_received': int(report.total_days_received),
            'missing_days': int(report.missing_days),
            'zero_volume_days': int(report.zero_volume_days),
            'price_gaps_over_50pct': int(report.price_gaps_over_50pct),
            'negative_prices': int(report.negative_prices),
            'start_date': report.start_date,
            'end_date': report.end_date,
            'issues': report.issues
        })
        
        cursor.execute("""
            INSERT OR REPLACE INTO data_quality 
            (ticker, download_date, quality_report, passed_check)
            VALUES (?, ?, ?, ?)
        """, (
            report.ticker,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            report_json,
            1 if report.passed_quality_check else 0
        ))
        
        self.conn.commit()
    
    def get_daily_bars(self, ticker: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Retrieve daily bars for a ticker"""
        query = "SELECT * FROM daily_bars WHERE ticker = ?"
        params = [ticker]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date"
        
        df = pd.read_sql_query(query, self.conn, params=params)
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        
        return df
    
    def get_all_tickers(self, active_only: bool = True) -> List[str]:
        """Get list of all tickers in database"""
        query = "SELECT DISTINCT ticker FROM daily_bars"
        df = pd.read_sql_query(query, self.conn)
        return df['ticker'].tolist()
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


class DataQualityChecker:
    """
    Performs quality checks on downloaded data
    
    Checks:
    1. Missing days (compare to trading calendar)
    2. Zero volume days (potential bad data)
    3. Extreme price gaps (>50% jumps, potential splits not adjusted)
    4. Negative prices (data error)
    5. Sufficient history (need minimum data for research)
    """
    
    @staticmethod
    def check_data_quality(ticker: str, df: pd.DataFrame, 
                          start_date: datetime, end_date: datetime) -> DataQualityReport:
        """
        Run all quality checks on a dataframe
        
        Returns DataQualityReport with pass/fail and issues
        """
        issues = []
        
        # Expected trading days (approximately)
        days_diff = (end_date - start_date).days
        expected_trading_days = int(days_diff * 252 / 365)  # ~252 trading days per year
        
        total_days_received = int(len(df))  # Convert to Python int
        missing_days = int(max(0, expected_trading_days - total_days_received))
        
        # Check 1: Too many missing days (>20% missing)
        if missing_days > expected_trading_days * 0.2:
            issues.append(f"Missing {missing_days} days ({missing_days/expected_trading_days*100:.1f}% of expected)")
        
        # Check 2: Zero volume days
        zero_volume_days = int((df['Volume'] == 0).sum())  # Convert to Python int
        if zero_volume_days > 0:
            issues.append(f"{zero_volume_days} days with zero volume")
        
        # Check 3: Extreme price gaps (potential unadjusted splits)
        price_changes = df['Close'].pct_change().abs()
        extreme_gaps = int((price_changes > 0.5).sum())  # Convert to Python int
        if extreme_gaps > 0:
            issues.append(f"{extreme_gaps} price gaps >50%")
        
        # Check 4: Negative prices (data error)
        negative_prices = int(((df['Open'] < 0) | (df['High'] < 0) | 
                          (df['Low'] < 0) | (df['Close'] < 0)).sum())  # Convert to Python int
        if negative_prices > 0:
            issues.append(f"{negative_prices} bars with negative prices")
        
        # Check 5: Minimum data requirement
        min_required_days = 252 * 2  # Need at least 2 years
        if total_days_received < min_required_days:
            issues.append(f"Insufficient data: {total_days_received} days (need {min_required_days})")
        
        # Overall pass/fail
        critical_issues = ['negative prices', 'Insufficient data']
        has_critical_issue = any(issue for issue in issues 
                                if any(crit in issue for crit in critical_issues))
        
        passed = not has_critical_issue and missing_days < expected_trading_days * 0.3
        
        return DataQualityReport(
            ticker=ticker,
            total_days_expected=int(expected_trading_days),
            total_days_received=total_days_received,
            missing_days=missing_days,
            zero_volume_days=zero_volume_days,
            price_gaps_over_50pct=extreme_gaps,
            negative_prices=negative_prices,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            passed_quality_check=passed,
            issues=issues
        )


class MarketDataDownloader:
    """
    Downloads historical market data with proper error handling and quality checks
    """
    
    def __init__(self, storage: MarketDataStorage):
        self.storage = storage
        self.quality_checker = DataQualityChecker()
    
    def download_ticker(self, ticker: str, years_back: int = 10) -> Tuple[bool, Optional[DataQualityReport]]:
        """
        Download data for a single ticker
        
        Returns: (success, quality_report)
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years_back)
        
        logger.info(f"Downloading {ticker}: {start_date.date()} to {end_date.date()}")
        
        try:
            # Download from yfinance
            df = yf.download(
                ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False,
                auto_adjust=False  # Keep unadjusted and adjusted prices separate
            )
            
            if df.empty:
                logger.warning(f"{ticker}: No data returned")
                return False, None
            
            # Handle MultiIndex columns (yfinance sometimes returns this)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Run quality checks
            quality_report = self.quality_checker.check_data_quality(
                ticker, df, start_date, end_date
            )
            
            # Store if passed quality check
            if quality_report.passed_quality_check:
                self.storage.store_daily_bars(ticker, df)
                self.storage.store_quality_report(quality_report)
                logger.info(f"{ticker}: ✓ Passed quality check, stored {len(df)} bars")
                return True, quality_report
            else:
                self.storage.store_quality_report(quality_report)
                logger.warning(f"{ticker}: ✗ Failed quality check: {quality_report.issues}")
                return False, quality_report
        
        except Exception as e:
            logger.error(f"{ticker}: Error downloading - {str(e)}")
            return False, None
    
    def download_universe(self, tickers: List[str], years_back: int = 10, 
                         delay_seconds: float = 0.5) -> Dict[str, any]:
        """
        Download data for entire universe of tickers
        
        Args:
            tickers: List of ticker symbols
            years_back: How many years of history to download
            delay_seconds: Delay between downloads to avoid rate limits
        
        Returns:
            Dictionary with download statistics
        """
        logger.info(f"Starting download of {len(tickers)} tickers, {years_back} years history")
        logger.info(f"Rate limit: {delay_seconds}s delay between tickers")
        
        results = {
            'total': len(tickers),
            'successful': 0,
            'failed': 0,
            'failed_quality': 0,
            'tickers_passed': [],
            'tickers_failed': [],
            'tickers_failed_quality': [],
            'start_time': datetime.now()
        }
        
        for i, ticker in enumerate(tickers, 1):
            logger.info(f"\n[{i}/{len(tickers)}] Processing {ticker}...")
            
            success, quality_report = self.download_ticker(ticker, years_back)
            
            if success:
                results['successful'] += 1
                results['tickers_passed'].append(ticker)
            elif quality_report is not None:
                results['failed_quality'] += 1
                results['tickers_failed_quality'].append(ticker)
            else:
                results['failed'] += 1
                results['tickers_failed'].append(ticker)
            
            # Progress update every 50 tickers
            if i % 50 == 0:
                elapsed = (datetime.now() - results['start_time']).total_seconds()
                rate = i / elapsed * 60  # tickers per minute
                remaining = len(tickers) - i
                eta_minutes = remaining / rate if rate > 0 else 0
                
                logger.info(f"\n{'='*60}")
                logger.info(f"Progress: {i}/{len(tickers)} ({i/len(tickers)*100:.1f}%)")
                logger.info(f"Successful: {results['successful']}, Failed: {results['failed']}, Failed quality: {results['failed_quality']}")
                logger.info(f"Rate: {rate:.1f} tickers/min, ETA: {eta_minutes:.1f} minutes")
                logger.info(f"{'='*60}\n")
            
            # Rate limiting
            time.sleep(delay_seconds)
        
        results['end_time'] = datetime.now()
        results['duration_seconds'] = (results['end_time'] - results['start_time']).total_seconds()
        
        return results


def main():
    """Main execution for building data infrastructure"""
    
    print("\n" + "="*80)
    print("MARKET DATA INFRASTRUCTURE - INITIAL BUILD")
    print("="*80)
    print("\nThis will:")
    print("1. Download 10 years of daily data for 300+ tickers")
    print("2. Run quality checks on each ticker")
    print("3. Store clean data in SQLite database")
    print("4. Generate quality report")
    print("\nEstimated time: 30-60 minutes (with rate limiting)")
    print("="*80 + "\n")
    
    # Load ticker universe
    ticker_file = Path("data/ticker_universe_300.csv")
    if not ticker_file.exists():
        print(f"ERROR: Ticker universe file not found at {ticker_file}")
        print("Using fallback list of liquid tickers...")
        tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
            'JPM', 'JNJ', 'V', 'WMT', 'PG', 'MA', 'UNH', 'HD', 'BAC', 'XOM',
            'DIS', 'ADBE', 'NFLX', 'COST', 'CRM', 'PEP', 'CSCO', 'ABT', 'TMO',
            'AVGO', 'ACN', 'MRK', 'NKE', 'TXN', 'LIN', 'ORCL', 'MDT', 'DHR'
        ]
    else:
        df = pd.read_csv(ticker_file)
        tickers = df['ticker'].tolist()
    
    print(f"Ticker universe: {len(tickers)} tickers\n")
    
    # Initialize infrastructure
    storage = MarketDataStorage()
    downloader = MarketDataDownloader(storage)
    
    # Download data
    results = downloader.download_universe(tickers, years_back=10, delay_seconds=0.5)
    
    # Print final report
    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE - FINAL REPORT")
    print("="*80)
    print(f"\nTotal tickers processed: {results['total']}")
    print(f"Successful: {results['successful']} ({results['successful']/results['total']*100:.1f}%)")
    print(f"Failed (no data): {results['failed']}")
    print(f"Failed (quality): {results['failed_quality']}")
    print(f"\nDuration: {results['duration_seconds']/60:.1f} minutes")
    print(f"Database: {storage.db_path}")
    print(f"Database size: {storage.db_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    if results['tickers_failed']:
        print(f"\nFailed tickers (no data): {results['tickers_failed'][:10]}...")
    
    if results['tickers_failed_quality']:
        print(f"\nFailed quality checks: {results['tickers_failed_quality'][:10]}...")
    
    print("\n" + "="*80)
    print("Next steps:")
    print("1. Review quality report for issues")
    print("2. Investigate failed tickers")
    print("3. Use storage.get_daily_bars(ticker) to query data")
    print("="*80 + "\n")
    
    storage.close()


if __name__ == "__main__":
    main()
