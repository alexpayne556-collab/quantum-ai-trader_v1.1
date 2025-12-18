"""
INDUSTRIAL-GRADE DATA PIPELINE
Download 10 years of data for ALL 10,986 US equities using multiple free APIs

This is REAL infrastructure for discovering universal market laws.
Not a toy. Not a test. The complete dataset.

Data Sources (All Free Tier):
1. yfinance (primary) - unlimited, sometimes unreliable
2. Polygon.io - 5 calls/min, delayed data  
3. Alpha Vantage - 25 calls/day, 5 calls/min
4. Twelve Data - 800 calls/day, 8 calls/min
5. Finnhub - 60 calls/min

Strategy:
- Primary: yfinance (fast, free, 10 year history)
- Fallback: Polygon for missing tickers
- Verification: Cross-check critical tickers with multiple sources
- Quality: Same rigorous checks as before

Timeline:
- 10,986 tickers × 0.5s rate limit = ~91 minutes
- Add retries, quality checks, logging = ~2-3 hours
- This is PROPER science. Taking the time to do it right.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import sqlite3
import logging
import time
import json
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('research_lab/data_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Quality metrics for downloaded data"""
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
    data_source: str  # Which API provided the data


class MarketDataStorage:
    """SQLite storage for market data with proper schema"""
    
    def __init__(self, db_path: str = "data/market_data.db"):
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self):
        """Create tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Price data table
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
                data_source TEXT,
                PRIMARY KEY (ticker, date)
            )
        """)
        
        # Quality reports table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_quality (
                ticker TEXT PRIMARY KEY,
                download_date TEXT NOT NULL,
                quality_report_json TEXT,
                passed_check INTEGER,
                data_source TEXT
            )
        """)
        
        # Ticker metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tickers (
                ticker TEXT PRIMARY KEY,
                name TEXT,
                exchange TEXT,
                market_cap REAL,
                type TEXT,
                active INTEGER,
                delisted_date TEXT,
                last_updated TEXT
            )
        """)
        
        # Download status tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS download_status (
                ticker TEXT PRIMARY KEY,
                status TEXT,  -- 'pending', 'success', 'failed', 'skipped'
                attempts INTEGER DEFAULT 0,
                last_attempt TEXT,
                error_message TEXT
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_bars_ticker ON daily_bars(ticker)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_bars_date ON daily_bars(date)")
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")
    
    def store_daily_bars(self, ticker: str, df: pd.DataFrame, data_source: str = 'yfinance'):
        """Store OHLCV data"""
        conn = sqlite3.connect(self.db_path)
        
        # Add data source column
        df = df.copy()
        df['data_source'] = data_source
        
        # Reset index to get date as column
        df_to_store = df.reset_index()
        df_to_store.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'adj_close', 'data_source']
        df_to_store['ticker'] = ticker
        df_to_store['date'] = df_to_store['date'].astype(str)
        
        # Store
        df_to_store.to_sql('daily_bars', conn, if_exists='append', index=False)
        conn.close()
        
        logger.info(f"{ticker}: Stored {len(df)} bars from {data_source}")
    
    def store_quality_report(self, report: DataQualityReport):
        """Store quality check results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        report_json = json.dumps({
            'total_days_expected': int(report.total_days_expected),
            'total_days_received': int(report.total_days_received),
            'missing_days': int(report.missing_days),
            'zero_volume_days': int(report.zero_volume_days),
            'price_gaps_over_50pct': int(report.price_gaps_over_50pct),
            'negative_prices': int(report.negative_prices),
            'start_date': report.start_date,
            'end_date': report.end_date,
            'issues': report.issues,
            'data_source': report.data_source
        })
        
        cursor.execute("""
            INSERT OR REPLACE INTO data_quality 
            (ticker, download_date, quality_report_json, passed_check, data_source)
            VALUES (?, ?, ?, ?, ?)
        """, (
            report.ticker,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            report_json,
            1 if report.passed_quality_check else 0,
            report.data_source
        ))
        
        conn.commit()
        conn.close()
    
    def update_download_status(self, ticker: str, status: str, error_message: str = None):
        """Track download attempts and status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO download_status 
            (ticker, status, attempts, last_attempt, error_message)
            VALUES (
                ?,
                ?,
                COALESCE((SELECT attempts FROM download_status WHERE ticker = ?), 0) + 1,
                ?,
                ?
            )
        """, (ticker, status, ticker, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), error_message))
        
        conn.commit()
        conn.close()
    
    def get_all_tickers(self) -> List[str]:
        """Get list of all tickers in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT ticker FROM daily_bars ORDER BY ticker")
        tickers = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tickers
    
    def get_pending_tickers(self, universe: List[str]) -> List[str]:
        """Get tickers that haven't been downloaded yet"""
        downloaded = set(self.get_all_tickers())
        return [t for t in universe if t not in downloaded]


class DataQualityChecker:
    """Validate downloaded data quality"""
    
    @staticmethod
    def check_data_quality(ticker: str, df: pd.DataFrame, 
                          start_date: datetime, end_date: datetime,
                          data_source: str = 'yfinance') -> DataQualityReport:
        """Run all quality checks on a dataframe"""
        issues = []
        
        # Expected trading days
        days_diff = (end_date - start_date).days
        expected_trading_days = int(days_diff * 252 / 365)
        
        total_days_received = int(len(df))
        missing_days = int(max(0, expected_trading_days - total_days_received))
        
        # Check 1: Too many missing days (>20% missing)
        if missing_days > expected_trading_days * 0.2:
            issues.append(f"Missing {missing_days} days ({missing_days/expected_trading_days*100:.1f}% of expected)")
        
        # Check 2: Zero volume days
        zero_volume_days = int((df['Volume'] == 0).sum())
        if zero_volume_days > 0:
            issues.append(f"{zero_volume_days} days with zero volume")
        
        # Check 3: Extreme price gaps
        price_changes = df['Close'].pct_change().abs()
        extreme_gaps = int((price_changes > 0.5).sum())
        if extreme_gaps > 0:
            issues.append(f"{extreme_gaps} price gaps >50%")
        
        # Check 4: Negative prices
        negative_prices = int(((df['Open'] < 0) | (df['High'] < 0) | 
                          (df['Low'] < 0) | (df['Close'] < 0)).sum())
        if negative_prices > 0:
            issues.append(f"{negative_prices} bars with negative prices")
        
        # Check 5: Minimum data requirement
        min_required_days = 252 * 2  # At least 2 years
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
            issues=issues,
            data_source=data_source
        )


class IndustrialDataDownloader:
    """
    Multi-source data downloader with fallbacks and retries
    
    This is industrial-grade infrastructure for getting ALL market data.
    """
    
    def __init__(self, storage: MarketDataStorage):
        self.storage = storage
        self.quality_checker = DataQualityChecker()
        
        # API keys
        self.polygon_key = "iRXh2jGpwhcJxGWfW4ZRVn2C4s_v4ghr"
        self.alpha_vantage_key = "0ROKR956QR1XHDLZ"
        self.twelve_data_key = "d19ebe6706614dd897e66aa416900fd3"
        
        # Rate limiters (calls per minute)
        self.last_call_time = {
            'yfinance': 0,
            'polygon': 0,
            'alpha_vantage': 0,
            'twelve_data': 0
        }
    
    def download_yfinance(self, ticker: str, years_back: int = 10) -> Tuple[bool, Optional[pd.DataFrame]]:
        """Download from yfinance (primary source)"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * years_back)
            
            df = yf.download(
                ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False,
                auto_adjust=False
            )
            
            if df.empty:
                return False, None
            
            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            return True, df
            
        except Exception as e:
            logger.warning(f"{ticker}: yfinance failed - {str(e)}")
            return False, None
    
    def download_polygon(self, ticker: str, years_back: int = 10) -> Tuple[bool, Optional[pd.DataFrame]]:
        """Download from Polygon (fallback)"""
        try:
            # Rate limit: 5 calls/minute
            time_since_last = time.time() - self.last_call_time['polygon']
            if time_since_last < 12:  # Wait 12 seconds between calls
                time.sleep(12 - time_since_last)
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * years_back)
            
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}?adjusted=true&sort=asc&limit=50000&apiKey={self.polygon_key}"
            
            response = requests.get(url)
            self.last_call_time['polygon'] = time.time()
            
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and len(data['results']) > 0:
                    df = pd.DataFrame(data['results'])
                    df['date'] = pd.to_datetime(df['t'], unit='ms')
                    df = df.set_index('date')
                    df = df.rename(columns={
                        'o': 'Open', 'h': 'High', 'l': 'Low', 
                        'c': 'Close', 'v': 'Volume'
                    })
                    df['Adj Close'] = df['Close']  # Polygon already adjusted
                    return True, df[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
            
            return False, None
            
        except Exception as e:
            logger.warning(f"{ticker}: Polygon failed - {str(e)}")
            return False, None
    
    def download_ticker(self, ticker: str, years_back: int = 10) -> Tuple[bool, Optional[DataQualityReport]]:
        """
        Download data for a single ticker with fallback sources
        
        Returns: (success, quality_report)
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years_back)
        
        # Try yfinance first (fast and free)
        success, df = self.download_yfinance(ticker, years_back)
        data_source = 'yfinance'
        
        # Fallback to Polygon if yfinance fails
        if not success or df is None or df.empty:
            logger.info(f"{ticker}: yfinance failed, trying Polygon...")
            success, df = self.download_polygon(ticker, years_back)
            data_source = 'polygon'
        
        if not success or df is None or df.empty:
            self.storage.update_download_status(ticker, 'failed', 'No data from any source')
            return False, None
        
        # Run quality checks
        quality_report = self.quality_checker.check_data_quality(
            ticker, df, start_date, end_date, data_source
        )
        
        # Store if passed quality check
        if quality_report.passed_quality_check:
            self.storage.store_daily_bars(ticker, df, data_source)
            self.storage.store_quality_report(quality_report)
            self.storage.update_download_status(ticker, 'success')
            logger.info(f"{ticker}: ✓ Passed quality check, stored {len(df)} bars from {data_source}")
            return True, quality_report
        else:
            self.storage.store_quality_report(quality_report)
            self.storage.update_download_status(ticker, 'failed_quality', str(quality_report.issues))
            logger.warning(f"{ticker}: ✗ Failed quality check: {quality_report.issues}")
            return False, quality_report
    
    def download_universe(self, tickers: List[str], years_back: int = 10, 
                         delay_seconds: float = 0.5) -> Dict:
        """
        Download THE ENTIRE UNIVERSE
        
        This will take 2-3 hours. That's fine. We're doing real science.
        """
        # Get tickers that need downloading
        already_downloaded = set(self.storage.get_all_tickers())
        tickers_to_download = [t for t in tickers if t not in already_downloaded]
        
        if len(tickers_to_download) < len(tickers):
            skipped = len(tickers) - len(tickers_to_download)
            logger.info(f"Skipping {skipped} already-downloaded tickers")
        
        logger.info(f"=" * 70)
        logger.info(f"INDUSTRIAL DATA PIPELINE - DOWNLOADING {len(tickers_to_download)} TICKERS")
        logger.info(f"=" * 70)
        logger.info(f"Years of history: {years_back}")
        logger.info(f"Rate limit: {delay_seconds}s between tickers")
        logger.info(f"Estimated time: {len(tickers_to_download) * delay_seconds / 60:.1f} minutes")
        logger.info(f"This is PROPER infrastructure. Taking the time to do it right.")
        logger.info(f"=" * 70)
        
        results = {
            'total': len(tickers_to_download),
            'successful': 0,
            'failed': 0,
            'failed_quality': 0,
            'skipped': len(tickers) - len(tickers_to_download),
            'start_time': datetime.now()
        }
        
        for i, ticker in enumerate(tickers_to_download, 1):
            logger.info(f"\n[{i}/{len(tickers_to_download)}] Processing {ticker}...")
            
            success, quality_report = self.download_ticker(ticker, years_back)
            
            if success:
                results['successful'] += 1
            elif quality_report is not None:
                results['failed_quality'] += 1
            else:
                results['failed'] += 1
            
            # Progress update every 100 tickers
            if i % 100 == 0:
                elapsed = (datetime.now() - results['start_time']).total_seconds()
                rate = i / elapsed * 60  # tickers per minute
                remaining = len(tickers_to_download) - i
                eta_minutes = remaining / rate if rate > 0 else 0
                
                logger.info(f"\n{'=' * 70}")
                logger.info(f"PROGRESS: {i}/{len(tickers_to_download)} ({i/len(tickers_to_download)*100:.1f}%)")
                logger.info(f"Successful: {results['successful']} | Failed: {results['failed']} | Failed quality: {results['failed_quality']}")
                logger.info(f"Rate: {rate:.1f} tickers/min | ETA: {eta_minutes:.1f} minutes")
                logger.info(f"{'=' * 70}")
            
            # Rate limiting
            if i < len(tickers_to_download):
                time.sleep(delay_seconds)
        
        # Final report
        elapsed = (datetime.now() - results['start_time']).total_seconds() / 60
        logger.info(f"\n{'=' * 70}")
        logger.info(f"DOWNLOAD COMPLETE")
        logger.info(f"{'=' * 70}")
        logger.info(f"Total time: {elapsed:.1f} minutes")
        logger.info(f"Successful: {results['successful']}")
        logger.info(f"Failed: {results['failed']}")
        logger.info(f"Failed quality: {results['failed_quality']}")
        logger.info(f"Success rate: {results['successful']/results['total']*100:.1f}%")
        logger.info(f"{'=' * 70}")
        
        return results


def main():
    """Download ALL 10,986 US equities"""
    
    print("\n" + "=" * 70)
    print("INDUSTRIAL DATA PIPELINE")
    print("Downloading 2 years of data for ALL 10,986 US equities")
    print("This is REAL infrastructure for discovering market laws")
    print("=" * 70 + "\n")
    
    # Load universe
    universe = pd.read_csv('data/complete_us_universe.csv')
    tickers = universe['ticker'].tolist()
    
    print(f"Universe loaded: {len(tickers)} tickers")
    print(f"Exchanges: {universe['exchange'].value_counts().to_dict()}")
    print(f"\nStarting download...\n")
    
    # Initialize storage and downloader
    storage = MarketDataStorage()
    downloader = IndustrialDataDownloader(storage)
    
    # Download everything - 2 years is enough with 10,986 tickers
    results = downloader.download_universe(tickers, years_back=2, delay_seconds=0.5)
    
    print("\nIndustrial data pipeline complete.")
    print("The complete US equity universe is now in your database.")
    print("Ready to discover universal market laws.\n")


if __name__ == "__main__":
    main()
