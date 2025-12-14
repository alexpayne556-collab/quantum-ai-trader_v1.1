"""
Data Fetcher - Alpha Companion Trading System

Handles OHLCV data download from Yahoo Finance with:
- Rate limiting to avoid bans
- Retry logic for failed requests
- Data validation
- Efficient parquet storage

Usage:
    fetcher = DataFetcher()
    fetcher.fetch_ticker('AAPL')
    fetcher.fetch_all_tickers()
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time
import logging
from typing import Optional, List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .config import *
from .storage import DataStorage
from .validator import DataValidator

# Setup logging
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter to avoid API bans."""
    
    def __init__(self, requests_per_second: float = REQUESTS_PER_SECOND):
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0
    
    def wait_if_needed(self):
        """Wait if we're going too fast."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()


class DataFetcher:
    """
    Fetch OHLCV data from Yahoo Finance with rate limiting and error handling.
    
    Features:
    - Automatic rate limiting
    - Retry logic with exponential backoff
    - Data validation
    - Parquet storage
    - Progress tracking
    """
    
    def __init__(
        self,
        tickers: List[str] = None,
        start_date: str = DEFAULT_START_DATE,
        end_date: str = DEFAULT_END_DATE,
        storage_dir: Path = RAW_DATA_DIR
    ):
        """
        Initialize data fetcher.
        
        Args:
            tickers: List of ticker symbols (default: TRADING_TICKERS)
            start_date: Start date YYYY-MM-DD
            end_date: End date YYYY-MM-DD
            storage_dir: Where to save data
        """
        self.tickers = tickers or TRADING_TICKERS
        self.start_date = start_date
        self.end_date = end_date
        self.storage_dir = Path(storage_dir)
        
        self.rate_limiter = RateLimiter()
        self.storage = DataStorage()
        self.validator = DataValidator()
        
        self.success_count = 0
        self.failure_count = 0
        self.failed_tickers = []
        
        logger.info(f"DataFetcher initialized")
        logger.info(f"  Tickers: {len(self.tickers)}")
        logger.info(f"  Date range: {start_date} to {end_date}")
        logger.info(f"  Storage: {storage_dir}")
    
    def fetch_ticker(
        self,
        ticker: str,
        start_date: str = None,
        end_date: str = None,
        save: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data for a single ticker with retry logic.
        
        Args:
            ticker: Ticker symbol
            start_date: Override default start date
            end_date: Override default end date
            save: Whether to save to parquet
        
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        start = start_date or self.start_date
        end = end_date or self.end_date
        
        for attempt in range(MAX_RETRIES):
            try:
                # Rate limit
                self.rate_limiter.wait_if_needed()
                
                # Download data
                logger.debug(f"Fetching {ticker} (attempt {attempt + 1}/{MAX_RETRIES})")
                
                df = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    progress=False,
                    auto_adjust=True,
                    timeout=TIMEOUT_SECONDS
                )
                
                # Check if we got data
                if df is None or len(df) == 0:
                    logger.warning(f"{ticker}: No data returned")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY_SECONDS)
                        continue
                    return None
                
                # Flatten multi-level columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                
                # Add metadata
                df['ticker'] = ticker
                df['fetch_date'] = datetime.now()
                
                # Validate data
                is_valid, issues = self.validator.validate_ticker_data(df, ticker)
                
                if not is_valid:
                    logger.warning(f"{ticker}: Validation issues - {', '.join(issues)}")
                    # Continue anyway, but log issues
                
                # Save if requested
                if save:
                    filepath = self.storage.save_ticker_data(df, ticker, 'raw')
                    logger.info(f"✅ {ticker}: Saved {len(df)} days to {filepath.name}")
                
                self.success_count += 1
                return df
                
            except Exception as e:
                logger.error(f"{ticker}: Error on attempt {attempt + 1}: {str(e)}")
                
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    logger.error(f"❌ {ticker}: Failed after {MAX_RETRIES} attempts")
                    self.failure_count += 1
                    self.failed_tickers.append(ticker)
                    return None
    
    def fetch_all_tickers(
        self,
        parallel: bool = False,
        max_workers: int = 4
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all tickers.
        
        Args:
            parallel: Use parallel downloads (be careful with rate limiting)
            max_workers: Number of parallel workers
        
        Returns:
            Dict mapping ticker to DataFrame
        """
        logger.info(f"Fetching {len(self.tickers)} tickers...")
        logger.info(f"  Date range: {self.start_date} to {self.end_date}")
        logger.info(f"  Parallel: {parallel}")
        
        results = {}
        self.success_count = 0
        self.failure_count = 0
        self.failed_tickers = []
        
        start_time = time.time()
        
        if parallel:
            # Parallel execution (careful with rate limits)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_ticker = {
                    executor.submit(self.fetch_ticker, ticker): ticker 
                    for ticker in self.tickers
                }
                
                with tqdm(total=len(self.tickers), desc="Fetching tickers") as pbar:
                    for future in as_completed(future_to_ticker):
                        ticker = future_to_ticker[future]
                        try:
                            df = future.result()
                            if df is not None:
                                results[ticker] = df
                        except Exception as e:
                            logger.error(f"{ticker}: Exception in parallel fetch: {e}")
                        pbar.update(1)
        else:
            # Sequential execution (safer)
            for ticker in tqdm(self.tickers, desc="Fetching tickers"):
                df = self.fetch_ticker(ticker)
                if df is not None:
                    results[ticker] = df
        
        elapsed = time.time() - start_time
        
        # Summary
        logger.info("="*80)
        logger.info("FETCH SUMMARY")
        logger.info("="*80)
        logger.info(f"✅ Success: {self.success_count}/{len(self.tickers)}")
        logger.info(f"❌ Failed: {self.failure_count}/{len(self.tickers)}")
        logger.info(f"⏱️  Time: {elapsed:.1f}s ({elapsed/len(self.tickers):.1f}s per ticker)")
        
        if self.failed_tickers:
            logger.warning(f"Failed tickers: {', '.join(self.failed_tickers[:10])}")
            if len(self.failed_tickers) > 10:
                logger.warning(f"  ... and {len(self.failed_tickers) - 10} more")
        
        return results
    
    def update_ticker(self, ticker: str, last_date: str = None) -> Optional[pd.DataFrame]:
        """
        Update data for a ticker (fetch only new data since last fetch).
        
        Args:
            ticker: Ticker symbol
            last_date: Last date in existing data (auto-detect if None)
        
        Returns:
            Updated DataFrame or None
        """
        # Try to load existing data
        existing_df = self.storage.load_ticker_data(ticker, 'raw')
        
        if existing_df is not None and len(existing_df) > 0:
            # Get last date from existing data
            last_date = existing_df.index.max().strftime('%Y-%m-%d')
            logger.info(f"{ticker}: Updating from {last_date}")
            
            # Fetch new data
            new_df = self.fetch_ticker(
                ticker,
                start_date=last_date,
                end_date=self.end_date,
                save=False
            )
            
            if new_df is not None and len(new_df) > 0:
                # Combine old and new (remove duplicates)
                combined = pd.concat([existing_df, new_df])
                combined = combined[~combined.index.duplicated(keep='last')]
                combined = combined.sort_index()
                
                # Save updated data
                self.storage.save_ticker_data(combined, ticker, 'raw')
                logger.info(f"✅ {ticker}: Updated with {len(new_df)} new days")
                return combined
            else:
                logger.info(f"{ticker}: No new data available")
                return existing_df
        else:
            # No existing data, fetch full history
            logger.info(f"{ticker}: No existing data, fetching full history")
            return self.fetch_ticker(ticker)
    
    def get_summary(self) -> Dict:
        """Get summary of fetched data."""
        return {
            'total_tickers': len(self.tickers),
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'success_rate': self.success_count / len(self.tickers) if self.tickers else 0,
            'failed_tickers': self.failed_tickers
        }


if __name__ == '__main__':
    # Test the fetcher
    print("🧪 Testing DataFetcher...")
    print("="*80)
    
    # Test single ticker
    print("\n1. Testing single ticker fetch (AAPL)...")
    fetcher = DataFetcher(tickers=['AAPL'])
    df = fetcher.fetch_ticker('AAPL')
    
    if df is not None:
        print(f"   ✅ Got {len(df)} days of data")
        print(f"   Date range: {df.index.min()} to {df.index.max()}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Sample:\n{df.tail(3)}")
    else:
        print("   ❌ Failed to fetch data")
    
    # Test multiple tickers
    print("\n2. Testing multiple ticker fetch (5 tickers)...")
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META']
    fetcher = DataFetcher(tickers=test_tickers)
    results = fetcher.fetch_all_tickers(parallel=False)
    
    print(f"   ✅ Fetched {len(results)}/{len(test_tickers)} tickers")
    for ticker, df in results.items():
        print(f"   {ticker}: {len(df)} days")
    
    # Print summary
    print("\n3. Summary:")
    summary = fetcher.get_summary()
    print(f"   Success rate: {summary['success_rate']:.1%}")
    
    print("\n" + "="*80)
    print("✅ DataFetcher test complete")
