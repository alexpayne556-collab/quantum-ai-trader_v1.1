"""
ROBUST DATA FETCHER - Production Ready
Features:
- yfinance primary data source (free, no API key)
- Alpha Vantage fallback (with API key rotation)
- Finnhub fallback (with API key)
- Local caching (24hr TTL to avoid rate limits)
- Retry logic with exponential backoff
- Data validation and cleaning
- Automatic MultiIndex column flattening
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
import json
from datetime import datetime, timedelta
import time
import requests
from typing import Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')


class DataFetcher:
    """Production-grade data fetcher with fallbacks and caching"""
    
    def __init__(self, cache_dir: str = 'data/cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # API keys (rotate if one hits rate limit)
        self.alpha_vantage_keys = [
            'demo',  # Free tier, limited
            # Add your keys here: 'YOUR_KEY_1', 'YOUR_KEY_2', etc.
        ]
        self.av_key_index = 0
        
        self.finnhub_keys = [
            'demo',  # Free tier
            # Add your keys here
        ]
        self.fh_key_index = 0
        
        # Cache settings
        self.cache_ttl_hours = 24
        
        # Rate limiting
        self.last_request_time = {}
        self.min_request_interval = 1.0  # seconds between requests
    
    def fetch_ohlcv(self, ticker: str, period: str = '60d', interval: str = '1d') -> Optional[pd.DataFrame]:
        """
        Main entry point - fetch OHLCV data with fallbacks
        
        Returns clean DataFrame with columns: Open, High, Low, Close, Volume (no MultiIndex)
        """
        # Check cache first
        cached_df = self._load_from_cache(ticker, period)
        if cached_df is not None:
            print(f"📦 Loaded {ticker} from cache")
            return cached_df
        
        # Try yfinance (primary, free, no API key needed)
        df = self._fetch_yfinance(ticker, period, interval)
        if df is not None:
            df = self._clean_dataframe(df)
            self._save_to_cache(ticker, period, df)
            return df
        
        # Try Alpha Vantage (fallback 1)
        print(f"⚠️  yfinance failed for {ticker}, trying Alpha Vantage...")
        df = self._fetch_alpha_vantage(ticker, period)
        if df is not None:
            df = self._clean_dataframe(df)
            self._save_to_cache(ticker, period, df)
            return df
        
        # Try Finnhub (fallback 2)
        print(f"⚠️  Alpha Vantage failed for {ticker}, trying Finnhub...")
        df = self._fetch_finnhub(ticker, period)
        if df is not None:
            df = self._clean_dataframe(df)
            self._save_to_cache(ticker, period, df)
            return df
        
        print(f"❌ All data sources failed for {ticker}")
        return None
    
    def _fetch_yfinance(self, ticker: str, period: str, interval: str, retries: int = 3) -> Optional[pd.DataFrame]:
        """Fetch from yfinance with retry logic"""
        for attempt in range(retries):
            try:
                self._rate_limit('yfinance')
                
                # Download data (auto_adjust=False to avoid MultiIndex)
                df = yf.download(ticker, period=period, interval=interval, 
                                progress=False, auto_adjust=False)
                
                if len(df) == 0:
                    print(f"⚠️  yfinance returned no data for {ticker} (attempt {attempt + 1}/{retries})")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                
                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # Validate required columns
                required = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in df.columns for col in required):
                    print(f"⚠️  Missing columns in {ticker} data: {df.columns.tolist()}")
                    continue
                
                print(f"✅ yfinance: {ticker} - {len(df)} rows")
                return df[required]
            
            except Exception as e:
                print(f"⚠️  yfinance error for {ticker} (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                continue
        
        return None
    
    def _fetch_alpha_vantage(self, ticker: str, period: str) -> Optional[pd.DataFrame]:
        """Fetch from Alpha Vantage API"""
        try:
            self._rate_limit('alpha_vantage')
            
            api_key = self.alpha_vantage_keys[self.av_key_index]
            
            # Determine outputsize based on period
            outputsize = 'full' if '2y' in period or '5y' in period else 'compact'
            
            url = f'https://www.alphavantage.co/query'
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': ticker,
                'apikey': api_key,
                'outputsize': outputsize,
                'datatype': 'json'
            }
            
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            if 'Time Series (Daily)' not in data:
                print(f"⚠️  Alpha Vantage: No data for {ticker}")
                # Rotate API key if rate limited
                if 'Note' in data or 'Information' in data:
                    self.av_key_index = (self.av_key_index + 1) % len(self.alpha_vantage_keys)
                return None
            
            # Convert to DataFrame
            time_series = data['Time Series (Daily)']
            df = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Rename columns
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df.astype(float)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Trim to period
            days = int(period.replace('d', '').replace('y', ''))
            if 'y' in period:
                days *= 365
            df = df.tail(days)
            
            print(f"✅ Alpha Vantage: {ticker} - {len(df)} rows")
            return df
        
        except Exception as e:
            print(f"⚠️  Alpha Vantage error: {e}")
            return None
    
    def _fetch_finnhub(self, ticker: str, period: str) -> Optional[pd.DataFrame]:
        """Fetch from Finnhub API"""
        try:
            self._rate_limit('finnhub')
            
            api_key = self.finnhub_keys[self.fh_key_index]
            
            # Calculate date range
            end_date = datetime.now()
            days = int(period.replace('d', '').replace('y', ''))
            if 'y' in period:
                days *= 365
            start_date = end_date - timedelta(days=days)
            
            url = 'https://finnhub.io/api/v1/stock/candle'
            params = {
                'symbol': ticker,
                'resolution': 'D',
                'from': int(start_date.timestamp()),
                'to': int(end_date.timestamp()),
                'token': api_key
            }
            
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            if data.get('s') != 'ok':
                print(f"⚠️  Finnhub: No data for {ticker}")
                if 'error' in data:
                    self.fh_key_index = (self.fh_key_index + 1) % len(self.finnhub_keys)
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame({
                'Open': data['o'],
                'High': data['h'],
                'Low': data['l'],
                'Close': data['c'],
                'Volume': data['v']
            })
            df.index = pd.to_datetime(data['t'], unit='s')
            df = df.sort_index()
            
            print(f"✅ Finnhub: {ticker} - {len(df)} rows")
            return df
        
        except Exception as e:
            print(f"⚠️  Finnhub error: {e}")
            return None
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate DataFrame:
        - Remove NaN values
        - Ensure proper column names
        - Validate OHLCV relationships
        - Remove outliers
        """
        # Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Ensure standard column names
        column_map = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'Adj Close': 'Close'  # Replace Adj Close with Close
        }
        df = df.rename(columns=column_map)
        
        # Keep only required columns
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df[[col for col in required if col in df.columns]]
        
        # Remove rows with NaN
        df = df.dropna()
        
        # Validate OHLC relationships
        df = df[(df['High'] >= df['Low']) & 
                (df['High'] >= df['Open']) & 
                (df['High'] >= df['Close']) &
                (df['Low'] <= df['Open']) &
                (df['Low'] <= df['Close'])]
        
        # Remove zero/negative values
        df = df[(df['Open'] > 0) & (df['High'] > 0) & 
                (df['Low'] > 0) & (df['Close'] > 0) & 
                (df['Volume'] >= 0)]
        
        # Remove extreme outliers (3 standard deviations)
        for col in ['Open', 'High', 'Low', 'Close']:
            mean = df[col].mean()
            std = df[col].std()
            df = df[(df[col] >= mean - 3*std) & (df[col] <= mean + 3*std)]
        
        return df
    
    def _rate_limit(self, source: str):
        """Enforce rate limiting between requests"""
        now = time.time()
        last_time = self.last_request_time.get(source, 0)
        elapsed = now - last_time
        
        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            time.sleep(sleep_time)
        
        self.last_request_time[source] = time.time()
    
    def _load_from_cache(self, ticker: str, period: str) -> Optional[pd.DataFrame]:
        """Load data from local cache if fresh"""
        cache_file = self.cache_dir / f'{ticker}_{period}.parquet'
        
        if not cache_file.exists():
            return None
        
        # Check cache age
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if file_age > timedelta(hours=self.cache_ttl_hours):
            print(f"🗑️  Cache expired for {ticker} (age: {file_age})")
            return None
        
        try:
            df = pd.read_parquet(cache_file)
            return df
        except Exception as e:
            print(f"⚠️  Cache read error: {e}")
            return None
    
    def _save_to_cache(self, ticker: str, period: str, df: pd.DataFrame):
        """Save data to local cache"""
        try:
            cache_file = self.cache_dir / f'{ticker}_{period}.parquet'
            df.to_parquet(cache_file)
            print(f"💾 Cached {ticker} data ({len(df)} rows)")
        except Exception as e:
            print(f"⚠️  Cache write error: {e}")
    
    def clear_cache(self, ticker: Optional[str] = None):
        """Clear cache for specific ticker or all"""
        if ticker:
            for file in self.cache_dir.glob(f'{ticker}_*.parquet'):
                file.unlink()
                print(f"🗑️  Cleared cache for {ticker}")
        else:
            for file in self.cache_dir.glob('*.parquet'):
                file.unlink()
            print(f"🗑️  Cleared all cache")
    
    def batch_fetch(self, tickers: List[str], period: str = '60d') -> Dict[str, pd.DataFrame]:
        """
        Fetch multiple tickers with rate limiting
        Returns dict of {ticker: DataFrame}
        """
        results = {}
        
        for i, ticker in enumerate(tickers):
            print(f"\n[{i+1}/{len(tickers)}] Fetching {ticker}...")
            df = self.fetch_ohlcv(ticker, period)
            
            if df is not None:
                results[ticker] = df
            else:
                print(f"❌ Failed to fetch {ticker}")
            
            # Rate limit between tickers
            if i < len(tickers) - 1:
                time.sleep(1)
        
        return results


if __name__ == '__main__':
    print("Testing DataFetcher with fallbacks and caching...")
    
    fetcher = DataFetcher()
    
    # Test portfolio tickers
    tickers = ['MU', 'IONQ', 'APLD', 'ANNX']
    
    results = fetcher.batch_fetch(tickers, period='60d')
    
    print(f"\n{'='*70}")
    print(f"RESULTS:")
    print(f"{'='*70}")
    
    for ticker, df in results.items():
        print(f"\n{ticker}:")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        print(f"  Latest close: ${df['Close'].iloc[-1]:.2f}")
        print(f"  Data quality: {df.isnull().sum().sum()} NaN values")
    
    print(f"\n✅ DataFetcher test complete!")
    print(f"   Cached: {len(list(fetcher.cache_dir.glob('*.parquet')))} files")
