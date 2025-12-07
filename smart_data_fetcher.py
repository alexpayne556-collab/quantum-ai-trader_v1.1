"""
Smart Data Fetcher with API Rotation and Rate Limiting
Rotates between multiple API providers to stay within free tier limits.
Uses yfinance as reliable backup when all APIs exhausted.
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import pandas as pd
import yfinance as yf
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class APIRateLimiter:
    """Track API usage and enforce rate limits"""
    
    def __init__(self):
        self.usage = {}  # {api_name: [(timestamp, count), ...]}
        
    def can_use(self, api_name: str, limits: Dict) -> bool:
        """Check if API can be used based on rate limits"""
        now = time.time()
        
        # Clean old entries
        if api_name not in self.usage:
            self.usage[api_name] = []
        
        # Remove entries older than 24 hours
        self.usage[api_name] = [
            (ts, count) for ts, count in self.usage[api_name]
            if now - ts < 86400  # 24 hours
        ]
        
        # Check daily limit
        daily_count = sum(count for ts, count in self.usage[api_name])
        if daily_count >= limits.get('daily', float('inf')):
            return False
        
        # Check per-minute limit
        minute_ago = now - 60
        minute_count = sum(
            count for ts, count in self.usage[api_name]
            if ts >= minute_ago
        )
        if minute_count >= limits.get('per_minute', float('inf')):
            return False
        
        return True
    
    def record_usage(self, api_name: str, count: int = 1):
        """Record API usage"""
        if api_name not in self.usage:
            self.usage[api_name] = []
        self.usage[api_name].append((time.time(), count))
    
    def get_stats(self) -> Dict:
        """Get usage statistics"""
        now = time.time()
        stats = {}
        
        for api_name, entries in self.usage.items():
            # Daily usage (last 24 hours)
            daily = sum(count for ts, count in entries if now - ts < 86400)
            # Per-minute usage
            minute = sum(count for ts, count in entries if now - ts < 60)
            stats[api_name] = {'daily': daily, 'per_minute': minute}
        
        return stats


class SmartDataFetcher:
    """
    Smart data fetcher that rotates between multiple API providers.
    
    API Priority Order (by reliability and limits):
    1. Finnhub - 60 calls/min (best for real-time)
    2. Financial Modeling Prep - 250 calls/day
    3. Polygon - 5 calls/min (delayed data)
    4. Alpha Vantage - 25 calls/day
    5. EOD Historical Data - 20 calls/day
    6. yfinance - Unlimited, reliable backup
    """
    
    # API Rate Limits (free tier)
    API_LIMITS = {
        'finnhub': {'daily': 86400, 'per_minute': 60},  # 60/min = 86400/day theoretical
        'fmp': {'daily': 250, 'per_minute': 60},
        'polygon': {'daily': 7200, 'per_minute': 5},  # 5/min = 7200/day theoretical
        'alpha_vantage': {'daily': 25, 'per_minute': 5},
        'eodhd': {'daily': 20, 'per_minute': 20},
    }
    
    def __init__(self):
        self.rate_limiter = APIRateLimiter()
        
        # Load API keys
        self.finnhub_key = os.getenv('FINNHUB_API_KEY')
        self.fmp_key = os.getenv('FMP_API_KEY')
        self.polygon_key = os.getenv('POLYGON_API_KEY')
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.eodhd_key = os.getenv('EODHD_API_TOKEN')
        
        # Cache for reducing duplicate requests
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
        logger.info("🔄 Smart Data Fetcher initialized with API rotation")
    
    def get_stock_data(
        self,
        ticker: str,
        period: str = '1y',
        interval: str = '1d'
    ) -> Optional[pd.DataFrame]:
        """
        Fetch stock data with automatic API rotation.
        
        Args:
            ticker: Stock symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        cache_key = f"{ticker}_{period}_{interval}"
        
        # Check cache first
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_duration:
                logger.debug(f"📦 Cache hit for {ticker}")
                return cached_data
        
        # Try APIs in priority order
        apis = [
            ('finnhub', self._fetch_finnhub),
            ('fmp', self._fetch_fmp),
            ('polygon', self._fetch_polygon),
            ('alpha_vantage', self._fetch_alpha_vantage),
            ('eodhd', self._fetch_eodhd),
        ]
        
        for api_name, fetch_func in apis:
            if self.rate_limiter.can_use(api_name, self.API_LIMITS[api_name]):
                try:
                    logger.info(f"🔄 Trying {api_name.upper()} for {ticker}")
                    df = fetch_func(ticker, period, interval)
                    
                    if df is not None and not df.empty:
                        self.rate_limiter.record_usage(api_name)
                        self.cache[cache_key] = (time.time(), df)
                        logger.info(f"✅ {api_name.upper()} success for {ticker}")
                        return df
                    
                except Exception as e:
                    logger.warning(f"❌ {api_name.upper()} failed for {ticker}: {e}")
                    continue
            else:
                logger.debug(f"⏭️  {api_name.upper()} rate limit reached, skipping")
        
        # Fallback to yfinance (unlimited, reliable)
        logger.info(f"🔄 Using yfinance fallback for {ticker}")
        try:
            df = self._fetch_yfinance(ticker, period, interval)
            if df is not None and not df.empty:
                self.cache[cache_key] = (time.time(), df)
                logger.info(f"✅ yfinance success for {ticker}")
                return df
        except Exception as e:
            logger.error(f"❌ yfinance failed for {ticker}: {e}")
        
        logger.error(f"❌ All APIs failed for {ticker}")
        return None
    
    def _fetch_finnhub(self, ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Fetch from Finnhub (60 calls/min)"""
        if not self.finnhub_key:
            return None
        
        # Convert period to timestamps
        end = int(datetime.now().timestamp())
        start = int((datetime.now() - self._period_to_timedelta(period)).timestamp())
        
        # Map interval to Finnhub resolution
        resolution_map = {
            '1d': 'D', '1h': '60', '30m': '30', '15m': '15', '5m': '5', '1m': '1'
        }
        resolution = resolution_map.get(interval, 'D')
        
        url = f"https://finnhub.io/api/v1/stock/candle"
        params = {
            'symbol': ticker,
            'resolution': resolution,
            'from': start,
            'to': end,
            'token': self.finnhub_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get('s') == 'ok':
            df = pd.DataFrame({
                'Open': data['o'],
                'High': data['h'],
                'Low': data['l'],
                'Close': data['c'],
                'Volume': data['v']
            }, index=pd.to_datetime(data['t'], unit='s'))
            return df
        
        return None
    
    def _fetch_fmp(self, ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Fetch from Financial Modeling Prep (250 calls/day)"""
        if not self.fmp_key:
            return None
        
        # FMP supports daily data best
        if interval != '1d':
            return None
        
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}"
        params = {'apikey': self.fmp_key}
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'historical' in data:
            df = pd.DataFrame(data['historical'])
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            # Filter by period
            cutoff = datetime.now() - self._period_to_timedelta(period)
            df = df[df.index >= cutoff]
            
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        return None
    
    def _fetch_polygon(self, ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Fetch from Polygon (5 calls/min, delayed data)"""
        if not self.polygon_key:
            return None
        
        # Map interval to Polygon timespan
        timespan_map = {
            '1d': 'day', '1h': 'hour', '30m': 'minute', '15m': 'minute',
            '5m': 'minute', '1m': 'minute'
        }
        timespan = timespan_map.get(interval, 'day')
        
        # Map interval to multiplier
        multiplier = 1
        if interval in ['30m', '15m', '5m']:
            multiplier = int(interval[:-1])
        
        # Calculate date range
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - self._period_to_timedelta(period)).strftime('%Y-%m-%d')
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
        params = {'apiKey': self.polygon_key, 'limit': 50000}
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get('results'):
            df = pd.DataFrame(data['results'])
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            df = df.set_index('timestamp')
            df = df.rename(columns={
                'o': 'Open',
                'h': 'High',
                'l': 'Low',
                'c': 'Close',
                'v': 'Volume'
            })
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        return None
    
    def _fetch_alpha_vantage(self, ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Fetch from Alpha Vantage (25 calls/day)"""
        if not self.alpha_vantage_key:
            return None
        
        # Alpha Vantage function based on interval
        if interval == '1d':
            function = 'TIME_SERIES_DAILY'
            key_prefix = 'Time Series (Daily)'
        else:
            return None  # Only daily supported for simplicity
        
        url = "https://www.alphavantage.co/query"
        params = {
            'function': function,
            'symbol': ticker,
            'apikey': self.alpha_vantage_key,
            'outputsize': 'full' if period in ['1y', '2y', '5y', 'max'] else 'compact'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if key_prefix in data:
            df = pd.DataFrame.from_dict(data[key_prefix], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df = df.rename(columns={
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            })
            df = df.astype(float)
            
            # Filter by period
            cutoff = datetime.now() - self._period_to_timedelta(period)
            df = df[df.index >= cutoff]
            
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        return None
    
    def _fetch_eodhd(self, ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Fetch from EOD Historical Data (20 calls/day)"""
        if not self.eodhd_key:
            return None
        
        # EODHD supports daily data best
        if interval != '1d':
            return None
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - self._period_to_timedelta(period)).strftime('%Y-%m-%d')
        
        url = f"https://eodhd.com/api/eod/{ticker}.US"
        params = {
            'api_token': self.eodhd_key,
            'from': start_date,
            'to': end_date,
            'fmt': 'json'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data and isinstance(data, list):
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        return None
    
    def _fetch_yfinance(self, ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Fetch from yfinance (reliable backup, unlimited)"""
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        
        if df.empty:
            return None
        
        # Ensure consistent column names
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    def _period_to_timedelta(self, period: str) -> timedelta:
        """Convert period string to timedelta"""
        period_map = {
            '1d': timedelta(days=1),
            '5d': timedelta(days=5),
            '1mo': timedelta(days=30),
            '3mo': timedelta(days=90),
            '6mo': timedelta(days=180),
            '1y': timedelta(days=365),
            '2y': timedelta(days=730),
            '5y': timedelta(days=1825),
            '10y': timedelta(days=3650),
            'ytd': timedelta(days=(datetime.now() - datetime(datetime.now().year, 1, 1)).days),
            'max': timedelta(days=7300)  # ~20 years
        }
        return period_map.get(period, timedelta(days=365))
    
    def get_usage_stats(self) -> Dict:
        """Get API usage statistics"""
        stats = self.rate_limiter.get_stats()
        
        # Add limits for context
        for api_name, limits in self.API_LIMITS.items():
            if api_name not in stats:
                stats[api_name] = {'daily': 0, 'per_minute': 0}
            stats[api_name]['daily_limit'] = limits['daily']
            stats[api_name]['per_minute_limit'] = limits['per_minute']
            stats[api_name]['daily_remaining'] = limits['daily'] - stats[api_name]['daily']
        
        return stats
    
    def print_usage_stats(self):
        """Print usage statistics in a nice format"""
        stats = self.get_usage_stats()
        
        print("\n📊 API Usage Statistics:")
        print("=" * 80)
        print(f"{'API':<20} {'Daily Used':<15} {'Daily Limit':<15} {'Remaining':<15}")
        print("-" * 80)
        
        for api_name in sorted(stats.keys()):
            s = stats[api_name]
            used = s['daily']
            limit = s['daily_limit']
            remaining = s['daily_remaining']
            pct = (used / limit * 100) if limit > 0 else 0
            
            print(f"{api_name:<20} {used:<15} {limit:<15} {remaining:<15} ({pct:.1f}%)")
        
        print("=" * 80)


# Global instance for convenience
_fetcher = None

def get_fetcher() -> SmartDataFetcher:
    """Get or create global SmartDataFetcher instance"""
    global _fetcher
    if _fetcher is None:
        _fetcher = SmartDataFetcher()
    return _fetcher


def fetch_stock_data(ticker: str, period: str = '1y', interval: str = '1d') -> Optional[pd.DataFrame]:
    """
    Convenience function to fetch stock data.
    
    Example:
        df = fetch_stock_data('AAPL', period='1y', interval='1d')
    """
    fetcher = get_fetcher()
    return fetcher.get_stock_data(ticker, period, interval)


if __name__ == "__main__":
    # Test the smart data fetcher
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 Testing Smart Data Fetcher with API Rotation\n")
    
    fetcher = SmartDataFetcher()
    
    # Test with multiple tickers
    test_tickers = ['AAPL', 'NVDA', 'TSLA', 'MSFT', 'GOOGL']
    
    for ticker in test_tickers:
        print(f"\n📈 Fetching {ticker}...")
        df = fetcher.get_stock_data(ticker, period='1mo', interval='1d')
        
        if df is not None:
            print(f"✅ Success! Got {len(df)} rows")
            print(f"   Date range: {df.index[0]} to {df.index[-1]}")
            print(f"   Latest close: ${df['Close'].iloc[-1]:.2f}")
        else:
            print(f"❌ Failed to fetch {ticker}")
    
    # Print usage statistics
    print("\n" + "="*80)
    fetcher.print_usage_stats()
    print("\n✅ Test complete!")
