"""
API Key Manager - Free Tier Rotation System

Manages multiple data sources with automatic failover:
1. Primary: Twelve Data (800/day, 8/min) - NEW!
2. Backup 1: Finnhub (60/min)
3. Backup 2: Polygon (5/min)
4. Backup 3: Alpha Vantage (25/day)
5. Backup 4: FMP (250/day)
6. Backup 5: EODHD (20/day)

Regime Data:
- FRED (unlimited) for VIX, yield curve, SPY returns - NEW!

Automatically rotates through sources when rate limits hit.
"""

import os
from dotenv import load_dotenv
import time
from typing import Optional, Dict, List
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta

load_dotenv()


class DataSourceManager:
    """
    Intelligent data source rotation with rate limit management
    """
    
    def __init__(self):
        # Load all API keys
        self.twelve_data_key = os.getenv('TWELVE_DATA_API_KEY')
        self.finnhub_key = os.getenv('FINNHUB_API_KEY')
        self.polygon_key = os.getenv('POLYGON_API_KEY')
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.fmp_key = os.getenv('FMP_API_KEY')
        self.eodhd_token = os.getenv('EODHD_API_TOKEN')
        self.fred_key = os.getenv('FRED_API_KEY')
        
        # Rate limit tracking
        self.call_counts = {
            'twelve_data': {'count': 0, 'reset_time': time.time() + 86400, 'limit': 800},
            'finnhub': {'count': 0, 'reset_time': time.time() + 60, 'limit': 60},
            'polygon': {'count': 0, 'reset_time': time.time() + 60, 'limit': 5},
            'alpha_vantage': {'count': 0, 'reset_time': time.time() + 86400, 'limit': 25},
            'fmp': {'count': 0, 'reset_time': time.time() + 86400, 'limit': 250},
            'eodhd': {'count': 0, 'reset_time': time.time() + 86400, 'limit': 20},
        }
        
        # Source priority order (best to worst)
        self.source_priority = [
            'twelve_data',  # NEW: 800/day, excellent rate limit
            'finnhub',      # 60/min, real-time
            'polygon',      # 5/min, good quality
            'yfinance',     # Unlimited, but slower
            'alpha_vantage',# 25/day, backup
            'fmp',          # 250/day, fundamentals
            'eodhd'         # 20/day, last resort
        ]
        
        print("✅ Data Source Manager initialized")
        print(f"   Available sources: {len([s for s in self.source_priority if self._has_key(s)])}")
    
    def _has_key(self, source: str) -> bool:
        """Check if API key exists for source"""
        key_map = {
            'twelve_data': self.twelve_data_key,
            'finnhub': self.finnhub_key,
            'polygon': self.polygon_key,
            'alpha_vantage': self.alpha_vantage_key,
            'fmp': self.fmp_key,
            'eodhd': self.eodhd_token,
            'yfinance': True  # Always available
        }
        return bool(key_map.get(source))
    
    def _check_rate_limit(self, source: str) -> bool:
        """Check if source has capacity"""
        if source == 'yfinance':
            return True  # No rate limit
        
        tracker = self.call_counts.get(source)
        if not tracker:
            return True
        
        # Reset counter if time window passed
        if time.time() > tracker['reset_time']:
            tracker['count'] = 0
            # Reset time window
            if source in ['finnhub', 'polygon']:
                tracker['reset_time'] = time.time() + 60  # Per minute
            else:
                tracker['reset_time'] = time.time() + 86400  # Per day
        
        # Check if under limit
        return tracker['count'] < tracker['limit']
    
    def _increment_count(self, source: str):
        """Increment usage counter"""
        if source in self.call_counts:
            self.call_counts[source]['count'] += 1
    
    def get_best_source(self) -> str:
        """Get the best available data source right now"""
        for source in self.source_priority:
            if self._has_key(source) and self._check_rate_limit(source):
                return source
        
        # If all rate limited, return yfinance as ultimate fallback
        return 'yfinance'
    
    def fetch_ohlcv(self, ticker: str, period: str = '3mo', interval: str = '1h') -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data with automatic source rotation
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            period: Time period ('1d', '5d', '1mo', '3mo', '1y', '2y')
            interval: Bar interval ('1m', '5m', '1h', '1d')
        
        Returns:
            DataFrame with OHLCV data or None if all sources fail
        """
        
        # Try sources in priority order
        for source in self.source_priority:
            if not self._has_key(source):
                continue
            
            if not self._check_rate_limit(source):
                print(f"⚠️  {source} rate limit reached, trying next...")
                continue
            
            try:
                print(f"📡 Fetching {ticker} from {source}...")
                
                if source == 'twelve_data':
                    df = self._fetch_twelve_data(ticker, period, interval)
                elif source == 'finnhub':
                    df = self._fetch_finnhub(ticker, period, interval)
                elif source == 'polygon':
                    df = self._fetch_polygon(ticker, period, interval)
                elif source == 'alpha_vantage':
                    df = self._fetch_alpha_vantage(ticker, interval)
                elif source == 'fmp':
                    df = self._fetch_fmp(ticker, period)
                elif source == 'yfinance':
                    df = self._fetch_yfinance(ticker, period, interval)
                else:
                    continue
                
                if df is not None and len(df) > 0:
                    self._increment_count(source)
                    print(f"✅ Got {len(df)} bars from {source}")
                    return df
                    
            except Exception as e:
                print(f"❌ {source} failed: {e}")
                continue
        
        print(f"❌ All sources failed for {ticker}")
        return None
    
    def _fetch_twelve_data(self, ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Fetch from Twelve Data (NEW - 800/day)"""
        # Map interval
        interval_map = {'1m': '1min', '5m': '5min', '1h': '1h', '1d': '1day'}
        td_interval = interval_map.get(interval, '1h')
        
        # Map period to outputsize
        outputsize = 5000 if period in ['1y', '2y'] else 1000
        
        url = f"https://api.twelvedata.com/time_series"
        params = {
            'symbol': ticker,
            'interval': td_interval,
            'outputsize': outputsize,
            'apikey': self.twelve_data_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if 'values' in data:
            df = pd.DataFrame(data['values'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            df = df.set_index('datetime').sort_index()
            return df
        
        return None
    
    def _fetch_finnhub(self, ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Fetch from Finnhub (60/min)"""
        # Calculate time range
        end = int(datetime.now().timestamp())
        period_map = {'1d': 1, '5d': 5, '1mo': 30, '3mo': 90, '1y': 365, '2y': 730}
        days = period_map.get(period, 90)
        start = int((datetime.now() - timedelta(days=days)).timestamp())
        
        # Map interval
        resolution_map = {'1m': '1', '5m': '5', '1h': '60', '1d': 'D'}
        resolution = resolution_map.get(interval, '60')
        
        url = f"https://finnhub.io/api/v1/stock/candle"
        params = {
            'symbol': ticker,
            'resolution': resolution,
            'from': start,
            'to': end,
            'token': self.finnhub_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data.get('s') == 'ok':
            df = pd.DataFrame({
                'datetime': pd.to_datetime(data['t'], unit='s'),
                'open': data['o'],
                'high': data['h'],
                'low': data['l'],
                'close': data['c'],
                'volume': data['v']
            })
            df = df.set_index('datetime').sort_index()
            return df
        
        return None
    
    def _fetch_polygon(self, ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Fetch from Polygon (5/min)"""
        # Map interval
        multiplier_map = {'1m': (1, 'minute'), '5m': (5, 'minute'), '1h': (1, 'hour'), '1d': (1, 'day')}
        multiplier, timespan = multiplier_map.get(interval, (1, 'hour'))
        
        # Calculate dates
        end_date = datetime.now().strftime('%Y-%m-%d')
        period_map = {'1d': 1, '5d': 5, '1mo': 30, '3mo': 90, '1y': 365, '2y': 730}
        days = period_map.get(period, 90)
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
        params = {'apiKey': self.polygon_key}
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data.get('results'):
            df = pd.DataFrame(data['results'])
            df['datetime'] = pd.to_datetime(df['t'], unit='ms')
            df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
            df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
            df = df.set_index('datetime').sort_index()
            return df
        
        return None
    
    def _fetch_alpha_vantage(self, ticker: str, interval: str) -> Optional[pd.DataFrame]:
        """Fetch from Alpha Vantage (25/day)"""
        function_map = {'1m': 'TIME_SERIES_INTRADAY', '5m': 'TIME_SERIES_INTRADAY', 
                        '1h': 'TIME_SERIES_INTRADAY', '1d': 'TIME_SERIES_DAILY'}
        function = function_map.get(interval, 'TIME_SERIES_INTRADAY')
        
        url = "https://www.alphavantage.co/query"
        params = {
            'function': function,
            'symbol': ticker,
            'interval': interval if interval != '1d' else None,
            'outputsize': 'full',
            'apikey': self.alpha_vantage_key
        }
        params = {k: v for k, v in params.items() if v is not None}
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        # Parse response based on function type
        time_series_key = [k for k in data.keys() if 'Time Series' in k]
        if time_series_key:
            time_series = data[time_series_key[0]]
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.rename(columns={
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume'
            })
            df = df.astype(float)
            return df.sort_index()
        
        return None
    
    def _fetch_fmp(self, ticker: str, period: str) -> Optional[pd.DataFrame]:
        """Fetch from FMP (250/day)"""
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}"
        params = {'apikey': self.fmp_key}
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if 'historical' in data:
            df = pd.DataFrame(data['historical'])
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            df = df.rename(columns={'adjClose': 'close'})
            return df[['open', 'high', 'low', 'close', 'volume']]
        
        return None
    
    def _fetch_yfinance(self, ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Fetch from yfinance (unlimited fallback)"""
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        
        if len(df) > 0:
            df = df.reset_index()
            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]
            
            df = df.rename(columns={'datetime': 'date'})
            if 'date' in df.columns:
                df = df.set_index('date')
            
            return df[['open', 'high', 'low', 'close', 'volume']]
        
        return None
    
    def get_fred_data(self, series_id: str) -> Optional[pd.Series]:
        """
        Fetch economic data from FRED (unlimited, free)
        
        Common series_ids:
        - 'VIXCLS': VIX (volatility index)
        - 'DGS10': 10-Year Treasury Constant Maturity Rate
        - 'DGS2': 2-Year Treasury Constant Maturity Rate
        - 'UNRATE': Unemployment Rate
        - 'CPIAUCSL': Consumer Price Index
        
        Returns:
            pandas Series with datetime index
        """
        url = f"https://api.stlouisfed.org/fred/series/observations"
        params = {
            'series_id': series_id,
            'api_key': self.fred_key,
            'file_type': 'json'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'observations' in data:
                obs = data['observations']
                df = pd.DataFrame(obs)
                df['date'] = pd.to_datetime(df['date'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df = df.dropna(subset=['value'])
                series = df.set_index('date')['value']
                
                print(f"✅ Got {len(series)} observations for {series_id} from FRED")
                return series
        
        except Exception as e:
            print(f"❌ FRED fetch failed for {series_id}: {e}")
        
        return None
    
    def get_usage_stats(self) -> Dict:
        """Get current usage statistics for all sources"""
        stats = {}
        for source, tracker in self.call_counts.items():
            remaining = tracker['limit'] - tracker['count']
            pct_used = (tracker['count'] / tracker['limit']) * 100
            stats[source] = {
                'used': tracker['count'],
                'limit': tracker['limit'],
                'remaining': remaining,
                'pct_used': f"{pct_used:.1f}%"
            }
        return stats


# Example usage
if __name__ == "__main__":
    manager = DataSourceManager()
    
    # Test data fetching with rotation
    print("\n" + "="*60)
    print("Testing Data Source Rotation")
    print("="*60)
    
    # Fetch some data
    df = manager.fetch_ohlcv('AAPL', period='5d', interval='1h')
    if df is not None:
        print(f"\n✅ Successfully fetched {len(df)} bars")
        print(f"Latest close: ${float(df['close'].iloc[-1]):.2f}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Test FRED data
    print("\n" + "="*60)
    print("Testing FRED Economic Data")
    print("="*60)
    
    vix = manager.get_fred_data('VIXCLS')
    if vix is not None:
        print(f"Current VIX: {vix.iloc[-1]:.2f}")
    
    # Show usage stats
    print("\n" + "="*60)
    print("API Usage Statistics")
    print("="*60)
    
    stats = manager.get_usage_stats()
    for source, stat in stats.items():
        print(f"{source:15s}: {stat['used']:4d}/{stat['limit']:4d} ({stat['pct_used']} used)")
