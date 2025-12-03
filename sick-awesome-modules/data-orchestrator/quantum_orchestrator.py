"""
Quantum Data Orchestrator
Async data fetching engine with intelligent fallback routing and unified output format.
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import pandas as pd

try:
    from quantum_api_config import get_config, APISource
except ImportError:
    from backend.quantum_api_config import get_config, APISource

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FetchResult:
    """Unified result format for data fetching."""
    ticker: str
    source: str
    success: bool
    timestamp: datetime
    data: Optional[pd.DataFrame] = None
    candles: int = 0
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            'ticker': self.ticker,
            'source': self.source,
            'success': self.success,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data.to_dict('records') if self.data is not None else None,
            'candles': self.candles,
            'error': self.error,
            'metadata': self.metadata,
            'status': 'success' if self.success else 'failed'
        }


class QuantumOrchestrator:
    """
    Intelligent data orchestration engine.
    Fetches market data from multiple sources with automatic fallback,
    parallel processing, and unified output format.
    """
    
    def __init__(self):
        """Initialize the orchestrator with API configuration."""
        self.config = get_config()
        self.session: Optional[aiohttp.ClientSession] = None
        self._request_counts: Dict[str, int] = {}
        self._last_reset: datetime = datetime.now()
        
        if not self.config.has_valid_sources():
            logger.error("⚠️  No valid API sources available!")
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def _check_rate_limit(self, source: APISource) -> bool:
        """
        Check if we're within rate limits for a source.
        
        Args:
            source: APISource to check
            
        Returns:
            True if we can make a request, False if rate limited
        """
        now = datetime.now()
        
        # Reset counters every minute
        if (now - self._last_reset).total_seconds() >= 60:
            self._request_counts.clear()
            self._last_reset = now
        
        current_count = self._request_counts.get(source.name, 0)
        
        if current_count >= source.rate_limit_per_minute:
            logger.warning(f"⚠️  Rate limit reached for {source.name} ({current_count}/{source.rate_limit_per_minute})")
            return False
        
        return True
    
    def _increment_request_count(self, source: APISource):
        """Increment request counter for a source."""
        self._request_counts[source.name] = self._request_counts.get(source.name, 0) + 1
    
    async def _fetch_polygon(self, ticker: str, source: APISource, days: int = 90) -> FetchResult:
        """
        Fetch data from Polygon.io.
        
        Args:
            ticker: Stock symbol
            source: APISource configuration
            days: Number of days of historical data
            
        Returns:
            FetchResult with data or error
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            url = f"{source.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            params = {'apiKey': source.key, 'adjusted': 'true', 'sort': 'asc'}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('resultsCount', 0) > 0:
                        results = data['results']
                        df = pd.DataFrame(results)
                        
                        # Normalize column names
                        df.rename(columns={
                            't': 'timestamp',
                            'o': 'open',
                            'h': 'high',
                            'l': 'low',
                            'c': 'close',
                            'v': 'volume'
                        }, inplace=True)
                        
                        # Convert timestamp to datetime
                        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df.set_index('date', inplace=True)
                        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
                        
                        self._increment_request_count(source)
                        
                        return FetchResult(
                            ticker=ticker,
                            source='Polygon',
                            success=True,
                            timestamp=datetime.now(),
                            data=df,
                            candles=len(df),
                            metadata={'days_requested': days, 'api_response_count': data.get('resultsCount')}
                        )
                    else:
                        return FetchResult(
                            ticker=ticker,
                            source='Polygon',
                            success=False,
                            timestamp=datetime.now(),
                            error='No data returned from API'
                        )
                else:
                    error_text = await response.text()
                    return FetchResult(
                        ticker=ticker,
                        source='Polygon',
                        success=False,
                        timestamp=datetime.now(),
                        error=f'HTTP {response.status}: {error_text[:100]}'
                    )
                    
        except Exception as e:
            logger.error(f"Polygon fetch error for {ticker}: {e}")
            return FetchResult(
                ticker=ticker,
                source='Polygon',
                success=False,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    async def _fetch_fmp(self, ticker: str, source: APISource, days: int = 90) -> FetchResult:
        """
        Fetch data from Financial Modeling Prep.
        
        Args:
            ticker: Stock symbol
            source: APISource configuration
            days: Number of days of historical data
            
        Returns:
            FetchResult with data or error
        """
        try:
            url = f"{source.base_url}/historical-price-full/{ticker}"
            params = {'apikey': source.key}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'historical' in data and len(data['historical']) > 0:
                        df = pd.DataFrame(data['historical'])
                        
                        # Convert date to datetime and set as index
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                        df.sort_index(inplace=True)
                        
                        # Filter to requested days
                        cutoff_date = datetime.now() - timedelta(days=days)
                        df = df[df.index >= cutoff_date]
                        
                        # Ensure standard columns
                        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
                        
                        self._increment_request_count(source)
                        
                        return FetchResult(
                            ticker=ticker,
                            source='FMP',
                            success=True,
                            timestamp=datetime.now(),
                            data=df,
                            candles=len(df),
                            metadata={'days_requested': days, 'symbol': data.get('symbol')}
                        )
                    else:
                        return FetchResult(
                            ticker=ticker,
                            source='FMP',
                            success=False,
                            timestamp=datetime.now(),
                            error='No historical data in response'
                        )
                else:
                    error_text = await response.text()
                    return FetchResult(
                        ticker=ticker,
                        source='FMP',
                        success=False,
                        timestamp=datetime.now(),
                        error=f'HTTP {response.status}: {error_text[:100]}'
                    )
                    
        except Exception as e:
            logger.error(f"FMP fetch error for {ticker}: {e}")
            return FetchResult(
                ticker=ticker,
                source='FMP',
                success=False,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    async def _fetch_alphavantage(self, ticker: str, source: APISource, days: int = 90) -> FetchResult:
        """
        Fetch data from Alpha Vantage.
        
        Args:
            ticker: Stock symbol
            source: APISource configuration
            days: Number of days of historical data
            
        Returns:
            FetchResult with data or error
        """
        try:
            url = f"{source.base_url}/query"
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': ticker,
                'apikey': source.key,
                'outputsize': 'full'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'Time Series (Daily)' in data:
                        time_series = data['Time Series (Daily)']
                        
                        # Convert to DataFrame
                        df = pd.DataFrame.from_dict(time_series, orient='index')
                        df.index = pd.to_datetime(df.index)
                        df.sort_index(inplace=True)
                        
                        # Rename columns
                        df.rename(columns={
                            '1. open': 'open',
                            '2. high': 'high',
                            '3. low': 'low',
                            '4. close': 'close',
                            '6. volume': 'volume'
                        }, inplace=True)
                        
                        # Convert to numeric
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            df[col] = pd.to_numeric(df[col])
                        
                        # Filter to requested days
                        cutoff_date = datetime.now() - timedelta(days=days)
                        df = df[df.index >= cutoff_date]
                        
                        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
                        
                        self._increment_request_count(source)
                        
                        return FetchResult(
                            ticker=ticker,
                            source='AlphaVantage',
                            success=True,
                            timestamp=datetime.now(),
                            data=df,
                            candles=len(df),
                            metadata={'days_requested': days}
                        )
                    else:
                        error_msg = data.get('Note') or data.get('Error Message') or 'Unknown error'
                        return FetchResult(
                            ticker=ticker,
                            source='AlphaVantage',
                            success=False,
                            timestamp=datetime.now(),
                            error=error_msg
                        )
                else:
                    error_text = await response.text()
                    return FetchResult(
                        ticker=ticker,
                        source='AlphaVantage',
                        success=False,
                        timestamp=datetime.now(),
                        error=f'HTTP {response.status}: {error_text[:100]}'
                    )
                    
        except Exception as e:
            logger.error(f"AlphaVantage fetch error for {ticker}: {e}")
            return FetchResult(
                ticker=ticker,
                source='AlphaVantage',
                success=False,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    async def _fetch_eodhd(self, ticker: str, source: APISource, days: int = 90) -> FetchResult:
        """
        Fetch data from EODHD.
        
        Args:
            ticker: Stock symbol
            source: APISource configuration
            days: Number of days of historical data
            
        Returns:
            FetchResult with data or error
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            url = f"{source.base_url}/eod/{ticker}.US"
            params = {
                'api_token': source.key,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'fmt': 'json'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if isinstance(data, list) and len(data) > 0:
                        df = pd.DataFrame(data)
                        
                        # Convert date to datetime and set as index
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                        df.sort_index(inplace=True)
                        
                        # Ensure standard columns
                        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
                        
                        self._increment_request_count(source)
                        
                        return FetchResult(
                            ticker=ticker,
                            source='EODHD',
                            success=True,
                            timestamp=datetime.now(),
                            data=df,
                            candles=len(df),
                            metadata={'days_requested': days}
                        )
                    else:
                        return FetchResult(
                            ticker=ticker,
                            source='EODHD',
                            success=False,
                            timestamp=datetime.now(),
                            error='No data returned or invalid format'
                        )
                else:
                    error_text = await response.text()
                    return FetchResult(
                        ticker=ticker,
                        source='EODHD',
                        success=False,
                        timestamp=datetime.now(),
                        error=f'HTTP {response.status}: {error_text[:100]}'
                    )
                    
        except Exception as e:
            logger.error(f"EODHD fetch error for {ticker}: {e}")
            return FetchResult(
                ticker=ticker,
                source='EODHD',
                success=False,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    async def fetch_with_fallback(self, ticker: str, days: int = 90) -> FetchResult:
        """
        Fetch data with intelligent fallback through priority-ordered sources.
        
        Args:
            ticker: Stock symbol
            days: Number of days of historical data
            
        Returns:
            FetchResult from first successful source, or final failure
        """
        sources = self.config.get_sources_by_priority()
        
        if not sources:
            return FetchResult(
                ticker=ticker,
                source='None',
                success=False,
                timestamp=datetime.now(),
                error='No valid API sources configured'
            )
        
        logger.info(f"Fetching {ticker} with {len(sources)} sources available...")
        
        for source in sources:
            # Check rate limit
            if not self._check_rate_limit(source):
                logger.warning(f"Skipping {source.name} due to rate limit")
                continue
            
            # Route to appropriate fetcher
            if source.name == 'Polygon':
                result = await self._fetch_polygon(ticker, source, days)
            elif source.name == 'FMP':
                result = await self._fetch_fmp(ticker, source, days)
            elif source.name == 'AlphaVantage':
                result = await self._fetch_alphavantage(ticker, source, days)
            elif source.name == 'EODHD':
                result = await self._fetch_eodhd(ticker, source, days)
            else:
                continue
            
            if result.success:
                logger.info(f"✓ {ticker}: {result.candles} candles from {result.source}")
                return result
            else:
                logger.warning(f"✗ {source.name} failed for {ticker}: {result.error}")
        
        # All sources failed
        return FetchResult(
            ticker=ticker,
            source='All',
            success=False,
            timestamp=datetime.now(),
            error=f'All {len(sources)} sources failed'
        )
    
    async def fetch_multiple(self, tickers: List[str], days: int = 90) -> Dict[str, FetchResult]:
        """
        Fetch data for multiple tickers in parallel.
        
        Args:
            tickers: List of stock symbols
            days: Number of days of historical data
            
        Returns:
            Dictionary mapping tickers to FetchResults
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"PARALLEL FETCH: {len(tickers)} tickers")
        logger.info(f"{'='*70}\n")
        
        tasks = [self.fetch_with_fallback(ticker, days) for ticker in tickers]
        results = await asyncio.gather(*tasks)
        
        result_dict = {result.ticker: result for result in results}
        
        # Summary
        successful = sum(1 for r in results if r.success)
        logger.info(f"\n{'='*70}")
        logger.info(f"FETCH COMPLETE: {successful}/{len(tickers)} successful")
        logger.info(f"{'='*70}\n")
        
        return result_dict


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def fetch_ticker(ticker: str, days: int = 90) -> FetchResult:
    """
    Convenience function to fetch a single ticker.
    
    Args:
        ticker: Stock symbol
        days: Number of days of historical data
        
    Returns:
        FetchResult with data or error
    """
    async with QuantumOrchestrator() as orchestrator:
        return await orchestrator.fetch_with_fallback(ticker, days)


async def fetch_tickers(tickers: List[str], days: int = 90) -> Dict[str, FetchResult]:
    """
    Convenience function to fetch multiple tickers in parallel.
    
    Args:
        tickers: List of stock symbols
        days: Number of days of historical data
        
    Returns:
        Dictionary mapping tickers to FetchResults
    """
    async with QuantumOrchestrator() as orchestrator:
        return await orchestrator.fetch_multiple(tickers, days)


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

async def test_single_ticker():
    """Test fetching a single ticker."""
    print("\n" + "="*70)
    print("TEST 1: SINGLE TICKER FETCH (SPY)")
    print("="*70 + "\n")
    
    result = await fetch_ticker("SPY", days=30)
    
    print(f"\nResult:")
    print(f"  Ticker: {result.ticker}")
    print(f"  Source: {result.source}")
    print(f"  Success: {result.success}")
    print(f"  Candles: {result.candles}")
    print(f"  Error: {result.error}")
    
    if result.success and result.data is not None:
        print(f"\nData Preview:")
        print(result.data.head())
        print(f"\nData Shape: {result.data.shape}")
        print(f"Date Range: {result.data.index[0]} to {result.data.index[-1]}")


async def test_multiple_tickers():
    """Test fetching multiple tickers in parallel."""
    print("\n" + "="*70)
    print("TEST 2: PARALLEL MULTI-TICKER FETCH")
    print("="*70 + "\n")
    
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    results = await fetch_tickers(tickers, days=30)
    
    print(f"\n📊 RESULTS SUMMARY:")
    for ticker, result in results.items():
        status = "✓" if result.success else "✗"
        candles = f"{result.candles} candles" if result.success else result.error
        print(f"  {status} {ticker:6} | {result.source:15} | {candles}")


async def test_fallback_mechanism():
    """Test the fallback mechanism with an obscure ticker."""
    print("\n" + "="*70)
    print("TEST 3: FALLBACK MECHANISM")
    print("="*70 + "\n")
    
    # Use a ticker that might fail on some sources
    result = await fetch_ticker("INVALID_TICKER_XYZ", days=30)
    
    print(f"\nFallback Test Result:")
    print(f"  Ticker: {result.ticker}")
    print(f"  Source: {result.source}")
    print(f"  Success: {result.success}")
    print(f"  Error: {result.error}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("QUANTUM ORCHESTRATOR - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    async def run_all_tests():
        await test_single_ticker()
        await test_multiple_tickers()
        await test_fallback_mechanism()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS COMPLETE")
        print("="*70 + "\n")
    
    asyncio.run(run_all_tests())
