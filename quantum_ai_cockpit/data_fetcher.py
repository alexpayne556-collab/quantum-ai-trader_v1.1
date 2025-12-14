"""
UNIFIED DATA FETCHER v1.0
Centralized data access for all Quantum AI Trader modules.
Tests API keys on startup and provides fallback to free sources.

Usage:
    from quantum_ai_cockpit.data_fetcher import DataFetcher
    
    fetcher = DataFetcher()
    df = fetcher.get_stock_data("AAPL", days=30)
    news = fetcher.get_news("AAPL")
    econ = fetcher.get_economic_data("FEDFUNDS")
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

import pandas as pd
import numpy as np
import requests

# Free libraries (pip install)
import yfinance as yf

# Load environment
from dotenv import load_dotenv

_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")


class DataFetcher:
    """
    Unified data fetcher with API key validation and fallback support.
    Tests all configured APIs and uses working ones with priority ordering.
    Falls back to free sources (yfinance) when APIs fail.
    """

    def __init__(self, verbose: bool = True, test_apis: bool = True):
        self.verbose = verbose
        self.api_status: Dict[str, bool] = {}
        self._load_api_keys()
        if test_apis:
            self._test_all_apis()

    def _load_api_keys(self) -> None:
        """Load all API keys from environment."""
        self.keys = {
            # Data Providers (Stock/Crypto)
            "POLYGON": os.getenv("POLYGON_API_KEY"),
            "FMP": os.getenv("FMP_API_KEY"),
            "FINNHUB": os.getenv("FINNHUB_API_KEY"),
            "ALPHAVANTAGE": os.getenv("ALPHAVANTAGE_API_KEY"),
            "EODHD": os.getenv("EODHD_API_TOKEN"),
            "TWELVEDATA": os.getenv("TWELVEDATA_API_KEY"),
            "TIINGO": os.getenv("TIINGO_API_KEY"),
            "INTRINIO": os.getenv("INTRINIO_API_KEY"),
            # News APIs
            "NEWSAPI": os.getenv("NEWSAPI_API_KEY"),
            "NEWSDATA": os.getenv("NEWSDATA_API_KEY"),
            "MARKETAUX": os.getenv("MARKETAUX_API_KEY"),
            # Economic Data
            "FRED": os.getenv("FRED_API_KEY"),
            # AI/ML
            "OPENAI": os.getenv("OPENAI_API_KEY"),
        }

    def _test_all_apis(self) -> None:
        """Test all API keys and report status."""
        if self.verbose:
            print("\n" + "=" * 60)
            print("🔑 API KEY VALIDATION")
            print("=" * 60)

        # Test each API
        self.api_status["POLYGON"] = self._test_polygon()
        self.api_status["FMP"] = self._test_fmp()
        self.api_status["FINNHUB"] = self._test_finnhub()
        self.api_status["ALPHAVANTAGE"] = self._test_alphavantage()
        self.api_status["EODHD"] = self._test_eodhd()
        self.api_status["TWELVEDATA"] = self._test_twelvedata()
        self.api_status["NEWSAPI"] = self._test_newsapi()
        self.api_status["NEWSDATA"] = self._test_newsdata()
        self.api_status["MARKETAUX"] = self._test_marketaux()
        self.api_status["FRED"] = self._test_fred()
        self.api_status["YFINANCE"] = self._test_yfinance()  # Always test free source

        # Summary
        working = sum(1 for v in self.api_status.values() if v)
        if self.verbose:
            print("-" * 60)
            print(f"✅ Working APIs: {working}/{len(self.api_status)}")
            print("=" * 60 + "\n")

    def _log(self, msg: str) -> None:
        """Print message only if verbose mode is enabled."""
        if self.verbose:
            print(msg)

    def _test_polygon(self) -> bool:
        """Test Polygon API."""
        if not self.keys.get("POLYGON"):
            self._log("  POLYGON          ❌ No key")
            return False
        try:
            url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/prev?apiKey={self.keys['POLYGON']}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200 and resp.json().get("resultsCount", 0) > 0:
                self._log("  POLYGON          ✅ Working")
                return True
            self._log(f"  POLYGON          ❌ Error: {resp.status_code}")
            return False
        except Exception as e:
            self._log(f"  POLYGON          ❌ {str(e)[:30]}")
            return False

    def _test_fmp(self) -> bool:
        """Test Financial Modeling Prep API."""
        if not self.keys.get("FMP"):
            self._log("  FMP              ❌ No key")
            return False
        try:
            url = f"https://financialmodelingprep.com/api/v3/quote/AAPL?apikey={self.keys['FMP']}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200 and isinstance(resp.json(), list) and len(resp.json()) > 0:
                self._log("  FMP              ✅ Working")
                return True
            self._log(f"  FMP              ❌ Error: {resp.status_code}")
            return False
        except Exception as e:
            self._log(f"  FMP              ❌ {str(e)[:30]}")
            return False

    def _test_finnhub(self) -> bool:
        """Test Finnhub API."""
        if not self.keys.get("FINNHUB"):
            self._log("  FINNHUB          ❌ No key")
            return False
        try:
            url = f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={self.keys['FINNHUB']}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200 and resp.json().get("c", 0) > 0:
                self._log("  FINNHUB          ✅ Working")
                return True
            self._log(f"  FINNHUB          ❌ Error: {resp.status_code}")
            return False
        except Exception as e:
            self._log(f"  FINNHUB          ❌ {str(e)[:30]}")
            return False

    def _test_alphavantage(self) -> bool:
        """Test Alpha Vantage API."""
        if not self.keys.get("ALPHAVANTAGE"):
            self._log("  ALPHAVANTAGE     ❌ No key")
            return False
        try:
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey={self.keys['ALPHAVANTAGE']}"
            resp = requests.get(url, timeout=10)
            data = resp.json()
            if resp.status_code == 200 and "Global Quote" in data and data["Global Quote"]:
                self._log("  ALPHAVANTAGE     ✅ Working")
                return True
            if "Note" in data:  # Rate limit
                self._log("  ALPHAVANTAGE     ⚠️  Rate limited")
                return True  # Key is valid, just rate limited
            self._log("  ALPHAVANTAGE     ❌ Error")
            return False
        except Exception as e:
            self._log(f"  ALPHAVANTAGE     ❌ {str(e)[:30]}")
            return False

    def _test_eodhd(self) -> bool:
        """Test EODHD API."""
        if not self.keys.get("EODHD"):
            self._log("  EODHD            ❌ No key")
            return False
        try:
            url = f"https://eodhd.com/api/real-time/AAPL.US?api_token={self.keys['EODHD']}&fmt=json"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200 and resp.json().get("close", 0) > 0:
                self._log("  EODHD            ✅ Working")
                return True
            self._log(f"  EODHD            ❌ Error: {resp.status_code}")
            return False
        except Exception as e:
            self._log(f"  EODHD            ❌ {str(e)[:30]}")
            return False

    def _test_twelvedata(self) -> bool:
        """Test TwelveData API."""
        if not self.keys.get("TWELVEDATA"):
            self._log("  TWELVEDATA       ❌ No key")
            return False
        try:
            url = f"https://api.twelvedata.com/quote?symbol=AAPL&apikey={self.keys['TWELVEDATA']}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200 and resp.json().get("close"):
                self._log("  TWELVEDATA       ✅ Working")
                return True
            self._log(f"  TWELVEDATA       ❌ Error: {resp.status_code}")
            return False
        except Exception as e:
            self._log(f"  TWELVEDATA       ❌ {str(e)[:30]}")
            return False

    def _test_newsapi(self) -> bool:
        """Test NewsAPI."""
        if not self.keys.get("NEWSAPI"):
            self._log("  NEWSAPI          ❌ No key")
            return False
        try:
            url = f"https://newsapi.org/v2/everything?q=stock&pageSize=1&apiKey={self.keys['NEWSAPI']}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200 and resp.json().get("status") == "ok":
                self._log("  NEWSAPI          ✅ Working")
                return True
            self._log(f"  NEWSAPI          ❌ Error: {resp.status_code}")
            return False
        except Exception as e:
            self._log(f"  NEWSAPI          ❌ {str(e)[:30]}")
            return False

    def _test_newsdata(self) -> bool:
        """Test NewsData.io API."""
        if not self.keys.get("NEWSDATA"):
            self._log("  NEWSDATA         ❌ No key")
            return False
        try:
            url = f"https://newsdata.io/api/1/news?apikey={self.keys['NEWSDATA']}&q=stock&language=en"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200 and resp.json().get("status") == "success":
                self._log("  NEWSDATA         ✅ Working")
                return True
            self._log(f"  NEWSDATA         ❌ Error: {resp.status_code}")
            return False
        except Exception as e:
            self._log(f"  NEWSDATA         ❌ {str(e)[:30]}")
            return False

    def _test_marketaux(self) -> bool:
        """Test MarketAux API."""
        if not self.keys.get("MARKETAUX"):
            self._log("  MARKETAUX        ❌ No key")
            return False
        try:
            url = f"https://api.marketaux.com/v1/news/all?api_token={self.keys['MARKETAUX']}&limit=1"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200 and resp.json().get("data"):
                self._log("  MARKETAUX        ✅ Working")
                return True
            self._log(f"  MARKETAUX        ❌ Error: {resp.status_code}")
            return False
        except Exception as e:
            self._log(f"  MARKETAUX        ❌ {str(e)[:30]}")
            return False

    def _test_fred(self) -> bool:
        """Test FRED API."""
        if not self.keys.get("FRED"):
            self._log("  FRED             ❌ No key")
            return False
        try:
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": "FEDFUNDS",
                "api_key": self.keys["FRED"],
                "file_type": "json",
                "limit": 1,
            }
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200 and "observations" in resp.json():
                self._log("  FRED             ✅ Working")
                return True
            self._log(f"  FRED             ❌ Error: {resp.status_code}")
            return False
        except Exception as e:
            self._log(f"  FRED             ❌ {str(e)[:30]}")
            return False

    def _test_yfinance(self) -> bool:
        """Test yfinance (free, no key needed)."""
        try:
            df = yf.download("AAPL", period="5d", progress=False)
            if len(df) > 0:
                self._log("  YFINANCE (free)  ✅ Working")
                return True
            self._log("  YFINANCE (free)  ❌ No data")
            return False
        except Exception as e:
            self._log(f"  YFINANCE (free)  ❌ {str(e)[:30]}")
            return False

    # =========================================================================
    # STOCK DATA METHODS
    # =========================================================================

    def get_stock_data(
        self,
        symbol: str,
        days: int = 365,
        source: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Get historical stock data with automatic fallback.
        
        Priority: Polygon → FMP → EODHD → AlphaVantage → yfinance (free)
        
        Args:
            symbol: Stock ticker (e.g., "AAPL")
            days: Number of days of history
            source: Force specific source (optional)
        
        Returns:
            DataFrame with OHLCV data or None
        """
        sources = ["POLYGON", "FMP", "EODHD", "ALPHAVANTAGE", "YFINANCE"]
        
        if source:
            sources = [source.upper()]

        for src in sources:
            if src != "YFINANCE" and not self.api_status.get(src, False):
                continue

            try:
                if src == "POLYGON":
                    df = self._fetch_polygon(symbol, days)
                elif src == "FMP":
                    df = self._fetch_fmp(symbol, days)
                elif src == "EODHD":
                    df = self._fetch_eodhd(symbol, days)
                elif src == "ALPHAVANTAGE":
                    df = self._fetch_alphavantage(symbol, days)
                elif src == "YFINANCE":
                    df = self._fetch_yfinance(symbol, days)
                else:
                    continue

                if df is not None and len(df) > 0:
                    df.attrs["source"] = src
                    return df
            except Exception:
                continue

        return None

    def _fetch_polygon(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """Fetch from Polygon.io."""
        end = datetime.now()
        start = end - timedelta(days=days)
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/"
            f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
            f"?apiKey={self.keys['POLYGON']}&limit=50000"
        )
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not data.get("results"):
            return None
        df = pd.DataFrame(data["results"])
        df["Date"] = pd.to_datetime(df["t"], unit="ms")
        df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
        df = df.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
        return df.sort_index()

    def _fetch_fmp(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """Fetch from Financial Modeling Prep."""
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?apikey={self.keys['FMP']}"
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if "historical" not in data:
            return None
        df = pd.DataFrame(data["historical"])
        df["Date"] = pd.to_datetime(df["date"])
        df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
        df = df.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
        df = df.sort_index()
        return df.tail(days)

    def _fetch_eodhd(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """Fetch from EODHD."""
        end = datetime.now()
        start = end - timedelta(days=days)
        url = (
            f"https://eodhd.com/api/eod/{symbol}.US"
            f"?api_token={self.keys['EODHD']}&fmt=json"
            f"&from={start.strftime('%Y-%m-%d')}&to={end.strftime('%Y-%m-%d')}"
        )
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not data or not isinstance(data, list):
            return None
        df = pd.DataFrame(data)
        df["Date"] = pd.to_datetime(df["date"])
        df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
        df = df.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
        return df.sort_index()

    def _fetch_alphavantage(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """Fetch from Alpha Vantage."""
        url = (
            f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY"
            f"&symbol={symbol}&outputsize=full&apikey={self.keys['ALPHAVANTAGE']}"
        )
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        ts = data.get("Time Series (Daily)", {})
        if not ts:
            return None
        df = pd.DataFrame.from_dict(ts, orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns={
            "1. open": "Open", "2. high": "High", "3. low": "Low",
            "4. close": "Close", "5. volume": "Volume"
        })
        df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        df = df.sort_index()
        return df.tail(days)

    def _fetch_yfinance(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """Fetch from yfinance (free)."""
        end = datetime.now()
        start = end - timedelta(days=days)
        df = yf.download(symbol, start=start, end=end, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Open", "High", "Low", "Close", "Volume"]]

    # =========================================================================
    # NEWS DATA METHODS
    # =========================================================================

    def get_news(self, query: str, days: int = 7, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get news articles for a query (ticker or topic).
        
        Args:
            query: Search term (e.g., "AAPL" or "stock market")
            days: How many days back to search
            limit: Max articles to return
        
        Returns:
            List of article dicts with title, url, source, publishedAt
        """
        if self.api_status.get("NEWSAPI"):
            return self._fetch_newsapi(query, days, limit)
        return []

    def _fetch_newsapi(self, query: str, days: int, limit: int) -> List[Dict[str, Any]]:
        """Fetch from NewsAPI."""
        try:
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            url = (
                f"https://newsapi.org/v2/everything"
                f"?q={query}&from={from_date}&sortBy=publishedAt"
                f"&pageSize={limit}&apiKey={self.keys['NEWSAPI']}"
            )
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                return []
            data = resp.json()
            articles = data.get("articles", [])
            return [
                {
                    "title": a.get("title"),
                    "url": a.get("url"),
                    "source": a.get("source", {}).get("name"),
                    "publishedAt": a.get("publishedAt"),
                    "description": a.get("description"),
                }
                for a in articles
            ]
        except Exception:
            return []

    # =========================================================================
    # ECONOMIC DATA METHODS
    # =========================================================================

    def get_economic_data(
        self,
        series_id: str = "FEDFUNDS",
        limit: int = 100,
    ) -> Optional[pd.DataFrame]:
        """
        Get economic data from FRED.
        
        Common series:
            - FEDFUNDS: Federal Funds Rate
            - UNRATE: Unemployment Rate
            - CPIAUCSL: Consumer Price Index
            - GDP: Gross Domestic Product
            - DGS10: 10-Year Treasury Rate
        
        Args:
            series_id: FRED series ID
            limit: Number of observations
        
        Returns:
            DataFrame with date and value columns
        """
        if not self.api_status.get("FRED"):
            return None

        try:
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": series_id,
                "api_key": self.keys["FRED"],
                "file_type": "json",
                "limit": limit,
                "sort_order": "desc",
            }
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                return None
            data = resp.json()
            obs = data.get("observations", [])
            if not obs:
                return None
            df = pd.DataFrame(obs)
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df[["date", "value"]].dropna()
            df = df.set_index("date").sort_index()
            df.attrs["series_id"] = series_id
            return df
        except Exception:
            return None

    # =========================================================================
    # QUOTE / REAL-TIME DATA
    # =========================================================================

    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get real-time quote for a symbol.
        
        Returns dict with: price, change, changePercent, volume, high, low
        """
        # Try Finnhub first (fast real-time)
        if self.api_status.get("FINNHUB"):
            try:
                url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={self.keys['FINNHUB']}"
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("c", 0) > 0:
                        return {
                            "symbol": symbol,
                            "price": data["c"],
                            "change": data["d"],
                            "changePercent": data["dp"],
                            "high": data["h"],
                            "low": data["l"],
                            "open": data["o"],
                            "previousClose": data["pc"],
                            "source": "FINNHUB",
                        }
            except Exception:
                pass

        # Fallback to FMP
        if self.api_status.get("FMP"):
            try:
                url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={self.keys['FMP']}"
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    if data and len(data) > 0:
                        q = data[0]
                        return {
                            "symbol": symbol,
                            "price": q.get("price"),
                            "change": q.get("change"),
                            "changePercent": q.get("changesPercentage"),
                            "high": q.get("dayHigh"),
                            "low": q.get("dayLow"),
                            "open": q.get("open"),
                            "previousClose": q.get("previousClose"),
                            "volume": q.get("volume"),
                            "source": "FMP",
                        }
            except Exception:
                pass

        # Fallback to yfinance
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                "symbol": symbol,
                "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "change": info.get("regularMarketChange"),
                "changePercent": info.get("regularMarketChangePercent"),
                "high": info.get("dayHigh"),
                "low": info.get("dayLow"),
                "open": info.get("open"),
                "previousClose": info.get("previousClose"),
                "volume": info.get("volume"),
                "source": "YFINANCE",
            }
        except Exception:
            return None


# =============================================================================
# CONVENIENCE FUNCTIONS (can be imported directly)
# =============================================================================

_default_fetcher: Optional[DataFetcher] = None


def get_fetcher(verbose: bool = False) -> DataFetcher:
    """Get or create the default DataFetcher instance."""
    global _default_fetcher
    if _default_fetcher is None:
        _default_fetcher = DataFetcher(verbose=verbose)
    return _default_fetcher


def fetch_stock(symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
    """Quick function to fetch stock data."""
    return get_fetcher().get_stock_data(symbol, days)


def fetch_quote(symbol: str) -> Optional[Dict[str, Any]]:
    """Quick function to fetch real-time quote."""
    return get_fetcher().get_quote(symbol)


def fetch_news(query: str, days: int = 7) -> List[Dict[str, Any]]:
    """Quick function to fetch news."""
    return get_fetcher().get_news(query, days)


def fetch_economic(series_id: str = "FEDFUNDS") -> Optional[pd.DataFrame]:
    """Quick function to fetch economic data."""
    return get_fetcher().get_economic_data(series_id)


# =============================================================================
# TEST ON IMPORT
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧪 DATA FETCHER TEST")
    print("=" * 60)

    fetcher = DataFetcher(verbose=True)

    # Test stock data
    print("\n📈 Testing stock data (AAPL)...")
    df = fetcher.get_stock_data("AAPL", days=30)
    if df is not None:
        print(f"   ✅ Got {len(df)} days from {df.attrs.get('source', 'unknown')}")
        print(f"   Latest close: ${df['Close'].iloc[-1]:.2f}")
    else:
        print("   ❌ Failed to get stock data")

    # Test quote
    print("\n💰 Testing real-time quote (MSFT)...")
    quote = fetcher.get_quote("MSFT")
    if quote:
        print(f"   ✅ {quote['symbol']}: ${quote['price']:.2f} ({quote.get('changePercent', 0):.2f}%)")
        print(f"   Source: {quote.get('source')}")
    else:
        print("   ❌ Failed to get quote")

    # Test news
    print("\n📰 Testing news (stock market)...")
    news = fetcher.get_news("stock market", days=3, limit=3)
    if news:
        print(f"   ✅ Got {len(news)} articles")
        for n in news[:2]:
            print(f"   - {n['title'][:50]}...")
    else:
        print("   ❌ Failed to get news")

    # Test economic data
    print("\n📊 Testing economic data (FEDFUNDS)...")
    econ = fetcher.get_economic_data("FEDFUNDS", limit=5)
    if econ is not None:
        print(f"   ✅ Got {len(econ)} observations")
        print(f"   Latest: {econ['value'].iloc[-1]}%")
    else:
        print("   ❌ Failed to get economic data")

    print("\n" + "=" * 60)
    print("✅ DATA FETCHER TEST COMPLETE")
    print("=" * 60 + "\n")
