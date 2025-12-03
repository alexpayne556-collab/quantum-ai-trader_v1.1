"""Finnhub API integration for historical and real-time price collection with error handling."""
import requests
from datetime import datetime, timedelta
import time

class PriceCollector:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1"
        self.last_call = 0
        self.min_interval = 1.0  # seconds

    def _rate_limit(self):
        now = time.time()
        wait = self.min_interval - (now - self.last_call)
        if wait > 0:
            time.sleep(wait)
        self.last_call = time.time()

    def fetch_historical(self, symbol: str, days: int = 30, timeframe: str = "D"):
        self._rate_limit()
        end = int(datetime.now().timestamp())
        start = int((datetime.now() - timedelta(days=days)).timestamp())
        url = f"{self.base_url}/stock/candle"
        params = {
            "symbol": symbol,
            "resolution": timeframe,
            "from": start,
            "to": end,
            "token": self.api_key
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("s") != "ok":
                print(f"[PriceCollector] Finnhub error: {data.get('s')}")
                return []
            candles = []
            for i in range(len(data["t"])):
                candles.append({
                    "symbol": symbol,
                    "date": datetime.fromtimestamp(data["t"][i]).strftime("%Y-%m-%d"),
                    "open": data["o"][i],
                    "high": data["h"][i],
                    "low": data["l"][i],
                    "close": data["c"][i],
                    "volume": data["v"][i],
                    "timeframe": timeframe
                })
            return candles
        except Exception as e:
            print(f"[PriceCollector] Error fetching historical: {e}")
            return []

    def fetch_quote(self, symbol: str):
        self._rate_limit()
        url = f"{self.base_url}/quote"
        params = {"symbol": symbol, "token": self.api_key}
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if "c" not in data:
                print(f"[PriceCollector] Finnhub quote error: {data}")
                return None
            return {
                "symbol": symbol,
                "price": data["c"],
                "high": data["h"],
                "low": data["l"],
                "open": data["o"],
                "previous_close": data["pc"],
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"[PriceCollector] Error fetching quote: {e}")
            return None
