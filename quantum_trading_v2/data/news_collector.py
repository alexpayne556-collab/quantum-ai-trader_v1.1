"""NewsAPI integration with sentiment analysis and symbol extraction."""
import requests
import re
from typing import List, Dict, Any
from datetime import datetime

class NewsCollector:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"

    def fetch_news(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        params = {"q": query, "pageSize": limit, "apiKey": self.api_key}
        try:
            resp = requests.get(self.base_url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles", [])
            return [self._process_article(a) for a in articles]
        except Exception as e:
            print(f"[NewsCollector] Error fetching news: {e}")
            return []

    def _process_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        headline = article.get("title", "")
        symbol = self._extract_symbol(headline)
        sentiment = self._score_sentiment(headline)
        return {
            "symbol": symbol,
            "headline": headline,
            "sentiment": sentiment,
            "published_date": article.get("publishedAt", datetime.now().isoformat()),
            "source": article.get("source", {}).get("name", "")
        }

    def _extract_symbol(self, headline: str) -> str:
        # Simple symbol extraction: look for $SYMBOL or uppercase tickers
        match = re.search(r"\$([A-Z]{2,5})", headline)
        if match:
            return match.group(1)
        # Fallback: find all-caps words
        words = re.findall(r"\b[A-Z]{2,5}\b", headline)
        return words[0] if words else "UNKNOWN"

    def _score_sentiment(self, text: str) -> str:
        # Simple rule-based sentiment scoring
        text = text.lower()
        if any(w in text for w in ["surge", "beat", "rise", "growth", "up", "record"]):
            return "positive"
        if any(w in text for w in ["fall", "miss", "drop", "down", "loss", "decline"]):
            return "negative"
        return "neutral"
