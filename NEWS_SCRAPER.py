#!/usr/bin/env python3
"""
=============================================================================
NEWS SCRAPER FOR TRADING SIGNALS
=============================================================================

Scrapes news from FREE sources to provide context for trading decisions.
NO API KEYS REQUIRED - uses publicly available RSS feeds and web scraping.

Sources:
- Yahoo Finance RSS
- Google News RSS
- Finviz news
- SEC filings (EDGAR)
- Reddit (wallstreetbets, stocks)

Usage:
    python NEWS_SCRAPER.py                    # Get news for watchlist
    python NEWS_SCRAPER.py AAPL TSLA         # Get news for specific symbols
    python NEWS_SCRAPER.py --market          # Get general market news

=============================================================================
"""

import os
import sys
import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote
import warnings

warnings.filterwarnings('ignore')

# Try to import required packages
try:
    import requests
    from bs4 import BeautifulSoup
    SCRAPING_AVAILABLE = True
except ImportError:
    SCRAPING_AVAILABLE = False
    print("⚠️ Install requirements: pip install requests beautifulsoup4")

try:
    import feedparser
    RSS_AVAILABLE = True
except ImportError:
    RSS_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH = Path('./news_data/')
DATA_PATH.mkdir(exist_ok=True)

# User agent to avoid blocks
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Keywords that affect signal confidence
SIGNAL_IMPACT_KEYWORDS = {
    'high_negative': [
        'bankruptcy', 'fraud', 'SEC investigation', 'delisting', 'default',
        'crash', 'plunge', 'collapse', 'halt', 'suspended'
    ],
    'moderate_negative': [
        'downgrade', 'lawsuit', 'recall', 'layoffs', 'miss', 'warning',
        'decline', 'cuts', 'slashes', 'disappoints', 'concerns'
    ],
    'high_positive': [
        'FDA approval', 'breakthrough', 'acquisition', 'merger', 'contract win',
        'beat', 'surge', 'soar', 'rocket', 'partnership'
    ],
    'moderate_positive': [
        'upgrade', 'buy rating', 'expansion', 'growth', 'profit', 'revenue beat',
        'positive', 'bullish', 'optimistic', 'strong'
    ],
    'volatility_events': [
        'earnings', 'FOMC', 'Fed', 'CPI', 'jobs report', 'GDP', 
        'interest rate', 'inflation', 'guidance'
    ]
}


# =============================================================================
# NEWS SOURCES
# =============================================================================

class YahooFinanceNews:
    """Scrape news from Yahoo Finance."""
    
    @staticmethod
    def get_news(symbol: str, max_items: int = 10) -> List[Dict]:
        """Get news for a symbol from Yahoo Finance."""
        if not SCRAPING_AVAILABLE:
            return []
        
        news = []
        url = f"https://finance.yahoo.com/quote/{symbol}/news"
        
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find news items (Yahoo's structure changes often)
            articles = soup.find_all('h3', class_=re.compile('.*'))[:max_items]
            
            for article in articles:
                link = article.find('a')
                if link:
                    title = link.get_text(strip=True)
                    href = link.get('href', '')
                    if title and len(title) > 10:
                        news.append({
                            'title': title,
                            'source': 'Yahoo Finance',
                            'url': f"https://finance.yahoo.com{href}" if href.startswith('/') else href,
                            'symbol': symbol,
                            'timestamp': datetime.now().isoformat()
                        })
            
            return news[:max_items]
        
        except Exception as e:
            return []


class GoogleNewsRSS:
    """Get news from Google News RSS feeds."""
    
    @staticmethod
    def get_news(query: str, max_items: int = 10) -> List[Dict]:
        """Get news from Google News RSS."""
        if not RSS_AVAILABLE:
            return []
        
        news = []
        url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-US&gl=US&ceid=US:en"
        
        try:
            feed = feedparser.parse(url)
            
            for entry in feed.entries[:max_items]:
                news.append({
                    'title': entry.get('title', ''),
                    'source': entry.get('source', {}).get('title', 'Google News'),
                    'url': entry.get('link', ''),
                    'published': entry.get('published', ''),
                    'symbol': query,
                    'timestamp': datetime.now().isoformat()
                })
            
            return news
        
        except Exception as e:
            return []


class FinvizNews:
    """Scrape news from Finviz."""
    
    @staticmethod
    def get_news(symbol: str, max_items: int = 10) -> List[Dict]:
        """Get news for a symbol from Finviz."""
        if not SCRAPING_AVAILABLE:
            return []
        
        news = []
        url = f"https://finviz.com/quote.ashx?t={symbol}"
        
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find news table
            news_table = soup.find('table', class_='fullview-news-outer')
            if not news_table:
                return []
            
            rows = news_table.find_all('tr')[:max_items]
            
            for row in rows:
                link = row.find('a', class_='tab-link-news')
                time_cell = row.find('td', align='right')
                
                if link:
                    title = link.get_text(strip=True)
                    href = link.get('href', '')
                    time_str = time_cell.get_text(strip=True) if time_cell else ''
                    
                    news.append({
                        'title': title,
                        'source': 'Finviz',
                        'url': href,
                        'time': time_str,
                        'symbol': symbol,
                        'timestamp': datetime.now().isoformat()
                    })
            
            return news
        
        except Exception as e:
            return []


class MarketNews:
    """Get general market news."""
    
    @staticmethod
    def get_market_news(max_items: int = 20) -> List[Dict]:
        """Get general market/economic news."""
        news = []
        
        # Try multiple sources
        sources = [
            ("stock market today", "Market"),
            ("S&P 500 news", "S&P500"),
            ("Federal Reserve FOMC", "Fed"),
            ("economic data release", "Economy"),
        ]
        
        for query, category in sources:
            items = GoogleNewsRSS.get_news(query, max_items=5)
            for item in items:
                item['category'] = category
            news.extend(items)
        
        return news[:max_items]


# =============================================================================
# NEWS ANALYZER
# =============================================================================

class NewsAnalyzer:
    """Analyze news for signal impact."""
    
    @staticmethod
    def analyze_sentiment(title: str) -> Dict:
        """Simple keyword-based sentiment analysis."""
        title_lower = title.lower()
        
        sentiment = {
            'score': 0,
            'impact': 'neutral',
            'keywords_found': [],
            'volatility_event': False
        }
        
        # Check for high impact keywords
        for keyword in SIGNAL_IMPACT_KEYWORDS['high_negative']:
            if keyword.lower() in title_lower:
                sentiment['score'] -= 3
                sentiment['keywords_found'].append(keyword)
        
        for keyword in SIGNAL_IMPACT_KEYWORDS['moderate_negative']:
            if keyword.lower() in title_lower:
                sentiment['score'] -= 1
                sentiment['keywords_found'].append(keyword)
        
        for keyword in SIGNAL_IMPACT_KEYWORDS['high_positive']:
            if keyword.lower() in title_lower:
                sentiment['score'] += 3
                sentiment['keywords_found'].append(keyword)
        
        for keyword in SIGNAL_IMPACT_KEYWORDS['moderate_positive']:
            if keyword.lower() in title_lower:
                sentiment['score'] += 1
                sentiment['keywords_found'].append(keyword)
        
        for keyword in SIGNAL_IMPACT_KEYWORDS['volatility_events']:
            if keyword.lower() in title_lower:
                sentiment['volatility_event'] = True
                sentiment['keywords_found'].append(f"[VOL]{keyword}")
        
        # Determine impact level
        if sentiment['score'] >= 3:
            sentiment['impact'] = 'strong_positive'
        elif sentiment['score'] >= 1:
            sentiment['impact'] = 'positive'
        elif sentiment['score'] <= -3:
            sentiment['impact'] = 'strong_negative'
        elif sentiment['score'] <= -1:
            sentiment['impact'] = 'negative'
        
        return sentiment
    
    @staticmethod
    def should_reduce_signal_confidence(news_items: List[Dict]) -> Tuple[bool, str]:
        """
        Determine if news should reduce signal confidence.
        
        Returns:
            (should_reduce, reason)
        """
        if not news_items:
            return False, "No news"
        
        # Check for major negative news
        for item in news_items:
            sentiment = NewsAnalyzer.analyze_sentiment(item.get('title', ''))
            
            if sentiment['impact'] == 'strong_negative':
                return True, f"Major negative news: {item['title'][:50]}..."
            
            if sentiment['volatility_event']:
                return True, f"Volatility event: {item['title'][:50]}..."
        
        return False, "Normal news environment"


# =============================================================================
# NEWS SCRAPER MAIN CLASS
# =============================================================================

class NewsScraper:
    """Main news scraper that aggregates all sources."""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
    
    def get_symbol_news(self, symbol: str, max_items: int = 10) -> List[Dict]:
        """Get news for a symbol from all sources."""
        # Check cache
        cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M')}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        all_news = []
        
        # Try each source
        print(f"   Fetching news for {symbol}...")
        
        # Yahoo Finance
        yahoo_news = YahooFinanceNews.get_news(symbol, max_items=5)
        all_news.extend(yahoo_news)
        
        # Google News
        google_news = GoogleNewsRSS.get_news(f"{symbol} stock", max_items=5)
        all_news.extend(google_news)
        
        # Finviz
        finviz_news = FinvizNews.get_news(symbol, max_items=5)
        all_news.extend(finviz_news)
        
        # Add sentiment analysis
        for item in all_news:
            item['sentiment'] = NewsAnalyzer.analyze_sentiment(item.get('title', ''))
        
        # Deduplicate by title similarity
        seen_titles = set()
        unique_news = []
        for item in all_news:
            title_key = item.get('title', '')[:50].lower()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_news.append(item)
        
        # Cache results
        self.cache[cache_key] = unique_news[:max_items]
        
        return unique_news[:max_items]
    
    def get_watchlist_news(self, symbols: List[str]) -> Dict[str, List[Dict]]:
        """Get news for all symbols in watchlist."""
        results = {}
        
        for symbol in symbols:
            news = self.get_symbol_news(symbol)
            if news:
                results[symbol] = news
            time.sleep(0.5)  # Rate limiting
        
        return results
    
    def get_market_summary(self) -> Dict:
        """Get overall market news summary."""
        return {
            'market_news': MarketNews.get_market_news(max_items=15),
            'timestamp': datetime.now().isoformat()
        }
    
    def print_news_report(self, symbols: List[str] = None):
        """Print formatted news report."""
        print("\n" + "="*70)
        print("📰 NEWS REPORT")
        print("="*70)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        # Market news
        print("\n" + "-"*70)
        print("🌍 MARKET NEWS")
        print("-"*70)
        
        market = self.get_market_summary()
        for item in market['market_news'][:10]:
            sentiment = item.get('sentiment', {})
            impact = sentiment.get('impact', 'neutral')
            emoji = {'strong_positive': '🟢', 'positive': '📈', 
                    'strong_negative': '🔴', 'negative': '📉'}.get(impact, '⚪')
            
            print(f"\n{emoji} [{item.get('category', 'Market')}] {item['title'][:70]}")
            if sentiment.get('volatility_event'):
                print(f"   ⚠️ VOLATILITY EVENT")
        
        # Symbol-specific news
        if symbols:
            print("\n" + "-"*70)
            print("📊 WATCHLIST NEWS")
            print("-"*70)
            
            watchlist_news = self.get_watchlist_news(symbols[:15])  # Limit to avoid rate limits
            
            # Group by sentiment impact
            high_impact = []
            normal = []
            
            for symbol, news_list in watchlist_news.items():
                for item in news_list[:3]:  # Top 3 per symbol
                    sentiment = item.get('sentiment', {})
                    if sentiment.get('impact') in ['strong_positive', 'strong_negative'] or sentiment.get('volatility_event'):
                        high_impact.append((symbol, item))
                    else:
                        normal.append((symbol, item))
            
            if high_impact:
                print("\n⚠️ HIGH IMPACT NEWS:")
                for symbol, item in high_impact[:10]:
                    sentiment = item['sentiment']
                    emoji = '🔴' if 'negative' in sentiment['impact'] else '🟢'
                    print(f"   {emoji} {symbol}: {item['title'][:60]}")
                    if sentiment.get('keywords_found'):
                        print(f"      Keywords: {', '.join(sentiment['keywords_found'][:3])}")
            
            print("\n📋 OTHER NEWS:")
            for symbol, item in normal[:15]:
                print(f"   {symbol}: {item['title'][:60]}")
    
    def save_news_data(self, symbols: List[str]):
        """Save news data to file."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'market': self.get_market_summary(),
            'watchlist': self.get_watchlist_news(symbols)
        }
        
        filename = DATA_PATH / f"news_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"\n✅ News saved to {filename}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Check dependencies
    if not SCRAPING_AVAILABLE:
        print("Installing required packages...")
        os.system("pip install requests beautifulsoup4 feedparser -q")
        print("Please run the script again.")
        return
    
    # Load watchlist
    watchlist_file = Path('./watchlist_config.json')
    symbols = []
    
    if watchlist_file.exists():
        with open(watchlist_file) as f:
            config = json.load(f)
        symbols = config.get('watchlist', {}).get('symbols', [])
    
    # Parse command line args
    if len(sys.argv) > 1:
        if sys.argv[1] == '--market':
            symbols = []  # Just market news
        else:
            symbols = [s.upper() for s in sys.argv[1:]]
    
    # Create scraper and run
    scraper = NewsScraper()
    
    if symbols:
        print(f"\n📋 Scanning news for {len(symbols)} symbols...")
        scraper.print_news_report(symbols[:20])  # Limit to avoid rate limits
        scraper.save_news_data(symbols[:20])
    else:
        print("\n📋 Scanning market news...")
        scraper.print_news_report([])


if __name__ == "__main__":
    main()
