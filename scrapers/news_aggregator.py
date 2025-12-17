"""
NEWS AGGREGATOR - Multi-Source News Scraping
Combines: Yahoo Finance, Google News RSS, Finnhub, Alpha Vantage
Returns unified format: [{ticker, headline, source, sentiment_score, timestamp}]
"""

import requests
import feedparser
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import time
import re
from bs4 import BeautifulSoup
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

class NewsAggregator:
    """
    Multi-source news aggregator with sentiment scoring.
    Sources: Yahoo Finance, Google News RSS, Finnhub, Alpha Vantage
    """
    
    def __init__(self, finnhub_key: str = None, alpha_vantage_key: str = None):
        self.finnhub_key = finnhub_key
        self.alpha_vantage_key = alpha_vantage_key
        self.cache = {}
        self.rate_limit_delay = 1  # seconds between calls
        
    def get_yahoo_news(self, ticker: str) -> List[Dict]:
        """Scrape Yahoo Finance news for a ticker"""
        news_items = []
        try:
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
            feed = feedparser.parse(url)
            
            for entry in feed.entries[:10]:  # Last 10 articles
                sentiment = self._calculate_sentiment(entry.title)
                news_items.append({
                    'ticker': ticker,
                    'headline': entry.title,
                    'summary': entry.get('summary', '')[:200],
                    'source': 'Yahoo Finance',
                    'sentiment_score': sentiment,
                    'sentiment_label': self._label_sentiment(sentiment),
                    'timestamp': entry.get('published', datetime.now().isoformat()),
                    'url': entry.get('link', '')
                })
        except Exception as e:
            print(f"Yahoo News error for {ticker}: {e}")
        
        return news_items
    
    def get_google_news(self, query: str) -> List[Dict]:
        """Scrape Google News RSS for a search query"""
        news_items = []
        try:
            # URL encode the query
            encoded_query = query.replace(' ', '+')
            url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(url)
            
            for entry in feed.entries[:10]:
                sentiment = self._calculate_sentiment(entry.title)
                # Extract source from title (Google News format: "Title - Source")
                title_parts = entry.title.rsplit(' - ', 1)
                headline = title_parts[0] if len(title_parts) > 1 else entry.title
                source = title_parts[1] if len(title_parts) > 1 else 'Google News'
                
                news_items.append({
                    'ticker': query,
                    'headline': headline,
                    'summary': '',
                    'source': f'Google News ({source})',
                    'sentiment_score': sentiment,
                    'sentiment_label': self._label_sentiment(sentiment),
                    'timestamp': entry.get('published', datetime.now().isoformat()),
                    'url': entry.get('link', '')
                })
        except Exception as e:
            print(f"Google News error for {query}: {e}")
        
        return news_items
    
    def get_finnhub_news(self, ticker: str) -> List[Dict]:
        """Get news from Finnhub API (requires API key)"""
        news_items = []
        if not self.finnhub_key:
            return news_items
            
        try:
            end = datetime.now()
            start = end - timedelta(days=7)
            url = f"https://finnhub.io/api/v1/company-news"
            params = {
                'symbol': ticker,
                'from': start.strftime('%Y-%m-%d'),
                'to': end.strftime('%Y-%m-%d'),
                'token': self.finnhub_key
            }
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                for article in data[:10]:
                    sentiment = self._calculate_sentiment(article.get('headline', ''))
                    news_items.append({
                        'ticker': ticker,
                        'headline': article.get('headline', ''),
                        'summary': article.get('summary', '')[:200],
                        'source': f"Finnhub ({article.get('source', 'Unknown')})",
                        'sentiment_score': sentiment,
                        'sentiment_label': self._label_sentiment(sentiment),
                        'timestamp': datetime.fromtimestamp(article.get('datetime', 0)).isoformat(),
                        'url': article.get('url', '')
                    })
        except Exception as e:
            print(f"Finnhub error for {ticker}: {e}")
            
        return news_items
    
    def get_alpha_vantage_news(self, ticker: str) -> List[Dict]:
        """Get news from Alpha Vantage API (requires API key)"""
        news_items = []
        if not self.alpha_vantage_key:
            return news_items
            
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': ticker,
                'apikey': self.alpha_vantage_key
            }
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                for article in data.get('feed', [])[:10]:
                    # Alpha Vantage provides its own sentiment
                    av_sentiment = float(article.get('overall_sentiment_score', 0))
                    news_items.append({
                        'ticker': ticker,
                        'headline': article.get('title', ''),
                        'summary': article.get('summary', '')[:200],
                        'source': f"Alpha Vantage ({article.get('source', 'Unknown')})",
                        'sentiment_score': av_sentiment,
                        'sentiment_label': self._label_sentiment(av_sentiment),
                        'timestamp': article.get('time_published', ''),
                        'url': article.get('url', '')
                    })
        except Exception as e:
            print(f"Alpha Vantage error for {ticker}: {e}")
            
        return news_items
    
    def get_all_news(self, ticker: str, company_name: str = None) -> pd.DataFrame:
        """
        Aggregate news from ALL sources for a ticker.
        Returns DataFrame sorted by timestamp (newest first).
        """
        all_news = []
        
        # Yahoo Finance (always available)
        all_news.extend(self.get_yahoo_news(ticker))
        time.sleep(self.rate_limit_delay)
        
        # Google News (search by company name if provided)
        search_term = company_name if company_name else ticker
        all_news.extend(self.get_google_news(f"{search_term} stock"))
        time.sleep(self.rate_limit_delay)
        
        # Finnhub (if key available)
        all_news.extend(self.get_finnhub_news(ticker))
        time.sleep(self.rate_limit_delay)
        
        # Alpha Vantage (if key available)
        all_news.extend(self.get_alpha_vantage_news(ticker))
        
        if not all_news:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_news)
        
        # Remove duplicates based on headline similarity
        df = df.drop_duplicates(subset=['headline'])
        
        # Sort by timestamp (newest first)
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp', ascending=False)
        except:
            pass
        
        return df
    
    def get_news_for_portfolio(self, tickers: List[str]) -> pd.DataFrame:
        """Get news for multiple tickers, sorted by sentiment score"""
        all_news = []
        
        print(f"Fetching news for {len(tickers)} tickers...")
        for i, ticker in enumerate(tickers):
            print(f"  [{i+1}/{len(tickers)}] {ticker}...", end=" ")
            df = self.get_all_news(ticker)
            if len(df) > 0:
                all_news.append(df)
                print(f"{len(df)} articles")
            else:
                print("no articles")
            time.sleep(0.5)  # Rate limiting
        
        if not all_news:
            return pd.DataFrame()
        
        combined = pd.concat(all_news, ignore_index=True)
        return combined
    
    def get_sentiment_summary(self, tickers: List[str]) -> pd.DataFrame:
        """
        Get aggregated sentiment summary for multiple tickers.
        Returns DataFrame with: ticker, avg_sentiment, article_count, sentiment_trend
        """
        summaries = []
        
        for ticker in tickers:
            df = self.get_all_news(ticker)
            if len(df) > 0:
                avg_sentiment = df['sentiment_score'].mean()
                article_count = len(df)
                positive_pct = (df['sentiment_score'] > 0.1).sum() / len(df) * 100
                negative_pct = (df['sentiment_score'] < -0.1).sum() / len(df) * 100
                
                summaries.append({
                    'ticker': ticker,
                    'avg_sentiment': round(avg_sentiment, 3),
                    'article_count': article_count,
                    'positive_pct': round(positive_pct, 1),
                    'negative_pct': round(negative_pct, 1),
                    'sentiment_label': self._label_sentiment(avg_sentiment)
                })
            time.sleep(0.5)
        
        if not summaries:
            return pd.DataFrame()
        
        df = pd.DataFrame(summaries)
        return df.sort_values('avg_sentiment', ascending=False)
    
    def _calculate_sentiment(self, text: str) -> float:
        """Calculate sentiment score using TextBlob (-1 to 1)"""
        if not text:
            return 0.0
        try:
            blob = TextBlob(text)
            return round(blob.sentiment.polarity, 3)
        except:
            return 0.0
    
    def _label_sentiment(self, score: float) -> str:
        """Convert sentiment score to label"""
        if score > 0.2:
            return 'VERY_POSITIVE'
        elif score > 0.05:
            return 'POSITIVE'
        elif score > -0.05:
            return 'NEUTRAL'
        elif score > -0.2:
            return 'NEGATIVE'
        else:
            return 'VERY_NEGATIVE'


class SectorNewsAggregator:
    """
    Aggregate news by sector to detect sector-wide momentum.
    """
    
    SECTORS = {
        'QUANTUM': ['IONQ', 'RGTI', 'QUBT', 'QMCO', 'ARQQ'],
        'CRYPTO': ['MARA', 'RIOT', 'CLSK', 'COIN', 'HUT'],
        'SPACE': ['RKLB', 'ASTS', 'SPIR', 'LUNR', 'BKSY'],
        'BIOTECH': ['NTLA', 'BEAM', 'CRSP', 'RXRX', 'AKRO', 'VKTX'],
        'AI_TECH': ['NVDA', 'AMD', 'PLTR', 'SMCI', 'AI', 'PATH'],
        'EV': ['TSLA', 'RIVN', 'LCID', 'QS', 'JOBY'],
        'CLEAN_ENERGY': ['PLUG', 'FCEL', 'ENPH', 'RUN'],
        'FINTECH': ['SOFI', 'UPST', 'AFRM', 'HOOD', 'SQ']
    }
    
    def __init__(self, base_aggregator: NewsAggregator = None):
        self.aggregator = base_aggregator or NewsAggregator()
    
    def get_sector_sentiment(self, sector: str) -> Dict:
        """Get aggregated sentiment for a sector"""
        if sector not in self.SECTORS:
            return {'error': f'Unknown sector: {sector}'}
        
        tickers = self.SECTORS[sector]
        sentiment_df = self.aggregator.get_sentiment_summary(tickers)
        
        if len(sentiment_df) == 0:
            return {'sector': sector, 'sentiment': 0, 'articles': 0}
        
        return {
            'sector': sector,
            'avg_sentiment': round(sentiment_df['avg_sentiment'].mean(), 3),
            'total_articles': sentiment_df['article_count'].sum(),
            'tickers_analyzed': len(sentiment_df),
            'most_positive': sentiment_df.iloc[0]['ticker'] if len(sentiment_df) > 0 else None,
            'most_negative': sentiment_df.iloc[-1]['ticker'] if len(sentiment_df) > 0 else None,
            'details': sentiment_df.to_dict('records')
        }
    
    def rank_sectors(self) -> pd.DataFrame:
        """Rank all sectors by sentiment (positive to negative)"""
        sector_data = []
        
        print("Ranking sectors by sentiment...")
        for sector in self.SECTORS:
            print(f"  Analyzing {sector}...")
            result = self.get_sector_sentiment(sector)
            if 'error' not in result:
                sector_data.append({
                    'sector': sector,
                    'avg_sentiment': result['avg_sentiment'],
                    'article_count': result['total_articles'],
                    'most_positive_ticker': result['most_positive'],
                    'most_negative_ticker': result['most_negative']
                })
        
        df = pd.DataFrame(sector_data)
        return df.sort_values('avg_sentiment', ascending=False)


# Quick test function
def quick_test():
    """Quick test of news aggregator"""
    print("="*60)
    print("NEWS AGGREGATOR TEST")
    print("="*60)
    
    agg = NewsAggregator()
    
    # Test single ticker
    print("\n1. Testing single ticker (NVDA)...")
    df = agg.get_all_news('NVDA', 'NVIDIA')
    print(f"   Found {len(df)} articles")
    if len(df) > 0:
        print(f"   Avg sentiment: {df['sentiment_score'].mean():.3f}")
        print(f"   Sample headlines:")
        for _, row in df.head(3).iterrows():
            print(f"   - {row['headline'][:60]}... ({row['sentiment_label']})")
    
    return df


if __name__ == "__main__":
    quick_test()
