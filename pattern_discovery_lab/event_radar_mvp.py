"""
EVENT RADAR MVP - Month 1 Prototype

This is NOT pattern discovery. This surfaces FACTS:
- Insider bought stock (fact)
- Company beat earnings (fact)  
- Volume spiked (fact)

User decides what to do with facts.

Data sources (all free):
- OpenInsider: Insider transactions
- Finnhub: Earnings, news
- yfinance: Price, volume
"""

import yfinance as yf
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional
import json


@dataclass
class MarketEvent:
    """A factual market event (not a prediction)"""
    event_type: str
    ticker: str
    description: str
    timestamp: datetime
    data: dict
    source: str
    relevance_score: float = 0.0  # 0-1, how relevant to user's style
    

class EventRadar:
    """
    Surfaces market EVENTS - not predictions.
    
    Key principle: We show WHAT HAPPENED, never WHAT WILL HAPPEN.
    """
    
    def __init__(self, finnhub_api_key: Optional[str] = None):
        self.finnhub_key = finnhub_api_key
        
    # =========================================================================
    # INSIDER BUYING (Research-backed signal: Seyhun 1998)
    # =========================================================================
    
    def get_insider_buys(self, min_value: int = 100_000) -> List[MarketEvent]:
        """
        Get recent insider purchases.
        
        WHY THIS MATTERS (not a prediction, just research):
        - Seyhun (1998): Insiders have asymmetric information
        - Insider BUYS are more informative than sells
        - Large buys (>$100K) by officers are most significant
        
        WE ARE NOT SAYING: "Buy because insider bought"
        WE ARE SAYING: "Insider bought. You might want to look."
        """
        events = []
        
        # OpenInsider scraping (free)
        try:
            # In production, would scrape openinsider.com
            # For MVP, using placeholder data structure
            
            # Example of what we'd return:
            sample_events = [
                {
                    'ticker': 'EXAMPLE',
                    'insider_name': 'John CEO',
                    'title': 'CEO',
                    'transaction_type': 'Purchase',
                    'shares': 10000,
                    'price': 50.00,
                    'value': 500000,
                    'date': datetime.now() - timedelta(days=1),
                }
            ]
            
            for txn in sample_events:
                if txn['value'] >= min_value and txn['transaction_type'] == 'Purchase':
                    events.append(MarketEvent(
                        event_type='insider_buy',
                        ticker=txn['ticker'],
                        description=f"{txn['title']} bought ${txn['value']:,.0f} worth",
                        timestamp=txn['date'],
                        data=txn,
                        source='OpenInsider',
                    ))
                    
        except Exception as e:
            print(f"Error fetching insider data: {e}")
            
        return events
    
    # =========================================================================
    # EARNINGS SURPRISES (Research-backed: Bernard & Thomas 1989)
    # =========================================================================
    
    def get_earnings_surprises(self, min_beat_pct: float = 5.0) -> List[MarketEvent]:
        """
        Get recent earnings beats.
        
        WHY THIS MATTERS (research, not prediction):
        - Bernard & Thomas (1989): Post-earnings announcement drift
        - Market underreacts to earnings surprises initially
        - Drift continues for ~60 days
        
        WE ARE NOT SAYING: "Buy because they beat"
        WE ARE SAYING: "They beat by X%. Historically, drift exists. Research more."
        """
        events = []
        
        if not self.finnhub_key:
            return events
            
        try:
            # Finnhub earnings calendar
            url = f"https://finnhub.io/api/v1/calendar/earnings"
            params = {
                'from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                'to': datetime.now().strftime('%Y-%m-%d'),
                'token': self.finnhub_key,
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            for earning in data.get('earningsCalendar', []):
                actual = earning.get('actual')
                estimate = earning.get('estimate')
                
                if actual and estimate and estimate != 0:
                    beat_pct = ((actual - estimate) / abs(estimate)) * 100
                    
                    if beat_pct >= min_beat_pct:
                        events.append(MarketEvent(
                            event_type='earnings_beat',
                            ticker=earning['symbol'],
                            description=f"Beat estimates by {beat_pct:.1f}%",
                            timestamp=datetime.strptime(earning['date'], '%Y-%m-%d'),
                            data={
                                'actual': actual,
                                'estimate': estimate,
                                'beat_pct': beat_pct,
                            },
                            source='Finnhub',
                        ))
                        
        except Exception as e:
            print(f"Error fetching earnings data: {e}")
            
        return events
    
    # =========================================================================
    # UNUSUAL VOLUME (Research-backed: Karpoff 1987)
    # =========================================================================
    
    def get_unusual_volume(self, 
                           tickers: List[str], 
                           volume_multiplier: float = 3.0) -> List[MarketEvent]:
        """
        Find stocks with unusual volume.
        
        WHY THIS MATTERS (research, not prediction):
        - Karpoff (1987): Volume precedes price moves
        - Unusual volume often indicates informed trading
        - No news + high volume = something is brewing
        
        WE ARE NOT SAYING: "Buy because volume spiked"
        WE ARE SAYING: "Volume is 3x normal. Investigate why."
        """
        events = []
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period='30d')
                
                if len(hist) < 20:
                    continue
                    
                avg_volume = hist['Volume'][:-1].mean()
                latest_volume = hist['Volume'].iloc[-1]
                
                if avg_volume > 0 and latest_volume > avg_volume * volume_multiplier:
                    ratio = latest_volume / avg_volume
                    
                    events.append(MarketEvent(
                        event_type='unusual_volume',
                        ticker=ticker,
                        description=f"Volume {ratio:.1f}x normal average",
                        timestamp=datetime.now(),
                        data={
                            'latest_volume': int(latest_volume),
                            'avg_volume': int(avg_volume),
                            'multiplier': ratio,
                        },
                        source='yfinance',
                    ))
                    
            except Exception as e:
                print(f"Error checking volume for {ticker}: {e}")
                
        return events
    
    # =========================================================================
    # SECTOR ROTATION (Observable money flows)
    # =========================================================================
    
    def get_sector_flows(self) -> List[MarketEvent]:
        """
        Track money flowing between sectors.
        
        WHY THIS MATTERS:
        - Sector rotation is observable (not predictive)
        - Money flowing INTO a sector = attention there
        - If user trades tech, they should know money is moving to energy
        
        WE ARE NOT SAYING: "Buy energy because money flowing there"
        WE ARE SAYING: "Money is moving to energy this week. Aware?"
        """
        events = []
        
        sector_etfs = {
            'XLK': 'Technology',
            'XLF': 'Financials', 
            'XLE': 'Energy',
            'XLV': 'Healthcare',
            'XLY': 'Consumer Discretionary',
            'XLP': 'Consumer Staples',
            'XLI': 'Industrials',
            'XLB': 'Materials',
            'XLU': 'Utilities',
            'XLRE': 'Real Estate',
        }
        
        performances = {}
        
        for etf, sector in sector_etfs.items():
            try:
                stock = yf.Ticker(etf)
                hist = stock.history(period='5d')
                
                if len(hist) >= 2:
                    pct_change = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100
                    performances[sector] = pct_change
                    
            except Exception as e:
                print(f"Error fetching {etf}: {e}")
                
        if performances:
            # Find leaders and laggards
            sorted_sectors = sorted(performances.items(), key=lambda x: x[1], reverse=True)
            
            # Top 2 sectors
            for sector, perf in sorted_sectors[:2]:
                events.append(MarketEvent(
                    event_type='sector_inflow',
                    ticker=sector,
                    description=f"Leading sector: {perf:+.1f}% this week",
                    timestamp=datetime.now(),
                    data={'performance': perf, 'rank': 'leader'},
                    source='yfinance',
                ))
                
            # Bottom 2 sectors  
            for sector, perf in sorted_sectors[-2:]:
                events.append(MarketEvent(
                    event_type='sector_outflow',
                    ticker=sector,
                    description=f"Lagging sector: {perf:+.1f}% this week",
                    timestamp=datetime.now(),
                    data={'performance': perf, 'rank': 'laggard'},
                    source='yfinance',
                ))
                
        return events


class StyleFilter:
    """
    Filter events to match user's trading style.
    
    Key principle: Show RELEVANT events, not ALL events.
    Too many events = noise = ignored.
    """
    
    def __init__(self, user_style: dict):
        """
        user_style example:
        {
            'market_cap': 'small',  # small, mid, large
            'sectors': ['Technology', 'Healthcare'],
            'holding_period': 'swing',  # day, swing, position
        }
        """
        self.style = user_style
        
    def filter_events(self, events: List[MarketEvent], max_events: int = 5) -> List[MarketEvent]:
        """
        Filter and rank events by relevance to user's style.
        
        MAX 5 EVENTS to prevent overload (research: alert fatigue)
        """
        scored_events = []
        
        for event in events:
            score = self._calculate_relevance(event)
            event.relevance_score = score
            scored_events.append(event)
            
        # Sort by relevance, take top N
        scored_events.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return scored_events[:max_events]
    
    def _calculate_relevance(self, event: MarketEvent) -> float:
        """
        Score 0-1 based on style match.
        """
        score = 0.5  # Base score
        
        # Boost for matching sectors
        if event.event_type == 'sector_inflow':
            if event.ticker in self.style.get('sectors', []):
                score += 0.3
                
        # Event type preferences
        event_weights = {
            'insider_buy': 0.8,  # High signal
            'earnings_beat': 0.7,
            'unusual_volume': 0.6,
            'sector_inflow': 0.5,
            'sector_outflow': 0.4,
        }
        
        score *= event_weights.get(event.event_type, 0.5)
        
        return min(score, 1.0)


class EventRadarDisplay:
    """
    Format events for user display.
    
    Key principle: Show facts + context, never predictions.
    """
    
    @staticmethod
    def format_event(event: MarketEvent) -> str:
        """Format single event for display"""
        
        templates = {
            'insider_buy': """
📊 INSIDER PURCHASE: ${ticker}
   {description}
   
   Why this matters (not a prediction):
   - Insiders have information you don't
   - Large buys by officers are significant
   
   Your job: Research why they bought. Decide if you agree.
   [Research] [Add to Watchlist] [Pass]
""",
            'earnings_beat': """
📈 EARNINGS BEAT: ${ticker}
   {description}
   
   Why this matters (not a prediction):
   - Market often underreacts initially (drift effect)
   - Not all beats are equal - was guidance raised?
   
   Your job: Check the call. Sustainable or one-time?
   [Research] [Add to Watchlist] [Pass]
""",
            'unusual_volume': """
🔊 UNUSUAL VOLUME: ${ticker}
   {description}
   
   Why this matters (not a prediction):
   - Volume often precedes price moves
   - No news + high volume = investigate
   
   Your job: Find out WHY volume spiked.
   [Research] [Add to Watchlist] [Pass]
""",
            'sector_inflow': """
💰 SECTOR STRENGTH: {ticker}
   {description}
   
   Why this matters (not a prediction):
   - Money is flowing into this sector
   - Your watchlist may have exposure here
   
   Your job: Are your positions aligned?
   [Review Exposure] [Pass]
""",
        }
        
        template = templates.get(event.event_type, "{description}")
        return template.format(
            ticker=event.ticker,
            description=event.description,
        )
    
    @staticmethod
    def format_daily_digest(events: List[MarketEvent]) -> str:
        """Format daily digest of events"""
        
        header = f"""
╔══════════════════════════════════════════════════════════════╗
║  📰 DAILY EVENT RADAR - {datetime.now().strftime('%Y-%m-%d')}                      ║
║                                                              ║
║  These are FACTS, not predictions.                          ║
║  We show what happened. You decide what it means.           ║
╚══════════════════════════════════════════════════════════════╝

Events matching your style ({len(events)} found, showing top 5):
"""
        
        body = ""
        for i, event in enumerate(events[:5], 1):
            body += f"\n{'─'*60}\n"
            body += f"#{i} | {event.event_type.upper()} | Relevance: {event.relevance_score:.0%}\n"
            body += EventRadarDisplay.format_event(event)
            
        footer = f"""
{'─'*60}

Remember:
• These events don't predict returns
• Use them to LOOK, not to BUY
• Always do your own research
• Most events won't lead to trades (that's good)
"""
        
        return header + body + footer


# =============================================================================
# DEMO / TESTING
# =============================================================================

def demo_event_radar():
    """
    Demonstrate the Event Radar MVP
    """
    print("\n" + "="*70)
    print(" EVENT RADAR MVP DEMO")
    print(" Showing EVENTS (facts), not PREDICTIONS")
    print("="*70)
    
    # Initialize radar
    radar = EventRadar(finnhub_api_key=None)  # No API key for demo
    
    # Define user style
    user_style = {
        'market_cap': 'small',
        'sectors': ['Technology', 'Healthcare'],
        'holding_period': 'swing',
    }
    
    style_filter = StyleFilter(user_style)
    
    # Collect events
    all_events = []
    
    # 1. Sector flows (works without API key)
    print("\n📊 Checking sector rotation...")
    sector_events = radar.get_sector_flows()
    all_events.extend(sector_events)
    print(f"   Found {len(sector_events)} sector events")
    
    # 2. Volume check on sample watchlist
    print("\n🔊 Checking unusual volume...")
    sample_watchlist = ['AAPL', 'MSFT', 'NVDA', 'AMD', 'TSLA']
    volume_events = radar.get_unusual_volume(sample_watchlist)
    all_events.extend(volume_events)
    print(f"   Found {len(volume_events)} volume events")
    
    # 3. Insider buys (placeholder data for demo)
    print("\n📊 Checking insider buys...")
    insider_events = radar.get_insider_buys()
    all_events.extend(insider_events)
    print(f"   Found {len(insider_events)} insider events")
    
    # Filter by style
    print("\n🎯 Filtering to your style...")
    filtered_events = style_filter.filter_events(all_events, max_events=5)
    print(f"   Showing {len(filtered_events)} most relevant events")
    
    # Display
    display = EventRadarDisplay()
    digest = display.format_daily_digest(filtered_events)
    print(digest)
    
    # Summary
    print("\n" + "="*70)
    print(" KEY PRINCIPLES")
    print("="*70)
    print("""
    1. We showed EVENTS (facts), not PREDICTIONS
    2. We filtered to YOUR STYLE (not everything)
    3. We capped at 5 events (prevent overload)
    4. Each event says "Your job: Research"
    5. We never said "BUY" or "SELL"
    
    This is NOT pattern discovery.
    This is information delivery.
    
    Next steps:
    - Add Finnhub API key for earnings data
    - Scrape OpenInsider for real insider data
    - Build the Watchlist Expander (Month 2)
    """)


if __name__ == "__main__":
    demo_event_radar()
