"""
🔭 MARKET INTELLIGENCE SYSTEM - Prototype

This is what we're building: A system that helps you SEE more than you can see alone.

NOT prediction. INFORMATION GATHERING.

Components:
1. Event Radar - What happened overnight?
2. Buzz Tracker - What's getting attention?
3. Supply Chain Web - If X moves, who else?
4. Theme Detector - What's emerging?
5. Morning Brief - Your 5-minute update

This prototype shows the VISION. We'll build each piece over 30 days.
"""

import yfinance as yf
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import requests
import json


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class IntelligenceItem:
    """A piece of market intelligence"""
    category: str           # 'event', 'buzz', 'supply_chain', 'theme', 'macro'
    importance: int         # 1-5 (5 = most important)
    ticker: Optional[str]   # Related ticker(s)
    headline: str           # What happened
    detail: str            # Why it matters
    source: str            # Where we got this
    timestamp: datetime     # When
    action_prompt: str      # What user should consider doing


@dataclass  
class UserProfile:
    """User's trading style and watchlist"""
    watchlist: List[str]
    sectors_of_interest: List[str]
    market_cap_preference: str  # 'small', 'mid', 'large', 'any'
    style: str  # 'momentum', 'value', 'growth', 'swing'
    risk_tolerance: str  # 'conservative', 'moderate', 'aggressive'


# =============================================================================
# SUPPLY CHAIN MAP (The Innovation)
# =============================================================================

class SupplyChainWeb:
    """
    Maps company relationships so you can see:
    - If AAPL moves, who else might move?
    - Who supplies NVDA?
    - Who are TSLA's customers?
    
    This is HAND-CURATED for MVP. Later we'll parse 10-K filings.
    """
    
    # Major company supply chain relationships
    # Format: company -> [suppliers, customers, competitors]
    SUPPLY_CHAIN = {
        'AAPL': {
            'suppliers': ['TSM', 'QRVO', 'SWKS', 'AVGO', 'MU', 'CRUS', 'LRCX', 'AMAT'],
            'customers': [],  # B2C
            'competitors': ['GOOGL', 'MSFT', 'SAMSUNG'],
            'themes': ['smartphones', 'wearables', 'services'],
        },
        'NVDA': {
            'suppliers': ['TSM', 'ASML', 'AMAT', 'LRCX', 'KLAC', 'MU', 'SK Hynix'],
            'customers': ['MSFT', 'GOOGL', 'META', 'AMZN', 'TSLA'],  # AI buyers
            'competitors': ['AMD', 'INTC', 'QCOM'],
            'themes': ['AI', 'data center', 'gaming', 'autonomous vehicles'],
        },
        'TSLA': {
            'suppliers': ['PCRFY', 'ALB', 'SQM', 'LTHM', 'LAC'],  # Battery materials
            'customers': [],  # B2C
            'competitors': ['F', 'GM', 'RIVN', 'LCID', 'NIO', 'BYD'],
            'themes': ['EV', 'autonomous', 'energy storage', 'robotics'],
        },
        'MSFT': {
            'suppliers': ['NVDA', 'AMD', 'INTC'],
            'customers': [],  # B2B/B2C mixed
            'competitors': ['GOOGL', 'AMZN', 'CRM'],
            'themes': ['cloud', 'AI', 'enterprise software', 'gaming'],
        },
        'AMZN': {
            'suppliers': ['NVDA', 'AMD', 'INTC'],  # AWS
            'customers': [],  # Platform
            'competitors': ['WMT', 'MSFT', 'GOOGL', 'SHOP'],
            'themes': ['e-commerce', 'cloud', 'advertising', 'logistics'],
        },
        'META': {
            'suppliers': ['NVDA', 'AMD'],  # AI infrastructure
            'customers': [],  # Platform
            'competitors': ['GOOGL', 'SNAP', 'PINS', 'TIKTOK'],
            'themes': ['social', 'advertising', 'VR/AR', 'AI'],
        },
        'TSM': {
            'suppliers': ['ASML', 'AMAT', 'LRCX', 'KLAC', 'TOELY'],
            'customers': ['AAPL', 'NVDA', 'AMD', 'QCOM', 'AVGO'],
            'competitors': ['INTC', 'SAMSUNG'],
            'themes': ['semiconductors', 'advanced manufacturing'],
        },
        'ASML': {
            'suppliers': ['ZEISS', 'CYMER'],
            'customers': ['TSM', 'INTC', 'SAMSUNG'],
            'competitors': [],  # Near monopoly in EUV
            'themes': ['semiconductors', 'lithography', 'advanced manufacturing'],
        },
    }
    
    def get_related_tickers(self, ticker: str) -> Dict:
        """Given a ticker, return all related companies"""
        if ticker not in self.SUPPLY_CHAIN:
            return {'ticker': ticker, 'message': 'Supply chain not mapped yet'}
            
        data = self.SUPPLY_CHAIN[ticker]
        return {
            'ticker': ticker,
            'if_this_moves_check': {
                'suppliers': data['suppliers'],
                'reasoning': f"If {ticker} is doing well, demand for suppliers may increase",
            },
            'also_affected': {
                'customers': data['customers'],
                'reasoning': f"If {ticker} has news, it may affect these customers",
            },
            'compare_to': {
                'competitors': data['competitors'],
                'reasoning': f"If {ticker} gains, competitors may lose (or sector is hot)",
            },
            'themes': data['themes'],
        }
    
    def find_by_theme(self, theme: str) -> List[str]:
        """Find all companies related to a theme"""
        related = []
        for ticker, data in self.SUPPLY_CHAIN.items():
            if theme.lower() in [t.lower() for t in data['themes']]:
                related.append(ticker)
        return related
    
    def alert_cascade(self, ticker: str, event: str) -> List[IntelligenceItem]:
        """
        When something happens to a company, alert about related companies.
        
        Example: "AAPL beats earnings" -> Alert about QRVO, SWKS, etc.
        """
        alerts = []
        related = self.get_related_tickers(ticker)
        
        if 'message' in related:
            return alerts
            
        # Alert about suppliers
        for supplier in related['if_this_moves_check']['suppliers'][:3]:
            alerts.append(IntelligenceItem(
                category='supply_chain',
                importance=3,
                ticker=supplier,
                headline=f"📦 {ticker} event may affect {supplier}",
                detail=f"{ticker}: {event}\n{supplier} is a supplier to {ticker}. Check if relevant.",
                source='Supply Chain Map',
                timestamp=datetime.now(),
                action_prompt=f"Research {supplier}'s exposure to {ticker}",
            ))
            
        # Alert about competitors
        for competitor in related['compare_to']['competitors'][:2]:
            alerts.append(IntelligenceItem(
                category='supply_chain',
                importance=2,
                ticker=competitor,
                headline=f"⚔️ {ticker} event - check competitor {competitor}",
                detail=f"{ticker}: {event}\n{competitor} competes with {ticker}. Relative play?",
                source='Supply Chain Map',
                timestamp=datetime.now(),
                action_prompt=f"Compare {ticker} vs {competitor} positioning",
            ))
            
        return alerts


# =============================================================================
# THEME DETECTOR
# =============================================================================

class ThemeDetector:
    """
    Detect investment themes and map to stocks.
    
    Future: Parse arXiv, patents, VC funding
    Now: Manual theme definitions
    """
    
    THEMES = {
        'AI': {
            'description': 'Artificial intelligence and machine learning',
            'leaders': ['NVDA', 'MSFT', 'GOOGL', 'META', 'AMZN'],
            'picks_and_shovels': ['ASML', 'TSM', 'AMAT', 'LRCX', 'MU'],
            'emerging': ['SMCI', 'PLTR', 'AI', 'PATH', 'SNOW'],
            'signals_to_watch': [
                'GPU demand / shortages',
                'Cloud capex announcements',
                'AI model releases (GPT-5, etc.)',
                'Enterprise AI adoption news',
            ],
        },
        'EV': {
            'description': 'Electric vehicles and battery technology',
            'leaders': ['TSLA', 'BYD', 'RIVN', 'LCID'],
            'picks_and_shovels': ['ALB', 'SQM', 'LTHM', 'LAC', 'MP'],
            'emerging': ['CHPT', 'EVGO', 'BLNK'],
            'signals_to_watch': [
                'Battery material prices (lithium, cobalt)',
                'EV sales numbers by region',
                'Charging infrastructure buildout',
                'Government incentive changes',
            ],
        },
        'cybersecurity': {
            'description': 'Security software and services',
            'leaders': ['CRWD', 'PANW', 'ZS', 'FTNT'],
            'picks_and_shovels': [],
            'emerging': ['S', 'NET', 'DDOG'],
            'signals_to_watch': [
                'Major breach news',
                'Government security mandates',
                'Enterprise security spending',
            ],
        },
        'obesity_drugs': {
            'description': 'GLP-1 and weight loss medications',
            'leaders': ['LLY', 'NVO'],
            'picks_and_shovels': ['TMO', 'DHR'],  # Lab equipment
            'emerging': ['VKTX', 'ALT'],
            'signals_to_watch': [
                'Clinical trial results',
                'Insurance coverage decisions',
                'FDA approvals',
                'Supply chain news',
            ],
        },
        'nuclear_renaissance': {
            'description': 'Nuclear energy revival for AI data centers',
            'leaders': ['CEG', 'VST', 'CCJ'],
            'picks_and_shovels': ['CCJ', 'UEC', 'UUUU'],  # Uranium
            'emerging': ['SMR', 'OKLO', 'NNE'],
            'signals_to_watch': [
                'Data center power deals',
                'Uranium prices',
                'Regulatory changes',
                'Tech company nuclear announcements',
            ],
        },
    }
    
    def get_theme_stocks(self, theme: str) -> Dict:
        """Get all stocks related to a theme"""
        if theme.lower() not in [t.lower() for t in self.THEMES.keys()]:
            return {'error': f'Theme {theme} not found'}
            
        theme_key = [k for k in self.THEMES.keys() if k.lower() == theme.lower()][0]
        return self.THEMES[theme_key]
    
    def get_all_themes(self) -> List[str]:
        """List all tracked themes"""
        return list(self.THEMES.keys())
    
    def theme_for_stock(self, ticker: str) -> List[str]:
        """Find which themes a stock belongs to"""
        themes = []
        for theme_name, theme_data in self.THEMES.items():
            all_stocks = (theme_data['leaders'] + 
                         theme_data['picks_and_shovels'] + 
                         theme_data['emerging'])
            if ticker in all_stocks:
                themes.append(theme_name)
        return themes


# =============================================================================
# BUZZ TRACKER (Social Sentiment)
# =============================================================================

class BuzzTracker:
    """
    Track social media mentions and sentiment.
    
    Future: Reddit API, Twitter, StockTwits
    Now: Placeholder structure
    """
    
    def __init__(self):
        self.buzz_scores = {}  # ticker -> score
        
    def get_trending(self, min_score: int = 50) -> List[Dict]:
        """
        Get trending tickers by social buzz.
        
        In production: 
        - Pull from Reddit (r/wallstreetbets, r/stocks)
        - Pull from StockTwits API
        - Calculate velocity of mentions (not just count)
        """
        # Placeholder - would be real API calls
        return [
            {
                'ticker': 'EXAMPLE',
                'buzz_score': 75,
                'mention_velocity': '+200% vs yesterday',
                'sentiment': 'bullish',
                'source': 'Reddit + StockTwits',
                'warning': 'High buzz can mean crowded/late',
            }
        ]
    
    def check_ticker_buzz(self, ticker: str) -> Dict:
        """Check buzz for a specific ticker"""
        return {
            'ticker': ticker,
            'reddit_mentions_24h': 'N/A (API not connected)',
            'stocktwits_sentiment': 'N/A (API not connected)',
            'google_trends': 'N/A (API not connected)',
            'status': 'Buzz tracking requires API integration',
        }


# =============================================================================
# MORNING BRIEF GENERATOR
# =============================================================================

class MorningBrief:
    """
    Generate a personalized morning briefing.
    
    "If you only have 5 minutes, here's what matters for YOUR positions."
    """
    
    def __init__(self, user: UserProfile):
        self.user = user
        self.supply_chain = SupplyChainWeb()
        self.themes = ThemeDetector()
        self.buzz = BuzzTracker()
        
    def generate_brief(self) -> str:
        """Generate the morning brief"""
        
        sections = []
        
        # Header
        sections.append(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  🌅 MORNING BRIEF - {datetime.now().strftime('%A, %B %d, %Y')}                        
║  
║  Personalized for: {self.user.style.upper()} style, {self.user.market_cap_preference} cap focus
║  Watchlist: {', '.join(self.user.watchlist[:5])}{'...' if len(self.user.watchlist) > 5 else ''}
╚══════════════════════════════════════════════════════════════════════╝
""")
        
        # Section 1: Watchlist Movers
        sections.append(self._watchlist_movers())
        
        # Section 2: Sector Check
        sections.append(self._sector_check())
        
        # Section 3: Supply Chain Alerts
        sections.append(self._supply_chain_alerts())
        
        # Section 4: Theme Updates
        sections.append(self._theme_updates())
        
        # Section 5: Action Items
        sections.append(self._action_items())
        
        return '\n'.join(sections)
    
    def _watchlist_movers(self) -> str:
        """Check pre-market/overnight moves on watchlist"""
        output = """
┌─────────────────────────────────────────────────────────────────────┐
│  📊 YOUR WATCHLIST - OVERNIGHT MOVES                               │
└─────────────────────────────────────────────────────────────────────┘
"""
        
        for ticker in self.user.watchlist[:10]:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                current = info.get('regularMarketPrice', 0)
                prev_close = info.get('regularMarketPreviousClose', 0)
                
                if prev_close > 0:
                    change_pct = ((current - prev_close) / prev_close) * 100
                    emoji = "🟢" if change_pct > 0 else "🔴" if change_pct < 0 else "⚪"
                    output += f"  {emoji} {ticker}: ${current:.2f} ({change_pct:+.1f}%)\n"
                else:
                    output += f"  ⚪ {ticker}: Data unavailable\n"
                    
            except Exception as e:
                output += f"  ⚪ {ticker}: Error fetching data\n"
                
        return output
    
    def _sector_check(self) -> str:
        """Check sector performance"""
        output = """
┌─────────────────────────────────────────────────────────────────────┐
│  🏭 SECTOR PERFORMANCE (5-Day)                                      │
└─────────────────────────────────────────────────────────────────────┘
"""
        
        sectors = {
            'XLK': 'Technology',
            'XLF': 'Financials',
            'XLE': 'Energy',
            'XLV': 'Healthcare',
            'XLY': 'Consumer Disc',
            'XLI': 'Industrials',
        }
        
        performances = []
        for etf, name in sectors.items():
            try:
                stock = yf.Ticker(etf)
                hist = stock.history(period='5d')
                if len(hist) >= 2:
                    change = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100
                    performances.append((name, change))
            except:
                pass
                
        # Sort by performance
        performances.sort(key=lambda x: x[1], reverse=True)
        
        for name, change in performances:
            emoji = "🟢" if change > 0 else "🔴"
            bar = "█" * int(abs(change) * 2) if abs(change) < 5 else "█" * 10
            output += f"  {emoji} {name:15} {change:+5.1f}% {bar}\n"
            
        return output
    
    def _supply_chain_alerts(self) -> str:
        """Supply chain connections for watchlist"""
        output = """
┌─────────────────────────────────────────────────────────────────────┐
│  🔗 SUPPLY CHAIN CONNECTIONS                                        │
└─────────────────────────────────────────────────────────────────────┘
"""
        
        for ticker in self.user.watchlist[:5]:
            related = self.supply_chain.get_related_tickers(ticker)
            if 'message' not in related:
                suppliers = related['if_this_moves_check']['suppliers'][:3]
                competitors = related['compare_to']['competitors'][:2]
                output += f"\n  {ticker}:\n"
                output += f"    📦 Suppliers: {', '.join(suppliers) if suppliers else 'None mapped'}\n"
                output += f"    ⚔️ Competitors: {', '.join(competitors) if competitors else 'None mapped'}\n"
                
        output += "\n  💡 TIP: If your stock moves big, check these related names.\n"
        
        return output
    
    def _theme_updates(self) -> str:
        """Themes your watchlist is exposed to"""
        output = """
┌─────────────────────────────────────────────────────────────────────┐
│  🎯 THEME EXPOSURE                                                  │
└─────────────────────────────────────────────────────────────────────┘
"""
        
        # Find themes in watchlist
        theme_exposure = {}
        for ticker in self.user.watchlist:
            ticker_themes = self.themes.theme_for_stock(ticker)
            for theme in ticker_themes:
                if theme not in theme_exposure:
                    theme_exposure[theme] = []
                theme_exposure[theme].append(ticker)
                
        if theme_exposure:
            for theme, tickers in theme_exposure.items():
                output += f"\n  📍 {theme.upper()}: {', '.join(tickers)}\n"
                theme_data = self.themes.get_theme_stocks(theme)
                if 'signals_to_watch' in theme_data:
                    output += f"     Watch for: {theme_data['signals_to_watch'][0]}\n"
        else:
            output += "\n  No major theme exposure detected in watchlist.\n"
            output += "  Consider adding names from trending themes:\n"
            for theme in list(self.themes.THEMES.keys())[:3]:
                output += f"    • {theme}\n"
                
        return output
    
    def _action_items(self) -> str:
        """Suggested actions for today"""
        return """
┌─────────────────────────────────────────────────────────────────────┐
│  ✅ TODAY'S ACTION ITEMS                                            │
└─────────────────────────────────────────────────────────────────────┘

  1. 📰 Check overnight news on your biggest movers
  2. 🔍 Research any supply chain connections that moved
  3. 📊 Note sector leadership - is your sector hot or cold?
  4. 🎯 Review theme exposure - any catalyst today?
  5. ⚠️ Set alerts for key levels on top 3 positions

  Remember: The goal is INFORMED decisions, not MORE trades.

─────────────────────────────────────────────────────────────────────
"""


# =============================================================================
# DEMO
# =============================================================================

def run_demo():
    """Demonstrate the Market Intelligence System"""
    
    print("\n" + "="*75)
    print(" 🔭 MARKET INTELLIGENCE SYSTEM - PROTOTYPE DEMO")
    print("="*75)
    
    # Create sample user profile
    user = UserProfile(
        watchlist=['NVDA', 'AAPL', 'TSLA', 'MSFT', 'AMD', 'TSM', 'ASML', 'META'],
        sectors_of_interest=['Technology', 'Consumer Discretionary'],
        market_cap_preference='large',
        style='momentum',
        risk_tolerance='moderate',
    )
    
    # Generate morning brief
    brief = MorningBrief(user)
    print(brief.generate_brief())
    
    # Demo supply chain
    print("\n" + "="*75)
    print(" 🔗 SUPPLY CHAIN DEMO - What moves together?")
    print("="*75)
    
    chain = SupplyChainWeb()
    
    print("\n📦 If NVDA has news, also check:")
    nvda_related = chain.get_related_tickers('NVDA')
    print(f"   Suppliers: {nvda_related['if_this_moves_check']['suppliers']}")
    print(f"   Customers (AI buyers): {nvda_related['also_affected']['customers']}")
    print(f"   Competitors: {nvda_related['compare_to']['competitors']}")
    
    print("\n📦 If AAPL has news, also check:")
    aapl_related = chain.get_related_tickers('AAPL')
    print(f"   Suppliers: {aapl_related['if_this_moves_check']['suppliers']}")
    print(f"   Competitors: {aapl_related['compare_to']['competitors']}")
    
    # Demo cascade alerts
    print("\n" + "="*75)
    print(" 🚨 CASCADE ALERT DEMO - Event triggers related alerts")
    print("="*75)
    
    print("\nScenario: NVDA beats earnings by 20%")
    alerts = chain.alert_cascade('NVDA', 'Beat earnings by 20%, raised guidance')
    for alert in alerts[:5]:
        print(f"\n{alert.headline}")
        print(f"   {alert.detail}")
        print(f"   Action: {alert.action_prompt}")
    
    # Demo themes
    print("\n" + "="*75)
    print(" 🎯 THEME DETECTOR DEMO")
    print("="*75)
    
    themes = ThemeDetector()
    
    print("\nAvailable themes to track:")
    for theme in themes.get_all_themes():
        theme_data = themes.get_theme_stocks(theme)
        print(f"\n📍 {theme.upper()}")
        print(f"   Leaders: {', '.join(theme_data['leaders'][:3])}")
        print(f"   Picks & Shovels: {', '.join(theme_data['picks_and_shovels'][:3])}")
        print(f"   Watch for: {theme_data['signals_to_watch'][0]}")
    
    # Summary
    print("\n" + "="*75)
    print(" 💡 WHAT THIS SYSTEM DOES")
    print("="*75)
    print("""
    1. MORNING BRIEF - Personalized to YOUR watchlist
       • Shows your movers overnight
       • Sector context
       • Supply chain connections
       • Theme exposure

    2. SUPPLY CHAIN WEB - Connect the dots
       • If NVDA moves, check TSM, ASML, AMD
       • Second-derivative plays
       • Competitor analysis

    3. THEME TRACKER - Ride the waves
       • Map themes to stocks
       • Know your exposure
       • Watch for catalysts

    4. CASCADE ALERTS - Don't miss related moves
       • Event on Company A → Alert on suppliers B, C, D
       • Competitor check
       
    COMING NEXT:
    - Real-time event radar (insider buys, earnings, volume)
    - Social buzz tracking (Reddit, Twitter, StockTwits)
    - News aggregation (filtered to your watchlist)
    - Pattern-in-YOUR-behavior analysis
    """)


if __name__ == "__main__":
    run_demo()
