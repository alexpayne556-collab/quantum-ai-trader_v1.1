"""
🤖 PERPLEXITY AI CHAT INTEGRATION
AI-powered market analysis and recommendation assistant
"""

import os
import json
import requests
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PerplexityAIChat:
    """
    Perplexity AI integration for market analysis and trading recommendations
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with API key"""
        self.api_key = api_key or os.getenv('PERPLEXITY_API_KEY')
        
        if not self.api_key:
            print("⚠️ No Perplexity API key found. Set PERPLEXITY_API_KEY in .env")
            self.enabled = False
        else:
            self.enabled = True
            print("✅ Perplexity AI Chat initialized")
        
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.conversation_history = []
    
    def chat(self, message: str, context: Optional[Dict] = None) -> str:
        """
        Send a message to Perplexity AI with optional trading context
        
        Args:
            message: User's question or request
            context: Trading context (portfolio, positions, signals, etc.)
        
        Returns:
            AI response as string
        """
        if not self.enabled:
            return "❌ Perplexity API key not configured. Add PERPLEXITY_API_KEY to .env"
        
        # Build system message with context
        system_message = self._build_system_message(context)
        
        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": message
        })
        
        # Prepare API request
        messages = [
            {"role": "system", "content": system_message}
        ] + self.conversation_history[-10:]  # Last 10 messages for context
        
        try:
            payload = {
                "model": "sonar",  # Perplexity's current model
                "messages": messages,
            }
            
            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            ai_response = result['choices'][0]['message']['content']
            
            # Add to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": ai_response
            })
            
            return ai_response
            
        except requests.exceptions.RequestException as e:
            return f"❌ Error communicating with Perplexity AI: {str(e)}"
        except Exception as e:
            return f"❌ Unexpected error: {str(e)}"
    
    def analyze_ticker(self, ticker: str, recommendation: Optional[Dict] = None) -> str:
        """
        Get AI analysis for a specific ticker
        
        Args:
            ticker: Stock ticker symbol
            recommendation: Our system's recommendation dict
        
        Returns:
            AI analysis as string
        """
        if recommendation:
            message = f"""
            Analyze {ticker} given our system's recommendation:
            
            Action: {recommendation.get('action', 'N/A')}
            Confidence: {recommendation.get('confidence', 0)*100:.0f}%
            Entry Price: ${recommendation.get('entry_price', 0):.2f}
            Target: ${recommendation.get('target_price', 0):.2f}
            Stop Loss: ${recommendation.get('stop_loss', 0):.2f}
            Sector: {recommendation.get('sector', 'N/A')}
            Reasoning: {', '.join(recommendation.get('reasoning', []))}
            
            Provide:
            1. Market news/catalysts for this stock
            2. Sector trends affecting it
            3. Risk factors to consider
            4. Your opinion on our recommendation
            """
        else:
            message = f"""
            Provide a comprehensive analysis of {ticker}:
            
            1. Recent news and catalysts
            2. Technical analysis
            3. Sector trends
            4. Analyst opinions
            5. Trading recommendation (Buy/Hold/Sell)
            """
        
        return self.chat(message)
    
    def explain_recommendation(self, ticker: str, action: str, reasoning: List[str]) -> str:
        """
        Get AI explanation for why we're recommending a particular action
        
        Args:
            ticker: Stock ticker
            action: BUY/SELL/HOLD/TRIM
            reasoning: List of reasons from our system
        
        Returns:
            AI explanation
        """
        message = f"""
        Our system recommends {action} on {ticker} because:
        
        {chr(10).join([f"• {reason}" for reason in reasoning])}
        
        Please explain:
        1. Why these factors matter
        2. What could go wrong
        3. What could go right
        4. Alternative perspectives
        
        Keep it concise and actionable.
        """
        
        return self.chat(message)
    
    def portfolio_review(self, portfolio: Dict, actions: List[Dict]) -> str:
        """
        Get AI review of entire portfolio and recommendations
        
        Args:
            portfolio: Portfolio dict with positions, value, P&L
            actions: List of recommended actions
        
        Returns:
            AI portfolio review
        """
        # Format portfolio info
        positions_str = "\n".join([
            f"• {pos['ticker']}: ${pos['market_value']:,.0f} ({pos['pnl_percent']:+.1f}%)"
            for pos in portfolio.get('positions', [])
        ])
        
        # Format recommendations
        buys = [a for a in actions if a['action'] == 'BUY_NEW']
        sells = [a for a in actions if a['action'] == 'SELL']
        
        buys_str = "\n".join([f"• {a['ticker']} ({a['confidence']*100:.0f}%)" for a in buys[:5]])
        sells_str = "\n".join([f"• {a['ticker']}" for a in sells])
        
        message = f"""
        Review my trading portfolio:
        
        📊 PORTFOLIO:
        Total Value: ${portfolio.get('total_value', 0):,.2f}
        P&L: {portfolio.get('pnl_percent', 0):+.1f}%
        Cash: ${portfolio.get('cash', 0):,.2f}
        
        POSITIONS:
        {positions_str}
        
        🟢 BUY RECOMMENDATIONS:
        {buys_str or 'None'}
        
        🔴 SELL RECOMMENDATIONS:
        {sells_str or 'None'}
        
        Provide:
        1. Overall portfolio assessment
        2. Risk analysis
        3. Diversification feedback
        4. Top 3 action items
        """
        
        return self.chat(message, context={'portfolio': portfolio, 'actions': actions})
    
    def sector_analysis(self, sector: str, rotation_stage: str, strength: int) -> str:
        """
        Get AI analysis of a specific sector
        
        Args:
            sector: Sector name (TECH, FINANCE, etc.)
            rotation_stage: Current market rotation stage
            strength: Sector strength 0-100
        
        Returns:
            AI sector analysis
        """
        message = f"""
        Analyze the {sector} sector:
        
        Current Market Stage: {rotation_stage}
        Sector Strength: {strength}/100
        
        Provide:
        1. Current sector trends
        2. Key catalysts/risks
        3. Best stocks in sector
        4. Sector outlook (bullish/bearish)
        """
        
        return self.chat(message)
    
    def market_overview(self, rotation_stage: str, favored_sectors: List[str]) -> str:
        """
        Get AI market overview and outlook
        
        Args:
            rotation_stage: Current market rotation stage
            favored_sectors: List of currently favored sectors
        
        Returns:
            AI market overview
        """
        message = f"""
        Provide a market overview:
        
        Market Stage: {rotation_stage}
        Favored Sectors: {', '.join(favored_sectors)}
        
        Cover:
        1. Overall market sentiment
        2. Key macro factors
        3. Sector rotation outlook
        4. Top opportunities
        5. Main risks
        """
        
        return self.chat(message)
    
    def _build_system_message(self, context: Optional[Dict] = None) -> str:
        """Build system message with trading context"""
        
        base_message = """You are a professional trading analyst and AI assistant for the Quantum AI Trader system.

Your role:
- Provide expert market analysis and trading insights
- Explain trading recommendations clearly
- Identify risks and opportunities
- Give actionable advice
- Use current market data when relevant

Style:
- Be concise and direct
- Use bullet points for clarity
- Focus on actionable insights
- Cite recent news/data when relevant
- Be honest about uncertainty"""

        if context:
            # Add portfolio context
            if 'portfolio' in context:
                p = context['portfolio']
                base_message += f"""

CURRENT PORTFOLIO:
- Total Value: ${p.get('total_value', 0):,.2f}
- P&L: {p.get('pnl_percent', 0):+.1f}%
- Cash: ${p.get('cash', 0):,.2f} ({p.get('cash_percent', 0):.1f}%)
- Positions: {len(p.get('positions', []))}"""
            
            # Add watchlist
            if 'watchlist' in context:
                base_message += f"\n\nWATCHLIST: {', '.join(context['watchlist'][:10])}"
        
        return base_message
    
    def reset_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("✅ Conversation history cleared")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*100)
    print("🤖 PERPLEXITY AI CHAT - DEMO")
    print("="*100 + "\n")
    
    # Initialize
    ai = PerplexityAIChat()
    
    if not ai.enabled:
        print("❌ Set PERPLEXITY_API_KEY in .env to use this feature")
        print("\nTo get API key:")
        print("1. Go to https://www.perplexity.ai/settings/api")
        print("2. Sign up/login")
        print("3. Create API key")
        print("4. Add to .env: PERPLEXITY_API_KEY=your_key_here")
        exit(1)
    
    # Example 1: Simple chat
    print("📝 Example 1: Simple market question\n")
    response = ai.chat("What are the top 3 tech stocks to watch right now and why?")
    print(response)
    print("\n" + "-"*100 + "\n")
    
    # Example 2: Ticker analysis
    print("📊 Example 2: Ticker analysis\n")
    recommendation = {
        'action': 'BUY',
        'confidence': 0.78,
        'entry_price': 350.25,
        'target_price': 367.76,
        'stop_loss': 336.24,
        'sector': 'TECH',
        'reasoning': [
            'ML model says BUY (78%)',
            'Sector TECH in favor',
            'Strong sector (78/100)'
        ]
    }
    response = ai.analyze_ticker('NVDA', recommendation)
    print(response)
    print("\n" + "-"*100 + "\n")
    
    # Example 3: Portfolio review
    print("💼 Example 3: Portfolio review\n")
    portfolio = {
        'total_value': 156848.00,
        'cash': 50000.00,
        'cash_percent': 31.9,
        'pnl_percent': 8.9,
        'positions': [
            {'ticker': 'AAPL', 'market_value': 27878, 'pnl_percent': 1.4},
            {'ticker': 'MSFT', 'market_value': 24158, 'pnl_percent': 0.7},
            {'ticker': 'JPM', 'market_value': 24442, 'pnl_percent': 6.3}
        ]
    }
    actions = [
        {'ticker': 'NVDA', 'action': 'BUY_NEW', 'confidence': 0.78},
        {'ticker': 'GOOGL', 'action': 'BUY_NEW', 'confidence': 0.75},
    ]
    response = ai.portfolio_review(portfolio, actions)
    print(response)
    
    print("\n" + "="*100)
    print("✅ Demo complete!")
    print("\nIntegrate into your dashboard with:")
    print("  from perplexity_ai_chat import PerplexityAIChat")
    print("  ai = PerplexityAIChat()")
    print("  response = ai.chat('Your question here')")
