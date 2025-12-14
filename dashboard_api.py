"""
🚀 ENHANCED BACKEND API FOR SPARK DASHBOARD
Flask server with Perplexity AI integration + Portfolio tracking
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
import os
import json
from datetime import datetime

# Import our trading system
from PORTFOLIO_AWARE_TRADER import Portfolio, PortfolioAwareTrader
from perplexity_ai_chat import PerplexityAIChat

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configure CORS
cors_origins = os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://localhost:5173').split(',')
CORS(app, resources={r"/api/*": {"origins": cors_origins}})

# Load watchlist
with open('MY_WATCHLIST.txt', 'r') as f:
    WATCHLIST = [line.strip() for line in f if line.strip()]

# Initialize Perplexity AI
ai_chat = PerplexityAIChat()

# Initialize trader (will be refreshed on each request for now)
def get_trader():
    """Get trader instance with latest portfolio"""
    portfolio = Portfolio.load("MY_PORTFOLIO.json")
    trader = PortfolioAwareTrader(WATCHLIST, portfolio)
    try:
        trader._load_models()
    except:
        pass  # Use default models
    return trader


# ============================================================================
# PORTFOLIO ENDPOINTS
# ============================================================================

@app.route('/api/portfolio/status', methods=['GET'])
def portfolio_status():
    """Get current portfolio status"""
    try:
        trader = get_trader()
        trader.update_portfolio_prices()
        
        return jsonify({
            'success': True,
            'portfolio': {
                'total_value': trader.portfolio.total_equity,
                'cash': trader.portfolio.cash,
                'cash_percent': trader.portfolio.cash_percent,
                'invested': trader.portfolio.invested_capital,
                'pnl_dollars': trader.portfolio.total_pnl,
                'pnl_percent': trader.portfolio.total_pnl_percent,
                'num_positions': len(trader.portfolio.positions)
            },
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/portfolio/positions', methods=['GET'])
def get_positions():
    """Get all current positions"""
    try:
        trader = get_trader()
        trader.update_portfolio_prices()
        
        positions = [
            {
                'ticker': p.ticker,
                'entry_price': p.entry_price,
                'current_price': p.current_price,
                'shares': p.shares,
                'market_value': p.market_value,
                'cost_basis': p.cost_basis,
                'pnl_dollars': p.pnl_dollars,
                'pnl_percent': p.pnl_percent,
                'days_held': p.days_held,
                'sector': p.sector,
                'stop_loss': p.stop_loss,
                'target_price': p.target_price
            }
            for p in trader.portfolio.positions
        ]
        
        return jsonify({
            'success': True,
            'positions': positions,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/portfolio/sector-allocation', methods=['GET'])
def sector_allocation():
    """Get sector allocation breakdown"""
    try:
        trader = get_trader()
        trader.update_portfolio_prices()
        
        allocation = trader.portfolio.get_sector_allocation()
        
        return jsonify({
            'success': True,
            'allocation': allocation,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# RECOMMENDATIONS ENDPOINTS
# ============================================================================

@app.route('/api/recommendations', methods=['GET'])
def get_recommendations():
    """Get all portfolio and watchlist recommendations"""
    try:
        trader = get_trader()
        actions = trader.analyze_portfolio_and_watchlist()
        
        # Save portfolio
        trader.portfolio.save("MY_PORTFOLIO.json")
        
        # Format actions
        formatted_actions = [
            {
                'ticker': a.ticker,
                'action': a.action,
                'confidence': a.confidence,
                'reasoning': a.reasoning,
                'suggested_dollars': a.suggested_dollars,
                'suggested_shares': a.suggested_shares,
                'current_position_size': a.current_position_size,
                'sector': a.sector,
                'sector_allocation': a.sector_allocation,
                'days_held': a.days_held,
                'pnl_percent': a.pnl_percent,
                'stop_loss': a.stop_loss,
                'target_price': a.target_price
            }
            for a in actions
        ]
        
        # Categorize
        sells = [a for a in formatted_actions if a['action'] == 'SELL']
        trims = [a for a in formatted_actions if a['action'] == 'TRIM']
        holds = [a for a in formatted_actions if a['action'] == 'HOLD']
        buys = [a for a in formatted_actions if a['action'] == 'BUY_NEW']
        waits = [a for a in formatted_actions if a['action'] == 'WAIT']
        
        return jsonify({
            'success': True,
            'recommendations': {
                'urgent_sells': sells,
                'trims': trims,
                'holds': holds,
                'high_confidence_buys': [b for b in buys if b['confidence'] > 0.75],
                'moderate_buys': [b for b in buys if 0.70 <= b['confidence'] <= 0.75],
                'waits': waits
            },
            'summary': {
                'sells': len(sells),
                'trims': len(trims),
                'holds': len(holds),
                'buys': len(buys),
                'waits': len(waits)
            },
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/recommendations/<ticker>', methods=['GET'])
def get_ticker_recommendation(ticker):
    """Get recommendation for specific ticker"""
    try:
        trader = get_trader()
        
        # Check if ticker in portfolio
        position = trader.portfolio.get_position(ticker)
        
        if position:
            action = trader._analyze_position(position)
        elif ticker in WATCHLIST:
            action = trader._analyze_watchlist_ticker(ticker)
        else:
            return jsonify({
                'success': False,
                'error': f'{ticker} not in portfolio or watchlist'
            }), 404
        
        return jsonify({
            'success': True,
            'recommendation': {
                'ticker': action.ticker,
                'action': action.action,
                'confidence': action.confidence,
                'reasoning': action.reasoning,
                'suggested_dollars': action.suggested_dollars,
                'suggested_shares': action.suggested_shares,
                'sector': action.sector,
                'pnl_percent': action.pnl_percent
            },
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# PERPLEXITY AI ENDPOINTS
# ============================================================================

@app.route('/api/ai/chat', methods=['POST'])
def ai_chat_endpoint():
    """Chat with Perplexity AI"""
    try:
        data = request.get_json()
        message = data.get('message')
        
        if not message:
            return jsonify({'success': False, 'error': 'No message provided'}), 400
        
        # Get portfolio context
        trader = get_trader()
        context = {
            'portfolio': {
                'total_value': trader.portfolio.total_equity,
                'cash': trader.portfolio.cash,
                'cash_percent': trader.portfolio.cash_percent,
                'pnl_percent': trader.portfolio.total_pnl_percent,
                'positions': [
                    {'ticker': p.ticker, 'market_value': p.market_value, 'pnl_percent': p.pnl_percent}
                    for p in trader.portfolio.positions
                ]
            },
            'watchlist': WATCHLIST
        }
        
        response = ai_chat.chat(message, context)
        
        return jsonify({
            'success': True,
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/ai/analyze-ticker', methods=['POST'])
def ai_analyze_ticker():
    """Get AI analysis for a ticker"""
    try:
        data = request.get_json()
        ticker = data.get('ticker')
        
        if not ticker:
            return jsonify({'success': False, 'error': 'No ticker provided'}), 400
        
        # Get our recommendation
        trader = get_trader()
        position = trader.portfolio.get_position(ticker)
        
        recommendation = None
        if position:
            action = trader._analyze_position(position)
            recommendation = {
                'action': action.action,
                'confidence': action.confidence,
                'reasoning': action.reasoning
            }
        elif ticker in WATCHLIST:
            action = trader._analyze_watchlist_ticker(ticker)
            if action:
                recommendation = {
                    'action': action.action,
                    'confidence': action.confidence,
                    'reasoning': action.reasoning,
                    'entry_price': action.suggested_dollars / action.suggested_shares if action.suggested_shares else 0
                }
        
        response = ai_chat.analyze_ticker(ticker, recommendation)
        
        return jsonify({
            'success': True,
            'ticker': ticker,
            'analysis': response,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/ai/portfolio-review', methods=['GET'])
def ai_portfolio_review():
    """Get AI review of entire portfolio"""
    try:
        trader = get_trader()
        trader.update_portfolio_prices()
        actions = trader.analyze_portfolio_and_watchlist()
        
        # Format for AI
        portfolio = {
            'total_value': trader.portfolio.total_equity,
            'cash': trader.portfolio.cash,
            'pnl_percent': trader.portfolio.total_pnl_percent,
            'positions': [
                {
                    'ticker': p.ticker,
                    'market_value': p.market_value,
                    'pnl_percent': p.pnl_percent
                }
                for p in trader.portfolio.positions
            ]
        }
        
        formatted_actions = [
            {
                'ticker': a.ticker,
                'action': a.action,
                'confidence': a.confidence
            }
            for a in actions
        ]
        
        response = ai_chat.portfolio_review(portfolio, formatted_actions)
        
        return jsonify({
            'success': True,
            'review': response,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# WATCHLIST ENDPOINTS
# ============================================================================

@app.route('/api/watchlist', methods=['GET'])
def get_watchlist():
    """Get current watchlist"""
    return jsonify({
        'success': True,
        'watchlist': WATCHLIST,
        'count': len(WATCHLIST),
        'timestamp': datetime.now().isoformat()
    })


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'features': {
            'portfolio_tracking': True,
            'ml_recommendations': True,
            'sector_analysis': True,
            'perplexity_ai': ai_chat.enabled,
            'paper_trading': os.getenv('ENABLE_PAPER_TRADING', 'false') == 'true'
        },
        'timestamp': datetime.now().isoformat()
    })


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == '__main__':
    port = int(os.getenv('API_PORT', 5000))
    debug = os.getenv('ENVIRONMENT', 'development') == 'development'
    
    print("\n" + "="*100)
    print("🚀 QUANTUM AI TRADER - BACKEND API")
    print("="*100 + "\n")
    print(f"✅ Server starting on http://localhost:{port}")
    print(f"✅ Watchlist: {len(WATCHLIST)} tickers")
    print(f"✅ Perplexity AI: {'Enabled' if ai_chat.enabled else 'Disabled (set PERPLEXITY_API_KEY)'}")
    print(f"\n📡 Available endpoints:")
    print(f"   GET  /api/health")
    print(f"   GET  /api/portfolio/status")
    print(f"   GET  /api/portfolio/positions")
    print(f"   GET  /api/portfolio/sector-allocation")
    print(f"   GET  /api/recommendations")
    print(f"   GET  /api/recommendations/<ticker>")
    print(f"   POST /api/ai/chat")
    print(f"   POST /api/ai/analyze-ticker")
    print(f"   GET  /api/ai/portfolio-review")
    print(f"   GET  /api/watchlist")
    print(f"\n" + "="*100 + "\n")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
