"""
📊 PAPER TRADING DASHBOARD - AI vs HUMAN COMPETITION
====================================================
Real-time competition between AI and your Robinhood trades.

Features:
- Log your trades as you make them
- AI makes its own decisions
- Daily scoreboard
- Performance tracking
- Who's winning? Who's learning?
"""

import json
import os
from datetime import datetime, timedelta
from flask import Flask, render_template_string, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np

app = Flask(__name__)

# ============================================================================
# STATE MANAGEMENT
# ============================================================================

class CompetitionState:
    """Manages the AI vs Human competition state"""
    
    def __init__(self, state_file='competition_state.json'):
        self.state_file = state_file
        self.starting_cash = 10000
        
        # Load or initialize state
        if os.path.exists(state_file):
            self.load()
        else:
            self.reset()
    
    def reset(self):
        """Reset competition"""
        self.state = {
            'started_at': datetime.now().isoformat(),
            'starting_cash': self.starting_cash,
            
            # AI state
            'ai': {
                'cash': self.starting_cash,
                'positions': {},
                'trades': [],
                'daily_values': [],
            },
            
            # Human state
            'human': {
                'cash': self.starting_cash,
                'positions': {},
                'trades': [],
                'daily_values': [],
            },
            
            # Competition log
            'daily_log': [],
            'messages': [],
        }
        self.save()
    
    def save(self):
        """Save state to file"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)
    
    def load(self):
        """Load state from file"""
        with open(self.state_file, 'r') as f:
            self.state = json.load(f)
    
    def log_trade(self, trader, ticker, action, price, shares, reason=""):
        """Log a trade for AI or human"""
        trade = {
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker,
            'action': action,
            'price': float(price),
            'shares': int(shares),
            'reason': reason,
        }
        
        self.state[trader]['trades'].append(trade)
        
        # Update positions
        if action == 'buy':
            cost = float(price) * int(shares)
            self.state[trader]['cash'] -= cost
            
            if ticker in self.state[trader]['positions']:
                pos = self.state[trader]['positions'][ticker]
                total_shares = pos['shares'] + shares
                avg_price = (pos['shares'] * pos['entry'] + cost) / total_shares
                self.state[trader]['positions'][ticker] = {
                    'shares': total_shares,
                    'entry': avg_price,
                    'bought_at': pos['bought_at']
                }
            else:
                self.state[trader]['positions'][ticker] = {
                    'shares': shares,
                    'entry': float(price),
                    'bought_at': datetime.now().isoformat()
                }
        
        elif action == 'sell':
            if ticker in self.state[trader]['positions']:
                pos = self.state[trader]['positions'][ticker]
                proceeds = float(price) * int(shares)
                self.state[trader]['cash'] += proceeds
                
                # Calculate P&L
                pnl = (float(price) / pos['entry'] - 1) * 100
                trade['pnl'] = pnl
                
                if shares >= pos['shares']:
                    del self.state[trader]['positions'][ticker]
                else:
                    self.state[trader]['positions'][ticker]['shares'] -= shares
        
        self.save()
        return trade
    
    def get_portfolio_value(self, trader, prices):
        """Calculate total portfolio value"""
        value = self.state[trader]['cash']
        
        for ticker, pos in self.state[trader]['positions'].items():
            if ticker in prices:
                value += pos['shares'] * prices[ticker]
        
        return value
    
    def get_scoreboard(self, prices):
        """Get current scoreboard"""
        ai_value = self.get_portfolio_value('ai', prices)
        human_value = self.get_portfolio_value('human', prices)
        
        ai_return = (ai_value / self.starting_cash - 1) * 100
        human_return = (human_value / self.starting_cash - 1) * 100
        
        return {
            'ai': {
                'value': ai_value,
                'return': ai_return,
                'cash': self.state['ai']['cash'],
                'positions': len(self.state['ai']['positions']),
                'trades': len(self.state['ai']['trades']),
            },
            'human': {
                'value': human_value,
                'return': human_return,
                'cash': self.state['human']['cash'],
                'positions': len(self.state['human']['positions']),
                'trades': len(self.state['human']['trades']),
            },
            'leader': 'AI' if ai_return > human_return else 'HUMAN',
            'lead_by': abs(ai_return - human_return),
        }


# Initialize state
competition = CompetitionState()

# ============================================================================
# DASHBOARD HTML
# ============================================================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>🏆 AI vs HUMAN Competition</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Arial, sans-serif; 
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        
        header {
            text-align: center;
            padding: 30px;
            background: rgba(0,0,0,0.3);
            border-radius: 15px;
            margin-bottom: 30px;
        }
        header h1 { font-size: 2.5em; margin-bottom: 10px; }
        header h1 span { color: #00d4ff; }
        
        .scoreboard {
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .player-card {
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
        }
        .player-card.ai { border: 2px solid #00d4ff; }
        .player-card.human { border: 2px solid #ff6b6b; }
        .player-card h2 { font-size: 1.8em; margin-bottom: 15px; }
        .player-card.ai h2 { color: #00d4ff; }
        .player-card.human h2 { color: #ff6b6b; }
        
        .value { font-size: 2.5em; font-weight: bold; margin: 10px 0; }
        .return { font-size: 1.5em; }
        .return.positive { color: #00ff88; }
        .return.negative { color: #ff4444; }
        
        .stat { margin: 10px 0; color: #aaa; }
        .stat span { color: #fff; font-weight: bold; }
        
        .vs-box {
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 3em;
            font-weight: bold;
            color: #ffd700;
        }
        
        .leader-badge {
            background: linear-gradient(45deg, #ffd700, #ff8c00);
            color: #000;
            padding: 8px 20px;
            border-radius: 20px;
            font-weight: bold;
            margin-top: 15px;
            display: inline-block;
        }
        
        .sections {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }
        
        .section {
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 20px;
        }
        .section h3 {
            font-size: 1.3em;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        .trade-form {
            display: grid;
            gap: 15px;
        }
        .trade-form input, .trade-form select, .trade-form button {
            padding: 12px;
            border-radius: 8px;
            border: none;
            font-size: 1em;
        }
        .trade-form input, .trade-form select {
            background: rgba(255,255,255,0.1);
            color: #fff;
        }
        .trade-form button {
            cursor: pointer;
            font-weight: bold;
            transition: transform 0.2s;
        }
        .trade-form button:hover { transform: scale(1.02); }
        .btn-buy { background: #00ff88; color: #000; }
        .btn-sell { background: #ff4444; color: #fff; }
        
        .trades-list {
            max-height: 300px;
            overflow-y: auto;
        }
        .trade-item {
            padding: 10px;
            margin: 5px 0;
            border-radius: 8px;
            background: rgba(255,255,255,0.05);
            display: flex;
            justify-content: space-between;
        }
        .trade-item.buy { border-left: 3px solid #00ff88; }
        .trade-item.sell { border-left: 3px solid #ff4444; }
        
        .positions-list {
            max-height: 200px;
            overflow-y: auto;
        }
        .position-item {
            padding: 10px;
            margin: 5px 0;
            border-radius: 8px;
            background: rgba(255,255,255,0.05);
            display: flex;
            justify-content: space-between;
        }
        
        .refresh-btn {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 15px 30px;
            background: #00d4ff;
            color: #000;
            border: none;
            border-radius: 30px;
            font-size: 1.1em;
            font-weight: bold;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(0,212,255,0.4);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🏆 <span>AI</span> vs <span>HUMAN</span> Competition</h1>
            <p>Who will win? Paper trading showdown!</p>
        </header>
        
        <div class="scoreboard">
            <div class="player-card ai">
                <h2>🤖 AI TRADER</h2>
                <div class="value" id="ai-value">$10,000</div>
                <div class="return" id="ai-return">+0.00%</div>
                <div class="stat">Cash: $<span id="ai-cash">10,000</span></div>
                <div class="stat">Positions: <span id="ai-positions">0</span></div>
                <div class="stat">Trades: <span id="ai-trades">0</span></div>
                <div id="ai-leader"></div>
            </div>
            
            <div class="vs-box">VS</div>
            
            <div class="player-card human">
                <h2>👤 YOU (HUMAN)</h2>
                <div class="value" id="human-value">$10,000</div>
                <div class="return" id="human-return">+0.00%</div>
                <div class="stat">Cash: $<span id="human-cash">10,000</span></div>
                <div class="stat">Positions: <span id="human-positions">0</span></div>
                <div class="stat">Trades: <span id="human-trades">0</span></div>
                <div id="human-leader"></div>
            </div>
        </div>
        
        <div class="sections">
            <div class="section">
                <h3>📝 LOG YOUR TRADE</h3>
                <form class="trade-form" id="trade-form">
                    <input type="text" id="ticker" placeholder="Ticker (e.g., NVDA)" required>
                    <input type="number" id="price" placeholder="Price" step="0.01" required>
                    <input type="number" id="shares" placeholder="Shares" required>
                    <input type="text" id="reason" placeholder="Reason (optional)">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                        <button type="button" class="btn-buy" onclick="logTrade('buy')">BUY</button>
                        <button type="button" class="btn-sell" onclick="logTrade('sell')">SELL</button>
                    </div>
                </form>
            </div>
            
            <div class="section">
                <h3>📊 YOUR POSITIONS</h3>
                <div class="positions-list" id="human-positions-list">
                    <p style="color: #666;">No positions yet</p>
                </div>
            </div>
            
            <div class="section">
                <h3>📜 YOUR TRADE HISTORY</h3>
                <div class="trades-list" id="human-trades-list">
                    <p style="color: #666;">No trades yet</p>
                </div>
            </div>
            
            <div class="section">
                <h3>🤖 AI TRADE HISTORY</h3>
                <div class="trades-list" id="ai-trades-list">
                    <p style="color: #666;">No AI trades yet</p>
                </div>
            </div>
        </div>
    </div>
    
    <button class="refresh-btn" onclick="refresh()">🔄 Refresh</button>
    
    <script>
        function refresh() {
            fetch('/api/scoreboard')
                .then(r => r.json())
                .then(data => {
                    // Update AI
                    document.getElementById('ai-value').textContent = '$' + data.ai.value.toLocaleString(undefined, {minimumFractionDigits: 2});
                    document.getElementById('ai-return').textContent = (data.ai.return >= 0 ? '+' : '') + data.ai.return.toFixed(2) + '%';
                    document.getElementById('ai-return').className = 'return ' + (data.ai.return >= 0 ? 'positive' : 'negative');
                    document.getElementById('ai-cash').textContent = data.ai.cash.toLocaleString();
                    document.getElementById('ai-positions').textContent = data.ai.positions;
                    document.getElementById('ai-trades').textContent = data.ai.trades;
                    
                    // Update Human
                    document.getElementById('human-value').textContent = '$' + data.human.value.toLocaleString(undefined, {minimumFractionDigits: 2});
                    document.getElementById('human-return').textContent = (data.human.return >= 0 ? '+' : '') + data.human.return.toFixed(2) + '%';
                    document.getElementById('human-return').className = 'return ' + (data.human.return >= 0 ? 'positive' : 'negative');
                    document.getElementById('human-cash').textContent = data.human.cash.toLocaleString();
                    document.getElementById('human-positions').textContent = data.human.positions;
                    document.getElementById('human-trades').textContent = data.human.trades;
                    
                    // Leader badge
                    document.getElementById('ai-leader').innerHTML = data.leader === 'AI' ? '<div class="leader-badge">🏆 LEADING</div>' : '';
                    document.getElementById('human-leader').innerHTML = data.leader === 'HUMAN' ? '<div class="leader-badge">🏆 LEADING</div>' : '';
                });
            
            // Refresh trades lists
            fetch('/api/trades')
                .then(r => r.json())
                .then(data => {
                    // Human trades
                    const humanList = document.getElementById('human-trades-list');
                    if (data.human && data.human.length > 0) {
                        humanList.innerHTML = data.human.slice(-10).reverse().map(t => `
                            <div class="trade-item ${t.action}">
                                <span>${t.action.toUpperCase()} ${t.shares} ${t.ticker} @ $${t.price}</span>
                                <span>${t.pnl ? (t.pnl >= 0 ? '+' : '') + t.pnl.toFixed(1) + '%' : ''}</span>
                            </div>
                        `).join('');
                    }
                    
                    // AI trades
                    const aiList = document.getElementById('ai-trades-list');
                    if (data.ai && data.ai.length > 0) {
                        aiList.innerHTML = data.ai.slice(-10).reverse().map(t => `
                            <div class="trade-item ${t.action}">
                                <span>${t.action.toUpperCase()} ${t.shares} ${t.ticker} @ $${t.price}</span>
                                <span>${t.pnl ? (t.pnl >= 0 ? '+' : '') + t.pnl.toFixed(1) + '%' : ''}</span>
                            </div>
                        `).join('');
                    }
                });
            
            // Refresh positions
            fetch('/api/positions')
                .then(r => r.json())
                .then(data => {
                    const list = document.getElementById('human-positions-list');
                    if (data.human && Object.keys(data.human).length > 0) {
                        list.innerHTML = Object.entries(data.human).map(([ticker, pos]) => `
                            <div class="position-item">
                                <span>${ticker}</span>
                                <span>${pos.shares} shares @ $${pos.entry.toFixed(2)}</span>
                            </div>
                        `).join('');
                    }
                });
        }
        
        function logTrade(action) {
            const ticker = document.getElementById('ticker').value.toUpperCase();
            const price = parseFloat(document.getElementById('price').value);
            const shares = parseInt(document.getElementById('shares').value);
            const reason = document.getElementById('reason').value;
            
            if (!ticker || !price || !shares) {
                alert('Please fill in ticker, price, and shares');
                return;
            }
            
            fetch('/api/trade', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    trader: 'human',
                    ticker: ticker,
                    action: action,
                    price: price,
                    shares: shares,
                    reason: reason
                })
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('ticker').value = '';
                    document.getElementById('price').value = '';
                    document.getElementById('shares').value = '';
                    document.getElementById('reason').value = '';
                    refresh();
                } else {
                    alert('Error: ' + data.error);
                }
            });
        }
        
        // Initial load
        refresh();
        // Auto-refresh every 30 seconds
        setInterval(refresh, 30000);
    </script>
</body>
</html>
"""

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route('/api/scoreboard')
def get_scoreboard():
    """Get current scoreboard with live prices"""
    # Get current prices for all positions
    all_tickers = set()
    all_tickers.update(competition.state['ai']['positions'].keys())
    all_tickers.update(competition.state['human']['positions'].keys())
    
    prices = {}
    for ticker in all_tickers:
        try:
            data = yf.download(ticker, period='1d', progress=False)
            if len(data) > 0:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                prices[ticker] = float(data['Close'].iloc[-1])
        except:
            pass
    
    return jsonify(competition.get_scoreboard(prices))


@app.route('/api/trade', methods=['POST'])
def log_trade():
    """Log a trade"""
    data = request.json
    
    try:
        trade = competition.log_trade(
            trader=data['trader'],
            ticker=data['ticker'],
            action=data['action'],
            price=data['price'],
            shares=data['shares'],
            reason=data.get('reason', '')
        )
        return jsonify({'success': True, 'trade': trade})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/trades')
def get_trades():
    """Get trade history"""
    return jsonify({
        'ai': competition.state['ai']['trades'],
        'human': competition.state['human']['trades']
    })


@app.route('/api/positions')
def get_positions():
    """Get current positions"""
    return jsonify({
        'ai': competition.state['ai']['positions'],
        'human': competition.state['human']['positions']
    })


@app.route('/api/reset', methods=['POST'])
def reset_competition():
    """Reset competition"""
    competition.reset()
    return jsonify({'success': True})


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("🏆 AI vs HUMAN COMPETITION DASHBOARD")
    print("="*70)
    print("\nStarting dashboard on http://localhost:5001")
    print("\nFeatures:")
    print("  • Log your Robinhood trades")
    print("  • Compare against AI decisions")
    print("  • Track who's winning")
    print("  • Learn from each other")
    print("="*70)
    
    app.run(host='0.0.0.0', port=5001, debug=True)
