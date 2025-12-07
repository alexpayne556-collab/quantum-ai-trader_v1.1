# 📊 Portfolio-Aware Swing Trading System

## Overview

Your AI trading system now **knows your watchlist and portfolio** - it makes context-aware decisions based on what you hold, sector allocations, P&L, and risk management rules.

---

## 🎯 Key Features

### 1. **Watchlist Training**
- Trains ML models specifically on YOUR watchlist tickers
- System becomes familiar with the stocks you care about
- Better predictions because it learns your universe

### 2. **Portfolio Awareness**
- Tracks all your positions (entry price, shares, P&L)
- Updates current prices automatically
- Calculates portfolio value and allocation

### 3. **Context-Aware Decisions**

#### For Held Positions:
- **HOLD** - Keep position, still bullish
- **TRIM** - Take partial profits (hit target or oversized)
- **SELL** - Exit now (stop loss, bearish signal, cut losses)

#### For Watchlist Tickers:
- **BUY_NEW** - High confidence entry signal
- **WAIT** - Signal present but confidence too low or can't add

### 4. **Risk Management**
- **Position Size Limits**: Max 20% per ticker
- **Sector Allocation Limits**: Max 40% per sector
- **Stop Loss Triggers**: Auto-recommends sell on stop hit
- **Target Price Alerts**: Recommends trim/sell on target hit
- **P&L Monitoring**: Cut losses at -8% if bearish

### 5. **Smart Entry Logic**
- Checks if you have enough cash
- Validates position won't exceed limits
- Validates sector won't be overweight
- Calculates optimal position size
- Provides entry, target, stop loss prices

---

## 📁 Files

### `PORTFOLIO_AWARE_TRADER.py`
Main system integrating:
- Portfolio tracking (positions, cash, P&L)
- Watchlist management
- ML training on your tickers
- Context-aware decision engine
- Risk management rules
- Sector allocation tracking

### `demo_portfolio_analysis.py`
Example showing realistic portfolio with:
- Winning positions (AAPL +1.4%, MSFT +0.7%)
- Position near stop (XOM -1.2% near $115 stop)
- Position at target (JPM +6.3% hit $242 target)
- Watchlist opportunities to consider

---

## 🚀 Quick Start

### 1. Define Your Watchlist

```python
from PORTFOLIO_AWARE_TRADER import PortfolioAwareTrader, Portfolio

# YOUR watchlist - tickers you want to track
MY_WATCHLIST = [
    'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META',  # Tech
    'JPM', 'BAC', 'GS',  # Finance
    'XOM', 'CVX',  # Energy
    'JNJ', 'UNH',  # Healthcare
    'WMT', 'HD'  # Consumer
]
```

### 2. Create/Load Portfolio

```python
# Option A: Create new portfolio
portfolio = Portfolio(cash=100000)

# Option B: Load saved portfolio
portfolio = Portfolio.load("my_portfolio.json")
```

### 3. Initialize Trader

```python
trader = PortfolioAwareTrader(
    watchlist=MY_WATCHLIST,
    portfolio=portfolio
)
```

### 4. Train on Watchlist (Optional)

```python
# Train ML models specifically on YOUR tickers
# Only need to do this once (takes a few minutes)
trader.train_on_watchlist(period='2y')
```

### 5. Analyze Everything

```python
# Get recommendations for all positions + watchlist
actions = trader.analyze_portfolio_and_watchlist()

# Save portfolio
trader.portfolio.save("my_portfolio.json")
```

---

## 📊 Output Format

### Held Position Example:

```
🟡 AAPL - HOLD (75%)
   Position: 17.8% of portfolio
   P&L: 🟢 +1.4%
   Days Held: 15
   Sector: TECH (33.2% allocated)
   • ✅ Strong profit: +1.4%
   • 🟢 ML model still bullish (75%)
   • ✅ Sector TECH in favor
```

### Watchlist Opportunity Example:

```
🟢 NVDA - BUY_NEW (78%)
   Suggested: $20,000 (35 shares)
   Sector: TECH (33.2% allocated)
   • 🟢 ML says BUY (78%)
   • ✅ Sector TECH in favor (Growth Phase)
   • 💪 Strong sector (78/100)
```

### Trim/Sell Example:

```
🟠 JPM - TRIM (75%)
   Position: 15.6% of portfolio
   P&L: 🟢 +6.3%
   Days Held: 12
   Sector: FINANCE (20.1% allocated)
   • ✅ Strong profit: +6.3%
   • 🎯 Target reached ($242.00)
   • 🟢 ML model still bullish (75%)
```

---

## 🎛️ Risk Management Settings

### Position Limits
```python
portfolio.max_position_size = 0.20  # Max 20% per ticker
portfolio.max_sector_allocation = 0.40  # Max 40% per sector
```

### Auto-Triggers
- **Stop Loss Hit**: Immediate SELL recommendation
- **Target Hit + Bearish Signal**: SELL recommendation
- **Target Hit + Still Bullish**: TRIM recommendation
- **Loss > 8% + Bearish**: SELL recommendation (cut losses)
- **Position > 30%**: TRIM recommendation (reduce concentration)

---

## 📈 Decision Logic

### For Existing Positions:

1. **Update current price** from yfinance
2. **Calculate P&L** ($ and %)
3. **Check triggers**:
   - Stop loss hit? → SELL
   - Target hit? → TRIM or SELL
   - Heavy loss + bearish? → SELL
   - Oversized position? → TRIM
4. **Get ML signal** from sector-aware system
5. **Make decision**: HOLD / TRIM / SELL

### For Watchlist Tickers:

1. **Get ML signal** from sector-aware system
2. **Skip if not BUY** signal
3. **Check sector rotation** - is sector favored?
4. **Calculate position size** (max 20% of equity)
5. **Validate limits**:
   - Enough cash?
   - Won't exceed position limit?
   - Won't exceed sector limit?
6. **Make decision**: BUY_NEW / WAIT

---

## 💾 Portfolio Persistence

### Save Portfolio
```python
trader.portfolio.save("my_portfolio.json")
```

### Load Portfolio
```python
portfolio = Portfolio.load("my_portfolio.json")
```

### JSON Format
```json
{
  "cash": 50000.0,
  "positions": [
    {
      "ticker": "AAPL",
      "entry_price": 275.00,
      "shares": 100,
      "entry_date": "2025-11-22T00:00:00",
      "sector": "TECH",
      "current_price": 278.78,
      "stop_loss": 265.00,
      "target_price": 290.00
    }
  ],
  "max_position_size": 0.20,
  "max_sector_allocation": 0.40
}
```

---

## 🔄 Workflow Example

### Daily Routine:

```python
# 1. Load your portfolio
portfolio = Portfolio.load("my_portfolio.json")

# 2. Initialize trader with your watchlist
trader = PortfolioAwareTrader(
    watchlist=MY_WATCHLIST,
    portfolio=portfolio
)

# 3. Analyze everything
actions = trader.analyze_portfolio_and_watchlist()

# 4. Review recommendations
for action in actions:
    if action.action == 'SELL':
        print(f"🔴 SELL {action.ticker}: {action.reasoning}")
    elif action.action == 'BUY_NEW' and action.confidence > 0.75:
        print(f"🟢 BUY {action.ticker}: ${action.suggested_dollars:,.0f}")

# 5. Save updated portfolio
trader.portfolio.save("my_portfolio.json")
```

### Weekly Routine:

```python
# Re-train models on your watchlist with latest data
trader.train_on_watchlist(period='2y')
```

---

## 🎯 Integration with Spark Dashboard

### Backend API Endpoints:

```python
from flask import Flask, jsonify, request
from PORTFOLIO_AWARE_TRADER import PortfolioAwareTrader, Portfolio

app = Flask(__name__)

@app.route('/api/portfolio/analyze', methods=['GET'])
def analyze_portfolio():
    """Analyze entire portfolio and watchlist"""
    portfolio = Portfolio.load("portfolio.json")
    trader = PortfolioAwareTrader(WATCHLIST, portfolio)
    actions = trader.analyze_portfolio_and_watchlist()
    
    return jsonify({
        'portfolio': {
            'total_value': portfolio.total_equity,
            'cash': portfolio.cash,
            'invested': portfolio.invested_capital,
            'pnl': portfolio.total_pnl,
            'pnl_percent': portfolio.total_pnl_percent
        },
        'actions': [
            {
                'ticker': a.ticker,
                'action': a.action,
                'confidence': a.confidence,
                'reasoning': a.reasoning,
                'suggested_dollars': a.suggested_dollars,
                'suggested_shares': a.suggested_shares,
                'pnl_percent': a.pnl_percent
            }
            for a in actions
        ]
    })

@app.route('/api/portfolio/positions', methods=['GET'])
def get_positions():
    """Get all current positions"""
    portfolio = Portfolio.load("portfolio.json")
    trader = PortfolioAwareTrader(WATCHLIST, portfolio)
    trader.update_portfolio_prices()
    
    return jsonify({
        'positions': [
            {
                'ticker': p.ticker,
                'entry_price': p.entry_price,
                'current_price': p.current_price,
                'shares': p.shares,
                'market_value': p.market_value,
                'pnl_dollars': p.pnl_dollars,
                'pnl_percent': p.pnl_percent,
                'days_held': p.days_held,
                'sector': p.sector
            }
            for p in portfolio.positions
        ]
    })

@app.route('/api/portfolio/sector-allocation', methods=['GET'])
def get_sector_allocation():
    """Get sector allocation breakdown"""
    portfolio = Portfolio.load("portfolio.json")
    return jsonify({
        'allocation': portfolio.get_sector_allocation()
    })
```

### React Components:

```jsx
// PortfolioSummary.tsx
const PortfolioSummary = () => {
  const [portfolio, setPortfolio] = useState(null);
  
  useEffect(() => {
    fetch('/api/portfolio/analyze')
      .then(r => r.json())
      .then(data => setPortfolio(data));
  }, []);
  
  if (!portfolio) return <Loader />;
  
  return (
    <Card>
      <h2>Portfolio Value: ${portfolio.portfolio.total_value.toLocaleString()}</h2>
      <p>P&L: {portfolio.portfolio.pnl_percent > 0 ? '🟢' : '🔴'} 
         ${portfolio.portfolio.pnl.toLocaleString()} 
         ({portfolio.portfolio.pnl_percent.toFixed(1)}%)
      </p>
      
      <h3>Actionable Recommendations</h3>
      {portfolio.actions
        .filter(a => a.action === 'SELL' || a.action === 'BUY_NEW')
        .map(action => (
          <ActionCard key={action.ticker} action={action} />
        ))}
    </Card>
  );
};
```

---

## 🎓 Training Process

When you run `train_on_watchlist()`:

1. **Downloads 2 years of data** for each ticker
2. **Engineers features**: momentum, volatility, moving averages
3. **Creates labels**: BUY (>3% gain), SELL (<-3% loss), HOLD (else)
4. **Trains ensemble**: LightGBM, XGBoost, HistGB
5. **Saves models** to `models/` directory
6. **Ready to use** - loads automatically next time

This makes your system **specialized** on your watchlist!

---

## 🚀 Next Steps

### Immediate:
1. ✅ **Define your watchlist** - 10-20 tickers you want to trade
2. ✅ **Create portfolio** - with current positions or start fresh
3. ✅ **Run analysis** - see what system recommends
4. ✅ **Save portfolio** - persist your positions

### Soon:
1. 🔄 **Train on watchlist** - takes 5-10 minutes
2. 📊 **Build API** - integrate with Spark dashboard
3. 🎨 **Create UI** - visualize portfolio and recommendations
4. 📱 **Add alerts** - notify on SELL/BUY signals

### Advanced:
1. 🤖 **Paper trading** - auto-execute in simulation
2. 📈 **Performance tracking** - monitor win rate, Sharpe ratio
3. 🔔 **Push notifications** - SMS/email on critical signals
4. 🎯 **Strategy optimization** - backtest different settings

---

## 📊 Example Output

```
====================================================================================================
📊 PORTFOLIO SUMMARY
====================================================================================================

💰 Total Value: $156,848.00
   Cash: $50,000.00 (31.9%)
   Invested: $106,848.00
   P&L: $8,748.00 (+8.9%)

📊 Sector Allocation:
   TECH                 ██████░░░░░░░░░░░░░░ 33.2%
   FINANCE              ████░░░░░░░░░░░░░░░░ 20.1%
   ENERGY               ██░░░░░░░░░░░░░░░░░░ 14.9%

📈 Recommended Actions:
   HOLD: 3
   TRIM: 1
   BUY_NEW: 2
   WAIT: 10

====================================================================================================
🎯 ACTIONABLE RECOMMENDATIONS
====================================================================================================

🟠 CONSIDER TRIMMING:
   JPM - Strong profit: +6.3%, target reached

🟢 NEW POSITIONS TO ADD:
   NVDA - $20,000 (35 shares)
      ML says BUY (78%), Sector TECH in favor

====================================================================================================
```

---

## ✅ Benefits

### 1. **Context-Aware**
- Knows what you hold
- Understands your risk exposure
- Makes portfolio-level decisions

### 2. **Risk-Managed**
- Position size limits
- Sector diversification
- Stop loss enforcement
- Profit taking triggers

### 3. **Specialized**
- Trained on YOUR tickers
- Familiar with your universe
- Better predictions

### 4. **Actionable**
- Clear BUY/SELL/HOLD/TRIM signals
- Specific dollar amounts
- Reasoning for each decision
- Portfolio impact analysis

### 5. **Persistent**
- Saves portfolio state
- Tracks P&L over time
- Maintains history

---

## 🎯 Summary

Your AI trading system now:

✅ **Knows your watchlist** - trained specifically on tickers you care about
✅ **Tracks your portfolio** - positions, P&L, sector allocation
✅ **Makes smart decisions** - HOLD/TRIM/SELL for positions, BUY_NEW/WAIT for opportunities
✅ **Manages risk** - position limits, sector limits, stop losses, profit targets
✅ **Provides reasoning** - explains every recommendation
✅ **Stays persistent** - saves/loads portfolio state

**Ready to integrate with Spark dashboard for full visual trading experience!**
