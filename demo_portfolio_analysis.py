"""
Demo showing how Portfolio-Aware Trader handles existing positions
"""

from PORTFOLIO_AWARE_TRADER import Portfolio, Position, PortfolioAwareTrader
from datetime import datetime, timedelta

# Create a realistic portfolio
portfolio = Portfolio(cash=50000)

# Add some winning positions
portfolio.positions = [
    # Tech winners
    Position(
        ticker='AAPL',
        entry_price=275.00,
        shares=100,
        entry_date=datetime.now() - timedelta(days=15),
        sector='TECH',
        current_price=278.78,
        stop_loss=265.00,
        target_price=290.00
    ),
    Position(
        ticker='MSFT',
        entry_price=480.00,
        shares=50,
        entry_date=datetime.now() - timedelta(days=10),
        sector='TECH',
        current_price=483.16,
        stop_loss=470.00,
        target_price=500.00
    ),
    
    # Position near stop loss
    Position(
        ticker='XOM',
        entry_price=118.00,
        shares=200,
        entry_date=datetime.now() - timedelta(days=20),
        sector='ENERGY',
        current_price=116.54,
        stop_loss=115.00,
        target_price=125.00
    ),
    
    # Position at target
    Position(
        ticker='JPM',
        entry_price=230.00,
        shares=100,
        entry_date=datetime.now() - timedelta(days=12),
        sector='FINANCE',
        current_price=244.42,
        stop_loss=225.00,
        target_price=242.00
    ),
]

# Define watchlist with these positions + new opportunities
MY_WATCHLIST = [
    # Current positions
    'AAPL', 'MSFT', 'XOM', 'JPM',
    
    # New opportunities to consider
    'NVDA', 'GOOGL', 'META',  # More tech
    'BAC', 'GS',  # More finance
    'CVX',  # More energy
    'JNJ', 'UNH',  # Healthcare
    'WMT', 'HD'  # Consumer
]

print("\n" + "="*100)
print("📊 PORTFOLIO-AWARE TRADER - REALISTIC DEMO")
print("="*100 + "\n")

print("📈 Starting Portfolio:")
print(f"   Cash: ${portfolio.cash:,.2f}")
print(f"   Positions: {len(portfolio.positions)}")
print(f"   Total Value: ${portfolio.total_equity:,.2f}\n")

for pos in portfolio.positions:
    pnl = pos.pnl_percent
    emoji = "🟢" if pnl > 0 else "🔴"
    print(f"   {emoji} {pos.ticker:5} ${pos.entry_price:.2f} → ${pos.current_price:.2f} ({pnl:+.1f}%) - {pos.days_held} days")

print("\n" + "="*100 + "\n")

# Initialize trader
trader = PortfolioAwareTrader(
    watchlist=MY_WATCHLIST,
    portfolio=portfolio
)

# Analyze everything
actions = trader.analyze_portfolio_and_watchlist()

# Show actionable summary
print("\n" + "="*100)
print("🎯 ACTIONABLE RECOMMENDATIONS")
print("="*100 + "\n")

sells = [a for a in actions if a.action == 'SELL']
trims = [a for a in actions if a.action == 'TRIM']
buys = [a for a in actions if a.action == 'BUY_NEW' and a.confidence > 0.70]

if sells:
    print("🔴 SELL THESE NOW:")
    for action in sells:
        print(f"   {action.ticker} - {action.reasoning[0]}")

if trims:
    print("\n🟠 CONSIDER TRIMMING:")
    for action in trims:
        print(f"   {action.ticker} - {action.reasoning[0]}")

if buys:
    print("\n🟢 NEW POSITIONS TO ADD:")
    for action in buys:
        print(f"   {action.ticker} - ${action.suggested_dollars:,.0f} ({action.suggested_shares} shares)")
        print(f"      {action.reasoning[0]}")

# Save portfolio
trader.portfolio.save("realistic_portfolio.json")

print("\n✅ Analysis complete! Portfolio saved to realistic_portfolio.json")
