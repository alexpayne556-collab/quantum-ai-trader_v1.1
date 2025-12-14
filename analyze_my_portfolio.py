"""
📊 ANALYZE MY PORTFOLIO
Daily script to get recommendations for YOUR portfolio and watchlist
"""

import json
from PORTFOLIO_AWARE_TRADER import Portfolio, PortfolioAwareTrader

# Load watchlist
with open('MY_WATCHLIST.txt', 'r') as f:
    MY_WATCHLIST = [line.strip() for line in f if line.strip()]

# Load portfolio
try:
    portfolio = Portfolio.load("MY_PORTFOLIO.json")
    print(f"✅ Loaded portfolio from MY_PORTFOLIO.json")
except FileNotFoundError:
    print("❌ MY_PORTFOLIO.json not found. Run train_my_system.py first!")
    exit(1)

# Initialize trader
trader = PortfolioAwareTrader(
    watchlist=MY_WATCHLIST,
    portfolio=portfolio
)

# Try to load trained models
try:
    trader._load_models()
    print(f"✅ Loaded trained models from models/")
except:
    print("⚠️ No trained models found. Using default models (train with train_my_system.py)")

# Analyze everything
print("\n" + "="*100)
print("📊 ANALYZING PORTFOLIO & WATCHLIST")
print("="*100 + "\n")

actions = trader.analyze_portfolio_and_watchlist()

# Save updated portfolio
trader.portfolio.save("MY_PORTFOLIO.json")

# ============================================================================
# ACTIONABLE SUMMARY
# ============================================================================

print("\n" + "="*100)
print("🎯 TODAY'S ACTIONABLE RECOMMENDATIONS")
print("="*100 + "\n")

# 1. URGENT SELLS
sells = [a for a in actions if a.action == 'SELL']
if sells:
    print("🔴 URGENT: SELL THESE NOW")
    for action in sells:
        print(f"   {action.ticker:6} - {action.reasoning[0]}")
        if action.pnl_percent:
            pnl_emoji = "🟢" if action.pnl_percent > 0 else "🔴"
            print(f"           Current P&L: {pnl_emoji} {action.pnl_percent:+.1f}%")
    print()

# 2. TRIM POSITIONS
trims = [a for a in actions if a.action == 'TRIM']
if trims:
    print("🟠 CONSIDER TRIMMING (Take Partial Profits)")
    for action in trims:
        print(f"   {action.ticker:6} - {action.reasoning[0]}")
        if action.pnl_percent:
            print(f"           Current P&L: 🟢 {action.pnl_percent:+.1f}%")
    print()

# 3. HIGH-CONFIDENCE BUYS
high_conf_buys = [a for a in actions if a.action == 'BUY_NEW' and a.confidence > 0.75]
if high_conf_buys:
    print("🟢 HIGH-CONFIDENCE BUY OPPORTUNITIES (>75%)")
    for action in sorted(high_conf_buys, key=lambda x: x.confidence, reverse=True):
        print(f"   {action.ticker:6} - {action.confidence*100:.0f}% confidence")
        print(f"           Buy: ${action.suggested_dollars:,.0f} ({action.suggested_shares} shares)")
        print(f"           {action.reasoning[0]}")
    print()

# 4. MODERATE BUYS
moderate_buys = [a for a in actions if a.action == 'BUY_NEW' and 0.70 <= a.confidence <= 0.75]
if moderate_buys:
    print("🟡 MODERATE BUY OPPORTUNITIES (70-75%)")
    for action in sorted(moderate_buys, key=lambda x: x.confidence, reverse=True):
        print(f"   {action.ticker:6} - {action.confidence*100:.0f}% confidence")
        print(f"           Buy: ${action.suggested_dollars:,.0f} ({action.suggested_shares} shares)")
    print()

# 5. HOLDS
holds = [a for a in actions if a.action == 'HOLD']
if holds:
    print(f"🟡 HOLDING ({len(holds)} positions)")
    for action in holds:
        pnl_emoji = "🟢" if action.pnl_percent and action.pnl_percent > 0 else "🔴"
        pnl_str = f"{action.pnl_percent:+.1f}%" if action.pnl_percent else "N/A"
        print(f"   {action.ticker:6} - {pnl_emoji} {pnl_str} - {action.reasoning[0]}")
    print()

# Summary stats
if not any([sells, trims, high_conf_buys]):
    print("⚪ No urgent actions required today. All positions stable.\n")

print("="*100)
print(f"💰 Portfolio Value: ${trader.portfolio.total_equity:,.2f}")
print(f"   Cash: ${trader.portfolio.cash:,.2f} ({trader.portfolio.cash_percent:.1f}%)")
if trader.portfolio.total_pnl != 0:
    pnl_emoji = "🟢" if trader.portfolio.total_pnl > 0 else "🔴"
    print(f"   P&L: {pnl_emoji} ${trader.portfolio.total_pnl:,.2f} ({trader.portfolio.total_pnl_percent:+.1f}%)")
print("="*100 + "\n")

# Export to JSON for dashboard
recommendations = {
    'timestamp': str(trader.portfolio.positions[0].entry_date if trader.portfolio.positions else "N/A"),
    'portfolio_value': trader.portfolio.total_equity,
    'cash': trader.portfolio.cash,
    'pnl': trader.portfolio.total_pnl,
    'pnl_percent': trader.portfolio.total_pnl_percent,
    'urgent_sells': [{'ticker': a.ticker, 'reason': a.reasoning[0]} for a in sells],
    'trims': [{'ticker': a.ticker, 'reason': a.reasoning[0], 'pnl': a.pnl_percent} for a in trims],
    'high_confidence_buys': [
        {
            'ticker': a.ticker,
            'confidence': a.confidence,
            'dollars': a.suggested_dollars,
            'shares': a.suggested_shares,
            'reason': a.reasoning[0]
        }
        for a in high_conf_buys
    ],
    'holds': len(holds)
}

with open('daily_recommendations.json', 'w') as f:
    json.dump(recommendations, f, indent=2)

print("✅ Recommendations exported to daily_recommendations.json")
print("   (Use this for Spark dashboard integration)\n")
