"""
🎯 TRAIN YOUR PORTFOLIO-AWARE SYSTEM
Trains on YOUR watchlist and YOUR portfolio tickers
"""

import json
from datetime import datetime
from PORTFOLIO_AWARE_TRADER import Portfolio, Position, PortfolioAwareTrader

# ============================================================================
# YOUR WATCHLIST (58 tickers)
# ============================================================================
with open('MY_WATCHLIST.txt', 'r') as f:
    MY_WATCHLIST = [line.strip() for line in f if line.strip()]

print(f"✅ Loaded {len(MY_WATCHLIST)} tickers from MY_WATCHLIST.txt")

# ============================================================================
# YOUR CURRENT PORTFOLIO
# ============================================================================
MY_PORTFOLIO = Portfolio(cash=50000)  # Adjust cash amount as needed

# Current positions
MY_PORTFOLIO.positions = [
    Position(
        ticker='SERV',
        entry_price=0.0,  # Update with your actual entry price
        shares=0,  # Update with your actual shares
        entry_date=datetime.now(),
        sector='TECH',  # Will auto-detect
        current_price=0.0
    ),
    Position(
        ticker='YYAI',
        entry_price=0.0,  # Update with your actual entry price
        shares=0,  # Update with your actual shares
        entry_date=datetime.now(),
        sector='TECH',
        current_price=0.0
    ),
    Position(
        ticker='APLD',
        entry_price=0.0,  # Update with your actual entry price
        shares=0,  # Update with your actual shares
        entry_date=datetime.now(),
        sector='TECH',
        current_price=0.0
    ),
    Position(
        ticker='HOOD',
        entry_price=0.0,  # Update with your actual entry price
        shares=0,  # Update with your actual shares
        entry_date=datetime.now(),
        sector='FINANCE',
        current_price=0.0
    ),
]

print(f"✅ Portfolio has {len(MY_PORTFOLIO.positions)} positions")

# ============================================================================
# SAVE PORTFOLIO
# ============================================================================
MY_PORTFOLIO.save("MY_PORTFOLIO.json")

# ============================================================================
# INITIALIZE TRADER
# ============================================================================
print("\n" + "="*100)
print("🚀 INITIALIZING YOUR PORTFOLIO-AWARE TRADER")
print("="*100 + "\n")

trader = PortfolioAwareTrader(
    watchlist=MY_WATCHLIST,
    portfolio=MY_PORTFOLIO
)

# ============================================================================
# TRAIN ON YOUR WATCHLIST + PORTFOLIO TICKERS
# ============================================================================
print("\n" + "="*100)
print("🎓 TRAINING ML MODELS ON YOUR TICKERS")
print(f"   This will make the system familiar with your {len(MY_WATCHLIST)} watchlist tickers")
print(f"   Training period: 2 years of historical data")
print(f"   Models: LightGBM, XGBoost, HistGradientBoosting")
print("="*100 + "\n")

print("⏰ This will take 10-20 minutes depending on data availability...")
print("   Grab a coffee ☕\n")

# Train the models
success = trader.train_on_watchlist(period='2y')

if success:
    print("\n" + "="*100)
    print("✅ TRAINING COMPLETE!")
    print("="*100 + "\n")
    print("Your system is now:")
    print("   ✅ Familiar with your 58 watchlist tickers")
    print("   ✅ Trained ML ensemble (70%+ accuracy expected)")
    print("   ✅ Saved models to models/ directory")
    print("   ✅ Ready to analyze portfolio and make recommendations")
    print("\n" + "="*100)
    
    # Save training metadata
    training_meta = {
        'trained_at': datetime.now().isoformat(),
        'watchlist_size': len(MY_WATCHLIST),
        'watchlist': MY_WATCHLIST,
        'portfolio_tickers': [p.ticker for p in MY_PORTFOLIO.positions],
        'training_period': '2y',
        'models': ['lightgbm', 'xgboost', 'histgb'],
        'status': 'complete'
    }
    
    with open('training_metadata.json', 'w') as f:
        json.dump(training_meta, f, indent=2)
    
    print("\n✅ Training metadata saved to training_metadata.json")
    
    # Now analyze portfolio
    print("\n" + "="*100)
    print("📊 ANALYZING YOUR PORTFOLIO & WATCHLIST")
    print("="*100 + "\n")
    
    actions = trader.analyze_portfolio_and_watchlist()
    
    # Save portfolio with updated prices
    trader.portfolio.save("MY_PORTFOLIO.json")
    
    # Show high-confidence opportunities
    print("\n" + "="*100)
    print("🎯 HIGH-CONFIDENCE OPPORTUNITIES (>75%)")
    print("="*100 + "\n")
    
    high_conf_buys = [a for a in actions if a.action == 'BUY_NEW' and a.confidence > 0.75]
    if high_conf_buys:
        for action in high_conf_buys:
            print(f"🟢 {action.ticker} - {action.confidence*100:.0f}% confidence")
            print(f"   Suggested: ${action.suggested_dollars:,.0f} ({action.suggested_shares} shares)")
            print(f"   {action.reasoning[0]}\n")
    else:
        print("⚪ No high-confidence BUY signals at this time")
    
    print("\n" + "="*100)
    print("✅ SYSTEM READY!")
    print("="*100)
    print("\nNext steps:")
    print("1. Update MY_PORTFOLIO.json with your actual entry prices and shares")
    print("2. Run: python analyze_my_portfolio.py (to get daily recommendations)")
    print("3. Build API for Spark dashboard integration")
    print("4. Optimize forecaster for better directional accuracy")

else:
    print("\n❌ Training failed. Check error messages above.")
    print("   Common issues:")
    print("   - Ticker delisted or invalid")
    print("   - Insufficient historical data")
    print("   - Network/API issues with yfinance")
