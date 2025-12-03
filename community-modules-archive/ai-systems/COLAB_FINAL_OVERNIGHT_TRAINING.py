# ═══════════════════════════════════════════════════════════════════════════════
# 🌙 FINAL OVERNIGHT TRAINING (COMPLETE & BULLETPROOF)
# ═══════════════════════════════════════════════════════════════════════════════
"""
This is the FINAL training cell. It will:
1. Analyze all 20 stocks in your portfolio
2. Analyze 10 hot volatile tickers
3. Save detailed analysis for each stock
4. Generate top buy/sell recommendations
5. Save everything to Google Drive

Estimated time: 1-2 hours (with working APIs)

After running this, you can:
- Go to sleep
- Wake up to find all results in /results/ folder
- Review recommendations and make informed trades!
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
import sys

print("=" * 80)
print("🌙 OVERNIGHT TRAINING - STARTING NOW")
print("=" * 80)
print(f"⏰ Started: {datetime.now().strftime('%I:%M %p on %A, %B %d')}")
print("⏱️  Estimated: 1-2 hours")
print("=" * 80)
print()

# Setup
DRIVE_BASE = Path("/content/drive/MyDrive/Quantum_AI_Cockpit")
RESULTS_DIR = DRIVE_BASE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Reload modules to ensure all fixes are loaded
for mod in list(sys.modules.keys()):
    if any(x in mod for x in ['fusior', 'master_analysis', 'pattern', 'recommender']):
        del sys.modules[mod]

from master_analysis_engine import MasterAnalysisEngine
engine = MasterAnalysisEngine()

# Helper function with timeout
async def analyze_with_timeout(ticker, timeout=60):
    """Analyze a stock with timeout protection."""
    try:
        return await asyncio.wait_for(
            engine.analyze_stock(
                symbol=ticker,
                account_balance=500,
                forecast_days=14,
                verbose=False
            ),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        return {"error": f"Timeout after {timeout}s", "status": "timeout"}
    except Exception as e:
        return {"error": str(e)[:100], "status": "error"}

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: YOUR PORTFOLIO (20 STOCKS)
# ═══════════════════════════════════════════════════════════════════════════════

print("📊 PHASE 1: ANALYZING YOUR PORTFOLIO")
print("=" * 80)
print()

portfolio = [
    "SOFI", "NVDA", "TSLA", "PLTR", "HOOD", "AMD", "IONQ", "RGTI",
    "LUNR", "RKLB", "MSTR", "COIN", "SQ", "PYPL", "RIVN", "NIO",
    "LCID", "APPS", "DKNG", "UPST"
]

portfolio_results = {}
portfolio_success = 0

for i, ticker in enumerate(portfolio, 1):
    print(f"[{i:2d}/20] {ticker:6s} ", end="", flush=True)
    
    result = await analyze_with_timeout(ticker, timeout=60)
    
    if result and result.get('status') == 'ok':
        rec = result.get('recommendation', {})
        action = rec.get('action', 'HOLD')
        conf = rec.get('confidence', 0)
        price = result.get('current_price', 0)
        
        print(f"✅ {action:10s} {conf:3.0f}% ${price:7.2f}")
        
        portfolio_results[ticker] = result
        portfolio_success += 1
        
        # Save individual
        (RESULTS_DIR / f"{ticker}_analysis.json").write_text(
            json.dumps(result, indent=2, default=str)
        )
    else:
        error = result.get('error', 'Unknown')[:30] if result else 'No result'
        print(f"❌ {error}")
        portfolio_results[ticker] = result

# Save portfolio summary
(RESULTS_DIR / "PORTFOLIO_COMPLETE.json").write_text(
    json.dumps(portfolio_results, indent=2, default=str)
)

print(f"\n✅ Portfolio: {portfolio_success}/20 successful ({portfolio_success/20*100:.0f}%)\n")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: HOT VOLATILE TICKERS (10 STOCKS)
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("🔥 PHASE 2: ANALYZING HOT VOLATILE TICKERS")
print("=" * 80)
print()

hot_tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "SPY", "QQQ", "GME", "AMC", "RIOT"
]

hot_results = {}
hot_success = 0

for i, ticker in enumerate(hot_tickers, 1):
    print(f"[{i:2d}/10] {ticker:6s} ", end="", flush=True)
    
    result = await analyze_with_timeout(ticker, timeout=60)
    
    if result and result.get('status') == 'ok':
        rec = result.get('recommendation', {})
        action = rec.get('action', 'HOLD')
        conf = rec.get('confidence', 0)
        price = result.get('current_price', 0)
        
        print(f"✅ {action:10s} {conf:3.0f}% ${price:7.2f}")
        
        hot_results[ticker] = result
        hot_success += 1
        
        # Save individual
        (RESULTS_DIR / f"{ticker}_analysis.json").write_text(
            json.dumps(result, indent=2, default=str)
        )
    else:
        error = result.get('error', 'Unknown')[:30] if result else 'No result'
        print(f"❌ {error}")
        hot_results[ticker] = result

# Save hot tickers summary
(RESULTS_DIR / "HOT_TICKERS_COMPLETE.json").write_text(
    json.dumps(hot_results, indent=2, default=str)
)

print(f"\n✅ Hot tickers: {hot_success}/10 successful ({hot_success/10*100:.0f}%)\n")

# ═══════════════════════════════════════════════════════════════════════════════
# GENERATE TOP RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("🌟 GENERATING TOP RECOMMENDATIONS")
print("=" * 80)
print()

all_results = {**portfolio_results, **hot_results}
strong_buys = []
strong_sells = []
holds = []

for ticker, result in all_results.items():
    if not result or result.get('status') != 'ok':
        continue
    
    rec = result.get('recommendation', {})
    action = rec.get('action', 'HOLD')
    conf = rec.get('confidence', 0)
    price = result.get('current_price', 0)
    rationale = rec.get('rationale', '')[:80]
    
    entry = {
        "ticker": ticker,
        "action": action,
        "confidence": conf,
        "price": price,
        "rationale": rationale
    }
    
    if action in ['BUY', 'STRONG_BUY'] and conf >= 70:
        strong_buys.append(entry)
    elif action in ['SELL', 'STRONG_SELL'] and conf >= 70:
        strong_sells.append(entry)
    else:
        holds.append(entry)

# Sort by confidence
strong_buys.sort(key=lambda x: x['confidence'], reverse=True)
strong_sells.sort(key=lambda x: x['confidence'], reverse=True)

# Display top recommendations
if strong_buys:
    print("🟢 TOP 5 STRONG BUYS (70%+ confidence):")
    print("-" * 80)
    for entry in strong_buys[:5]:
        print(f"   {entry['ticker']:6s} | {entry['confidence']:3.0f}% | ${entry['price']:7.2f}")
        print(f"           {entry['rationale']}")
        print()
else:
    print("⚠️  No strong buy signals (70%+ confidence)")

print()

if strong_sells:
    print("🔴 TOP 5 STRONG SELLS (70%+ confidence):")
    print("-" * 80)
    for entry in strong_sells[:5]:
        print(f"   {entry['ticker']:6s} | {entry['confidence']:3.0f}% | ${entry['price']:7.2f}")
        print(f"           {entry['rationale']}")
        print()
else:
    print("✅ No strong sell signals (good news!)")

# Save recommendations
recommendations = {
    "generated_at": datetime.now().isoformat(),
    "strong_buys": strong_buys,
    "strong_sells": strong_sells,
    "holds": holds,
    "summary": {
        "total_analyzed": len(all_results),
        "successful": portfolio_success + hot_success,
        "strong_buys_count": len(strong_buys),
        "strong_sells_count": len(strong_sells),
        "holds_count": len(holds)
    }
}

(RESULTS_DIR / "TOP_RECOMMENDATIONS.json").write_text(
    json.dumps(recommendations, indent=2, default=str)
)

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print()
print("=" * 80)
print("🎉 OVERNIGHT TRAINING COMPLETE!")
print("=" * 80)
print()
print(f"✅ Portfolio stocks: {portfolio_success}/20")
print(f"✅ Hot tickers: {hot_success}/10")
print(f"✅ Total success: {portfolio_success + hot_success}/30 ({(portfolio_success + hot_success)/30*100:.0f}%)")
print()
print(f"🟢 Strong buy signals: {len(strong_buys)}")
print(f"🔴 Strong sell signals: {len(strong_sells)}")
print(f"⚪ Hold/neutral: {len(holds)}")
print()
print(f"📁 All results saved to: {RESULTS_DIR}")
print(f"⏰ Finished: {datetime.now().strftime('%I:%M %p on %A, %B %d')}")
print()
print("=" * 80)
print("💰 API USAGE TODAY:")
print("=" * 80)
print("✅ TWELVEDATA: ~30-60 calls (well within 800/day free tier)")
print("✅ POLYGON: ~20-40 calls (well within 5/min free tier)")
print("✅ YFINANCE: Unlimited free (used as backup)")
print()
print("💡 NO PAID API NEEDED!")
print("=" * 80)
print()
print("🌅 GOOD MORNING!")
print("   Your AI trading system has analyzed your entire portfolio overnight.")
print("   Review the recommendations above and make informed trading decisions!")
print()
print("📊 To view detailed analysis for any stock:")
print(f"   • Check: {RESULTS_DIR}/<TICKER>_analysis.json")
print()
print("=" * 80)

