"""
Real-world test for Watchlist Scanner
Tests all 20 tickers with real API data and technical analysis
"""
import asyncio
from watchlist_scanner import WatchlistScanner

async def test_watchlist_scanner():
    print("🚀 Starting Institutional-Grade Watchlist Scanner...")
    print(f"📊 Analyzing all 20 tickers with real-time data\n")
    
    scanner = WatchlistScanner()
    
    # Scan all symbols
    results = await scanner.scan_all_symbols()
    
    # Print full results
    scanner.print_scan_results()
    
    # Get top opportunities
    print("\n" + "="*80)
    print("🎯 TOP TRADING OPPORTUNITIES (Signal Strength >= 60)")
    print("="*80)
    
    opportunities = scanner.get_top_opportunities(min_strength=60.0, limit=10)
    
    if opportunities:
        for i, opp in enumerate(opportunities, 1):
            print(f"\n#{i} {opp.symbol} - {opp.recommendation}")
            print(f"   Price: ${opp.price:.2f}")
            print(f"   Signal Strength: {opp.signal_strength:.1f}/100")
            print(f"   RSI (5m/1h): {opp.rsi_5m:.1f} / {opp.rsi_1h:.1f}")
            print(f"   MACD: {opp.macd_signal}")
            print(f"   Bollinger: {opp.bb_signal}")
            print(f"   Volume Surge: {'YES ✓' if opp.volume_surge else 'NO'}")
            print(f"   Momentum: {opp.momentum_score:.1f}/100")
    else:
        print("\n⚠️ No high-strength opportunities found at this time")
    
    # Export results
    scanner.export_to_json("data/watchlist_scan_results.json")
    
    print("\n✅ Watchlist scan completed!")
    print(f"📈 Scanned {len(results)} symbols successfully")

if __name__ == "__main__":
    asyncio.run(test_watchlist_scanner())
