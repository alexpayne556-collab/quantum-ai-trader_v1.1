"""
MONDAY MORNING SCANNER - Ready to Run at Market Open
======================================================
Run this at 9am Monday to get instant top picks!

1. Scans 50+ stocks in 2-3 minutes
2. Returns top 10 opportunities (sorted by score)
3. Shows detailed analysis of top 3
4. Exports CSV for tracking

NO MANUAL WORK. JUST RUN AND TRADE.
"""

# =============================================================================
# IMPORTS & SETUP
# =============================================================================

from google.colab import drive
drive.mount('/content/drive')

import sys
import os
import importlib.util

# Clear cache
for module in list(sys.modules.keys()):
    if any(x in module for x in ['api_integrations', 'coordinator', 'daily_scanner']):
        del sys.modules[module]

# Add path
sys.path.insert(0, '/content/drive/MyDrive/QuantumAI/backend/modules')

print("🔧 Force-loading modules...")

# Force load modules
def load_module_directly(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

base = '/content/drive/MyDrive/QuantumAI/backend/modules/'

api_integrations = load_module_directly('api_integrations', base + 'api_integrations.py')
coordinator = load_module_directly('professional_signal_coordinator', base + 'professional_signal_coordinator.py')
daily_scanner = load_module_directly('daily_scanner', base + 'daily_scanner.py')

from daily_scanner import ProfessionalDailyScanner, Strategy

print("✅ Modules loaded!\n")

# =============================================================================
# STOCK UNIVERSES
# =============================================================================

# PENNY STOCKS - High explosiv potential (100-500% moves)
PENNY_UNIVERSE = [
    # Current hot tickers
    'IMPP', 'BBIG', 'HEMP', 'CIFR', 'DRMA', 'SAVA', 'DRUG', 'OXBR',
    'SNDL', 'ILUS', 'ATER', 'PHUN', 'BKKT', 'CLOV', 
    
    # EV/Battery plays
    'GOEV', 'WKHS', 'MULN', 'HYLN', 'AYRO',
    
    # Biotech movers
    'OCGN', 'NVAX', 'SENS', 'TNXP', 'GEVO', 'IZEA',
    
    # Crypto-related
    'MARA', 'RIOT', 'BTBT', 'SOS', 'CAN', 'EBON',
    
    # Small cap tech
    'WORX', 'GROM', 'DPLS'
]

# SWING TRADES - Larger caps (20-50% moves, safer)
SWING_UNIVERSE = [
    'NVDA', 'AMD', 'TSLA', 'PLTR', 'SOFI', 'NIO', 'LCID', 'RIVN',
    'META', 'GOOGL', 'AMZN', 'MSFT', 'AAPL', 'NFLX', 'DIS',
    'COIN', 'RIOT', 'MARA', 'PYPL', 'SHOP', 'UBER', 'DASH',
    'CRWD', 'NET', 'SNOW', 'ZM', 'DKNG', 'AFRM', 'U'
]

# =============================================================================
# MONDAY MORNING SCANNER
# =============================================================================

def monday_morning_scan(scan_pennies=True, scan_swings=True):
    """
    Complete Monday morning scan
    
    Args:
        scan_pennies: Scan penny stocks (100-500% potential)
        scan_swings: Scan swing trades (20-50% potential)
    """
    
    print("\n" + "="*80)
    print("🌅 MONDAY MORNING SCANNER - MARKET OPEN")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    all_results = []
    
    # Scan penny stocks
    if scan_pennies:
        print("🔥 SCANNING PENNY STOCKS (100-500% potential)")
        print("-"*80)
        
        penny_scanner = ProfessionalDailyScanner(
            strategy=Strategy.PENNY_STOCK,
            max_workers=4
        )
        
        penny_results = penny_scanner.scan_universe(PENNY_UNIVERSE, use_modules=False)
        
        print(f"\n✅ Penny scan complete: {len(penny_results)} stocks scored")
        print(f"   Strong Buys (85+): {len([r for r in penny_results if r.final_score >= 85])}")
        print(f"   Buys (70-85): {len([r for r in penny_results if 70 <= r.final_score < 85])}")
        print()
        
        all_results.extend(penny_results)
    
    # Scan swing trades
    if scan_swings:
        print("📈 SCANNING SWING TRADES (20-50% potential)")
        print("-"*80)
        
        swing_scanner = ProfessionalDailyScanner(
            strategy=Strategy.SWING_TRADE,
            max_workers=4
        )
        
        swing_results = swing_scanner.scan_universe(SWING_UNIVERSE, use_modules=False)
        
        print(f"\n✅ Swing scan complete: {len(swing_results)} stocks scored")
        print(f"   Strong Buys (82+): {len([r for r in swing_results if r.final_score >= 82])}")
        print(f"   Buys (70-82): {len([r for r in swing_results if 70 <= r.final_score < 82])}")
        print()
        
        all_results.extend(swing_results)
    
    # Sort all results by score
    all_results.sort(key=lambda x: x.final_score, reverse=True)
    
    # Top 10 opportunities
    top_10 = all_results[:10]
    
    print("\n" + "="*80)
    print("🏆 TOP 10 OPPORTUNITIES - READY TO TRADE")
    print("="*80)
    print(f"{'Rank':<6} {'Ticker':<8} {'Score':<8} {'Strategy':<12} {'Rec':<12} {'Signals'}")
    print("-"*80)
    
    for i, result in enumerate(top_10, 1):
        strategy_name = result.strategy.value.replace('_', ' ').title()
        print(f"{i:<6} {result.ticker:<8} {result.final_score:>6.1f}  "
              f"{strategy_name:<12} {result.recommendation.label:<12} {result.signals_used}/7")
    
    print("="*80)
    
    # Detailed analysis of top 3
    print("\n" + "="*80)
    print("🔍 DETAILED ANALYSIS - TOP 3 PICKS")
    print("="*80)
    
    for i, result in enumerate(top_10[:3], 1):
        print(f"\n{'─'*80}")
        print(f"#{i} - {result.ticker} ({result.strategy.value.replace('_', ' ').title()})")
        print('─'*80)
        print(f"AI Score: {result.final_score:.1f}/100 {result.recommendation.emoji}")
        print(f"Recommendation: {result.recommendation.label}")
        print(f"Confidence: {result.missing_signal_penalty:.0%}")
        print(f"\nTop Signals:")
        
        sorted_signals = sorted(
            result.normalized_signals.items(),
            key=lambda x: x[1] * result.weights.get(x[0], 0),
            reverse=True
        )[:3]
        
        for name, value in sorted_signals:
            weight = result.weights.get(name, 0)
            print(f"  • {name:15s}: {value:5.1f}/100 (weight: {weight:4.1%})")
        
        print(f"\n💡 Trade Setup:")
        if result.strategy.value == 'penny_stock':
            print(f"  • Entry: Current price")
            print(f"  • Target: 50-200% gain")
            print(f"  • Stop: 15-20% below entry")
            print(f"  • Position: 10-15% of account")
        else:
            print(f"  • Entry: Current price")
            print(f"  • Target: 20-50% gain")
            print(f"  • Stop: 7-8% below entry")
            print(f"  • Position: 5-8% of account")
    
    print("\n" + "="*80)
    
    # Export to CSV
    filename = f"monday_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    import pandas as pd
    data = [r.to_dict() for r in all_results]
    df = pd.DataFrame(data)
    df.to_csv(f"/content/drive/MyDrive/QuantumAI/{filename}", index=False)
    
    print(f"\n💾 Results exported: {filename}")
    print(f"   Location: /content/drive/MyDrive/QuantumAI/{filename}")
    
    print("\n" + "="*80)
    print("✅ SCAN COMPLETE - READY TO TRADE!")
    print("="*80)
    print("\n💡 NEXT STEPS:")
    print("   1. Review top 3 picks above")
    print("   2. Check Robinhood for current prices")
    print("   3. Look for news/catalysts")
    print("   4. Enter positions with proper stop losses")
    print("   5. Track performance in exported CSV")
    print("\n" + "="*80 + "\n")
    
    return top_10


# =============================================================================
# RUN MONDAY SCAN
# =============================================================================

from datetime import datetime
import time

print("⏰ Checking time...")
now = datetime.now()
print(f"Current time: {now.strftime('%A, %B %d, %Y - %I:%M %p')}")

if now.weekday() in [5, 6]:  # Saturday or Sunday
    print("⚠️  It's the weekend! Market is closed.")
    print("   Running demo scan anyway...")

print("\n🚀 Starting Monday Morning Scan...\n")

start_time = time.time()

top_picks = monday_morning_scan(scan_pennies=True, scan_swings=True)

elapsed = time.time() - start_time

print(f"\n⏱️  Total scan time: {elapsed:.1f} seconds")
print(f"📊 Average: {elapsed/10:.1f}s per top pick")

print("\n🎯 YOUR TOP PICK:")
if top_picks:
    top = top_picks[0]
    print(f"   {top.ticker} - {top.final_score:.1f}/100 {top.recommendation.emoji}")
    print(f"   {top.strategy.value.replace('_', ' ').title()} opportunity")
    print(f"   Open Robinhood and check this FIRST!")

