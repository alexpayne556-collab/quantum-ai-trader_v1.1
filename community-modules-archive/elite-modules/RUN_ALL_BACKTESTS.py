"""
MASTER BACKTEST & CONFIDENCE BUILDER
=====================================
Run ALL backtests and generate combined confidence report

This is your ONE-CLICK confidence builder before risking $500
"""

import subprocess
import sys
from datetime import datetime

print("="*80)
print("🚀 QUANTUM AI CONFIDENCE BUILDER")
print("="*80)
print("Build confidence through DATA before risking your $500")
print("="*80)
print()
print("This will:")
print("  1. Test penny stock explosion detection (20+ historical winners)")
print("  2. Test swing trade recommendations (20+ historical trades)")
print("  3. Generate combined confidence report")
print("  4. Tell you if you're ready for paper trading")
print()
print("=" *80)

input("\nPress ENTER to start backtesting...")

print("\n" + "="*80)
print("📊 PHASE 1: PENNY STOCK EXPLOSION BACKTEST")
print("="*80)
print("Testing if AI can detect 50-500% explosive moves BEFORE they happen...")
print()

try:
    import BACKTEST_PENNY_EXPLOSIONS
    from BACKTEST_PENNY_EXPLOSIONS import PennyStockBacktester
    
    penny_tester = PennyStockBacktester()
    penny_tester.load_modules()
    penny_tester.run_all_tests()
    
    # Get results
    if penny_tester.results:
        penny_detected = len([r for r in penny_tester.results if r['detected']])
        penny_total = len(penny_tester.results)
        penny_detection_rate = (penny_detected / penny_total * 100) if penny_total > 0 else 0
    else:
        penny_detection_rate = 0
        penny_total = 0
    
    print(f"\n✅ Phase 1 complete: {penny_detection_rate:.1f}% detection rate")
    
except Exception as e:
    print(f"\n❌ Phase 1 failed: {e}")
    penny_detection_rate = 0
    penny_total = 0

print("\n" + "="*80)
print("📈 PHASE 2: SWING TRADE BACKTEST")
print("="*80)
print("Testing if AI can identify profitable swing trades (3-4 month holds)...")
print()

try:
    import BACKTEST_SWING_TRADES
    from BACKTEST_SWING_TRADES import SwingTradeBacktester
    
    swing_tester = SwingTradeBacktester()
    swing_tester.load_modules()
    swing_tester.run_all_tests()
    
    # Get results
    if swing_tester.results:
        swing_recommended = [r for r in swing_tester.results if r['recommended']]
        swing_winners = [r for r in swing_recommended if r['win']]
        swing_win_rate = (len(swing_winners) / len(swing_recommended) * 100) if swing_recommended else 0
    else:
        swing_win_rate = 0
        swing_recommended = []
    
    print(f"\n✅ Phase 2 complete: {swing_win_rate:.1f}% win rate")
    
except Exception as e:
    print(f"\n❌ Phase 2 failed: {e}")
    swing_win_rate = 0
    swing_recommended = []

# Generate combined confidence report
print("\n" + "="*80)
print("🎯 COMBINED CONFIDENCE REPORT")
print("="*80)
print()

print(f"📊 PENNY STOCK STRATEGY:")
print(f"   Tests: {penny_total}")
print(f"   Detection rate: {penny_detection_rate:.1f}%")
if penny_detection_rate >= 60:
    print(f"   Status: ✅ READY FOR PAPER TRADING")
    penny_ready = True
elif penny_detection_rate >= 50:
    print(f"   Status: ⚠️  NEEDS TUNING")
    penny_ready = False
else:
    print(f"   Status: ❌ NOT READY")
    penny_ready = False

print()
print(f"📈 SWING TRADE STRATEGY:")
print(f"   Tests: {len(swing_recommended) if swing_recommended else 0}")
print(f"   Win rate: {swing_win_rate:.1f}%")
if swing_win_rate >= 60:
    print(f"   Status: ✅ READY FOR PAPER TRADING")
    swing_ready = True
elif swing_win_rate >= 50:
    print(f"   Status: ⚠️  NEEDS TUNING")
    swing_ready = False
else:
    print(f"   Status: ❌ NOT READY")
    swing_ready = False

print()
print("="*80)
print("🎯 OVERALL CONFIDENCE:")
print("="*80)

if penny_ready and swing_ready:
    print()
    print("   🔥 VERY HIGH CONFIDENCE!")
    print()
    print("   ✅ Both strategies tested well")
    print("   ✅ AI can detect penny explosions")
    print("   ✅ AI can identify swing trades")
    print()
    print("   📋 NEXT STEPS:")
    print("   1. Start 2-week paper trading (risk-free)")
    print("   2. Track every signal")
    print("   3. Build real-world experience")
    print("   4. After 2 weeks profitable → Go live with $100")
    print()
    overall_confidence = "VERY HIGH"
    
elif penny_ready or swing_ready:
    print()
    print("   ✅ MEDIUM-HIGH CONFIDENCE")
    print()
    if penny_ready:
        print("   ✅ Penny stock strategy looks good")
        print("   ⚠️  Swing trade strategy needs work")
        print()
        print("   📋 RECOMMENDATION:")
        print("   - Focus on penny stocks for now")
        print("   - Paper trade penny strategy")
        print("   - Improve swing trade scoring")
    else:
        print("   ⚠️  Penny stock strategy needs work")
        print("   ✅ Swing trade strategy looks good")
        print()
        print("   📋 RECOMMENDATION:")
        print("   - Focus on swing trades for now")
        print("   - Paper trade swing strategy")
        print("   - Improve penny detection")
    print()
    overall_confidence = "MEDIUM-HIGH"
    
else:
    print()
    print("   ⚠️  IMPROVEMENT NEEDED")
    print()
    print("   Both strategies below 60% target")
    print()
    print("   📋 RECOMMENDATION:")
    print("   - Review backtest details above")
    print("   - Identify why AI missed good opportunities")
    print("   - Tune thresholds and weights")
    print("   - Re-run backtests")
    print("   - DON'T risk $500 yet")
    print()
    overall_confidence = "LOW"

print("="*80)

# Save summary report
report_filename = f"confidence_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(report_filename, 'w') as f:
    f.write("QUANTUM AI CONFIDENCE REPORT\n")
    f.write("="*80 + "\n\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"PENNY STOCK STRATEGY:\n")
    f.write(f"  Detection rate: {penny_detection_rate:.1f}%\n")
    f.write(f"  Status: {'READY' if penny_ready else 'NOT READY'}\n\n")
    f.write(f"SWING TRADE STRATEGY:\n")
    f.write(f"  Win rate: {swing_win_rate:.1f}%\n")
    f.write(f"  Status: {'READY' if swing_ready else 'NOT READY'}\n\n")
    f.write(f"OVERALL CONFIDENCE: {overall_confidence}\n")

print(f"\n💾 Report saved: {report_filename}")
print()
print("="*80)
print("✅ BACKTEST COMPLETE!")
print("="*80)
print()
print("Review the detailed results above to understand:")
print("  • Which trades AI correctly identified")
print("  • Which trades AI missed")
print("  • Why the scoring worked or didn't work")
print()
print("Next: Based on confidence level, proceed to paper trading or tune AI")
print()
print("="*80)

