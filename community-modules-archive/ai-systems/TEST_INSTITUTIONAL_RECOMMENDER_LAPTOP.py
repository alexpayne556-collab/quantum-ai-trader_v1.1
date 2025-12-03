"""
═════════════════════════════════════════════════════════════════════════════
🧪 INSTITUTIONAL AI RECOMMENDER - VALIDATION TEST (LAPTOP)
═════════════════════════════════════════════════════════════════════════════
Run this test on your LAPTOP to verify the institutional-grade AI recommender
integration is working correctly with fusior_forecast.

Expected Output:
✅ Position Sizing with Kelly Criterion
✅ Risk:Reward ratios (should be >0.00:1)
✅ Entry/Exit strategies with tranches
✅ Detailed trade rationale

Run: python TEST_INSTITUTIONAL_RECOMMENDER_LAPTOP.py
═════════════════════════════════════════════════════════════════════════════
"""

import sys
import asyncio
from pathlib import Path

# Add backend/modules to path
MODULES_DIR = Path(__file__).resolve().parent / "backend" / "modules"
sys.path.insert(0, str(MODULES_DIR))

async def test_institutional_recommender():
    """Test institutional AI recommender on a single stock"""
    
    print("=" * 80)
    print("🧪 TESTING INSTITUTIONAL AI RECOMMENDER INTEGRATION")
    print("=" * 80)
    
    # Import forecaster
    from fusior_forecast import run as run_forecast
    
    # Test on a volatile stock
    symbol = "NVDA"
    print(f"\n🔮 Running institutional-grade forecast on {symbol}...\n")
    
    try:
        # Run forecast (this should call AI recommender with institutional enhancement)
        result = await run_forecast(symbol=symbol, horizon_days=21)
        
        # Extract AI recommendation
        ai_rec = result.get("ai_recommendation")
        
        if not ai_rec:
            print("❌ ERROR: No AI recommendation in result!")
            return
        
        # Check for institutional features
        inst_grade = ai_rec.get("institutional_grade")
        
        print("=" * 80)
        print(f"📊 RESULTS FOR {symbol}")
        print("=" * 80)
        
        # Basic recommendation
        print(f"\n🎯 RECOMMENDATION: {ai_rec.get('action', 'UNKNOWN')}")
        print(f"📈 CONFIDENCE: {ai_rec.get('confidence', 0.0):.1%}")
        print(f"💰 EXPECTED 5D MOVE: {ai_rec.get('expected_move_5d', 0.0):+.2f}%")
        print(f"💰 EXPECTED 20D MOVE: {ai_rec.get('expected_move_20d', 0.0):+.2f}%")
        
        # Check institutional features
        if not inst_grade:
            print("\n" + "=" * 80)
            print("❌ ERROR: INSTITUTIONAL FEATURES MISSING!")
            print("=" * 80)
            print("\nThe AI recommender did NOT include institutional-grade features.")
            print("This means the integration in fusior_forecast.py failed.")
            return
        
        print("\n" + "=" * 80)
        print("✅ INSTITUTIONAL FEATURES DETECTED!")
        print("=" * 80)
        
        # Position Sizing
        pos_sizing = inst_grade.get("position_sizing", {})
        if pos_sizing:
            print("\n📊 POSITION SIZING (Kelly Criterion):")
            print(f"   Risk Amount: ${pos_sizing.get('risk_amount', 0):.2f}")
            print(f"   Kelly %: {pos_sizing.get('kelly_pct', 0):.2%}")
            print(f"   Shares: {pos_sizing.get('shares', 0)}")
            print(f"   Position Value: ${pos_sizing.get('position_value', 0):.2f}")
            print(f"   Position %: {pos_sizing.get('position_pct', 0):.2f}%")
        else:
            print("\n⚠️ Position sizing missing!")
        
        # Risk:Reward
        rr = inst_grade.get("risk_reward", {})
        if rr:
            print("\n⚖️ RISK:REWARD:")
            print(f"   Entry: ${rr.get('entry', 0):.2f}")
            print(f"   Stop: ${rr.get('stop', 0):.2f}")
            print(f"   Target: ${rr.get('target', 0):.2f}")
            print(f"   Risk: {rr.get('risk_pct', 0):.2f}%")
            print(f"   Reward: {rr.get('reward_pct', 0):.2f}%")
            
            rr_ratio = rr.get('rr_ratio', 0)
            print(f"   R:R Ratio: {rr_ratio:.2f}:1 ", end="")
            if rr_ratio > 0:
                print("✅ VALID")
            else:
                print("❌ INVALID (should be >0)")
        else:
            print("\n⚠️ Risk:Reward missing!")
        
        # Entry Strategy
        entry_strat = inst_grade.get("entry_strategy", {})
        if entry_strat and entry_strat.get("tranches"):
            print("\n📍 ENTRY STRATEGY:")
            print(f"   Strategy Type: {entry_strat.get('strategy_type', 'unknown')}")
            print(f"   Timeframe: {entry_strat.get('timeframe', 'unknown')}")
            for tranche in entry_strat.get("tranches", [])[:3]:  # Show first 3
                print(f"   Tranche {tranche.get('tranche', '?')}: "
                      f"{tranche.get('allocation', 0)*100:.0f}% @ ${tranche.get('price', 0):.2f} "
                      f"({tranche.get('trigger', 'unknown')})")
        else:
            print("\n⚠️ Entry strategy missing!")
        
        # Exit Strategy
        exit_strat = inst_grade.get("exit_strategy", {})
        if exit_strat:
            print("\n🚪 EXIT STRATEGY:")
            
            # Profit levels
            profit_levels = exit_strat.get("profit_levels", [])
            if profit_levels:
                print("   Profit Targets:")
                for level in profit_levels[:3]:  # Show first 3
                    print(f"   Level {level.get('level', '?')}: "
                          f"{level.get('allocation', 0)*100:.0f}% @ ${level.get('price', 0):.2f} "
                          f"(+{level.get('gain_pct', 0):.1f}%)")
            
            # Stops
            initial_stop = exit_strat.get("initial_stop", {})
            if initial_stop:
                print(f"\n   Initial Stop: ${initial_stop.get('price', 0):.2f} "
                      f"({initial_stop.get('loss_pct', 0):.1f}%)")
            
            trailing_stop = exit_strat.get("trailing_stop", {})
            if trailing_stop:
                print(f"   Trailing Stop: {trailing_stop.get('type', 'unknown')} "
                      f"({trailing_stop.get('distance_atr', 0):.1f}x ATR)")
        else:
            print("\n⚠️ Exit strategy missing!")
        
        # Trade Rationale
        rationale = inst_grade.get("trade_rationale", "")
        if rationale:
            print("\n" + "=" * 80)
            print("📋 TRADE RATIONALE:")
            print("=" * 80)
            print(rationale)
        else:
            print("\n⚠️ Trade rationale missing!")
        
        # Final Assessment
        print("\n" + "=" * 80)
        print("🎓 FINAL ASSESSMENT")
        print("=" * 80)
        
        checks = {
            "Position Sizing": bool(pos_sizing and pos_sizing.get("shares", 0) > 0),
            "Risk:Reward Valid": bool(rr and rr.get("rr_ratio", 0) > 0),
            "Entry Strategy": bool(entry_strat and entry_strat.get("tranches")),
            "Exit Strategy": bool(exit_strat and exit_strat.get("profit_levels")),
            "Trade Rationale": bool(rationale and len(rationale) > 100)
        }
        
        passed = sum(checks.values())
        total = len(checks)
        
        for check_name, check_passed in checks.items():
            status = "✅" if check_passed else "❌"
            print(f"{status} {check_name}")
        
        print("\n" + "=" * 80)
        print(f"📊 SCORE: {passed}/{total} ({passed/total*100:.0f}%)")
        print("=" * 80)
        
        if passed == total:
            print("\n🎉 SUCCESS! Institutional AI Recommender is FULLY OPERATIONAL!")
        elif passed >= 3:
            print("\n⚠️ PARTIAL SUCCESS - Some features missing but core functionality works")
        else:
            print("\n❌ FAILURE - Major integration issues detected")
        
        print("\n✅ Test complete!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n🚀 Starting test...\n")
    asyncio.run(test_institutional_recommender())

