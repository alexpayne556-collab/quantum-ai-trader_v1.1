# ═══════════════════════════════════════════════════════════
# 🏆 TEST INSTITUTIONAL-GRADE AI RECOMMENDER (FIXED)
# ═══════════════════════════════════════════════════════════
import sys
from pathlib import Path
import asyncio

# Setup paths
backend_dir = Path.cwd()
if 'backend' not in str(backend_dir):
    backend_dir = backend_dir / 'backend'
if str(backend_dir / 'modules') not in sys.path:
    sys.path.insert(0, str(backend_dir / 'modules'))

print("╔════════════════════════════════════════════════════════╗")
print("║  🏆 INSTITUTIONAL-GRADE AI RECOMMENDER TEST            ║")
print("║  Professional trade recommendations with full analysis ║")
print("╚════════════════════════════════════════════════════════╝\n")

# Force reload of modules
for mod in list(sys.modules.keys()):
    if 'fusior' in mod or 'recommender' in mod:
        del sys.modules[mod]

from fusior_forecast import run as run_forecast

async def test_recommender():
    symbol = 'NVDA'
    result = await run_forecast(symbol, visualize=False, horizon_days=21)

    if result and result.get("status") == "ok":
        ai_rec = result.get("ai_recommendation")
        
        if ai_rec is None:
            print(f"❌ Error: No AI recommendation generated")
            print(f"Result keys: {result.keys()}")
            return
        
        inst_grade = ai_rec.get("institutional_grade", {})
        
        print("="*80)
        print(f"🎯 {symbol} - INSTITUTIONAL RECOMMENDATION")
        print("="*80)
        
        # Basic recommendation
        action = ai_rec.get("action", "HOLD")
        confidence = result.get("confidence", 0)
        
        print(f"\n📊 RECOMMENDATION: {action}")
        print(f"📈 CONFIDENCE: {confidence:.1%}")
        print(f"\n📋 RATIONALE:")
        print(ai_rec.get("rationale", "No rationale provided"))
        
        if inst_grade:
            # Position sizing
            pos_sizing = inst_grade.get("position_sizing", {})
            if pos_sizing:
                print(f"\n" + "="*80)
                print(f"💰 POSITION SIZING (Kelly Criterion):")
                print(f"="*80)
                print(f"   Account: ${pos_sizing.get('account_balance', 50000):,.0f}")
                print(f"   Risk Amount: ${pos_sizing.get('risk_amount', 0):,.2f}")
                print(f"   Kelly %: {pos_sizing.get('kelly_pct', 0)*100:.2f}%")
                print(f"   Shares: {pos_sizing.get('shares', 0)}")
                print(f"   Position Value: ${pos_sizing.get('position_value', 0):,.2f}")
                print(f"   Portfolio %: {pos_sizing.get('position_pct', 0):.2f}%")
            
            # Risk/Reward
            rr = inst_grade.get("risk_reward", {})
            if rr:
                print(f"\n" + "="*80)
                print(f"⚖️  RISK:REWARD ANALYSIS:")
                print(f"="*80)
                print(f"   Entry: ${rr.get('entry', 0):.2f}")
                print(f"   Stop Loss: ${rr.get('stop', 0):.2f} (-{rr.get('risk_pct', 0):.2f}%)")
                print(f"   Target: ${rr.get('target', 0):.2f} (+{rr.get('reward_pct', 0):.2f}%)")
                print(f"   R:R Ratio: {rr.get('rr_ratio', 0):.2f}:1 {'✅' if rr.get('rr_ratio', 0) > 2.5 else '⚠️'}")
            
            # Entry strategy
            entry_strat = inst_grade.get("entry_strategy", {})
            if entry_strat:
                tranches = entry_strat.get("tranches", [])
                print(f"\n" + "="*80)
                print(f"🎯 ENTRY STRATEGY ({entry_strat.get('strategy_type', 'standard').upper()}):")
                print(f"="*80)
                for tranche in tranches:
                    print(f"   • {tranche.get('description', '')}")
            
            # Exit strategy
            exit_strat = inst_grade.get("exit_strategy", {})
            if exit_strat:
                profit_levels = exit_strat.get("profit_levels", [])
                print(f"\n" + "="*80)
                print(f"📈 EXIT STRATEGY:")
                print(f"="*80)
                print("   Profit Taking Levels:")
                for level in profit_levels:
                    print(f"      • {level.get('description', '')}")
                
                initial_stop = exit_strat.get("initial_stop", {})
                if initial_stop:
                    print(f"\n   Stop Loss:")
                    print(f"      • {initial_stop.get('description', '')}")
                
                trailing = exit_strat.get("trailing_stop", {})
                if trailing:
                    print(f"      • {trailing.get('description', '')}")
            
            # Trade rationale
            rationale = inst_grade.get("trade_rationale", "")
            if rationale:
                print(f"\n" + "="*80)
                print("📋 DETAILED INSTITUTIONAL ANALYSIS:")
                print("="*80)
                print(rationale)
        
        else:
            print("\n⚠️  Institutional features not available")
            print(f"   (Likely due to HOLD recommendation or low confidence)")
        
        print("\n" + "="*80)
        print("✅ INSTITUTIONAL RECOMMENDATION COMPLETE!")
        print("="*80)
        
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")

# Run the test
asyncio.run(test_recommender())

