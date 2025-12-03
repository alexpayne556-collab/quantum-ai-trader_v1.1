"""
🔍 DEBUG ENSEMBLE VOTING - SEE WHY CONFIDENCE = 0%
"""

from google.colab import drive
import sys, asyncio

drive.mount('/content/drive', force_remount=False)

PROJECT_ROOT = "/content/drive/MyDrive/Quantum_AI_Cockpit"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, f"{PROJECT_ROOT}/backend/modules")

# Clear cache
for mod in list(sys.modules.keys()):
    if any(x in mod for x in ['fusior', 'ai_recommender', 'master_analysis', 'institutional', 'pattern', 'elite']):
        try:
            del sys.modules[mod]
        except:
            pass

print("✅ Setup complete\n")

async def debug_ensemble():
    from master_analysis_institutional import InstitutionalAnalysisEngine

    symbol = "AMD"
    engine = InstitutionalAnalysisEngine()

    print("="*80)
    print(f"🔍 DEBUGGING ENSEMBLE VOTING FOR {symbol}")
    print("="*80)
    print()

    # Get signals from all modules
    print("📊 GATHERING SIGNALS FROM ALL MODULES...")
    print("-" * 80)

    signals = await engine._gather_all_signals(symbol, forecast_days=5)

    for module_name, signal in signals.items():
        if signal.get('direction', 'neutral') != 'neutral':
            status = "✅ ACTIVE"
            emoji = "📈" if signal['direction'] in ['bullish', 'up'] else "📉"
            conf = signal.get('confidence', 0) * 100
            print(f"  {status} {module_name:15s}: {emoji} {signal['direction']:8s} ({conf:.1f}% confidence)")
        else:
            print(f"  ⚪ {module_name:15s}: Neutral/failed")

    print()
    print("⚖️  ENSEMBLE VOTING...")
    print("-" * 80)

    # Calculate ensemble vote
    ensemble_vote = engine._weighted_ensemble_vote(signals)

    print(f"  📊 Ensemble Decision: {ensemble_vote['action']} ({ensemble_vote['confidence']*100:.1f}% confidence)")
    print(f"  📊 Bullish Score: {ensemble_vote['bullish_score']:.3f}")
    print(f"  📊 Bearish Score: {ensemble_vote['bearish_score']:.3f}")

    print()
    print("🔧 MARKET CONTEXT ADJUSTMENT...")
    print("-" * 80)

    # Apply market context
    context_adjusted = engine._adjust_for_market_context(ensemble_vote, symbol)
    print(f"  📊 After Context: {context_adjusted['action']} ({context_adjusted['confidence']*100:.1f}% confidence)")

    if 'adjustments' in context_adjusted:
        for adj in context_adjusted['adjustments']:
            print(f"  📝 {adj}")

    print()
    print("🎯 FINAL ANALYSIS...")
    print("-" * 80)

    # Get full analysis
    analysis = await engine.analyze_with_ensemble(symbol, account_balance=10000, forecast_days=5)

    print(f"  💰 Price: ${analysis.get('current_price', 0):.2f}")
    print(f"  📊 Action: {analysis.get('action', 'N/A')}")
    print(f"  🎯 Confidence: {analysis.get('confidence', 0)*100:.1f}%")

    print()
    print("="*80)
    print("🔧 DIAGNOSTICS")
    print("="*80)

    # Check if any modules are contributing
    active_modules = sum(1 for s in signals.values() if s.get('direction', 'neutral') != 'neutral')
    print(f"  📊 Active Modules: {active_modules}/5")

    # Check ensemble weights
    print(f"  ⚖️  Module Weights:")
    for module, weight in engine.module_weights.items():
        print(f"      {module:15s}: {weight:.3f}")

    # Check if trade plan should be generated
    if analysis.get('action') in ['BUY', 'STRONG_BUY', 'BUY_THE_DIP']:
        print("  ✅ Trade plan should be generated")
    else:
        print("  ❌ No trade plan (action is not BUY/STRONG_BUY/BUY_THE_DIP)")

    print()
    print("="*80)
    print("💡 RECOMMENDATIONS")
    print("="*80)

    if active_modules == 0:
        print("  • All modules are neutral/failed - check individual module outputs")
        print("  • Try different symbol or debug each module separately")
    elif active_modules < 3:
        print("  • Only few modules active - ensemble voting too conservative")
        print("  • May need to lower voting thresholds or increase module weights")
    else:
        print("  • Multiple modules active but ensemble still neutral")
        print("  • Check if bullish/bearish signals are balanced")

    if analysis.get('confidence', 0) == 0:
        print("  • Confidence = 0% - ensemble voting threshold too high")
        print("  • Consider lowering the decision threshold in _weighted_ensemble_vote")

await debug_ensemble()
