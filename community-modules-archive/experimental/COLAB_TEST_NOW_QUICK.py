"""
QUICK TEST - See Results in 5 Minutes
======================================

This will test 3 stocks quickly so you can see how everything works
without waiting for the full overnight training.

Perfect for:
- Testing if everything is working
- Seeing the system in action
- Verifying modules before bed
"""

# ═══════════════════════════════════════════════════════════════════
# QUICK 3-STOCK TEST
# ═══════════════════════════════════════════════════════════════════

from google.colab import drive
import sys, asyncio, json, time
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("🚀 QUICK SYSTEM TEST - 3 Stocks")
print("="*80)

# Mount Drive
drive.mount('/content/drive', force_remount=False)
PROJECT_ROOT = "/content/drive/MyDrive/Quantum_AI_Cockpit"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, f"{PROJECT_ROOT}/backend/modules")

print(f"✅ {PROJECT_ROOT}\n")

# Import modules
print("📦 Loading modules...")
for mod in list(sys.modules.keys()):
    if any(x in mod for x in ['master_analysis', 'data_orchestrator', 'fusior', 'ai_recommender']):
        del sys.modules[mod]

from backend.modules.master_analysis_engine import MasterAnalysisEngine
from backend.modules.data_orchestrator import DataOrchestrator

print("✅ Modules loaded\n")

# Test stocks
TEST_STOCKS = ["AAPL", "NVDA", "TSLA"]

async def quick_validation(symbol: str):
    """Quick validation for one stock."""
    print(f"\n{'='*70}")
    print(f"🔬 Testing {symbol}")
    print(f"{'='*70}")
    
    start = time.time()
    engine = MasterAnalysisEngine()
    data = DataOrchestrator()
    
    try:
        # Get data
        df = await data.fetch_symbol_data(symbol, days=60)
        if df is None or len(df) < 30:
            print(f"   ⚠️  Insufficient data")
            return None
        
        df = df.sort_values('date').reset_index(drop=True)
        print(f"   ✅ Got {len(df)} days of data")
        
        # Test last 10 prediction points
        predictions = []
        for i in range(len(df) - 15, len(df) - 5):
            try:
                current_price = float(df.iloc[i]['close'])
                future_price = float(df.iloc[min(i + 5, len(df) - 1)]['close'])
                actual_return = ((future_price - current_price) / current_price) * 100
                
                result = await engine.analyze_stock(symbol, forecast_days=5, verbose=False)
                rec = result.get("recommendation", {})
                
                action = rec.get("action", "HOLD")
                confidence = rec.get("confidence", 0)
                
                # Check if correct
                correct = False
                if action in ["BUY", "STRONG_BUY", "BUY_THE_DIP"]:
                    correct = actual_return > 0
                elif action == "SELL":
                    correct = actual_return < 0
                elif action == "HOLD":
                    correct = abs(actual_return) < 2.0
                
                pnl = (actual_return / 100) * 10000 if action in ["BUY", "STRONG_BUY", "BUY_THE_DIP"] else 0
                
                predictions.append({
                    "action": action,
                    "confidence": float(confidence),
                    "actual_return": float(actual_return),
                    "correct": correct,
                    "pnl": float(pnl)
                })
            except:
                continue
        
        if not predictions:
            print(f"   ⚠️  No predictions generated")
            return None
        
        # Calculate metrics
        accuracy = (len([p for p in predictions if p['correct']]) / len(predictions)) * 100
        buy_preds = [p for p in predictions if p['action'] in ["BUY", "STRONG_BUY", "BUY_THE_DIP"]]
        buy_accuracy = (len([p for p in buy_preds if p['correct']]) / len(buy_preds)) * 100 if buy_preds else 0
        total_pnl = sum([p['pnl'] for p in predictions])
        
        elapsed = time.time() - start
        
        print(f"\n   📊 Results:")
        print(f"      Predictions: {len(predictions)}")
        print(f"      Overall Accuracy: {accuracy:.1f}%")
        print(f"      BUY Accuracy: {buy_accuracy:.1f}%")
        print(f"      Simulated P&L: ${total_pnl:,.0f}")
        print(f"      BUY signals: {len(buy_preds)}/{len(predictions)}")
        print(f"      Time: {elapsed:.1f}s")
        
        return {
            "symbol": symbol,
            "predictions": len(predictions),
            "accuracy": accuracy,
            "buy_accuracy": buy_accuracy,
            "pnl": total_pnl,
            "buy_count": len(buy_preds),
            "elapsed": elapsed
        }
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

# Run quick test
async def run_quick_test():
    print("\n⏱️  Starting quick validation...\n")
    
    results = []
    for symbol in TEST_STOCKS:
        result = await quick_validation(symbol)
        if result:
            results.append(result)
    
    if not results:
        print("\n❌ No valid results")
        return
    
    # Summary
    print("\n\n" + "="*80)
    print("📊 QUICK TEST SUMMARY")
    print("="*80)
    
    df = pd.DataFrame(results)
    
    print(f"\nTested: {len(df)} stocks")
    print(f"Avg Overall Accuracy: {df['accuracy'].mean():.1f}%")
    print(f"Avg BUY Accuracy: {df['buy_accuracy'].mean():.1f}%")
    print(f"Total Simulated P&L: ${df['pnl'].sum():,.0f}")
    print(f"Avg Time per Stock: {df['elapsed'].mean():.1f}s")
    
    # Quick visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('🎯 Quick Test Results', fontsize=16, fontweight='bold')
    
    # Chart 1: Accuracy
    ax1 = axes[0]
    colors = ['#2ecc71' if x >= 50 else '#e74c3c' for x in df['buy_accuracy']]
    bars = ax1.bar(df['symbol'], df['buy_accuracy'], color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.axhline(50, color='black', linestyle='--', linewidth=2, label='50% Threshold')
    ax1.set_ylabel('BUY Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('📈 BUY Signal Accuracy', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 2,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Chart 2: P&L
    ax2 = axes[1]
    colors2 = ['#2ecc71' if x > 0 else '#e74c3c' for x in df['pnl']]
    bars2 = ax2.bar(df['symbol'], df['pnl'], color=colors2, alpha=0.8, edgecolor='black', linewidth=2)
    ax2.axhline(0, color='black', linestyle='-', linewidth=2)
    ax2.set_ylabel('Simulated P&L ($)', fontsize=12, fontweight='bold')
    ax2.set_title('💰 Profitability', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + (50 if height > 0 else -50),
                f'${height:,.0f}', ha='center', va='bottom' if height > 0 else 'top',
                fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Assessment
    avg_buy_acc = df['buy_accuracy'].mean()
    total_pnl = df['pnl'].sum()
    
    print("\n" + "="*80)
    print("🎯 QUICK ASSESSMENT")
    print("="*80)
    
    if avg_buy_acc >= 60 and total_pnl > 0:
        print("\n✅ LOOKS GOOD!")
        print("   System is performing well on these stocks")
        print("   Ready to run full overnight training")
    elif avg_buy_acc >= 50:
        print("\n⚠️  DECENT PERFORMANCE")
        print("   System shows promise but needs calibration")
        print("   Run overnight training to test more stocks")
    else:
        print("\n❌ NEEDS WORK")
        print("   System accuracy is below target")
        print("   Check module configurations before overnight run")
    
    print("\n💡 Next Steps:")
    if avg_buy_acc >= 50:
        print("   1. Run the full overnight training (20 stocks)")
        print("   2. Let it run while you sleep")
        print("   3. Check visualizations in the morning")
    else:
        print("   1. Review module performance")
        print("   2. Check for errors in logs")
        print("   3. Verify data quality")
    
    print("\n✅ Quick test complete!")

# Run it
await run_quick_test()

# ═══════════════════════════════════════════════════════════════════
# LIVE PREDICTION TEST (Optional)
# ═══════════════════════════════════════════════════════════════════

print("\n\n" + "="*80)
print("🔮 LIVE PREDICTION TEST")
print("="*80)
print("\nGetting current recommendations for the 3 test stocks...\n")

async def get_live_prediction(symbol: str):
    """Get current live recommendation."""
    engine = MasterAnalysisEngine()
    
    try:
        result = await engine.analyze_stock(symbol, forecast_days=5, verbose=False)
        rec = result.get("recommendation", {})
        
        print(f"\n{'='*60}")
        print(f"📊 {symbol}")
        print(f"{'='*60}")
        print(f"💰 Price:      ${result.get('current_price', 0):.2f}")
        print(f"📈 Action:     {rec.get('action', 'UNKNOWN')}")
        print(f"🎯 Confidence: {rec.get('confidence', 0)*100:.0f}%")
        print(f"📝 Rationale:  {rec.get('rationale', 'No rationale')[:100]}")
        
        if rec.get('confidence', 0) >= 0.75:
            print(f"\n✅ HIGH CONFIDENCE - Consider this trade")
        elif rec.get('confidence', 0) >= 0.60:
            print(f"\n⚠️  MEDIUM CONFIDENCE - Use caution")
        else:
            print(f"\n❌ LOW CONFIDENCE - Skip this trade")
    
    except Exception as e:
        print(f"\n❌ {symbol}: Error - {e}")

for symbol in TEST_STOCKS:
    await get_live_prediction(symbol)

print("\n\n" + "="*80)
print("✅ ALL TESTS COMPLETE!")
print("="*80)
print("\n💤 If results look good, run the overnight training and go to sleep!")
print("☀️  Wake up to full validation across all 20 stocks + beautiful visualizations\n")

