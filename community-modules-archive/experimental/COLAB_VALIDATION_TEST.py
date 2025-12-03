# ═══════════════════════════════════════════════════════════════════════════════
# 🧪 COMPREHENSIVE VALIDATION TEST
# ═══════════════════════════════════════════════════════════════════════════════
"""
Test all fixes with 3 real stocks to ensure everything works before overnight training.
"""

import asyncio
from datetime import datetime
from master_analysis_engine import MasterAnalysisEngine

print("=" * 80)
print("🧪 VALIDATION TEST - 3 STOCKS")
print("=" * 80)
print(f"⏰ Started: {datetime.now().strftime('%I:%M %p')}")
print("=" * 80)
print()

# Initialize engine
engine = MasterAnalysisEngine()

# Test stocks
test_stocks = ["AAPL", "NVDA", "TSLA"]
results = {}
success_count = 0

for i, ticker in enumerate(test_stocks, 1):
    print(f"\n[{i}/3] Testing {ticker}...")
    print("-" * 80)
    
    try:
        result = await engine.analyze_stock(
            symbol=ticker,
            account_balance=500,
            forecast_days=14,
            verbose=False
        )
        
        if result and result.get('status') == 'ok':
            # Extract key metrics
            rec = result.get('recommendation', {})
            action = rec.get('action', 'HOLD')
            confidence = rec.get('confidence', 0)
            price = result.get('current_price', 0)
            
            # Check if price is extracted correctly (should NOT be $0.00)
            if price > 0:
                print(f"   ✅ {ticker}: {action} ({confidence:.0f}% confidence)")
                print(f"   💰 Current Price: ${price:.2f}")
                
                # Check forecast
                forecast = result.get('forecast', {})
                trend = forecast.get('trend', 'UNKNOWN')
                print(f"   📈 Forecast Trend: {trend}")
                
                # Check patterns
                patterns = result.get('patterns', {})
                if patterns and isinstance(patterns, dict):
                    summary = patterns.get('summary', {})
                    detected = summary.get('total_patterns_detected', 0)
                    print(f"   🎯 Patterns Detected: {detected}")
                else:
                    print(f"   ⚠️  Patterns: Not analyzed (error or insufficient data)")
                
                results[ticker] = {
                    "action": action,
                    "confidence": confidence,
                    "price": price,
                    "trend": trend,
                    "status": "SUCCESS"
                }
                success_count += 1
                print(f"   ✅ PASSED")
            else:
                print(f"   ❌ FAILED: Price is $0.00 (extraction issue)")
                results[ticker] = {"status": "PRICE_ERROR"}
        else:
            error = result.get('error', 'Unknown error') if result else 'No result'
            print(f"   ❌ FAILED: {error[:60]}")
            results[ticker] = {"status": "ERROR", "error": str(error)[:60]}
            
    except Exception as e:
        print(f"   ❌ EXCEPTION: {str(e)[:60]}")
        results[ticker] = {"status": "EXCEPTION", "error": str(e)[:60]}
        import traceback
        print(f"\n   Traceback (last 3 lines):")
        lines = traceback.format_exc().split('\n')
        for line in lines[-4:-1]:
            print(f"   {line}")

print("\n" + "=" * 80)
print("📊 TEST RESULTS")
print("=" * 80)
print()
print(f"✅ Success: {success_count}/3 ({success_count/3*100:.0f}%)")
print()

if success_count == 3:
    print("🎉 ALL TESTS PASSED!")
    print("=" * 80)
    print("✅ System is READY for overnight training!")
    print()
    print("💡 NEXT STEPS:")
    print("   1. Run the FINAL TRAINING cell below")
    print("   2. Go to sleep - it will train for 2-4 hours")
    print("   3. Wake up to fully trained models + recommendations!")
    print("=" * 80)
elif success_count >= 2:
    print("⚠️  MOSTLY WORKING")
    print("=" * 80)
    print(f"✅ {success_count}/3 stocks analyzed successfully")
    print("⚠️  1 stock failed - this might be OK (data issue)")
    print()
    print("💡 You can proceed with training, but check errors above")
    print("=" * 80)
else:
    print("❌ TOO MANY FAILURES")
    print("=" * 80)
    print(f"Only {success_count}/3 stocks worked")
    print()
    print("🔧 DEBUGGING INFO:")
    for ticker, result in results.items():
        status = result.get('status')
        if status != 'SUCCESS':
            error = result.get('error', 'Unknown')
            print(f"   • {ticker}: {status} - {error}")
    print()
    print("💡 Check the errors above and re-run the comprehensive fix")
    print("=" * 80)

print()
print(f"⏰ Finished: {datetime.now().strftime('%I:%M %p')}")
print("=" * 80)

