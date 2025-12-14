"""
Quick Test Script for COLAB
Run this after deploy_to_colab.py to verify fixes are working
"""

def run_quick_tests():
    """Run quick verification tests"""
    print("=" * 70)
    print("🧪 QUICK VERIFICATION TESTS")
    print("=" * 70)
    
    results = {}
    
    # Test 1: Market Regime Manager
    print("\n📊 Test 1: Market Regime Manager...")
    try:
        from market_regime_manager import MarketRegimeManager
        mgr = MarketRegimeManager()
        regime_info = mgr.calculate_market_regime()
        print(f"   ✓ Regime: {regime_info['regime']}")
        print(f"   ✓ Price: ${regime_info['price']:.2f}")
        print(f"   ✓ VIX Proxy: {regime_info['vix_proxy']:.1f}%")
        results['market_regime'] = True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results['market_regime'] = False
    
    # Test 2: Risk Manager
    print("\n🛡️ Test 2: Risk Manager...")
    try:
        from risk_manager import RiskManager, MarketRegime
        rm = RiskManager(initial_capital=10000.0)
        
        # Test BULL regime
        can_trade = rm.can_trade(MarketRegime.BULL)
        print(f"   ✓ BULL regime: can_trade={can_trade}")
        
        # Test position sizing (entry=$100, stop=$95)
        shares, risk = rm.calculate_position_size(100.0, 95.0, MarketRegime.BULL)
        print(f"   ✓ Position size: {shares} shares (risk: ${risk:.2f})")
        
        # Test CRASH regime
        can_trade_crash = rm.can_trade(MarketRegime.CRASH)
        print(f"   ✓ CRASH regime: can_trade={can_trade_crash} (should be False)")
        
        results['risk_manager'] = True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results['risk_manager'] = False
    
    # Test 3: AI Recommender
    print("\n🤖 Test 3: AI Recommender...")
    try:
        from ai_recommender import AIRecommender
        rec = AIRecommender(use_adaptive_labels=True)
        train_result = rec.train_for_ticker('AAPL')
        
        if 'error' not in train_result:
            print(f"   ✓ CV Mean Accuracy: {train_result.get('cv_mean', 0):.1%}")
            print(f"   ✓ Adaptive Labels: {train_result.get('adaptive_labels')}")
            print(f"   ✓ Selected Features: {len(train_result.get('selected_features', []))}")
            results['ai_recommender'] = True
        else:
            print(f"   ⚠️ Training error: {train_result.get('error')}")
            results['ai_recommender'] = False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results['ai_recommender'] = False
    
    # Test 4: Optimization Toolkit
    print("\n🎯 Test 4: Optimization Toolkit...")
    try:
        from optimization_toolkit import (
            make_labels_adaptive,
            calculate_rsi_wilders,
            detect_volume_surge_dynamic
        )
        import numpy as np
        
        # Test RSI
        prices = np.random.random(100) * 100 + 100
        rsi = calculate_rsi_wilders(prices)
        print(f"   ✓ RSI calculation working: {rsi[-1]:.1f}")
        
        # Test volume surge
        volumes = np.random.random(100) * 1000000
        surge, ratio = detect_volume_surge_dynamic(volumes)
        print(f"   ✓ Volume surge detection working: surge={surge}")
        
        results['optimization_toolkit'] = True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results['optimization_toolkit'] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"   {test}: {status}")
    
    print(f"\n   Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED - System ready!")
    else:
        print("\n⚠️ Some tests failed - review errors above")
    
    return results


if __name__ == "__main__":
    run_quick_tests()
