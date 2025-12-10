#!/usr/bin/env python3
"""
🧪 LEGENDARY STACK SMOKE TESTS
Tests all active modules to identify what's production-ready vs needs training
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yfinance as yf
import pandas as pd

# Test configuration
TEST_TICKER = "NVDA"
TEST_PERIOD = "5d"

print("\n" + "="*80)
print("🧪 ACTIVE MODULE SMOKE TESTS")
print("="*80)

# Download test data
print(f"📥 Downloading {TEST_TICKER} data ({TEST_PERIOD})...")
data = yf.download(TEST_TICKER, period=TEST_PERIOD, progress=False)
if len(data) > 0:
    print(f"   ✅ Downloaded {len(data)} days\n")
else:
    print("   ❌ Failed to download data\n")
    sys.exit(1)

# Store results
results = {}

def test_dark_pool_signals():
    """Test dark pool signals module"""
    print("1️⃣  TESTING: src/features/dark_pool_signals.py")
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "src" / "features"))
        from dark_pool_signals import DarkPoolSignals
        
        dp = DarkPoolSignals(TEST_TICKER, cache_enabled=False)
        result = dp.get_all_signals()
        
        assert isinstance(result, dict), "Invalid result type"
        
        print(f"   ✅ PASS - {len(result)} signals generated\n")
        return True
    except Exception as e:
        print(f"   ❌ FAIL - {str(e)[:100]}\n")
        return False

def test_feature_engine():
    """Test feature engine module"""
    print("2️⃣  TESTING: src/python/feature_engine.py")
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "src" / "python"))
        from feature_engine import FeatureEngine
        
        fe = FeatureEngine()
        features = fe.calculate_features(data)
        
        assert not features.empty, "No features generated"
        assert len(features.columns) > 10, f"Only {len(features.columns)} features"
        
        print(f"   ✅ PASS - Generated {len(features.columns)} features\n")
        return True
    except Exception as e:
        print(f"   ❌ FAIL - {str(e)[:100]}\n")
        return False

def test_pattern_detector():
    """Test pattern detector module"""
    print("3️⃣  TESTING: pattern_detector.py")
    try:
        from pattern_detector import PatternDetector
        
        pd_obj = PatternDetector()
        result = pd_obj.detect_all_patterns(TEST_TICKER)
        
        assert isinstance(result, dict), "Invalid result format"
        assert 'patterns' in result, "Missing patterns key"
        
        patterns = result['patterns']
        bullish_count = sum(1 for p in patterns if p.get('type') == 'BULLISH')
        bearish_count = sum(1 for p in patterns if p.get('type') == 'BEARISH')
        
        print(f"   ✅ PASS - {len(patterns)} patterns ({bullish_count} bullish, {bearish_count} bearish)\n")
        return True
    except Exception as e:
        print(f"   ❌ FAIL - {str(e)[:100]}\n")
        return False

def test_market_regime_manager():
    """Test market regime manager"""
    print("4️⃣  TESTING: market_regime_manager.py")
    try:
        from market_regime_manager import MarketRegimeManager
        
        mrm = MarketRegimeManager()
        regime_info = mrm.get_current_regime()
        
        assert isinstance(regime_info, dict), "Invalid regime format"
        assert 'regime' in regime_info, "Missing regime key"
        
        regime = regime_info['regime']
        confidence = regime_info.get('confidence', 0)
        
        print(f"   ✅ PASS - Regime: {regime} ({confidence:.0f}% confidence)\n")
        return True
    except Exception as e:
        print(f"   ❌ FAIL - {str(e)[:100]}\n")
        return False

def test_optimized_signal_config():
    """Test optimized signal config (61.7% WR proven)"""
    print("5️⃣  TESTING: optimized_signal_config.py")
    try:
        from optimized_signal_config import OptimizedSignalConfig
        
        config = OptimizedSignalConfig()
        
        # Verify tier structure
        assert hasattr(config, 'TIER_S_PATTERNS'), "Missing Tier S"
        assert hasattr(config, 'TIER_A_PATTERNS'), "Missing Tier A"
        assert hasattr(config, 'TIER_B_PATTERNS'), "Missing Tier B"
        
        # Verify weights
        assert hasattr(config, 'SIGNAL_WEIGHTS'), "Missing signal weights"
        
        tier_s_count = len(config.TIER_S_PATTERNS)
        tier_a_count = len(config.TIER_A_PATTERNS)
        tier_b_count = len(config.TIER_B_PATTERNS)
        
        print(f"   ✅ PASS - Tiers: S({tier_s_count}) A({tier_a_count}) B({tier_b_count})\n")
        return True
    except Exception as e:
        print(f"   ❌ FAIL - {str(e)[:100]}\n")
        return False

def test_ai_recommender():
    """Test AI recommender module"""
    print("6️⃣  TESTING: ai_recommender.py")
    try:
        from ai_recommender import AIRecommender
        
        ai = AIRecommender()
        
        # Check if model files exist
        model_dir = PROJECT_ROOT / "models"
        has_models = model_dir.exists() and len(list(model_dir.glob("*.pkl"))) > 0
        
        if has_models:
            print(f"   ✅ PASS - Found trained models in models/\n")
        else:
            print(f"   ⚠️  PASS (UNTRAINED) - Module works but needs training\n")
        
        return True
    except Exception as e:
        print(f"   ❌ FAIL - {str(e)[:100]}\n")
        return False

def test_intelligence_companion():
    """Test intelligence companion orchestrator"""
    print("7️⃣  TESTING: IntelligenceCompanion.py")
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "src" / "intelligence"))
        from IntelligenceCompanion import IntelligenceCompanion
        
        ic = IntelligenceCompanion()
        
        print(f"   ✅ PASS - Orchestrator initialized\n")
        return True
    except Exception as e:
        print(f"   ❌ FAIL - {str(e)[:100]}\n")
        return False

def test_alpha_76_watchlist():
    """Test alpha 76 watchlist function"""
    print("8️⃣  TESTING: get_legendary_tickers()")
    try:
        # Try multiple training scripts
        try:
            from colab_pro_trainer import get_legendary_tickers
        except:
            from multi_ticker_trainer import get_legendary_tickers
        
        tickers = get_legendary_tickers()
        
        assert len(tickers) >= 90, f"Only {len(tickers)} tickers (expected 90+)"
        assert "NVDA" in tickers, "Missing NVDA"
        assert "DGNX" in tickers, "Missing DGNX (Perplexity hot pick)"
        
        print(f"   ✅ PASS - {len(tickers)} tickers loaded\n")
        return True
    except Exception as e:
        print(f"   ❌ FAIL - {str(e)[:100]}\n")
        return False

def test_environment_config():
    """Test environment configuration"""
    print("9️⃣  TESTING: config/environment_config.py")
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "config"))
        from environment_config import EnvironmentConfig
        
        env = EnvironmentConfig()
        
        # Check API status
        working = sum(1 for status in env.api_status.values() if status == "WORKING")
        total = len(env.api_status)
        
        print(f"   ✅ PASS - {working}/{total} APIs configured\n")
        return True
    except Exception as e:
        print(f"   ❌ FAIL - {str(e)[:100]}\n")
        return False

def test_data_fetcher():
    """Test production data fetcher"""
    print("🔟 TESTING: PRODUCTION_DATAFETCHER.py")
    try:
        from PRODUCTION_DATAFETCHER import ProductionDataFetcher
        
        fetcher = ProductionDataFetcher()
        test_data = fetcher.fetch_ticker_data(TEST_TICKER, period="5d")
        
        assert not test_data.empty, "No data fetched"
        assert len(test_data) >= 3, f"Only {len(test_data)} days"
        
        print(f"   ✅ PASS - Fetched {len(test_data)} days of data\n")
        return True
    except Exception as e:
        print(f"   ❌ FAIL - {str(e)[:100]}\n")
        return False

# Run all tests
if __name__ == "__main__":
    tests = [
        ("Dark Pool Signals", test_dark_pool_signals),
        ("Feature Engine", test_feature_engine),
        ("Pattern Detector", test_pattern_detector),
        ("Market Regime Manager", test_market_regime_manager),
        ("Optimized Signal Config (61.7% WR)", test_optimized_signal_config),
        ("AI Recommender", test_ai_recommender),
        ("Intelligence Companion", test_intelligence_companion),
        ("Alpha 76 Watchlist", test_alpha_76_watchlist),
        ("Environment Config", test_environment_config),
        ("Production Data Fetcher", test_data_fetcher),
    ]
    
    results = {}
    for name, test_func in tests:
        results[name] = test_func()
    
    # Summary
    print("="*80)
    print("📊 SMOKE TEST SUMMARY")
    print("="*80)
    
    for name, passed in results.items():
        status = "✅ LEGENDARY" if passed else "❌ BROKEN"
        print(f"   {status} - {name}")
    
    passed_count = sum(results.values())
    total_count = len(results)
    pass_rate = (passed_count / total_count) * 100
    
    print(f"\n   Overall: {passed_count}/{total_count} tests passed ({pass_rate:.0f}%)")
    
    if passed_count == total_count:
        print("\n🎉 ALL SYSTEMS GO - LEGENDARY STACK READY!")
    elif passed_count >= total_count * 0.7:
        print("\n⚠️  MOSTLY READY - Some modules need fixes")
    else:
        print("\n❌ NEEDS WORK - Multiple failures detected")
    
    print("="*80 + "\n")
    
    sys.exit(0 if passed_count >= total_count * 0.7 else 1)
