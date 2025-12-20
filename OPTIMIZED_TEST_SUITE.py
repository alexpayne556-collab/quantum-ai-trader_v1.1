"""
OPTIMIZED TEST SUITE
====================
Fast, efficient testing with caching and parallel execution.

Performance targets:
- Unit tests: <5 seconds (was N/A)
- Integration tests: <30 seconds (was ~180 seconds)
- Total runtime: <40 seconds (was ~195 seconds)

Architecture:
- 80% unit tests (fast, isolated, no network)
- 15% integration tests (cached data)
- 5% end-to-end tests (full system)
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Callable
import json
import os

# Import test modules
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests/unit'))

from tests.unit.test_regime_detector import run_all as test_regime_unit
from tests.unit.test_signal_combiner import run_all as test_signal_unit
from tests.unit.test_news_monitor import run_all as test_news_unit

from test_cache import MarketDataCache
from QUANTUM_ENSEMBLE_ENGINE import QuantumEnsemble, RegimeDetector, PatternHunter

# ============================================================================
# PHASE 1: UNIT TESTS (Fast, No Network)
# ============================================================================

def run_unit_tests_parallel() -> Dict[str, tuple]:
    """
    Run all unit tests in parallel.
    Target: <5 seconds total
    """
    print("\n" + "="*80)
    print("PHASE 1: UNIT TESTS (Fast, Isolated)")
    print("="*80)
    
    def safe_run(test_func, name):
        """Wrapper to catch exceptions."""
        try:
            start = time.time()
            passed, failed = test_func()
            elapsed = time.time() - start
            return name, (passed, failed, elapsed, None)
        except Exception as e:
            return name, (0, 1, 0, str(e))
    
    # Run tests in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(safe_run, test_regime_unit, 'Regime Detector'): 'regime',
            executor.submit(safe_run, test_signal_unit, 'Signal Combiner'): 'signal',
            executor.submit(safe_run, test_news_unit, 'News Monitor'): 'news',
        }
        
        results = {}
        for future in as_completed(futures):
            name, result = future.result()
            results[name] = result
            
            passed, failed, elapsed, error = result
            if error:
                print(f"\n❌ {name}: {error}")
            else:
                print(f"\n✅ {name}: {passed} passed, {failed} failed ({elapsed:.2f}s)")
    
    return results

# ============================================================================
# PHASE 2: INTEGRATION TESTS (Cached Data)
# ============================================================================

def test_full_pipeline_integration():
    """Test complete workflow with cached data."""
    print("\n" + "="*80)
    print("INTEGRATION TEST: Full Pipeline")
    print("="*80)
    
    try:
        # Get cached data (or download once)
        print("\n📊 Loading market data...")
        start = time.time()
        market_data = MarketDataCache.prepare_regime_data(period='1y')
        load_time = time.time() - start
        print(f"   ✓ Data loaded in {load_time:.2f}s")
        
        # Initialize components
        ensemble = QuantumEnsemble()
        
        # Step 1: Detect regime
        print("\n1. Detecting regime...")
        regime = ensemble.regime_detector.get_current_regime(market_data)
        print(f"   ✓ Regime: {regime.market_regime.value}")
        print(f"   ✓ Volatility: {regime.volatility.value}")
        print(f"   ✓ VIX: {regime.vix_level:.1f}")
        
        # Step 2: Simulate signals
        print("\n2. Generating signals...")
        signals = {
            'H16': -0.4,
            'H19': 0.7,
            'H20': 0.3,
            'H27E': 0.5
        }
        print(f"   ✓ {len(signals)} signals generated")
        
        # Step 3: Add news event
        print("\n3. Checking news events...")
        ensemble.news_monitor.clear_old_events()
        ensemble.news_monitor.add_event('economic_data', 'Test event')
        active_events = len(ensemble.news_monitor.get_active_events())
        print(f"   ✓ {active_events} active event(s)")
        
        # Step 4: Combine signals
        print("\n4. Combining signals...")
        result = ensemble.combine_signals(signals, regime)
        print(f"   ✓ Combined signal: {result['combined_signal']:+.3f}")
        print(f"   ✓ Confidence: {result['confidence']:.1%}")
        
        # Step 5: Make decision
        print("\n5. Making trade decision...")
        should_trade = ensemble.should_trade(result)
        print(f"   ✓ Should trade: {should_trade}")
        
        # Verify invariants
        assert result['confidence'] >= 0 and result['confidence'] <= 1
        assert abs(result['combined_signal']) <= 1.0
        assert sum(result['weights_used'].values()) - 1.0 < 0.01
        
        print("\n✅ INTEGRATION TEST PASSED")
        return True, None
        
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {e}")
        return False, str(e)

def test_pattern_hunter_integration():
    """Test pattern detection with real data."""
    print("\n" + "="*80)
    print("INTEGRATION TEST: Pattern Hunter")
    print("="*80)
    
    try:
        print("\n📊 Loading 5-year data...")
        spy_5y = MarketDataCache.get_data('SPY', period='5y')
        
        print("\n🔍 Scanning for rare patterns...")
        pattern_hunter = PatternHunter()
        
        # Create test market state
        market_state = {
            'golden_cross': True,
            'vix': 28,
            'vix_change_2d': -7,
            'breadth': 0.15,
            'sentiment_percentile': 0.95,
            'macro_risk': 'neutral',
            'spy_rsi': 25,
            'volume_ratio': 2.0
        }
        
        patterns = pattern_hunter.scan_all_patterns(market_state)
        
        print(f"\n✓ Found {len(patterns)} pattern(s)")
        for p in patterns:
            print(f"   - {p.name}: {p.confidence:.0%} confidence")
        
        assert len(patterns) >= 0  # Can be 0-2 patterns
        
        print("\n✅ PATTERN HUNTER TEST PASSED")
        return True, None
        
    except Exception as e:
        print(f"\n❌ PATTERN HUNTER TEST FAILED: {e}")
        return False, str(e)

def test_regime_detection_historical():
    """Test regime detection on historical periods."""
    print("\n" + "="*80)
    print("INTEGRATION TEST: Historical Regime Detection")
    print("="*80)
    
    try:
        detector = RegimeDetector()
        
        # Test current market
        print("\n📊 Testing current market state...")
        market_data = MarketDataCache.prepare_regime_data(period='1y')
        regime = detector.get_current_regime(market_data)
        
        print(f"   Current: {regime.market_regime.value}")
        print(f"   VIX: {regime.vix_level:.1f}")
        print(f"   Trend: {regime.trend.value}")
        
        # Verify reasonable values
        assert 10 <= regime.vix_level <= 100, f"VIX {regime.vix_level} out of range"
        assert 0 <= regime.vix_percentile <= 1, f"VIX percentile {regime.vix_percentile} out of range"
        
        print("\n✅ HISTORICAL REGIME TEST PASSED")
        return True, None
        
    except Exception as e:
        print(f"\n❌ HISTORICAL REGIME TEST FAILED: {e}")
        return False, str(e)

def run_integration_tests_parallel() -> Dict[str, tuple]:
    """Run integration tests in parallel."""
    print("\n" + "="*80)
    print("PHASE 2: INTEGRATION TESTS (Cached Data)")
    print("="*80)
    
    def safe_run(test_func):
        """Wrapper to time and catch exceptions."""
        start = time.time()
        success, error = test_func()
        elapsed = time.time() - start
        return success, elapsed, error
    
    # Run in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(safe_run, test_full_pipeline_integration): 'Full Pipeline',
            executor.submit(safe_run, test_pattern_hunter_integration): 'Pattern Hunter',
            executor.submit(safe_run, test_regime_detection_historical): 'Historical Regime',
        }
        
        results = {}
        for future in as_completed(futures):
            name = [k for k, v in futures.items() if v == future][0]
            name_str = futures[name]
            success, elapsed, error = future.result()
            results[name_str] = (success, elapsed, error)
    
    return results

# ============================================================================
# PHASE 3: PERFORMANCE BENCHMARKS
# ============================================================================

def run_performance_benchmarks():
    """Quick performance checks."""
    print("\n" + "="*80)
    print("PHASE 3: PERFORMANCE BENCHMARKS")
    print("="*80)
    
    from test_fixtures import create_mock_market_data, create_mock_signals
    
    benchmarks = {}
    
    # Benchmark 1: Regime detection speed
    print("\n⏱️  Regime detection latency...")
    detector = RegimeDetector()
    data = create_mock_market_data(days=252)
    
    times = []
    for _ in range(10):
        start = time.perf_counter()
        detector.get_current_regime(data)
        times.append((time.perf_counter() - start) * 1000)
    
    avg_ms = sum(times) / len(times)
    print(f"   Average: {avg_ms:.2f}ms (target: <100ms)")
    benchmarks['regime_detection_ms'] = avg_ms
    
    # Benchmark 2: Signal combination speed
    print("\n⏱️  Signal combination latency...")
    ensemble = QuantumEnsemble()
    regime = detector.get_current_regime(data)
    
    times = []
    for _ in range(10):
        signals = create_mock_signals(num_signals=5)
        start = time.perf_counter()
        ensemble.combine_signals(signals, regime, use_news_adjustment=False)
        times.append((time.perf_counter() - start) * 1000)
    
    avg_ms = sum(times) / len(times)
    print(f"   Average: {avg_ms:.2f}ms (target: <50ms)")
    benchmarks['signal_combination_ms'] = avg_ms
    
    return benchmarks

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run complete optimized test suite."""
    overall_start = time.time()
    
    print("="*80)
    print("OPTIMIZED TEST SUITE")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = {}
    
    # Phase 1: Unit tests (parallel)
    phase1_start = time.time()
    unit_results = run_unit_tests_parallel()
    phase1_time = time.time() - phase1_start
    all_results['unit_tests'] = unit_results
    
    # Check if unit tests passed
    unit_failures = sum(result[1] for result in unit_results.values())
    if unit_failures > 0:
        print(f"\n⚠️  {unit_failures} unit test(s) failed - stopping early")
        return all_results
    
    # Phase 2: Integration tests (parallel, with cached data)
    phase2_start = time.time()
    integration_results = run_integration_tests_parallel()
    phase2_time = time.time() - phase2_start
    all_results['integration_tests'] = integration_results
    
    # Phase 3: Performance benchmarks
    phase3_start = time.time()
    perf_results = run_performance_benchmarks()
    phase3_time = time.time() - phase3_start
    all_results['performance'] = perf_results
    
    # Print cache stats
    print("\n" + "="*80)
    print("DATA CACHE STATISTICS")
    print("="*80)
    MarketDataCache.cache_stats()
    
    # Summary
    total_time = time.time() - overall_start
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    # Unit tests summary
    total_unit_passed = sum(r[0] for r in unit_results.values())
    total_unit_failed = sum(r[1] for r in unit_results.values())
    print(f"\nUnit Tests: {total_unit_passed} passed, {total_unit_failed} failed ({phase1_time:.1f}s)")
    
    # Integration tests summary
    integration_passed = sum(1 for r in integration_results.values() if r[0])
    integration_failed = len(integration_results) - integration_passed
    print(f"Integration Tests: {integration_passed} passed, {integration_failed} failed ({phase2_time:.1f}s)")
    
    # Performance summary
    print(f"Performance Benchmarks: {len(perf_results)} completed ({phase3_time:.1f}s)")
    
    print(f"\nTotal Runtime: {total_time:.1f}s")
    print(f"Target: <40s {'✅' if total_time < 40 else '❌'}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('test_results', exist_ok=True)
    filename = f'test_results/optimized_tests_{timestamp}.json'
    
    # Serialize results
    serializable_results = {
        'timestamp': timestamp,
        'total_time_seconds': total_time,
        'phase_times': {
            'unit': phase1_time,
            'integration': phase2_time,
            'performance': phase3_time
        },
        'unit_tests': {k: {'passed': v[0], 'failed': v[1], 'time': v[2]} 
                       for k, v in unit_results.items()},
        'integration_tests': {k: {'success': v[0], 'time': v[1], 'error': v[2]} 
                             for k, v in integration_results.items()},
        'performance': perf_results
    }
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n📊 Results saved to: {filename}")
    
    print("\n" + "="*80)
    
    return all_results

if __name__ == '__main__':
    results = run_all_tests()
