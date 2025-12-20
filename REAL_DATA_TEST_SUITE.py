
"""
REAL DATA TEST SUITE
====================
NO MOCKS - All tests use REAL market data.
Fast because we cache, not because we fake.
"""

import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os

from QUANTUM_ENSEMBLE_ENGINE import (
    QuantumEnsemble, RegimeDetector, NewsQuantum, 
    PatternHunter, CorrelationTracker, MarketRegime,
    VolatilityRegime, TrendRegime
)

# ============================================================================
# REAL DATA CACHE - Download once, test many times
# ============================================================================

class RealDataCache:
    """Cache REAL market data. No mocks, no fakes."""
    
    _data = {}
    
    @classmethod
    def get(cls, symbol, period='1y'):
        """Get real data with caching."""
        key = f"{symbol}_{period}"
        
        if key not in cls._data:
            print(f"📥 Downloading REAL {symbol} ({period})...")
            df = yf.download(symbol, period=period, progress=False, auto_adjust=True)
            
            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0].lower() for col in df.columns]
            else:
                df.columns = [col.lower() for col in df.columns]
            
            cls._data[key] = df
        
        return cls._data[key].copy()
    
    @classmethod
    def get_spy_vix(cls, period='1y'):
        """Get SPY and VIX with aligned indices."""
        spy = cls.get('SPY', period)
        vix = cls.get('^VIX', period)
        
        # Align
        common = spy.index.intersection(vix.index)
        
        return pd.DataFrame({
            'close': spy.loc[common, 'close'],
            'high': spy.loc[common, 'high'],
            'low': spy.loc[common, 'low'],
            'vix': vix.loc[common, 'close']
        })
    
    @classmethod
    def prefetch(cls):
        """Prefetch all data sequentially (parallel yfinance has bugs)."""
        symbols = [
            ('SPY', '1y'), ('SPY', '5y'),
            ('^VIX', '1y'), ('^VIX', '5y'),
            ('QQQ', '1y'), ('IWM', '1y')
        ]
        
        # Sequential prefetch - yfinance has thread safety issues
        for sym, per in symbols:
            cls.get(sym, per)
        
        print("✅ All data prefetched")

# ============================================================================
# REAL TESTS
# ============================================================================

def test_regime_detection_real():
    """Test regime detection with REAL market data."""
    print("\n" + "="*70)
    print("TEST: Regime Detection (REAL DATA)")
    print("="*70)
    
    detector = RegimeDetector()
    data = RealDataCache.get_spy_vix('1y')
    
    print(f"\n📊 Testing with {len(data)} days of REAL SPY/VIX data")
    
    # Detect current regime
    regime = detector.get_current_regime(data)
    
    print(f"\n   Market Regime: {regime.market_regime.value}")
    print(f"   Volatility: {regime.volatility.value}")
    print(f"   Trend: {regime.trend.value}")
    print(f"   VIX: {regime.vix_level:.1f}")
    print(f"   VIX Percentile: {regime.vix_percentile:.1%}")
    
    # Validate
    assert regime.vix_level > 0, "VIX should be positive"
    assert 0 <= regime.vix_percentile <= 1, "VIX percentile out of range"
    assert regime.market_regime in MarketRegime, "Invalid regime"
    
    print("\n✅ PASSED")
    return True

def test_regime_historical_real():
    """Test regime detection on multiple historical points."""
    print("\n" + "="*70)
    print("TEST: Historical Regime Detection (REAL DATA)")
    print("="*70)
    
    detector = RegimeDetector()
    
    # Use the 5y cache - ensure we get the right one
    data = RealDataCache.get_spy_vix('5y')
    
    print(f"\n📊 Testing regime detection over 5 years ({len(data)} days)")
    
    # Handle case where we somehow got less data
    if len(data) < 500:
        print("⚠️ Less than 500 days - redownloading...")
        spy = yf.download('SPY', period='5y', progress=False, auto_adjust=True)
        vix = yf.download('^VIX', period='5y', progress=False, auto_adjust=True)
        
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = [col[0].lower() for col in spy.columns]
            vix.columns = [col[0].lower() for col in vix.columns]
        else:
            spy.columns = [col.lower() for col in spy.columns]
            vix.columns = [col.lower() for col in vix.columns]
        
        common = spy.index.intersection(vix.index)
        data = pd.DataFrame({
            'close': spy.loc[common, 'close'],
            'high': spy.loc[common, 'high'],
            'low': spy.loc[common, 'low'],
            'vix': vix.loc[common, 'close']
        })
        print(f"   Got {len(data)} days")
    
    regimes_found = {r.value: 0 for r in MarketRegime}
    
    # Sample every 20 days, starting after enough history
    for i in range(252, len(data), 20):
        subset = data.iloc[:i]
        regime = detector.get_current_regime(subset)
        regimes_found[regime.market_regime.value] += 1
    
    print(f"\n   Regimes detected:")
    for regime, count in regimes_found.items():
        print(f"      {regime}: {count} occurrences")
    
    # Should find multiple regimes
    active_regimes = sum(1 for c in regimes_found.values() if c > 0)
    assert active_regimes >= 2, f"Only found {active_regimes} regime types"
    
    print("\n✅ PASSED")
    return True

def test_signal_combination_real():
    """Test signal combination with REAL regime detection."""
    print("\n" + "="*70)
    print("TEST: Signal Combination (REAL DATA)")
    print("="*70)
    
    ensemble = QuantumEnsemble()
    data = RealDataCache.get_spy_vix('1y')
    
    # Get REAL regime
    regime = ensemble.regime_detector.get_current_regime(data)
    print(f"\n📊 Current regime: {regime.market_regime.value}")
    
    # Test signal combinations
    test_cases = [
        {'H16': 0.8, 'H20': 0.7, 'H19': 0.6},  # All bullish
        {'H16': -0.5, 'H20': 0.3, 'H19': -0.4},  # Mixed
        {'H16': -0.8, 'H20': -0.6, 'H27E': -0.7},  # All bearish
    ]
    
    for i, signals in enumerate(test_cases):
        result = ensemble.combine_signals(signals, regime, use_news_adjustment=False)
        
        print(f"\n   Case {i+1}: {signals}")
        print(f"   Combined: {result['combined_signal']:+.3f}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   Weights: {result['weights_used']}")
        
        # Validate
        assert -1 <= result['combined_signal'] <= 1, "Signal out of bounds"
        assert 0 <= result['confidence'] <= 1, "Confidence out of bounds"
        assert abs(sum(result['weights_used'].values()) - 1.0) < 0.01, "Weights don't sum to 1"
    
    print("\n✅ PASSED")
    return True

def test_news_impact_real():
    """Test news monitoring with real confidence reduction."""
    print("\n" + "="*70)
    print("TEST: News Event Impact (REAL LOGIC)")
    print("="*70)
    
    monitor = NewsQuantum()
    
    events = [
        ('fomc', 'Fed rate decision'),
        ('earnings', 'AAPL earnings'),
        ('geopolitical', 'War escalation'),
        ('black_swan', 'Bank collapse'),
    ]
    
    base_confidence = 0.85
    print(f"\n📊 Base confidence: {base_confidence:.1%}")
    
    for event_type, desc in events:
        monitor.clear_old_events()
        monitor.add_event(event_type, desc)
        
        adjusted, reason = monitor.adjust_signal_confidence(base_confidence, 'test')
        reduction = (1 - adjusted/base_confidence) * 100
        
        print(f"\n   {event_type.upper()}: {adjusted:.1%} ({reduction:+.0f}% reduction)")
        
        # Validate events reduce confidence
        assert adjusted <= base_confidence, f"{event_type} should reduce confidence"
    
    print("\n✅ PASSED")
    return True

def test_correlation_adjustment_real():
    """Test correlation adjustment with real signal profiles."""
    print("\n" + "="*70)
    print("TEST: Correlation Adjustment (REAL LOGIC)")
    print("="*70)
    
    tracker = CorrelationTracker()
    
    # VIX signals are highly correlated
    vix_signals = {'H20': 1.0, 'H21': 1.0, 'H128': 1.0}
    adjusted = tracker.correlation_adjusted_weights(vix_signals)
    
    print(f"\n📊 VIX signals (highly correlated):")
    print(f"   Input: {vix_signals}")
    print(f"   Adjusted: {adjusted}")
    
    # Weights should differ (not all equal)
    weights = list(adjusted.values())
    assert not all(abs(w - weights[0]) < 0.01 for w in weights), "Correlation should differ weights"
    
    # Sum should be 1
    assert abs(sum(adjusted.values()) - 1.0) < 0.01, "Weights should sum to 1"
    
    print("\n✅ PASSED")
    return True

def test_pattern_hunter_real():
    """Test pattern recognition with real market conditions."""
    print("\n" + "="*70)
    print("TEST: Pattern Hunter (REAL CONDITIONS)")
    print("="*70)
    
    hunter = PatternHunter()
    data = RealDataCache.get_spy_vix('5y')
    
    # Calculate real indicators
    data['ma50'] = data['close'].rolling(50).mean()
    data['ma200'] = data['close'].rolling(200).mean()
    
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    data['rsi'] = 100 - (100 / (1 + gain/loss))
    
    print(f"\n📊 Scanning 5 years of REAL data for rare patterns...")
    
    patterns_found = []
    
    for i in range(252, len(data), 5):  # Every 5 days
        current = data.iloc[i]
        prev = data.iloc[i-1]
        
        market_state = {
            'golden_cross': current['ma50'] > current['ma200'] and prev['ma50'] <= prev['ma200'],
            'vix': current['vix'],
            'vix_change_2d': current['vix'] - data.iloc[i-2]['vix'] if i >= 2 else 0,
            'breadth': 0.5,
            'sentiment_percentile': (data['vix'].iloc[:i] < current['vix']).sum() / i,
            'macro_risk': 'neutral',
            'spy_rsi': current['rsi'],
            'volume_ratio': 1.5
        }
        
        patterns = hunter.scan_all_patterns(market_state)
        if patterns:
            patterns_found.append({
                'date': data.index[i],
                'patterns': [p.name for p in patterns]
            })
    
    print(f"\n   Found {len(patterns_found)} pattern occurrences")
    if patterns_found:
        print(f"   Recent: {patterns_found[-3:]}")
    
    print("\n✅ PASSED")
    return True

def test_full_integration_real():
    """Full system integration with REAL data."""
    print("\n" + "="*70)
    print("TEST: Full System Integration (REAL DATA)")
    print("="*70)
    
    ensemble = QuantumEnsemble()
    data = RealDataCache.get_spy_vix('1y')
    
    print("\n📊 Running full integration with real market data...")
    
    # Step 1: Regime
    print("\n1. Detecting regime...")
    regime = ensemble.regime_detector.get_current_regime(data)
    print(f"   ✓ {regime.market_regime.value} (VIX: {regime.vix_level:.1f})")
    
    # Step 2: Generate signals (would come from signal generators in production)
    print("\n2. Simulating signals...")
    signals = {'H16': -0.4, 'H19': 0.7, 'H20': 0.3, 'H27E': 0.5}
    print(f"   ✓ {len(signals)} signals")
    
    # Step 3: Add news event
    print("\n3. Adding news event...")
    ensemble.news_monitor.clear_old_events()
    ensemble.news_monitor.add_event('economic_data', 'Jobs report')
    print(f"   ✓ News event registered")
    
    # Step 4: Combine
    print("\n4. Combining signals...")
    result = ensemble.combine_signals(signals, regime)
    print(f"   ✓ Combined: {result['combined_signal']:+.3f}")
    print(f"   ✓ Confidence: {result['confidence']:.1%}")
    
    # Step 5: Decision
    print("\n5. Making decision...")
    should_trade = ensemble.should_trade(result)
    print(f"   ✓ Trade: {should_trade}")
    
    # Validate all steps
    assert regime is not None
    assert result is not None
    assert -1 <= result['combined_signal'] <= 1
    
    print("\n✅ PASSED")
    return True

def test_performance_benchmark():
    """Benchmark real performance."""
    print("\n" + "="*70)
    print("TEST: Performance Benchmark")
    print("="*70)
    
    detector = RegimeDetector()
    ensemble = QuantumEnsemble()
    data = RealDataCache.get_spy_vix('1y')
    
    # Benchmark regime detection
    print("\n📊 Benchmarking regime detection...")
    times = []
    for _ in range(100):
        start = time.perf_counter()
        detector.get_current_regime(data)
        times.append((time.perf_counter() - start) * 1000)
    
    avg_regime = np.mean(times)
    print(f"   Regime detection: {avg_regime:.2f}ms avg (100 runs)")
    
    # Benchmark signal combination
    print("\n📊 Benchmarking signal combination...")
    regime = detector.get_current_regime(data)
    signals = {'H16': 0.5, 'H20': 0.3, 'H19': 0.6}
    
    times = []
    for _ in range(100):
        start = time.perf_counter()
        ensemble.combine_signals(signals, regime, use_news_adjustment=False)
        times.append((time.perf_counter() - start) * 1000)
    
    avg_combine = np.mean(times)
    print(f"   Signal combination: {avg_combine:.2f}ms avg (100 runs)")
    
    # Validate performance
    assert avg_regime < 100, f"Regime detection too slow: {avg_regime:.2f}ms"
    assert avg_combine < 50, f"Signal combination too slow: {avg_combine:.2f}ms"
    
    print(f"\n   ✅ All benchmarks within targets")
    print("\n✅ PASSED")
    return True

# ============================================================================
# MAIN RUNNER
# ============================================================================

def run_all_real_tests():
    """Run all tests with REAL data."""
    print("="*70)
    print("REAL DATA TEST SUITE")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Prefetch data
    print("\n📥 Prefetching REAL market data...")
    start = time.time()
    RealDataCache.prefetch()
    prefetch_time = time.time() - start
    print(f"   Done in {prefetch_time:.1f}s")
    
    # Run tests
    tests = [
        ("Regime Detection", test_regime_detection_real),
        ("Historical Regime", test_regime_historical_real),
        ("Signal Combination", test_signal_combination_real),
        ("News Impact", test_news_impact_real),
        ("Correlation Adjustment", test_correlation_adjustment_real),
        ("Pattern Hunter", test_pattern_hunter_real),
        ("Full Integration", test_full_integration_real),
        ("Performance Benchmark", test_performance_benchmark),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            start = time.time()
            passed = test_func()
            elapsed = time.time() - start
            results.append((name, passed, elapsed, None))
        except Exception as e:
            results.append((name, False, 0, str(e)))
            print(f"\n❌ {name} FAILED: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)
    
    passed = sum(1 for r in results if r[1])
    failed = len(results) - passed
    total_time = sum(r[2] for r in results) + prefetch_time
    
    for name, success, elapsed, error in results:
        status = "✅" if success else "❌"
        print(f"{status} {name}: {elapsed:.1f}s")
        if error:
            print(f"   Error: {error}")
    
    print(f"\n{'='*70}")
    print(f"TOTAL: {passed}/{len(results)} passed ({passed/len(results)*100:.0f}%)")
    print(f"TIME: {total_time:.1f}s (data fetch: {prefetch_time:.1f}s)")
    print(f"{'='*70}")
    
    # Save results
    os.makedirs('test_results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    report = {
        'timestamp': timestamp,
        'total_time': total_time,
        'prefetch_time': prefetch_time,
        'passed': passed,
        'failed': failed,
        'tests': [
            {'name': r[0], 'passed': r[1], 'time': r[2], 'error': r[3]}
            for r in results
        ]
    }
    
    with open(f'test_results/real_tests_{timestamp}.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    return passed == len(results)

if __name__ == '__main__':
    success = run_all_real_tests()
    exit(0 if success else 1)
