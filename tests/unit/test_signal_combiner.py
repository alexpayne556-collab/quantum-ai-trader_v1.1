"""
UNIT TESTS - SIGNAL COMBINER
============================
Fast tests for signal combination logic.

Target: <50ms per test
"""

from QUANTUM_ENSEMBLE_ENGINE import QuantumEnsemble, CorrelationTracker, RegimeDetector, MarketRegime
from test_fixtures import create_mock_signals, create_mock_market_data

def test_signal_combination_single():
    """Test combination with single signal."""
    ensemble = QuantumEnsemble()
    detector = RegimeDetector()
    data = create_mock_market_data()
    regime = detector.get_current_regime(data)
    
    signals = {'H16': 0.8}
    result = ensemble.combine_signals(signals, regime, use_news_adjustment=False)
    
    # Single signal should pass through
    assert abs(result['combined_signal'] - 0.8) < 0.1
    assert result['confidence'] > 0

def test_signal_combination_agreement():
    """Test combination when all signals agree."""
    ensemble = QuantumEnsemble()
    detector = RegimeDetector()
    data = create_mock_market_data()
    regime = detector.get_current_regime(data)
    
    # All bullish
    signals = {'H16': 0.7, 'H19': 0.8, 'H20': 0.75}
    result = ensemble.combine_signals(signals, regime, use_news_adjustment=False)
    
    # Combined should be positive and strong
    assert result['combined_signal'] > 0.5
    # Confidence should be high (low variance = agreement)
    assert result['confidence'] > 0.6

def test_signal_combination_conflict():
    """Test combination when signals conflict."""
    ensemble = QuantumEnsemble()
    detector = RegimeDetector()
    data = create_mock_market_data()
    regime = detector.get_current_regime(data)
    
    # Conflicting signals
    signals = {'H16': 0.8, 'H19': -0.7, 'H20': 0.5}
    result = ensemble.combine_signals(signals, regime, use_news_adjustment=False)
    
    # Confidence should be lower (high variance = disagreement)
    assert result['confidence'] < 0.7

def test_weights_sum_to_one():
    """Test that signal weights always sum to 1.0."""
    ensemble = QuantumEnsemble()
    detector = RegimeDetector()
    data = create_mock_market_data()
    regime = detector.get_current_regime(data)
    
    for _ in range(10):  # Test with random signals
        signals = create_mock_signals(num_signals=5, bias='neutral')
        result = ensemble.combine_signals(signals, regime, use_news_adjustment=False)
        
        weight_sum = sum(result['weights_used'].values())
        assert abs(weight_sum - 1.0) < 0.001, f"Weights sum to {weight_sum}, expected 1.0"

def test_correlation_adjustment_identical():
    """Test correlation adjustment with identical signals."""
    tracker = CorrelationTracker()
    
    # Three identical signals
    signals = {'H16': 1.0, 'H19': 1.0, 'H20': 1.0}
    adjusted = tracker.correlation_adjusted_weights(signals)
    
    # Weights should not all be equal (correlation adjusted)
    weights = list(adjusted.values())
    assert not all(abs(w - weights[0]) < 0.001 for w in weights), "Weights should differ after correlation adjustment"

def test_correlation_adjustment_sum():
    """Test correlation-adjusted weights sum to 1.0."""
    tracker = CorrelationTracker()
    
    for _ in range(10):
        signals = create_mock_signals(num_signals=4)
        adjusted = tracker.correlation_adjusted_weights(signals)
        
        weight_sum = sum(adjusted.values())
        assert abs(weight_sum - 1.0) < 0.001, f"Adjusted weights sum to {weight_sum}"

def test_correlation_known_pairs():
    """Test known signal correlations."""
    tracker = CorrelationTracker()
    
    # H20 and H21 are known to be 80% correlated
    corr = tracker.get_correlation('H20', 'H21')
    assert corr == 0.8, f"Expected 0.8, got {corr}"
    
    # H62 is independent
    corr = tracker.get_correlation('H62', 'H16')
    assert corr == 0.1, f"Expected 0.1, got {corr}"

def test_regime_weights_crisis():
    """Test signal weights in crisis regime."""
    ensemble = QuantumEnsemble()
    detector = RegimeDetector()
    
    # Create crisis scenario
    data = create_mock_market_data(trend='bear', volatility='extreme')
    regime = detector.get_current_regime(data)
    regime.market_regime = MarketRegime.CRISIS  # Force crisis
    
    weights = ensemble.get_regime_weights(regime)
    
    # VIX signals should have high weight in crisis
    vix_weight = weights.get('H20', 0) + weights.get('H21', 0) + weights.get('H128', 0)
    assert vix_weight > 0.5, f"VIX signals should dominate in crisis, got {vix_weight:.1%}"

def test_regime_weights_range():
    """Test signal weights in range-bound regime."""
    ensemble = QuantumEnsemble()
    detector = RegimeDetector()
    
    # Create range-bound scenario
    data = create_mock_market_data(trend='sideways', volatility='low')
    regime = detector.get_current_regime(data)
    regime.market_regime = MarketRegime.RANGE_BOUND  # Force range-bound
    
    weights = ensemble.get_regime_weights(regime)
    
    # Mean reversion signals should have high weight
    # H19 (Bollinger) should be strong in ranges
    assert weights.get('H19', 0) > 0.2, "H19 should be strong in range-bound markets"

def test_should_trade_low_confidence():
    """Test that low confidence prevents trading."""
    ensemble = QuantumEnsemble()
    
    result = {
        'combined_signal': 0.8,
        'confidence': 0.3,  # Low confidence
        'individual_signals': {'H16': 0.8}
    }
    
    assert not ensemble.should_trade(result), "Should not trade with low confidence"

def test_should_trade_weak_signal():
    """Test that weak signal prevents trading."""
    ensemble = QuantumEnsemble()
    
    result = {
        'combined_signal': 0.2,  # Weak signal
        'confidence': 0.8,
        'individual_signals': {'H16': 0.2}
    }
    
    assert not ensemble.should_trade(result), "Should not trade with weak signal"

def test_should_trade_strong_setup():
    """Test that strong setup triggers trade."""
    ensemble = QuantumEnsemble()
    
    result = {
        'combined_signal': 0.8,  # Strong signal
        'confidence': 0.8,  # High confidence
        'individual_signals': {'H16': 0.8, 'H19': 0.75, 'H20': 0.85}
    }
    
    assert ensemble.should_trade(result), "Should trade with strong setup"

def run_all():
    """Run all unit tests for signal combiner."""
    tests = [
        test_signal_combination_single,
        test_signal_combination_agreement,
        test_signal_combination_conflict,
        test_weights_sum_to_one,
        test_correlation_adjustment_identical,
        test_correlation_adjustment_sum,
        test_correlation_known_pairs,
        test_regime_weights_crisis,
        test_regime_weights_range,
        test_should_trade_low_confidence,
        test_should_trade_weak_signal,
        test_should_trade_strong_setup,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
            print(f"✅ {test.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"❌ {test.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"❌ {test.__name__}: {type(e).__name__}: {e}")
    
    return passed, failed

if __name__ == '__main__':
    print("="*80)
    print("UNIT TESTS: Signal Combiner")
    print("="*80)
    passed, failed = run_all()
    print(f"\n{'='*80}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*80}")
