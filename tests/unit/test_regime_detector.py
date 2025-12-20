"""
UNIT TESTS - REGIME DETECTOR
=============================
Fast, isolated tests using mock data.
No network calls, no external dependencies.

Target: <100ms per test
"""

from QUANTUM_ENSEMBLE_ENGINE import RegimeDetector, VolatilityRegime, TrendRegime, MarketRegime
from test_fixtures import (
    create_mock_market_data, 
    create_crisis_scenario,
    create_bull_market_scenario,
    create_range_bound_scenario
)

def test_volatility_classification_low():
    """Test VIX < 15 = low volatility."""
    detector = RegimeDetector()
    assert detector.detect_volatility_regime(12, 0.2) == VolatilityRegime.LOW
    assert detector.detect_volatility_regime(14, 0.24) == VolatilityRegime.LOW

def test_volatility_classification_normal():
    """Test VIX 15-20 = normal volatility."""
    detector = RegimeDetector()
    assert detector.detect_volatility_regime(17, 0.4) == VolatilityRegime.NORMAL
    assert detector.detect_volatility_regime(19, 0.5) == VolatilityRegime.NORMAL

def test_volatility_classification_high():
    """Test VIX 20-25 = high volatility."""
    detector = RegimeDetector()
    assert detector.detect_volatility_regime(22, 0.7) == VolatilityRegime.HIGH
    assert detector.detect_volatility_regime(24, 0.76) == VolatilityRegime.HIGH

def test_volatility_classification_extreme():
    """Test VIX > 35 = extreme volatility."""
    detector = RegimeDetector()
    assert detector.detect_volatility_regime(40, 0.95) == VolatilityRegime.EXTREME
    assert detector.detect_volatility_regime(50, 0.99) == VolatilityRegime.EXTREME

def test_trend_classification_uptrend():
    """Test positive returns = uptrend."""
    detector = RegimeDetector()
    # 3% in 20 days, 5% in 50 days = uptrend
    assert detector.detect_trend_regime(0.03, 0.05, 0.001) == TrendRegime.UPTREND

def test_trend_classification_downtrend():
    """Test negative returns = downtrend."""
    detector = RegimeDetector()
    # -3% in 20 days, -5% in 50 days = downtrend
    assert detector.detect_trend_regime(-0.03, -0.05, -0.001) == TrendRegime.DOWNTREND

def test_trend_classification_sideways():
    """Test small returns = sideways."""
    detector = RegimeDetector()
    # Less than 2% move = sideways
    assert detector.detect_trend_regime(0.01, 0.01, 0.0001) == TrendRegime.SIDEWAYS
    assert detector.detect_trend_regime(-0.01, -0.01, -0.0001) == TrendRegime.SIDEWAYS

def test_market_regime_crisis():
    """Test high vol + downtrend = crisis."""
    detector = RegimeDetector()
    regime = detector.detect_market_regime(VolatilityRegime.HIGH, TrendRegime.DOWNTREND)
    assert regime == MarketRegime.CRISIS

def test_market_regime_recovery():
    """Test high vol + uptrend = recovery."""
    detector = RegimeDetector()
    regime = detector.detect_market_regime(VolatilityRegime.HIGH, TrendRegime.UPTREND)
    assert regime == MarketRegime.RECOVERY

def test_market_regime_bull_trending():
    """Test low vol + uptrend = bull trending."""
    detector = RegimeDetector()
    regime = detector.detect_market_regime(VolatilityRegime.LOW, TrendRegime.UPTREND)
    assert regime == MarketRegime.BULL_TRENDING

def test_market_regime_range_bound():
    """Test low vol + sideways = range bound."""
    detector = RegimeDetector()
    regime = detector.detect_market_regime(VolatilityRegime.LOW, TrendRegime.SIDEWAYS)
    assert regime == MarketRegime.RANGE_BOUND

def test_full_regime_detection_crisis():
    """Test full regime detection with crisis data."""
    detector = RegimeDetector()
    data = create_crisis_scenario()
    regime = detector.get_current_regime(data)
    
    # Should detect some form of stress
    assert regime.volatility in [VolatilityRegime.HIGH, VolatilityRegime.EXTREME]
    assert regime.market_regime in [MarketRegime.CRISIS, MarketRegime.VOLATILE_CHOPPY, MarketRegime.BEAR_TRENDING]

def test_full_regime_detection_bull():
    """Test full regime detection with bull market data."""
    detector = RegimeDetector()
    data = create_bull_market_scenario()
    regime = detector.get_current_regime(data)
    
    # Should detect bullish conditions
    assert regime.trend == TrendRegime.UPTREND
    assert regime.market_regime in [MarketRegime.BULL_TRENDING, MarketRegime.RECOVERY]

def test_full_regime_detection_range():
    """Test full regime detection with range-bound data."""
    detector = RegimeDetector()
    data = create_range_bound_scenario()
    regime = detector.get_current_regime(data)
    
    # Should detect sideways
    assert regime.trend == TrendRegime.SIDEWAYS
    assert regime.market_regime in [MarketRegime.RANGE_BOUND, MarketRegime.VOLATILE_CHOPPY]

def run_all():
    """Run all unit tests for regime detector."""
    tests = [
        test_volatility_classification_low,
        test_volatility_classification_normal,
        test_volatility_classification_high,
        test_volatility_classification_extreme,
        test_trend_classification_uptrend,
        test_trend_classification_downtrend,
        test_trend_classification_sideways,
        test_market_regime_crisis,
        test_market_regime_recovery,
        test_market_regime_bull_trending,
        test_market_regime_range_bound,
        test_full_regime_detection_crisis,
        test_full_regime_detection_bull,
        test_full_regime_detection_range,
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
    print("UNIT TESTS: Regime Detector")
    print("="*80)
    passed, failed = run_all()
    print(f"\n{'='*80}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*80}")
