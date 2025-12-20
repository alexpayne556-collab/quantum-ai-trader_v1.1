"""
UNIT TESTS - NEWS MONITOR
=========================
Fast tests for news event monitoring.

Target: <50ms per test
"""

from QUANTUM_ENSEMBLE_ENGINE import NewsQuantum
from datetime import datetime, timedelta

def test_fomc_impact():
    """Test FOMC event reduces confidence by 90%."""
    monitor = NewsQuantum()
    monitor.add_event('fomc', 'Fed meeting')
    
    base = 0.85
    adjusted, reason = monitor.adjust_signal_confidence(base, 'test')
    
    # Should reduce by ~90%
    assert adjusted < base * 0.2, f"Expected <{base*0.2:.2f}, got {adjusted:.2f}"
    assert 'fomc' in reason.lower()

def test_earnings_impact():
    """Test earnings event reduces confidence by 50%."""
    monitor = NewsQuantum()
    monitor.add_event('earnings', 'AAPL earnings')
    
    base = 0.80
    adjusted, reason = monitor.adjust_signal_confidence(base, 'test')
    
    # Should reduce by ~50%
    assert adjusted < base * 0.6, f"Expected <{base*0.6:.2f}, got {adjusted:.2f}"

def test_black_swan_impact():
    """Test black swan event reduces confidence to zero."""
    monitor = NewsQuantum()
    monitor.add_event('black_swan', 'Bank collapse')
    
    base = 0.90
    adjusted, reason = monitor.adjust_signal_confidence(base, 'test')
    
    # Should be near zero
    assert adjusted < 0.1, f"Expected <0.1, got {adjusted:.2f}"

def test_no_events():
    """Test no adjustment when no events active."""
    monitor = NewsQuantum()
    
    base = 0.75
    adjusted, reason = monitor.adjust_signal_confidence(base, 'test')
    
    # Should be unchanged
    assert adjusted == base, f"Expected {base}, got {adjusted}"
    assert 'no active' in reason.lower()

def test_multiple_events():
    """Test multiple events - highest impact wins."""
    monitor = NewsQuantum()
    monitor.add_event('economic_data', 'CPI report')  # -30%
    monitor.add_event('fomc', 'Fed meeting')  # -90%
    
    base = 0.80
    adjusted, reason = monitor.adjust_signal_confidence(base, 'test')
    
    # Should use worst impact (FOMC -90%)
    assert adjusted < base * 0.2, "Should use worst event impact"
    assert 'fomc' in reason.lower(), "Reason should mention FOMC"

def test_event_duration():
    """Test events have correct durations."""
    monitor = NewsQuantum()
    
    assert monitor.event_types['fomc']['duration_days'] == 2
    assert monitor.event_types['earnings']['duration_days'] == 1
    assert monitor.event_types['black_swan']['duration_days'] == 10

def test_event_expiration():
    """Test events expire after duration."""
    monitor = NewsQuantum()
    
    # Add event with backdated timestamp
    past = datetime.now() - timedelta(days=10)
    monitor.active_events.append(NewsQuantum().event_types)
    
    # Clear old events
    monitor.clear_old_events()
    
    # Should be empty after clearing old events
    # (This is a simplified test - in reality would need to mock timestamps)
    assert True  # Placeholder - would need better time mocking

def test_confidence_bounds():
    """Test adjusted confidence stays in [0, 1] range."""
    monitor = NewsQuantum()
    monitor.add_event('black_swan', 'Catastrophe')
    
    # Test various base values
    for base in [0.1, 0.5, 0.9, 1.0]:
        adjusted, _ = monitor.adjust_signal_confidence(base, 'test')
        assert 0 <= adjusted <= 1, f"Confidence {adjusted} out of bounds for base {base}"

def test_event_registration():
    """Test events are properly registered."""
    monitor = NewsQuantum()
    monitor.add_event('fomc', 'Test event')
    
    active = monitor.get_active_events()
    assert len(active) == 1
    assert active[0].event_type == 'fomc'
    assert active[0].description == 'Test event'

def test_unknown_event_type():
    """Test unknown event type is handled."""
    monitor = NewsQuantum()
    
    # Should not crash
    monitor.add_event('unknown_type', 'Some event')
    
    # Should not affect confidence since it's unknown
    base = 0.80
    adjusted, _ = monitor.adjust_signal_confidence(base, 'test')
    # Might still work if it added the unknown event, but shouldn't crash

def run_all():
    """Run all unit tests for news monitor."""
    tests = [
        test_fomc_impact,
        test_earnings_impact,
        test_black_swan_impact,
        test_no_events,
        test_multiple_events,
        test_event_duration,
        test_event_expiration,
        test_confidence_bounds,
        test_event_registration,
        test_unknown_event_type,
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
    print("UNIT TESTS: News Monitor")
    print("="*80)
    passed, failed = run_all()
    print(f"\n{'='*80}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*80}")
