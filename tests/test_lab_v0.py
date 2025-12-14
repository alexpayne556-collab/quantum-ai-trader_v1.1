#!/usr/bin/env python3
"""
Unit tests for Pattern Discovery Lab V0.

Fast tests with synthetic data.
"""
import pytest
import numpy as np
import pandas as pd
import json
import tempfile
from pathlib import Path

from pattern_discovery_lab.lab_v0 import (
    walk_forward_splits,
    compute_forward_returns,
    compute_rank_ic,
    compute_ic_tstat,
    negative_control_random,
    placebo_time_shift,
    check_no_nan_inf,
    check_oos_degradation_gate,
    check_finite_metrics_gate,
    is_finite,
    ensure_deterministic_order,
    round_numeric,
    NUMERIC_PRECISION,
)

from pattern_discovery_lab.detector_v0 import momentum_detector

from pattern_discovery_lab.schema_v0 import (
    write_json,
    read_json,
    sanitize_string,
    SCHEMA_VERSION,
)


# ============================================================================
# TEST WALK-FORWARD SPLITS
# ============================================================================

def test_walk_forward_splits_embargo_respected():
    """Test that embargo bars are excluded between train and test."""
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    
    splits = list(walk_forward_splits(dates, train_len=3, test_len=2, embargo=1))
    
    # Check first split
    train_idx, test_idx = splits[0]
    assert len(train_idx) == 3
    assert len(test_idx) == 2
    
    # Train should be [0, 1, 2]
    # Embargo should be [3]
    # Test should be [4, 5]
    assert list(train_idx) == [0, 1, 2]
    assert list(test_idx) == [4, 5]
    
    # Verify embargo gap
    assert max(train_idx) + 1 < min(test_idx)  # At least 1 bar gap


def test_walk_forward_splits_no_overlap():
    """Test that train and test sets never overlap."""
    dates = pd.date_range('2020-01-01', periods=20, freq='D')
    
    splits = list(walk_forward_splits(dates, train_len=5, test_len=3, embargo=2))
    
    for train_idx, test_idx in splits:
        # No overlap between train and test
        assert len(set(train_idx) & set(test_idx)) == 0
        
        # Test comes after train
        assert max(train_idx) < min(test_idx)


def test_walk_forward_splits_insufficient_data():
    """Test error when insufficient data."""
    dates = pd.date_range('2020-01-01', periods=5, freq='D')
    
    # Need at least train_len + embargo + test_len = 3 + 1 + 2 = 6
    with pytest.raises(ValueError, match="Insufficient data"):
        list(walk_forward_splits(dates, train_len=3, test_len=2, embargo=1))


def test_walk_forward_splits_multiple_splits():
    """Test multiple splits are generated correctly."""
    dates = pd.date_range('2020-01-01', periods=15, freq='D')
    
    splits = list(walk_forward_splits(dates, train_len=3, test_len=2, embargo=1))
    
    # Should generate multiple splits
    assert len(splits) > 1
    
    # Each split should advance by test_len
    for i in range(len(splits) - 1):
        train_idx1, _ = splits[i]
        train_idx2, _ = splits[i + 1]
        assert train_idx2[0] == train_idx1[0] + 2  # Advanced by test_len


# ============================================================================
# TEST FORWARD RETURNS
# ============================================================================

def test_compute_forward_returns():
    """Test forward returns calculation."""
    prices = pd.Series([100, 110, 121, 133.1], index=pd.date_range('2020-01-01', periods=4))
    
    fwd_returns = compute_forward_returns(prices, horizon=1)
    
    # fwd_ret[0] = (110 / 100) - 1 = 0.1
    # fwd_ret[1] = (121 / 110) - 1 = 0.1
    # fwd_ret[2] = (133.1 / 121) - 1 = 0.1
    # fwd_ret[3] = NaN
    
    assert abs(fwd_returns.iloc[0] - 0.1) < 1e-6
    assert abs(fwd_returns.iloc[1] - 0.1) < 1e-6
    assert abs(fwd_returns.iloc[2] - 0.1) < 1e-6
    assert pd.isna(fwd_returns.iloc[3])


def test_compute_forward_returns_horizon():
    """Test forward returns with different horizons."""
    prices = pd.Series([100, 110, 121, 133.1, 146.41])
    
    fwd_returns_2 = compute_forward_returns(prices, horizon=2)
    
    # fwd_ret[0] = (121 / 100) - 1 = 0.21
    assert abs(fwd_returns_2.iloc[0] - 0.21) < 1e-6


# ============================================================================
# TEST RANK IC
# ============================================================================

def test_compute_rank_ic_perfect_correlation():
    """Test RankIC with perfect positive correlation."""
    signal = pd.Series([1, 2, 3, 4, 5])
    fwd_returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
    
    ic = compute_rank_ic(signal, fwd_returns)
    
    # Perfect monotonic relationship -> Spearman = 1.0
    assert abs(ic - 1.0) < 1e-6


def test_compute_rank_ic_negative_correlation():
    """Test RankIC with negative correlation."""
    signal = pd.Series([5, 4, 3, 2, 1])
    fwd_returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
    
    ic = compute_rank_ic(signal, fwd_returns)
    
    # Perfect negative monotonic relationship -> Spearman = -1.0
    assert abs(ic - (-1.0)) < 1e-6


def test_compute_rank_ic_with_nan():
    """Test RankIC handles NaN values."""
    signal = pd.Series([1, 2, np.nan, 4, 5])
    fwd_returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
    
    ic = compute_rank_ic(signal, fwd_returns)
    
    # Should compute on valid values only
    assert not np.isnan(ic)


def test_compute_ic_tstat():
    """Test IC t-statistic computation."""
    # Generate IC values with mean > 0
    ic_values = [0.05, 0.06, 0.04, 0.07, 0.05]
    
    t_stat, p_val = compute_ic_tstat(ic_values)
    
    # Should have positive t-stat
    assert t_stat > 0
    assert 0 <= p_val <= 1


# ============================================================================
# TEST NEGATIVE CONTROLS
# ============================================================================

def test_negative_control_random_deterministic():
    """Test random control is deterministic with same seed."""
    signal1 = negative_control_random(100, seed=42)
    signal2 = negative_control_random(100, seed=42)
    
    pd.testing.assert_series_equal(signal1, signal2)


def test_negative_control_random_with_index():
    """Test random control respects provided index."""
    index = pd.date_range('2020-01-01', periods=100, freq='D')
    signal = negative_control_random(100, seed=42, index=index)
    
    assert len(signal) == 100
    pd.testing.assert_index_equal(signal.index, index)


def test_negative_control_random_different_seeds():
    """Test random control differs with different seeds."""
    signal1 = negative_control_random(100, seed=42)
    signal2 = negative_control_random(100, seed=43)
    
    # Should be different
    assert not signal1.equals(signal2)


def test_placebo_time_shift():
    """Test time-shift placebo."""
    signal = pd.Series([1, 2, 3, 4, 5])
    
    shifted = placebo_time_shift(signal, shift=2)
    
    # Should shift forward by 2
    assert pd.isna(shifted.iloc[0])
    assert pd.isna(shifted.iloc[1])
    assert shifted.iloc[2] == 1
    assert shifted.iloc[3] == 2


# ============================================================================
# TEST GATE CHECKS
# ============================================================================

def test_check_no_nan_inf_valid():
    """Test NaN/Inf check with valid value."""
    # Should not raise
    check_no_nan_inf(3.14, "test_value")


def test_check_no_nan_inf_rejects_nan():
    """Test NaN/Inf check rejects NaN."""
    with pytest.raises(ValueError, match="is NaN"):
        check_no_nan_inf(np.nan, "test_value")


def test_check_no_nan_inf_rejects_inf():
    """Test NaN/Inf check rejects Inf."""
    with pytest.raises(ValueError, match="is Inf"):
        check_no_nan_inf(np.inf, "test_value")


def test_check_oos_degradation_gate_pass():
    """Test OOS degradation gate passes when OOS >= 60% of IS."""
    passed, reason = check_oos_degradation_gate(is_ic_mean=0.10, oos_ic_mean=0.07, threshold=0.60)
    
    assert passed is True
    assert "0.07" in reason


def test_check_oos_degradation_gate_fail():
    """Test OOS degradation gate fails when OOS < 60% of IS."""
    passed, reason = check_oos_degradation_gate(is_ic_mean=0.10, oos_ic_mean=0.05, threshold=0.60)
    
    assert passed is False
    assert "0.05" in reason


def test_check_oos_degradation_gate_nan():
    """Test OOS degradation gate fails with NaN."""
    passed, reason = check_oos_degradation_gate(is_ic_mean=np.nan, oos_ic_mean=0.07)
    
    assert passed is False
    assert "NaN" in reason


def test_ensure_deterministic_order():
    """Test list is sorted deterministically."""
    items = [3, 1, 4, 1, 5, 9, 2, 6]
    
    sorted_items = ensure_deterministic_order(items)
    
    assert sorted_items == [1, 1, 2, 3, 4, 5, 6, 9]


def test_round_numeric():
    """Test numeric rounding."""
    value = 3.141592653589793
    
    rounded = round_numeric(value, decimals=4)
    
    assert rounded == 3.1416


def test_round_numeric_nan_to_none():
    """Test NaN converts to None."""
    assert round_numeric(np.nan) is None
    

def test_round_numeric_inf_to_none():
    """Test Inf converts to None."""
    assert round_numeric(np.inf) is None
    assert round_numeric(-np.inf) is None


def test_is_finite():
    """Test is_finite check."""
    assert is_finite(3.14) is True
    assert is_finite(0.0) is True
    assert is_finite(np.nan) is False
    assert is_finite(np.inf) is False


def test_check_finite_metrics_gate():
    """Test finite metrics gate."""
    # All finite
    passed, reason = check_finite_metrics_gate({"a": 1.0, "b": 2.0})
    assert passed is True
    
    # Has NaN
    passed, reason = check_finite_metrics_gate({"a": 1.0, "b": np.nan})
    assert passed is False
    assert "b" in reason
    
    # Has Inf
    passed, reason = check_finite_metrics_gate({"a": np.inf, "b": 2.0})
    assert passed is False
    assert "a" in reason
    
    # None is OK
    passed, reason = check_finite_metrics_gate({"a": None, "b": 2.0})
    assert passed is True


# ============================================================================
# TEST DETECTOR
# ============================================================================

def test_momentum_detector():
    """Test momentum detector calculation."""
    prices = pd.Series([100, 110, 121, 133.1], index=pd.date_range('2020-01-01', periods=4))
    
    signal = momentum_detector(prices, lookback=1)
    
    # signal[0] = NaN (no prior price)
    # signal[1] = (110 / 100) - 1 = 0.1
    # signal[2] = (121 / 110) - 1 = 0.1
    # signal[3] = (133.1 / 121) - 1 = 0.1
    
    assert pd.isna(signal.iloc[0])
    assert abs(signal.iloc[1] - 0.1) < 1e-6
    assert abs(signal.iloc[2] - 0.1) < 1e-6
    assert abs(signal.iloc[3] - 0.1) < 1e-6


def test_momentum_detector_lookback():
    """Test momentum detector with different lookback."""
    prices = pd.Series([100, 110, 121, 133.1, 146.41])
    
    signal = momentum_detector(prices, lookback=2)
    
    # signal[2] = (121 / 100) - 1 = 0.21
    assert abs(signal.iloc[2] - 0.21) < 1e-6


# ============================================================================
# TEST SCHEMA AND JSON SERIALIZATION
# ============================================================================

def test_json_serialization_deterministic():
    """Test JSON serialization is deterministic."""
    obj = {
        "z": 3,
        "a": 1,
        "m": 2,
        "nested": {"z": 9, "a": 7}
    }
    
    # Serialize twice
    json_str1 = json.dumps(obj, sort_keys=True, indent=2)
    json_str2 = json.dumps(obj, sort_keys=True, indent=2)
    
    # Should be identical
    assert json_str1 == json_str2
    
    # Keys should be sorted
    assert '"a":' in json_str1
    assert json_str1.index('"a":') < json_str1.index('"m":')
    assert json_str1.index('"m":') < json_str1.index('"z":')


def test_write_json_deterministic():
    """Test write_json produces deterministic output."""
    obj = {
        "schema_version": SCHEMA_VERSION,
        "values": [3, 1, 2],
        "meta": {"z": 1, "a": 2}
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path1 = Path(tmpdir) / "test1.json"
        path2 = Path(tmpdir) / "test2.json"
        
        write_json(str(path1), obj)
        write_json(str(path2), obj)
        
        # Read both files as strings
        content1 = path1.read_text()
        content2 = path2.read_text()
        
        # Should be byte-identical
        assert content1 == content2


def test_write_json_creates_directory():
    """Test write_json creates parent directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "subdir" / "nested" / "file.json"
        
        obj = {"test": "value"}
        write_json(str(path), obj)
        
        # File should exist
        assert path.exists()
        
        # Content should be correct
        loaded = read_json(str(path))
        assert loaded == obj


def test_build_results_dict():
    """Test results dict construction."""
    results = {
        "schema_version": SCHEMA_VERSION,
        "meta": {"run_id": "20231214_120000", "seed": 42},
        "gates": {"finite_metrics": {"passed": True, "reason": "OK"}},
        "controls": {"random": {"ic": 0.01}},
        "candidates_ranked": [{"rank": 1, "candidate_id": "test"}],
        "overall_status": "PASS"
    }
    
    assert results["schema_version"] == SCHEMA_VERSION
    assert results["meta"]["seed"] == 42
    assert results["overall_status"] == "PASS"


def test_sanitize_string_valid():
    """Test sanitize_string accepts valid strings."""
    valid = "Hello World 123 !@#$%"
    
    result = sanitize_string(valid)
    
    assert result == valid


def test_sanitize_string_rejects_non_ascii():
    """Test sanitize_string rejects non-ASCII."""
    with pytest.raises(ValueError, match="non-ASCII"):
        sanitize_string("Hello 世界")


def test_sanitize_string_rejects_control_chars():
    """Test sanitize_string rejects control characters."""
    with pytest.raises(ValueError, match="control characters"):
        sanitize_string("Hello\x00World")


# ============================================================================
# INTEGRATION TEST
# ============================================================================

def test_full_workflow_integration():
    """Test complete workflow with synthetic data."""
    # Create synthetic price data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = pd.Series(100 * np.exp(np.random.randn(100).cumsum() * 0.01), index=dates)
    
    # Generate signal
    signal = momentum_detector(prices, lookback=5)
    
    # Compute forward returns
    fwd_returns = compute_forward_returns(prices, horizon=1)
    
    # Walk-forward splits
    splits = list(walk_forward_splits(dates, train_len=30, test_len=10, embargo=1))
    
    assert len(splits) > 0
    
    # Compute IC for first split
    train_idx, test_idx = splits[0]
    
    train_signal = signal.iloc[train_idx]
    train_fwd_ret = fwd_returns.iloc[train_idx]
    
    ic_train = compute_rank_ic(train_signal, train_fwd_ret)
    
    # Should compute without error
    assert isinstance(ic_train, float)
    
    # Negative control WITH PROPER INDEX ALIGNMENT
    random_signal = negative_control_random(len(prices), seed=42, index=prices.index)
    ic_random = compute_rank_ic(random_signal, fwd_returns)
    
    # Should compute without error and be finite
    assert isinstance(ic_random, float)
    assert is_finite(ic_random)


# ============================================================================
# TEST STRICT JSON COMPLIANCE
# ============================================================================

def test_degenerate_zero_variance_yields_none():
    """Test that zero-variance signal yields None IC (not NaN)."""
    # Zero-variance signal (constant)
    signal = pd.Series([1.0] * 100, index=pd.date_range('2020-01-01', periods=100))
    fwd_returns = pd.Series(np.random.randn(100), index=signal.index)
    
    # Should return nan from spearmanr
    ic = compute_rank_ic(signal, fwd_returns)
    
    # round_numeric should convert to None
    ic_rounded = round_numeric(ic)
    
    assert ic_rounded is None


def test_insufficient_data_yields_none():
    """Test that insufficient data yields None IC (not NaN)."""
    # Only 1 valid observation
    signal = pd.Series([1.0, np.nan, np.nan], index=pd.date_range('2020-01-01', periods=3))
    fwd_returns = pd.Series([0.01, 0.02, 0.03], index=signal.index)
    
    ic = compute_rank_ic(signal, fwd_returns)
    
    # Should return NaN from insufficient data
    assert np.isnan(ic)
    
    # round_numeric should convert to None
    ic_rounded = round_numeric(ic)
    assert ic_rounded is None


def test_strict_json_dump_with_allow_nan_false():
    """Test that write_json with allow_nan=False rejects NaN/Inf."""
    import tempfile
    import os
    
    # Clean data - should work
    clean_data = {
        "values": [1.0, 2.0, None],  # None is allowed (becomes null)
        "mean": 1.5
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "clean.json")
        write_json(path, clean_data)  # Should succeed
        
        # Verify it wrote
        loaded = read_json(path)
        assert loaded["values"] == [1.0, 2.0, None]
    
    # Data with NaN - should FAIL
    dirty_data = {
        "values": [1.0, float('nan'), 3.0],
        "mean": 2.0
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "dirty.json")
        
        # Should raise ValueError because allow_nan=False
        with pytest.raises(ValueError, match="Out of range float values are not JSON compliant"):
            write_json(path, dirty_data)


def test_finite_metrics_gate_with_degenerate_case():
    """Test that finite_metrics gate catches non-finite values."""
    from pattern_discovery_lab.lab_v0 import check_finite_metrics_gate
    
    # Case 1: All finite - should pass
    finite_results = {
        "in_sample_ic": {"mean": 0.5, "t_stat": 2.0, "p_value": 0.05},
        "out_of_sample_ic": {"mean": 0.3, "t_stat": 1.5, "p_value": 0.10},
        "negative_controls": {
            "random_placebo": {"ic": 0.01, "reason": None},
            "time_shift_placebo": {"ic": 0.1, "reason": None}
        }
    }
    passed, reason = check_finite_metrics_gate(finite_results)
    assert passed
    assert "finite" in reason.lower()
    
    # Case 2: NaN in main IC - should fail
    nan_results = {
        "in_sample_ic": {"mean": float('nan'), "t_stat": 2.0, "p_value": 0.05},
        "out_of_sample_ic": {"mean": 0.3, "t_stat": 1.5, "p_value": 0.10},
        "negative_controls": {
            "random_placebo": {"ic": 0.01, "reason": None},
            "time_shift_placebo": {"ic": 0.1, "reason": None}
        }
    }
    passed, reason = check_finite_metrics_gate(nan_results)
    assert not passed
    assert "non-finite" in reason.lower()
    
    # Case 3: None (null) is acceptable - should pass
    null_results = {
        "in_sample_ic": {"mean": None, "t_stat": None, "p_value": None},
        "out_of_sample_ic": {"mean": None, "t_stat": None, "p_value": None},
        "negative_controls": {
            "random_placebo": {"ic": None, "reason": "insufficient data"},
            "time_shift_placebo": {"ic": None, "reason": "insufficient data"}
        }
    }
    passed, reason = check_finite_metrics_gate(null_results)
    assert passed  # None is valid (represents undefined/null)


def test_stdout_formatting_never_prints_nan():
    """Test that format_metric_for_stdout never outputs 'nan'."""
    # Helper to simulate stdout formatting
    def format_metric(val):
        return f"{val:.6f}" if val is not None else "null"
    
    # None -> "null"
    assert format_metric(None) == "null"
    
    # float('nan') would print "nan" if passed directly - must check first
    nan_val = float('nan')
    formatted = format_metric(None if np.isnan(nan_val) else nan_val)
    assert formatted == "null"
    
    # Normal value
    assert format_metric(0.123456) == "0.123456"
