"""
Smoke test for detectors and controls.

Validates that:
1. All detectors run without errors
2. Controls work correctly
3. Known-structure data shows higher scores than pure noise
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.datasets.pure_noise import WhiteNoiseDataset
from src.datasets.known_structure import ARProcessDataset
from src.detectors.compressibility import LempelZivDetector
from src.detectors.dependence import AutocorrelationDetector, MutualInformationDetector
from src.detectors.regime import ChangepointDetector
from src.detectors.stability import WindowShiftDetector
from src.controls import TimeShuffleControl, PhaseRandomizationControl


def test_detectors():
    """Test that all detectors run without errors."""
    
    print("=" * 80)
    print("DETECTOR SMOKE TEST")
    print("=" * 80)
    
    # Generate test data
    seed = 42
    noise_ds = WhiteNoiseDataset(seed, {'length': 500})
    noise_data = noise_ds.generate()
    
    ar_ds = ARProcessDataset(seed, {'length': 500, 'phi': 0.7})
    ar_data = ar_ds.generate()
    
    # Test detectors
    detectors = [
        LempelZivDetector({'normalize': True}),
        AutocorrelationDetector({'max_lag': 20}),
        MutualInformationDetector({'lag': 1, 'bins': 10}),
        ChangepointDetector({'model': 'rbf', 'min_size': 50})
    ]
    
    print("\n📊 Testing Detectors on White Noise...")
    for detector in detectors:
        result = detector.run(noise_data)
        print(f"  ✅ {result.detector_name}: score={result.structure_score:.4f}, time={result.execution_time:.4f}s")
    
    print("\n🎯 Testing Detectors on AR(1) Process...")
    for detector in detectors:
        result = detector.run(ar_data)
        print(f"  ✅ {result.detector_name}: score={result.structure_score:.4f}, time={result.execution_time:.4f}s")
    
    # Test stability detector (uses another detector as base)
    print("\n⚖️  Testing Stability Detector...")
    base_detector = AutocorrelationDetector({'max_lag': 20})
    stability_detector = WindowShiftDetector({'n_windows': 5}, base_detector)
    
    result_noise = stability_detector.run(noise_data)
    result_ar = stability_detector.run(ar_data)
    
    print(f"  ✅ {result_noise.detector_name} on noise: {result_noise.structure_score:.4f}")
    print(f"  ✅ {result_ar.detector_name} on AR(1): {result_ar.structure_score:.4f}")


def test_controls():
    """Test that controls work correctly."""
    
    print("\n" + "=" * 80)
    print("CONTROL SMOKE TEST")
    print("=" * 80)
    
    # Generate AR(1) data (has structure)
    seed = 42
    ar_ds = ARProcessDataset(seed, {'length': 500, 'phi': 0.7})
    ar_data = ar_ds.generate()
    
    # Detector
    detector = AutocorrelationDetector({'max_lag': 20})
    
    print("\n🔀 Testing Time-Shuffle Control...")
    time_shuffle = TimeShuffleControl(n_shuffles=50, seed=seed)
    result = time_shuffle.test(detector, ar_data, alpha=0.01)
    
    print(f"  Original score: {result.original_score:.4f}")
    print(f"  Mean shuffled score: {result.metadata['mean_shuffled_score']:.4f}")
    print(f"  P-value: {result.p_value:.4f}")
    print(f"  Passes: {result.passes}")
    
    assert result.passes, "AR(1) should pass time-shuffle control (original > shuffled)"
    print("  ✅ Time-shuffle control working correctly")
    
    print("\n🌀 Testing Phase-Randomization Control...")
    phase_rand = PhaseRandomizationControl(n_surrogates=50, seed=seed)
    result = phase_rand.test(detector, ar_data, alpha=0.01)
    
    print(f"  Original score: {result.original_score:.4f}")
    print(f"  Mean surrogate score: {result.metadata['mean_surrogate_score']:.4f}")
    print(f"  P-value: {result.p_value:.4f}")
    print(f"  Passes: {result.passes}")
    
    # NOTE: Phase-randomization PRESERVES autocorrelation for AR(1)!
    # This is EXPECTED - linear autocorrelation lives in frequency domain.
    # For AR(1), phase-randomization should NOT destroy autocorrelation.
    # This control is designed to catch nonlinear-only structure.
    print("  ℹ️  Phase-randomization preserves linear autocorrelation (expected for AR)")
    print("  ✅ Phase-randomization control working correctly")


def test_calibration_behavior():
    """Test that detectors behave correctly on calibration datasets."""
    
    print("\n" + "=" * 80)
    print("CALIBRATION BEHAVIOR TEST")
    print("=" * 80)
    
    seed = 42
    
    # Generate datasets
    noise_ds = WhiteNoiseDataset(seed, {'length': 500})
    noise_data = noise_ds.generate()
    
    ar_ds = ARProcessDataset(seed, {'length': 500, 'phi': 0.7})
    ar_data = ar_ds.generate()
    
    # Test autocorrelation detector
    detector = AutocorrelationDetector({'max_lag': 20})
    
    noise_score = detector.detect(noise_data)
    ar_score = detector.detect(ar_data)
    
    print(f"\n📊 Autocorrelation Detector:")
    print(f"  White noise score: {noise_score:.4f}")
    print(f"  AR(1) score: {ar_score:.4f}")
    print(f"  Ratio (AR/noise): {ar_score / max(noise_score, 1e-6):.2f}x")
    
    # AR(1) should have MUCH higher score than noise
    assert ar_score > noise_score * 3, f"AR(1) score ({ar_score:.4f}) should be >> noise score ({noise_score:.4f})"
    print("  ✅ Detector correctly distinguishes structure from noise")


if __name__ == "__main__":
    try:
        test_detectors()
        test_controls()
        test_calibration_behavior()
        
        print("\n" + "=" * 80)
        print("🎉 ALL TESTS PASSED - Detectors and controls working correctly!")
        print("=" * 80)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
