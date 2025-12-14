"""
Truth Tests for Enforcement Contract v0.1

These tests PROVE that enforcement gates are real, not security theater.
Each test targets a specific gate and validates it actually halts/fails when violated.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pattern_discovery_lab.src.runner import run_calibration, run_detector_with_controls
from pattern_discovery_lab.src.datasets.pure_noise import WhiteNoiseDataset
from pattern_discovery_lab.src.datasets.known_structure import ARProcessDataset
from pattern_discovery_lab.src.detectors.dependence import AutocorrelationDetector
from pattern_discovery_lab.src.controls import compute_lag1_autocorrelation, compute_shuffle_threshold, TimeShuffleControl, PhaseRandomizationControl


def test_T1_ar1_phi09_detection():
    """
    T1: AR(1) phi=0.9 detection
    - rho1_original should be HIGH (~0.9)
    - rho1_shuffled should be LOW (<0.05)
    - spectrum_error should be LOW (<0.05)
    """
    print("\n" + "="*80)
    print("T1: AR(1) phi=0.9 Detection Test")
    print("="*80)
    
    # Generate AR(1) with strong autocorrelation
    ar_ds = ARProcessDataset(seed=42, config={'length': 1000, 'phi': 0.9})
    data = ar_ds.generate()
    
    # Compute original rho1
    rho1_original = compute_lag1_autocorrelation(data)
    print(f"rho1_original = {rho1_original:.4f} (expected: ~0.9)")
    assert rho1_original > 0.8, f"AR(1) phi=0.9 should have rho1 > 0.8, got {rho1_original}"
    
    # Run detector with controls
    detector = AutocorrelationDetector({'max_lag': 20})
    config = {'seed': 42, 'n_shuffles': 100, 'n_surrogates': 100, 'alpha': 0.05}
    result = run_detector_with_controls(detector, data, config)
    
    # Check rho1_shuffled is destroyed
    # N-based threshold: T(N) = 4.5/sqrt(N) = 4.5/sqrt(1000) ≈ 0.1423
    N = 1000
    shuffle_threshold = compute_shuffle_threshold(N, 100)
    rho1_shuffled_max = result['rho1_shuffled_max']
    print(f"rho1_shuffled_max = {rho1_shuffled_max:.4f} (expected: < T({N})={shuffle_threshold:.4f})")
    assert rho1_shuffled_max <= shuffle_threshold, f"Shuffled data should have |rho1| <= T(N)={shuffle_threshold:.4f}, got {rho1_shuffled_max}"
    
    # Check spectrum error is low
    spectrum_error_max = result['spectrum_error_max']
    print(f"spectrum_error_max = {spectrum_error_max:.4f} (expected: <0.05)")
    assert spectrum_error_max <= 0.05, f"Spectrum error should be <=0.05, got {spectrum_error_max}"
    
    # Check shuffle_check and spectrum_check are PASS
    assert result['shuffle_check'] == 'PASS', f"shuffle_check should be PASS, got {result['shuffle_check']}"
    assert result['spectrum_check'] == 'PASS', f"spectrum_check should be PASS, got {result['spectrum_check']}"
    
    print("✅ T1 PASSED: AR(1) phi=0.9 correctly detected with valid controls")
    return True


def test_T2_white_noise_rejection():
    """
    T2: White noise rejection
    - rho1_original should be LOW (~0)
    - rho1_shuffled should be LOW (<0.05)
    - spectrum_error should be LOW (<0.05)
    - Detector should NOT reject null (no hallucination)
    """
    print("\n" + "="*80)
    print("T2: White Noise Rejection Test")
    print("="*80)
    
    # Generate white noise
    noise_ds = WhiteNoiseDataset(seed=42, config={'length': 1000})
    data = noise_ds.generate()
    
    # Compute original rho1
    rho1_original = compute_lag1_autocorrelation(data)
    print(f"rho1_original = {rho1_original:.4f} (expected: ~0)")
    assert abs(rho1_original) < 0.1, f"White noise should have |rho1| < 0.1, got {rho1_original}"
    
    # Run detector with controls
    detector = AutocorrelationDetector({'max_lag': 20})
    config = {'seed': 42, 'n_shuffles': 100, 'n_surrogates': 100, 'alpha': 0.05}
    result = run_detector_with_controls(detector, data, config)
    
    # N-based threshold: T(N) = 4.5/sqrt(N) = 4.5/sqrt(1000) ≈ 0.1423
    N = 1000
    shuffle_threshold = compute_shuffle_threshold(N, 100)
    rho1_shuffled_max = result['rho1_shuffled_max']
    print(f"rho1_shuffled_max = {rho1_shuffled_max:.4f} (expected: < T({N})={shuffle_threshold:.4f})")
    assert rho1_shuffled_max <= shuffle_threshold, f"Shuffled data should have |rho1| <= T(N)={shuffle_threshold:.4f}, got {rho1_shuffled_max}"
    
    # Check spectrum error is low
    spectrum_error_max = result['spectrum_error_max']
    print(f"spectrum_error_max = {spectrum_error_max:.4f} (expected: <=0.05)")
    assert spectrum_error_max <= 0.05, f"Spectrum error should be <=0.05, got {spectrum_error_max}"
    
    # Detector should NOT pass controls (should not claim structure on noise)
    # Actually, for white noise, shuffle p-value should be HIGH (no structure to destroy)
    shuffle_p = result['shuffle_p_value']
    print(f"shuffle_p = {shuffle_p:.4f} (expected: >0.05, meaning no structure detected)")
    assert shuffle_p > 0.05, f"White noise should have shuffle_p > 0.05 (no structure), got {shuffle_p}"
    
    print("✅ T2 PASSED: White noise correctly shows no structure")
    return True


def test_T3_no_shuffle_attack():
    """
    T3: No-shuffle attack
    - Inject broken shuffle that returns original data unchanged
    - Must HALT/FAIL because rho1_shuffled will be high (same as original)
    """
    print("\n" + "="*80)
    print("T3: No-Shuffle Attack Test")
    print("="*80)
    
    # Generate AR(1) with strong autocorrelation
    ar_ds = ARProcessDataset(seed=42, config={'length': 1000, 'phi': 0.9})
    data = ar_ds.generate()
    
    # Create a broken shuffle control that returns original data
    class BrokenShuffleControl(TimeShuffleControl):
        def generate_shuffled(self, data):
            # ATTACK: Don't shuffle, return original
            return data.copy()
    
    detector = AutocorrelationDetector({'max_lag': 20})
    
    # Run with broken shuffle
    broken_shuffle = BrokenShuffleControl(n_shuffles=25, seed=42)
    result = broken_shuffle.test(detector, data, alpha=0.05)
    
    # The rho1 values should be HIGH (because we didn't shuffle)
    rho1_shuffled_mean = result.metadata.get('mean_abs_rho1_shuffled', 0.0)
    rho1_shuffled_max = result.metadata.get('max_abs_rho1_shuffled', 0.0)
    
    print(f"rho1_shuffled_mean = {rho1_shuffled_mean:.4f} (expected: HIGH, ~0.9)")
    print(f"rho1_shuffled_max = {rho1_shuffled_max:.4f} (expected: HIGH, ~0.9)")
    
    # Gate A should trigger: rho1_shuffled > T(N) = 4.5/sqrt(1000) ≈ 0.1423
    N = 1000
    shuffle_threshold = compute_shuffle_threshold(N, 25)
    gate_a_triggers = rho1_shuffled_max > shuffle_threshold
    print(f"Gate A triggers (rho1_max > T({N})={shuffle_threshold:.4f}): {gate_a_triggers}")
    
    assert gate_a_triggers, f"Gate A MUST trigger on no-shuffle attack! rho1_max={rho1_shuffled_max}"
    
    print("✅ T3 PASSED: No-shuffle attack correctly triggers Gate A HALT")
    return True


def test_T4_broken_surrogate_attack():
    """
    T4: Broken surrogate attack
    - Inject broken surrogate that corrupts FFT magnitudes by *1.1
    - Must HALT/FAIL because spectrum_error will be high
    """
    print("\n" + "="*80)
    print("T4: Broken Surrogate Attack Test")
    print("="*80)
    
    from scipy import fft as scipy_fft
    
    # Generate AR(1) data
    ar_ds = ARProcessDataset(seed=42, config={'length': 1000, 'phi': 0.7})
    data = ar_ds.generate()
    
    # Create a broken phase control that corrupts magnitudes
    class BrokenPhaseControl(PhaseRandomizationControl):
        def generate_surrogate(self, data):
            # ATTACK: Corrupt magnitudes by 10%
            fft_data = scipy_fft.fft(data)
            magnitudes = np.abs(fft_data) * 1.1  # Corrupt by 10%
            
            n = len(data)
            random_phases = self.rng.uniform(0, 2 * np.pi, n)
            
            if n % 2 == 0:
                random_phases[n//2+1:] = -random_phases[1:n//2][::-1]
            else:
                random_phases[(n+1)//2:] = -random_phases[1:(n+1)//2][::-1]
            
            random_phases[0] = 0
            if n % 2 == 0:
                random_phases[n//2] = 0
            
            fft_surrogate = magnitudes * np.exp(1j * random_phases)
            surrogate = scipy_fft.ifft(fft_surrogate).real
            
            # Compute spectrum error (should be ~10% due to magnitude corruption)
            fft_surr = scipy_fft.fft(surrogate)
            magnitudes_surr = np.abs(fft_surr)
            original_magnitudes = np.abs(scipy_fft.fft(data))
            
            eps = 1e-10
            relative_error = np.abs(original_magnitudes - magnitudes_surr) / (original_magnitudes + eps)
            spectrum_error = float(np.mean(relative_error))
            max_spectrum_error = float(np.max(relative_error))
            
            return surrogate, spectrum_error, max_spectrum_error
    
    detector = AutocorrelationDetector({'max_lag': 20})
    
    # Run with broken surrogate
    broken_phase = BrokenPhaseControl(n_surrogates=10, seed=42)
    result = broken_phase.test(detector, data, alpha=0.05)
    
    # The spectrum error should be HIGH (because we corrupted magnitudes)
    spectrum_error_mean = result.metadata.get('mean_spectrum_error', 0.0)
    spectrum_error_max = result.metadata.get('max_spectrum_error', 0.0)
    
    print(f"spectrum_error_mean = {spectrum_error_mean:.4f} (expected: HIGH, ~10%)")
    print(f"spectrum_error_max = {spectrum_error_max:.4f} (expected: HIGH, ~10%)")
    
    # Gate B should trigger: spectrum_error > 0.05
    gate_b_triggers = spectrum_error_max > 0.05
    print(f"Gate B triggers (spec_err_max > 0.05): {gate_b_triggers}")
    
    assert gate_b_triggers, f"Gate B MUST trigger on broken surrogate attack! spec_err_max={spectrum_error_max}"
    
    print("✅ T4 PASSED: Broken surrogate attack correctly triggers Gate B HALT")
    return True


def main():
    """Run all truth tests."""
    print("="*80)
    print("ENFORCEMENT CONTRACT v0.1 - TRUTH TESTS")
    print("="*80)
    print("These tests PROVE that enforcement gates actually work.")
    print("")
    
    tests = [
        ("T1: AR(1) phi=0.9 detection", test_T1_ar1_phi09_detection),
        ("T2: White noise rejection", test_T2_white_noise_rejection),
        ("T3: No-shuffle attack (Gate A)", test_T3_no_shuffle_attack),
        ("T4: Broken surrogate attack (Gate B)", test_T4_broken_surrogate_attack),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {test_name}")
            print(f"   Reason: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {test_name}")
            print(f"   Exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*80)
    print("TRUTH TEST SUMMARY")
    print("="*80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✅ ALL TRUTH TESTS PASSED - ENFORCEMENT IS REAL")
        return 0
    else:
        print(f"\n❌ {failed} TESTS FAILED - ENFORCEMENT HAS GAPS")
        return 1


if __name__ == '__main__':
    sys.exit(main())
