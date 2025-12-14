"""
Pattern Discovery Lab - Gates v0.1
Enforcement gates for surrogate testing.
"""
import numpy as np
from typing import Dict, Any, Tuple


# ============================================================================
# GATE A: Shuffle Control (rho1)
# ============================================================================

def compute_rho1(x: np.ndarray) -> float:
    """
    Compute lag-1 autocorrelation (mean-centered, non-circular).
    
    Args:
        x: Time series data
        
    Returns:
        Lag-1 autocorrelation coefficient
    """
    x = np.asarray(x, dtype=float)
    x_centered = x - np.mean(x)
    
    # Non-circular: use first N-1 pairs
    numerator = np.sum(x_centered[:-1] * x_centered[1:])
    denominator = np.sum(x_centered ** 2)
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def gate_a_shuffle_control(data: np.ndarray, M: int = 100, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Gate A: Shuffle Control - Test if shuffled surrogates preserve low autocorrelation.
    
    Threshold: T(N) = 4.5 / sqrt(N)
    
    Args:
        data: Original time series
        M: Number of shuffles
        alpha: Significance level (for reporting)
        
    Returns:
        Dictionary with gate results
    """
    data = np.asarray(data, dtype=float)
    N = len(data)
    
    # Compute threshold
    threshold = 4.5 / np.sqrt(N)
    
    # Generate M shuffles and compute rho1 for each
    rho1_shuffled = []
    for _ in range(M):
        shuffled = np.random.permutation(data)
        rho1 = compute_rho1(shuffled)
        rho1_shuffled.append(rho1)
    
    rho1_shuffled = np.array(rho1_shuffled)
    abs_rho1 = np.abs(rho1_shuffled)
    
    # Statistics
    rho1_shuffled_max = np.max(abs_rho1)
    mean_abs_rho1 = np.mean(abs_rho1)
    p95_abs_rho1 = np.percentile(abs_rho1, 95)
    
    # Z-score equivalent (approximate)
    std_rho1 = np.std(rho1_shuffled)
    z_score_equiv = rho1_shuffled_max / std_rho1 if std_rho1 > 0 else 0.0
    
    # Verdict
    shuffle_check = "PASS" if rho1_shuffled_max <= threshold else "HALT"
    
    return {
        'N': int(N),
        'M': int(M),
        'rho1_shuffled_max': float(rho1_shuffled_max),
        'threshold': float(threshold),
        'shuffle_check': shuffle_check,
        'mean_abs_rho1': float(mean_abs_rho1),
        'p95_abs_rho1': float(p95_abs_rho1),
        'z_score_equiv': float(z_score_equiv),
    }


# ============================================================================
# GATE B: Spectrum Preservation
# ============================================================================

def gate_b_spectrum_error(original: np.ndarray, surrogate: np.ndarray, 
                          threshold: float = 0.05) -> Dict[str, Any]:
    """
    Gate B: Spectrum Preservation - Test if surrogate preserves original spectrum.
    
    Uses rfft to compute power spectrum, excludes DC and Nyquist.
    
    Args:
        original: Original time series
        surrogate: Surrogate time series
        threshold: Maximum allowed relative error (default 0.05 = 5%)
        
    Returns:
        Dictionary with gate results
    """
    original = np.asarray(original, dtype=float)
    surrogate = np.asarray(surrogate, dtype=float)
    
    # Mean-center
    orig_centered = original - np.mean(original)
    surr_centered = surrogate - np.mean(surrogate)
    
    # Compute power spectra using rfft
    fft_orig = np.fft.rfft(orig_centered)
    fft_surr = np.fft.rfft(surr_centered)
    
    power_orig = np.abs(fft_orig) ** 2
    power_surr = np.abs(fft_surr) ** 2
    
    # Exclude DC (index 0) and Nyquist (last element)
    # Use slice [1:N//2] to get positive frequencies excluding DC and Nyquist
    N = len(original)
    freq_slice = slice(1, N // 2)
    
    power_orig_slice = power_orig[freq_slice]
    power_surr_slice = power_surr[freq_slice]
    
    # Compute relative error with epsilon guard
    eps = 1e-10
    rel_errors = np.abs(power_orig_slice - power_surr_slice) / np.maximum(power_orig_slice, eps)
    
    spectrum_error_max = float(np.max(rel_errors))
    spectrum_error_mean = float(np.mean(rel_errors))
    
    # Verdict
    spectrum_check = "PASS" if spectrum_error_max <= threshold else "HALT"
    
    return {
        'spectrum_error_max': spectrum_error_max,
        'spectrum_error_mean': spectrum_error_mean,
        'spectrum_check': spectrum_check,
        'threshold': threshold,
    }


# ============================================================================
# Phase Randomization (for surrogate generation)
# ============================================================================

def generate_phase_surrogate(data: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Generate phase-randomized surrogate preserving power spectrum.
    
    Args:
        data: Original time series
        seed: Random seed for reproducibility
        
    Returns:
        Phase-randomized surrogate
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    data = np.asarray(data, dtype=float)
    N = len(data)
    
    # Mean-center
    data_centered = data - np.mean(data)
    
    # FFT
    fft_data = np.fft.rfft(data_centered)
    
    # Randomize phases (preserve DC and Nyquist)
    magnitudes = np.abs(fft_data)
    
    # Generate random phases
    random_phases = rng.uniform(0, 2 * np.pi, len(fft_data))
    
    # Keep DC phase unchanged
    random_phases[0] = np.angle(fft_data[0])
    
    # For real FFT, if N is even, keep Nyquist phase unchanged
    if N % 2 == 0:
        random_phases[-1] = np.angle(fft_data[-1])
    
    # Construct new FFT with randomized phases
    fft_randomized = magnitudes * np.exp(1j * random_phases)
    
    # Inverse FFT
    surrogate = np.fft.irfft(fft_randomized, n=N)
    
    # Restore original mean
    surrogate += np.mean(data)
    
    return surrogate


# ============================================================================
# Reporting Contract
# ============================================================================

EXPECTED_DECISIONS = {
    'autocorrelation_dependence': {
        'white_noise': 'ACCEPT',
        'ar1_phi_0p9': 'REJECT',
        'garch': 'ACCEPT',
    }
}

CERTIFIED_DETECTORS = {'autocorrelation_dependence'}
EXPERIMENTAL_DETECTORS = {'time_reversal_asymmetry'}


def get_detector_class(detector_name: str) -> str:
    """Return CERTIFIED or EXPERIMENTAL."""
    if detector_name in CERTIFIED_DETECTORS:
        return 'CERTIFIED'
    return 'EXPERIMENTAL'


def get_expected_decision(detector_name: str, dataset_name: str) -> str:
    """Get expected decision for dataset × detector combination."""
    return EXPECTED_DECISIONS.get(detector_name, {}).get(dataset_name, 'N/A')
