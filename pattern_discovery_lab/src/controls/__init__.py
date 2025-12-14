"""
Control Suite - Mandatory Null Tests

Every detector MUST pass through these controls:
1. Time-shuffle: Random permutation of time indices
2. Phase-randomization: Destroy temporal structure, preserve power spectrum
3. Null-world comparison: Compare to pure noise and finance-null datasets

If structure survives controls incorrectly → detector is miscalibrated.
"""

import numpy as np
from typing import Dict, Any, Callable
from scipy import fft as scipy_fft
from dataclasses import dataclass


def compute_lag1_autocorrelation(data: np.ndarray) -> float:
    """
    Compute lag-1 autocorrelation coefficient with MEAN CENTERING.
    
    Formula: rho1 = sum_{t=1..N-1} (x[t]-mu)*(x[t-1]-mu) / sum_{t=0..N-1} (x[t]-mu)^2
    Uses non-circular lag (no wrap).
    """
    if len(data) < 2:
        return 0.0
    N = len(data)
    mu = np.mean(data)
    x_centered = data - mu
    
    # Numerator: sum of (x[t]-mu)*(x[t-1]-mu) for t=1..N-1
    numerator = np.sum(x_centered[1:] * x_centered[:-1])
    
    # Denominator: sum of (x[t]-mu)^2 for t=0..N-1
    denominator = np.sum(x_centered ** 2)
    
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_shuffle_threshold(N: int, M: int = 100) -> float:
    """
    Compute principled, N-based threshold for shuffle control.
    
    T(N) = 4.5 / sqrt(N)
    
    This is a conservative bound for M up to ~100-1000 shuffles.
    """
    return 4.5 / np.sqrt(N)


@dataclass
class ControlResult:
    """
    Result from a control test.
    
    Attributes:
        control_name: Name of control (time_shuffle, phase_randomization, etc.)
        original_score: Structure score on original data
        control_scores: List of scores on controlled data
        passes: Whether control test passed
        p_value: Statistical significance
        metadata: Additional info
    """
    control_name: str
    original_score: float
    control_scores: list
    passes: bool
    p_value: float
    metadata: Dict[str, Any]


class TimeShuffleControl:
    """
    Time-shuffle control: randomly permute time indices.
    
    Expected behavior:
    - True structure should be DESTROYED by shuffling
    - If structure_score(shuffled) ≈ structure_score(original) → likely spurious
    
    Usage:
        control = TimeShuffleControl(n_shuffles=100)
        result = control.test(detector, data)
        assert result.passes  # Original score should be >> shuffled scores
    """
    
    def __init__(self, n_shuffles: int = 100, seed: int = 42):
        """
        Args:
            n_shuffles: Number of shuffled surrogates to generate
            seed: Random seed for reproducibility
        """
        self.n_shuffles = n_shuffles
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def generate_shuffled(self, data: np.ndarray) -> np.ndarray:
        """
        Generate one shuffled surrogate.
        
        Args:
            data: Original time-series
        
        Returns:
            Shuffled version (same values, random order)
        """
        shuffled = data.copy()
        self.rng.shuffle(shuffled)
        return shuffled
    
    def test(self, detector, data: np.ndarray, alpha: float = 0.01) -> ControlResult:
        """
        Run time-shuffle control test.
        
        Args:
            detector: Detector instance with detect() method
            data: Original time-series
            alpha: Significance level (default 0.01)
        
        Returns:
            ControlResult with pass/fail verdict
        """
        # Score on original data
        original_score = detector.detect(data)
        
        # Compute original rho1 for reference
        original_rho1 = compute_lag1_autocorrelation(data)
        
        # Scores on shuffled surrogates + compute rho1 for each
        shuffled_scores = []
        shuffled_rho1_values = []
        for _ in range(self.n_shuffles):
            shuffled = self.generate_shuffled(data)
            score = detector.detect(shuffled)
            shuffled_scores.append(score)
            shuffled_rho1_values.append(compute_lag1_autocorrelation(shuffled))
        
        # Compute p-value: fraction of shuffled scores >= original score
        # (one-tailed test: original should be HIGHER than shuffled)
        n_greater_equal = sum(s >= original_score for s in shuffled_scores)
        p_value = n_greater_equal / self.n_shuffles
        
        # Pass if original score is significantly higher than shuffled
        passes = p_value < alpha
        
        # Compute mean |rho1| post-shuffle for ENFORCE_NOW check
        mean_abs_rho1_shuffled = np.mean([abs(r) for r in shuffled_rho1_values])
        max_abs_rho1_shuffled = np.max([abs(r) for r in shuffled_rho1_values])
        
        return ControlResult(
            control_name='time_shuffle',
            original_score=original_score,
            control_scores=shuffled_scores,
            passes=passes,
            p_value=p_value,
            metadata={
                'n_shuffles': self.n_shuffles,
                'alpha': alpha,
                'mean_shuffled_score': np.mean(shuffled_scores),
                'std_shuffled_score': np.std(shuffled_scores),
                'original_rho1': float(original_rho1),
                'mean_abs_rho1_shuffled': float(mean_abs_rho1_shuffled),
                'max_abs_rho1_shuffled': float(max_abs_rho1_shuffled),
                'shuffled_rho1_values': [float(r) for r in shuffled_rho1_values]
            }
        )


class PhaseRandomizationControl:
    """
    Phase-randomization surrogate control.
    
    Method (Theiler et al., 1992):
    1. FFT of original data
    2. Randomize phases (keep magnitudes)
    3. Inverse FFT
    
    Result: Surrogate has same power spectrum but destroyed temporal structure.
    
    Expected behavior:
    - True temporal structure should be DESTROYED
    - If structure survives → may be frequency-domain artifact
    """
    
    def __init__(self, n_surrogates: int = 100, seed: int = 42):
        """
        Args:
            n_surrogates: Number of surrogates to generate
            seed: Random seed for reproducibility
        """
        self.n_surrogates = n_surrogates
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def generate_surrogate(self, data: np.ndarray) -> tuple:
        """
        Generate one phase-randomized surrogate.
        
        Args:
            data: Original time-series
        
        Returns:
            Tuple of (surrogate, spectrum_error_mean, spectrum_error_max)
            where spectrum_error is computed comparing SURROGATE spectrum to ORIGINAL spectrum
        """
        n = len(data)
        
        # Mean-center before spectrum to avoid DC domination
        data_centered = data - np.mean(data)
        
        # FFT of original (centered)
        fft_orig = scipy_fft.fft(data_centered)
        
        # Extract magnitudes (power spectrum) - these are what we preserve
        magnitudes_orig = np.abs(fft_orig)
        
        # Generate random phases (uniform on [0, 2π))
        random_phases = self.rng.uniform(0, 2 * np.pi, n)
        
        # For real-valued input, ensure Hermitian symmetry
        # (phases at negative frequencies must be conjugate)
        if n % 2 == 0:
            # Even length
            random_phases[n//2+1:] = -random_phases[1:n//2][::-1]
        else:
            # Odd length
            random_phases[(n+1)//2:] = -random_phases[1:(n+1)//2][::-1]
        
        random_phases[0] = 0  # DC component has zero phase
        if n % 2 == 0:
            random_phases[n//2] = 0  # Nyquist frequency has zero phase
        
        # Reconstruct FFT with randomized phases (using ORIGINAL magnitudes)
        fft_surrogate = magnitudes_orig * np.exp(1j * random_phases)
        
        # Inverse FFT
        surrogate = scipy_fft.ifft(fft_surrogate).real
        
        # Add back mean to surrogate
        surrogate = surrogate + np.mean(data)
        
        # Compute spectrum preservation error:
        # Compare SURROGATE power spectrum to ORIGINAL power spectrum
        # This should NOT be trivially 0 due to numerical precision and phase reconstruction
        surrogate_centered = surrogate - np.mean(surrogate)
        fft_surr = scipy_fft.fft(surrogate_centered)
        magnitudes_surr = np.abs(fft_surr)
        
        # Power spectra (|FFT|^2)
        P_orig = magnitudes_orig ** 2
        P_surr = magnitudes_surr ** 2
        
        # Compute relative error over positive frequencies excluding DC
        # rel_err[k] = |P_orig[k] - P_surr[k]| / max(P_orig[k], eps)
        eps = 1e-10
        n_positive = n // 2  # positive frequencies from index 1 to n//2
        
        rel_errors = []
        for k in range(1, n_positive + 1):
            if k < n:
                rel_err = abs(P_orig[k] - P_surr[k]) / max(P_orig[k], eps)
                rel_errors.append(rel_err)
        
        spectrum_error_mean = float(np.mean(rel_errors)) if rel_errors else 0.0
        spectrum_error_max = float(np.max(rel_errors)) if rel_errors else 0.0
        
        return surrogate, spectrum_error_mean, spectrum_error_max
    
    def test(self, detector, data: np.ndarray, alpha: float = 0.01) -> ControlResult:
        """
        Run phase-randomization control test.
        
        Args:
            detector: Detector instance
            data: Original time-series
            alpha: Significance level
        
        Returns:
            ControlResult with pass/fail verdict
        """
        original_score = detector.detect(data)
        
        surrogate_scores = []
        spectrum_errors = []
        max_spectrum_errors = []
        for _ in range(self.n_surrogates):
            surrogate, spec_err, max_spec_err = self.generate_surrogate(data)
            score = detector.detect(surrogate)
            surrogate_scores.append(score)
            spectrum_errors.append(spec_err)
            max_spectrum_errors.append(max_spec_err)
        
        # P-value using rank-based formula: (k+1)/(N+1)
        N = len(surrogate_scores)
        k = sum(1 for s in surrogate_scores if s >= original_score)
        p_value = (k + 1) / (N + 1)
        
        passes = p_value < alpha
        
        # Compute spectrum error stats
        mean_spectrum_error = float(np.mean(spectrum_errors))
        max_spectrum_error_overall = float(np.max(max_spectrum_errors))
        
        return ControlResult(
            control_name='phase_randomization',
            original_score=original_score,
            control_scores=surrogate_scores,
            passes=passes,
            p_value=p_value,
            metadata={
                'n_surrogates': self.n_surrogates,
                'N': N,
                'k': k,
                'alpha': alpha,
                'mean_surrogate_score': np.mean(surrogate_scores),
                'std_surrogate_score': np.std(surrogate_scores),
                'mean_spectrum_error': mean_spectrum_error,
                'max_spectrum_error': max_spectrum_error_overall,
                'spectrum_errors': spectrum_errors
            }
        )


class NullWorldComparison:
    """
    Null-world comparison control.
    
    Compare detector score on real data to scores on:
    1. Pure noise world (white noise, same length)
    2. Finance-null world (GARCH+jumps, same length)
    
    Expected behavior:
    - Real data score should be SIGNIFICANTLY HIGHER than null worlds
    - If not → "structure" is just noise or finance texture
    """
    
    def __init__(self, n_null_samples: int = 100, seed: int = 42):
        """
        Args:
            n_null_samples: Number of null-world samples to generate
            seed: Random seed
        """
        self.n_null_samples = n_null_samples
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def generate_pure_noise(self, length: int) -> np.ndarray:
        """Generate pure white noise of given length."""
        return self.rng.standard_normal(length)
    
    def generate_finance_null(self, length: int) -> np.ndarray:
        """
        Generate finance-null world (GARCH + fat tails).
        
        Simple GARCH(1,1) with Student-t innovations.
        """
        from scipy import stats
        
        # GARCH params (typical)
        omega = 0.1
        alpha = 0.1
        beta = 0.85
        dof = 5
        
        r = np.zeros(length)
        sigma2 = np.zeros(length)
        sigma2[0] = omega / (1 - alpha - beta)
        
        z = stats.t.rvs(df=dof, size=length, random_state=self.rng)
        z = z / np.sqrt(dof / (dof - 2))
        
        r[0] = np.sqrt(sigma2[0]) * z[0]
        
        for t in range(1, length):
            sigma2[t] = omega + alpha * r[t-1]**2 + beta * sigma2[t-1]
            r[t] = np.sqrt(sigma2[t]) * z[t]
        
        return r
    
    def test(self, detector, data: np.ndarray, alpha: float = 0.01) -> Dict[str, ControlResult]:
        """
        Run null-world comparison tests.
        
        Args:
            detector: Detector instance
            data: Original time-series
            alpha: Significance level
        
        Returns:
            Dict with 'pure_noise' and 'finance_null' ControlResults
        """
        original_score = detector.detect(data)
        length = len(data)
        
        results = {}
        
        # Test 1: Pure noise comparison
        pure_noise_scores = []
        for _ in range(self.n_null_samples):
            noise = self.generate_pure_noise(length)
            score = detector.detect(noise)
            pure_noise_scores.append(score)
        
        n_greater_equal = sum(s >= original_score for s in pure_noise_scores)
        p_value_noise = n_greater_equal / self.n_null_samples
        
        results['pure_noise'] = ControlResult(
            control_name='null_world_pure_noise',
            original_score=original_score,
            control_scores=pure_noise_scores,
            passes=(p_value_noise < alpha),
            p_value=p_value_noise,
            metadata={
                'n_samples': self.n_null_samples,
                'alpha': alpha,
                'mean_null_score': np.mean(pure_noise_scores),
                'std_null_score': np.std(pure_noise_scores)
            }
        )
        
        # Test 2: Finance-null comparison
        finance_null_scores = []
        for _ in range(self.n_null_samples):
            finance_null = self.generate_finance_null(length)
            score = detector.detect(finance_null)
            finance_null_scores.append(score)
        
        n_greater_equal = sum(s >= original_score for s in finance_null_scores)
        p_value_finance = n_greater_equal / self.n_null_samples
        
        results['finance_null'] = ControlResult(
            control_name='null_world_finance_null',
            original_score=original_score,
            control_scores=finance_null_scores,
            passes=(p_value_finance < alpha),
            p_value=p_value_finance,
            metadata={
                'n_samples': self.n_null_samples,
                'alpha': alpha,
                'mean_null_score': np.mean(finance_null_scores),
                'std_null_score': np.std(finance_null_scores)
            }
        )
        
        return results


def run_all_controls(detector, data: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run full control battery on a detector.
    
    Args:
        detector: Detector instance
        data: Time-series data
        config: Control configuration (n_shuffles, n_surrogates, alpha, seed)
    
    Returns:
        Dict with all control results and overall pass/fail
    """
    n_shuffles = config.get('n_shuffles', 100)
    n_surrogates = config.get('n_surrogates', 100)
    alpha = config.get('alpha', 0.01)
    seed = config.get('seed', 42)
    
    # Run controls
    time_shuffle = TimeShuffleControl(n_shuffles=n_shuffles, seed=seed)
    phase_rand = PhaseRandomizationControl(n_surrogates=n_surrogates, seed=seed)
    null_comp = NullWorldComparison(n_null_samples=n_surrogates, seed=seed)
    
    results = {
        'time_shuffle': time_shuffle.test(detector, data, alpha),
        'phase_randomization': phase_rand.test(detector, data, alpha),
        'null_comparison': null_comp.test(detector, data, alpha)
    }
    
    # Overall pass: ALL controls must pass
    all_pass = (
        results['time_shuffle'].passes and
        results['phase_randomization'].passes and
        results['null_comparison']['pure_noise'].passes and
        results['null_comparison']['finance_null'].passes
    )
    
    return {
        'results': results,
        'overall_pass': all_pass,
        'config': config
    }
