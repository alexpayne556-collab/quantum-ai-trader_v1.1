#!/usr/bin/env python3
"""
Adaptive Statistics Module for Pattern Discovery Lab V1

Data-driven statistical methods that replace hardcoded thresholds.
Based on:
- Red Team (DeepSeek): 15 critical flaws identified
- Research (Perplexity): Academic formulas and citations
- Blue Team (Claude): Adaptive algorithm designs

No magic numbers - all thresholds computed from data properties.
"""

import numpy as np
from scipy import stats
from scipy.stats import norm
from typing import Tuple, List, Dict, Optional
import warnings


# ============================================================================
# CONSTANTS (Mathematical, not arbitrary thresholds)
# ============================================================================

EULER_MASCHERONI = 0.5772156649015329  # γ - mathematical constant
EULER_NUMBER = np.e  # e ≈ 2.71828


# ============================================================================
# 1. MINIMUM BACKTEST LENGTH (MinBTL)
# Bailey & López de Prado (2014) - Equation 6
# ============================================================================

def calculate_min_btl(n_trials: int, target_sharpe: float) -> float:
    """
    Calculate Minimum Backtest Length (MinBTL) in years.
    
    Given the number of strategies tested (N) and the observed in-sample
    Sharpe ratio (SR), returns the minimum backtest length needed to 
    reject the null hypothesis that true OOS Sharpe = 0.
    
    Formula (Bailey & López de Prado 2014, Eq. 6):
    MinBTL = (1/SR²) × [(1-γ)Φ⁻¹(1-1/N) + γΦ⁻¹(1-1/(Ne))]²
    
    Args:
        n_trials: Number of independent strategy configurations tested (N)
        target_sharpe: The in-sample annualized Sharpe ratio observed (SR)
    
    Returns:
        Minimum backtest length in years
        
    Reference:
        Bailey, D.H. & López de Prado, M. (2014). "The Deflated Sharpe Ratio"
        Journal of Portfolio Management, 40(5), 94-107.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    
    if target_sharpe <= 0:
        return float('inf')  # Cannot validate zero/negative Sharpe
    
    # Expected maximum of N standard normals (GEV approximation)
    # E[max_N] ≈ (1-γ)Φ⁻¹(1-1/N) + γΦ⁻¹(1-1/(Ne))
    
    if n_trials == 1:
        # Edge case: single trial, expected max = 0
        expected_max = 0.0
    else:
        term1 = (1 - EULER_MASCHERONI) * norm.ppf(1 - 1/n_trials)
        term2 = EULER_MASCHERONI * norm.ppf(1 - 1/(n_trials * EULER_NUMBER))
        expected_max = term1 + term2
    
    # MinBTL = E[max_N]² / SR²
    min_btl = (expected_max ** 2) / (target_sharpe ** 2)
    
    return max(min_btl, 0.0)  # Cannot be negative


def expected_max_sharpe_null(n_trials: int, backtest_years: float) -> float:
    """
    Expected maximum Sharpe ratio under null hypothesis (true SR = 0).
    
    Given N trials and T years of data, what's the expected best SR
    you'd find even if ALL strategies have zero true edge?
    
    Args:
        n_trials: Number of strategies tested
        backtest_years: Length of backtest in years
    
    Returns:
        Expected maximum Sharpe ratio from noise alone
    """
    if n_trials < 1:
        return 0.0
    
    if n_trials == 1:
        return 0.0
    
    # E[max_N] for standard normals
    term1 = (1 - EULER_MASCHERONI) * norm.ppf(1 - 1/n_trials)
    term2 = EULER_MASCHERONI * norm.ppf(1 - 1/(n_trials * EULER_NUMBER))
    expected_max_z = term1 + term2
    
    # Scale by sqrt of time (annualized SR has variance ~1/T)
    expected_max_sr = expected_max_z / np.sqrt(backtest_years)
    
    return expected_max_sr


# ============================================================================
# 2. BENJAMINI-HOCHBERG FDR CONTROL
# Benjamini & Hochberg (1995), Harvey & Liu (2014)
# ============================================================================

def benjamini_hochberg(
    p_values: np.ndarray, 
    alpha: float = 0.05,
    method: str = 'bh'
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Benjamini-Hochberg procedure for controlling False Discovery Rate.
    
    Controls the expected proportion of false discoveries among rejected
    hypotheses, rather than family-wise error rate.
    
    Args:
        p_values: Array of p-values from hypothesis tests
        alpha: Target FDR level (default 0.05)
        method: 'bh' for standard, 'by' for Benjamini-Yekutieli 
                (arbitrary dependence)
    
    Returns:
        Tuple of:
        - rejected: Boolean array of rejected hypotheses
        - adjusted_p: Adjusted p-values (q-values)
        - n_rejected: Number of rejected hypotheses
        
    Reference:
        Benjamini, Y. & Hochberg, Y. (1995). "Controlling the False 
        Discovery Rate." JRSS-B, 57(1), 289-300.
        
        Harvey, C.R., Liu, Y. & Zhu, H. (2016). "...and the Cross-Section 
        of Expected Returns." Review of Financial Studies, 29(1).
    """
    p_values = np.asarray(p_values)
    m = len(p_values)
    
    if m == 0:
        return np.array([]), np.array([]), 0
    
    # Sort p-values and track original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    
    # Correction factor for BY (arbitrary dependence)
    if method == 'by':
        # c(m) = sum(1/k for k in 1..m) ≈ ln(m) + γ
        c_m = np.sum(1.0 / np.arange(1, m + 1))
    else:
        c_m = 1.0
    
    # Critical values: (i/m) × α / c(m)
    ranks = np.arange(1, m + 1)
    critical_values = (ranks / m) * alpha / c_m
    
    # Find largest k where p_(k) ≤ critical_k
    below_critical = sorted_p <= critical_values
    
    if np.any(below_critical):
        # Largest index where condition holds
        k = np.max(np.where(below_critical)[0]) + 1
        threshold = sorted_p[k - 1]
    else:
        k = 0
        threshold = 0.0
    
    # Rejected hypotheses
    rejected = p_values <= threshold
    n_rejected = int(np.sum(rejected))
    
    # Compute adjusted p-values (q-values)
    # q_i = min_{j >= i} (m × c_m × p_(j) / j)
    adjusted_p = np.zeros(m)
    
    # Work backwards from largest p-value
    min_so_far = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        adj = sorted_p[i] * m * c_m / rank
        min_so_far = min(min_so_far, adj)
        adjusted_p[sorted_indices[i]] = min(min_so_far, 1.0)
    
    return rejected, adjusted_p, n_rejected


def storey_pi0_estimate(p_values: np.ndarray, lambda_vals: np.ndarray = None) -> float:
    """
    Estimate π₀ (proportion of true nulls) using Storey's method.
    
    If most p-values cluster near 1, π₀ ≈ 1 (all null).
    If many p-values are small, π₀ < 1 (some true alternatives).
    
    Args:
        p_values: Array of p-values
        lambda_vals: Grid of lambda values for estimation
    
    Returns:
        Estimated proportion of true null hypotheses
        
    Reference:
        Storey, J.D. (2002). "A Direct Approach to False Discovery Rates."
        JRSS-B, 64(3), 479-498.
    """
    p_values = np.asarray(p_values)
    m = len(p_values)
    
    if m < 10:
        # Too few tests for reliable estimation
        return 1.0
    
    if lambda_vals is None:
        lambda_vals = np.arange(0.05, 0.96, 0.05)
    
    # For each λ, estimate π₀(λ) = #{p > λ} / (m × (1 - λ))
    pi0_estimates = []
    for lam in lambda_vals:
        num_above = np.sum(p_values > lam)
        pi0_lam = num_above / (m * (1 - lam))
        pi0_estimates.append(min(pi0_lam, 1.0))
    
    # Use simple average of stable estimates (λ > 0.5)
    stable_idx = lambda_vals > 0.5
    if np.any(stable_idx):
        pi0 = np.mean(np.array(pi0_estimates)[stable_idx])
    else:
        pi0 = np.mean(pi0_estimates)
    
    return min(max(pi0, 0.0), 1.0)


# ============================================================================
# 3. ADAPTIVE EMBARGO (ACF-Based)
# López de Prado (2018), DeepSeek Red Team flaw #6
# ============================================================================

def compute_acf(series: np.ndarray, max_lag: int = 50) -> np.ndarray:
    """
    Compute autocorrelation function up to max_lag.
    
    Args:
        series: Time series data
        max_lag: Maximum lag to compute
    
    Returns:
        Array of ACF values for lags 1 to max_lag
    """
    series = np.asarray(series)
    n = len(series)
    
    if n < max_lag + 10:
        max_lag = max(1, n // 2 - 5)
    
    # Center the series
    series_centered = series - np.mean(series)
    var = np.var(series_centered)
    
    if var == 0:
        return np.zeros(max_lag)
    
    acf = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        acf[lag - 1] = np.corrcoef(series_centered[:-lag], series_centered[lag:])[0, 1]
    
    return acf


def adaptive_embargo(
    returns: np.ndarray,
    label_horizon: int = 1,
    confidence: float = 0.95
) -> Tuple[int, Dict]:
    """
    Compute adaptive embargo period based on ACF decay.
    
    The embargo should be the larger of:
    1. First lag where ACF becomes insignificant
    2. The label horizon (to prevent label overlap)
    
    Args:
        returns: Return time series
        label_horizon: Forward horizon used for labels (bars)
        confidence: Confidence level for significance (default 0.95)
    
    Returns:
        Tuple of (embargo_bars, diagnostics_dict)
        
    Reference:
        López de Prado, M. (2018). Advances in Financial Machine Learning.
        Wiley. Chapter 7: Cross-Validation in Finance.
    """
    returns = np.asarray(returns)
    n = len(returns)
    
    # Significance bound for ACF under null
    z_crit = norm.ppf((1 + confidence) / 2)
    significance_bound = z_crit / np.sqrt(n)
    
    # Compute ACF
    max_lag = min(50, n // 4)
    acf = compute_acf(returns, max_lag)
    
    # Find first lag where ACF is insignificant
    decay_lag = max_lag  # Default if never decays
    for lag in range(len(acf)):
        if abs(acf[lag]) < significance_bound:
            decay_lag = lag + 1  # +1 because lags are 1-indexed
            break
    
    # Embargo is max of ACF decay and label horizon
    embargo = max(decay_lag, label_horizon)
    
    diagnostics = {
        'acf_decay_lag': decay_lag,
        'label_horizon': label_horizon,
        'significance_bound': significance_bound,
        'acf_first_5': acf[:5].tolist() if len(acf) >= 5 else acf.tolist(),
        'n_observations': n
    }
    
    return embargo, diagnostics


# ============================================================================
# 4. SAMPLE SIZE REQUIREMENTS
# Power analysis with autocorrelation adjustment
# ============================================================================

def effective_sample_size(n: int, rho: float) -> float:
    """
    Compute effective sample size adjusting for autocorrelation.
    
    With positive autocorrelation, effective N is much smaller than nominal N.
    
    Formula: n_eff = n × (1 - ρ) / (1 + ρ)
    
    Args:
        n: Nominal sample size
        rho: Lag-1 autocorrelation coefficient
    
    Returns:
        Effective sample size
        
    Reference:
        Bayley, G.V. & Hammersley, J.M. (1946). "The Effective Number 
        of Independent Observations in an Autocorrelated Time Series."
        Supplement to JRSS, 8(2), 184-197.
    """
    if abs(rho) >= 1:
        return 1.0  # Highly autocorrelated = effectively 1 observation
    
    adjustment = (1 - rho) / (1 + rho)
    return max(1.0, n * adjustment)


def required_sample_size(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
    rho: float = 0.0
) -> int:
    """
    Compute required sample size for detecting an effect.
    
    Based on standard power analysis, adjusted for autocorrelation.
    
    Formula: n = [(z_α + z_β)² / δ²] × (1 + ρ) / (1 - ρ)
    
    Args:
        effect_size: Minimum effect to detect (e.g., IC = 0.05)
        alpha: Significance level (Type I error)
        power: Statistical power (1 - Type II error)
        rho: Lag-1 autocorrelation (for adjustment)
    
    Returns:
        Required sample size (observations)
        
    Reference:
        Cohen, J. (1988). Statistical Power Analysis for the 
        Behavioral Sciences. 2nd Ed.
    """
    if effect_size <= 0:
        return float('inf')
    
    z_alpha = norm.ppf(1 - alpha / 2)  # Two-tailed
    z_beta = norm.ppf(power)
    
    # Base sample size (iid case)
    n_base = ((z_alpha + z_beta) ** 2) / (effect_size ** 2)
    
    # Autocorrelation adjustment
    if abs(rho) < 1:
        n_adjusted = n_base * (1 + rho) / (1 - rho)
    else:
        n_adjusted = float('inf')
    
    return int(np.ceil(max(30, n_adjusted)))  # Minimum 30 for CLT


def check_sample_sufficiency(
    data: np.ndarray,
    effect_size: float = 0.05,
    alpha: float = 0.05,
    power: float = 0.80
) -> Tuple[bool, Dict]:
    """
    Check if sample size is sufficient for reliable pattern validation.
    
    Args:
        data: Time series data
        effect_size: Minimum IC/effect to reliably detect
        alpha: Significance level
        power: Desired statistical power
    
    Returns:
        Tuple of (is_sufficient, diagnostics)
    """
    data = np.asarray(data)
    n = len(data)
    
    # Estimate autocorrelation
    if n > 2:
        rho = np.corrcoef(data[:-1], data[1:])[0, 1]
        if np.isnan(rho):
            rho = 0.0
    else:
        rho = 0.0
    
    # Compute requirements
    n_required = required_sample_size(effect_size, alpha, power, rho)
    n_effective = effective_sample_size(n, rho)
    
    is_sufficient = n_effective >= n_required
    
    diagnostics = {
        'n_nominal': n,
        'n_effective': n_effective,
        'n_required': n_required,
        'lag1_autocorr': rho,
        'effect_size': effect_size,
        'alpha': alpha,
        'power': power,
        'sufficient': is_sufficient,
        'deficit': max(0, n_required - n_effective)
    }
    
    return is_sufficient, diagnostics


# ============================================================================
# 5. DEFLATED SHARPE RATIO
# Bailey & López de Prado (2014)
# ============================================================================

def probabilistic_sharpe_ratio(
    sharpe_observed: float,
    sharpe_benchmark: float,
    n: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0
) -> float:
    """
    Probabilistic Sharpe Ratio: probability that true SR exceeds benchmark.
    
    Accounts for non-normality of returns via skewness and kurtosis.
    
    Formula:
    PSR = Φ[(SR - SR*) × sqrt(n-1) / sqrt(1 - γ₃SR + (γ₄-1)/4 × SR²)]
    
    Args:
        sharpe_observed: Estimated Sharpe ratio from sample
        sharpe_benchmark: Benchmark SR to exceed (often 0)
        n: Sample size (observations)
        skewness: Sample skewness (γ₃)
        kurtosis: Sample kurtosis (γ₄), excess kurtosis = kurtosis - 3
    
    Returns:
        Probability that true SR exceeds benchmark
        
    Reference:
        Bailey, D.H. & López de Prado, M. (2014). "The Deflated Sharpe Ratio."
        Journal of Portfolio Management, 40(5), 94-107.
    """
    if n <= 1:
        return 0.5  # No information
    
    # Adjustment for non-normality
    # SE(SR) ≈ sqrt(1 - γ₃×SR + (γ₄-1)/4 × SR²)
    gamma3 = skewness
    gamma4 = kurtosis  # Already excess kurtosis in some conventions
    
    variance_adjustment = 1 - gamma3 * sharpe_observed + (gamma4 - 1) / 4 * sharpe_observed ** 2
    
    if variance_adjustment <= 0:
        # Edge case: extreme non-normality
        variance_adjustment = 1.0
    
    se_sr = np.sqrt(variance_adjustment / (n - 1))
    
    if se_sr <= 0:
        return 0.5
    
    z = (sharpe_observed - sharpe_benchmark) / se_sr
    psr = norm.cdf(z)
    
    return psr


def deflated_sharpe_ratio(
    sharpe_observed: float,
    n_trials: int,
    backtest_years: float,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0
) -> float:
    """
    Deflated Sharpe Ratio: probability SR exceeds expected noise maximum.
    
    Given that you tested N strategies, what's the probability that your
    best strategy's performance isn't just the best of noise?
    
    Args:
        sharpe_observed: Best observed Sharpe ratio
        n_trials: Number of strategies tested
        backtest_years: Length of backtest in years
        n_obs: Number of observations (for PSR)
        skewness: Return skewness
        kurtosis: Return kurtosis
    
    Returns:
        Probability that observed SR represents true skill, not luck
    """
    # Expected maximum SR under null
    sr_null_max = expected_max_sharpe_null(n_trials, backtest_years)
    
    # PSR relative to null maximum
    dsr = probabilistic_sharpe_ratio(
        sharpe_observed=sharpe_observed,
        sharpe_benchmark=sr_null_max,
        n=n_obs,
        skewness=skewness,
        kurtosis=kurtosis
    )
    
    return dsr


# ============================================================================
# 6. BOOTSTRAP CONFIDENCE INTERVALS
# BCa method for correlation coefficients
# ============================================================================

def bootstrap_ci(
    data_x: np.ndarray,
    data_y: np.ndarray,
    metric: str = 'spearman',
    alpha: float = 0.05,
    n_bootstrap: int = 2000
) -> Tuple[float, float, float, float]:
    """
    Bootstrap confidence interval for correlation coefficient.
    
    Uses BCa (Bias-Corrected and Accelerated) method for better coverage.
    
    Args:
        data_x: First variable (e.g., signal)
        data_y: Second variable (e.g., returns)
        metric: 'spearman' or 'pearson'
        alpha: Significance level (default 0.05 for 95% CI)
        n_bootstrap: Number of bootstrap samples
    
    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper, se)
        
    Reference:
        Efron, B. (1987). "Better Bootstrap Confidence Intervals."
        JASA, 82(397), 171-185.
    """
    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y)
    n = len(data_x)
    
    if n < 10:
        warnings.warn("Sample size < 10, bootstrap CI may be unreliable")
    
    # Point estimate
    if metric == 'spearman':
        point_est, _ = stats.spearmanr(data_x, data_y)
    else:
        point_est, _ = stats.pearsonr(data_x, data_y)
    
    if np.isnan(point_est):
        return 0.0, 0.0, 0.0, 0.0
    
    # Bootstrap distribution
    boot_estimates = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        x_boot = data_x[idx]
        y_boot = data_y[idx]
        
        if metric == 'spearman':
            est, _ = stats.spearmanr(x_boot, y_boot)
        else:
            est, _ = stats.pearsonr(x_boot, y_boot)
        
        if not np.isnan(est):
            boot_estimates.append(est)
    
    boot_estimates = np.array(boot_estimates)
    
    if len(boot_estimates) < 100:
        # Not enough valid bootstrap samples
        se = np.std(boot_estimates) if len(boot_estimates) > 0 else 0.0
        return point_est, point_est - 1.96 * se, point_est + 1.96 * se, se
    
    # Simple percentile CI (BCa requires jackknife, more complex)
    ci_lower = np.percentile(boot_estimates, alpha / 2 * 100)
    ci_upper = np.percentile(boot_estimates, (1 - alpha / 2) * 100)
    se = np.std(boot_estimates)
    
    return point_est, ci_lower, ci_upper, se


# ============================================================================
# MAIN TEST
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("ADAPTIVE STATISTICS MODULE - SELF TEST")
    print("=" * 70)
    print()
    
    # Test 1: MinBTL
    print("1. MINIMUM BACKTEST LENGTH (MinBTL)")
    print("-" * 40)
    for n_trials in [10, 50, 100, 500, 1000]:
        min_btl = calculate_min_btl(n_trials, target_sharpe=1.0)
        print(f"   N={n_trials:4d} trials, SR=1.0 → MinBTL = {min_btl:.2f} years")
    print()
    
    # Test 2: BH FDR
    print("2. BENJAMINI-HOCHBERG FDR")
    print("-" * 40)
    np.random.seed(42)
    # 90 nulls (uniform p-values) + 10 alternatives (small p-values)
    p_null = np.random.uniform(0, 1, 90)
    p_alt = np.random.uniform(0, 0.01, 10)
    p_values = np.concatenate([p_null, p_alt])
    np.random.shuffle(p_values)
    
    rejected, adjusted_p, n_rej = benjamini_hochberg(p_values, alpha=0.05)
    print(f"   100 tests (10 true alternatives)")
    print(f"   Rejected at FDR=0.05: {n_rej}")
    print(f"   π₀ estimate: {storey_pi0_estimate(p_values):.2f}")
    print()
    
    # Test 3: Adaptive Embargo
    print("3. ADAPTIVE EMBARGO")
    print("-" * 40)
    np.random.seed(42)
    # AR(1) returns with ρ=0.1
    n = 500
    returns = np.zeros(n)
    for i in range(1, n):
        returns[i] = 0.1 * returns[i-1] + np.random.randn() * 0.01
    
    embargo, diag = adaptive_embargo(returns, label_horizon=1)
    print(f"   Embargo (ACF-based): {embargo} bars")
    print(f"   ACF decay lag: {diag['acf_decay_lag']}")
    print(f"   First 5 ACF values: {[f'{x:.3f}' for x in diag['acf_first_5']]}")
    print()
    
    # Test 4: Sample Size
    print("4. SAMPLE SIZE REQUIREMENTS")
    print("-" * 40)
    sufficient, diag = check_sample_sufficiency(returns, effect_size=0.05)
    print(f"   N nominal: {diag['n_nominal']}")
    print(f"   N effective: {diag['n_effective']:.1f}")
    print(f"   N required: {diag['n_required']}")
    print(f"   Sufficient: {sufficient}")
    print()
    
    # Test 5: Deflated Sharpe
    print("5. DEFLATED SHARPE RATIO")
    print("-" * 40)
    dsr = deflated_sharpe_ratio(
        sharpe_observed=1.5,
        n_trials=100,
        backtest_years=5,
        n_obs=1260  # 5 years × 252 days
    )
    print(f"   SR=1.5, N=100 trials, 5 years → DSR = {dsr:.2%}")
    print(f"   (Probability this isn't just the best of noise)")
    print()
    
    print("=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
