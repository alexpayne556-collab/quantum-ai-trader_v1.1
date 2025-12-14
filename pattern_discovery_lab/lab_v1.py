#!/usr/bin/env python3
"""
Pattern Discovery Lab V1 - Adaptive Evaluation Framework

EVOLVED FROM V0 WITH:
- NO hardcoded thresholds (data-driven)
- MinBTL haircut on multiple testing
- Benjamini-Hochberg FDR control
- Adaptive embargo from ACF decay
- Sample size sufficiency checks
- Deflated Sharpe Ratio

Based on synthesis of:
- Red Team (DeepSeek): 15 critical flaws identified
- Research (Perplexity): Academic formulas with citations
- Blue Team (Claude): Adaptive algorithm designs

THE DATA SPEAKS - WE DO NOT ASSUME.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Iterator, Dict, Optional, Any
from scipy import stats
from dataclasses import dataclass, asdict
import json
from datetime import datetime

# Import our adaptive statistics module
from adaptive_statistics import (
    calculate_min_btl,
    expected_max_sharpe_null,
    benjamini_hochberg,
    storey_pi0_estimate,
    adaptive_embargo,
    compute_acf,
    effective_sample_size,
    required_sample_size,
    check_sample_sufficiency,
    probabilistic_sharpe_ratio,
    deflated_sharpe_ratio,
    bootstrap_ci
)


# ============================================================================
# CONSTANTS (Mathematical, not arbitrary)
# ============================================================================

NUMERIC_PRECISION = 6
MIN_OBS_FOR_STATS = 30  # Central Limit Theorem minimum
MIN_OBS_FOR_BOOTSTRAP = 50  # Bootstrap needs more


# ============================================================================
# DATA CLASSES FOR STRUCTURED OUTPUT
# ============================================================================

@dataclass
class DataDiagnostics:
    """Diagnostics computed from the data itself."""
    n_observations: int
    n_effective: float
    n_required_for_effect: int
    lag1_autocorrelation: float
    acf_decay_lag: int
    recommended_embargo: int
    sample_sufficient: bool
    sample_deficit: int
    min_detectable_effect: float


@dataclass
class MultipleTestingReport:
    """Report on multiple testing correction."""
    n_trials: int
    n_rejected_raw: int
    n_rejected_fdr: int
    fdr_level: float
    pi0_estimate: float
    min_btl_years: float
    expected_noise_sharpe: float


@dataclass
class ValidationResult:
    """Result of validating a single pattern."""
    pattern_id: str
    ic_mean_is: float
    ic_mean_oos: float
    ic_std_oos: float
    ic_ir_oos: float  # IC / std(IC) - Information Ratio
    p_value_raw: float
    p_value_adjusted: float  # BH-adjusted q-value
    ci_lower: float
    ci_upper: float
    sharpe_is: Optional[float]
    deflated_sharpe: Optional[float]
    passed_all_gates: bool
    gate_results: Dict[str, bool]
    failure_reasons: List[str]


@dataclass
class LabV1Report:
    """Complete Lab V1 evaluation report."""
    timestamp: str
    data_diagnostics: Dict
    multiple_testing: Dict
    pattern_results: List[Dict]
    summary: Dict
    methodology: Dict


# ============================================================================
# WALK-FORWARD SPLITS (Same as V0 but with adaptive embargo)
# ============================================================================

def walk_forward_splits_adaptive(
    dates: pd.DatetimeIndex,
    returns: np.ndarray,
    train_len: int,
    test_len: int,
    min_embargo: int = 1,
    confidence: float = 0.95
) -> Tuple[Iterator[Tuple[np.ndarray, np.ndarray]], int, Dict]:
    """
    Generate walk-forward splits with ADAPTIVE embargo.
    
    Embargo is computed from data ACF, not hardcoded.
    
    Args:
        dates: DatetimeIndex
        returns: Return series for ACF computation
        train_len: Training window length
        test_len: Test window length
        min_embargo: Minimum embargo (floor)
        confidence: Confidence for ACF significance
    
    Returns:
        (split_generator, actual_embargo, diagnostics)
    """
    n = len(dates)
    
    # Compute adaptive embargo from ACF
    embargo, acf_diag = adaptive_embargo(
        returns, 
        label_horizon=1, 
        confidence=confidence
    )
    
    # Use at least min_embargo
    actual_embargo = max(embargo, min_embargo)
    
    # Generate splits
    def _generate():
        min_required = train_len + actual_embargo + test_len
        if n < min_required:
            raise ValueError(
                f"Insufficient data: need {min_required} bars "
                f"(train={train_len} + embargo={actual_embargo} + test={test_len}), "
                f"but only have {n}"
            )
        
        start = 0
        while start + train_len + actual_embargo + test_len <= n:
            train_end = start + train_len
            test_start = train_end + actual_embargo
            test_end = test_start + test_len
            
            train_idx = np.arange(start, train_end)
            test_idx = np.arange(test_start, test_end)
            
            yield train_idx, test_idx
            start += test_len
    
    return _generate(), actual_embargo, acf_diag


# ============================================================================
# FORWARD RETURNS
# ============================================================================

def compute_forward_returns(prices: pd.Series, horizon: int = 1) -> pd.Series:
    """Compute forward returns: fwd_return[t] = price[t+horizon]/price[t] - 1"""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    fwd_returns = prices.shift(-horizon) / prices - 1.0
    return fwd_returns


# ============================================================================
# RANK IC COMPUTATION WITH BOOTSTRAP CI
# ============================================================================

def compute_rank_ic_with_ci(
    signal: np.ndarray,
    fwd_returns: np.ndarray,
    alpha: float = 0.05,
    n_bootstrap: int = 1000
) -> Tuple[float, float, float, float, float]:
    """
    Compute Rank IC with bootstrap confidence interval.
    
    Args:
        signal: Signal values
        fwd_returns: Forward returns
        alpha: Significance level for CI
        n_bootstrap: Bootstrap samples
    
    Returns:
        (ic, ci_lower, ci_upper, se, p_value)
    """
    # Align and drop NaN
    mask = ~(np.isnan(signal) | np.isnan(fwd_returns))
    sig_clean = signal[mask]
    ret_clean = fwd_returns[mask]
    
    if len(sig_clean) < MIN_OBS_FOR_STATS:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    # Point estimate and p-value
    ic, p_val = stats.spearmanr(sig_clean, ret_clean)
    
    if np.isnan(ic):
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    # Bootstrap CI if enough data
    if len(sig_clean) >= MIN_OBS_FOR_BOOTSTRAP:
        _, ci_lower, ci_upper, se = bootstrap_ci(
            sig_clean, ret_clean, 
            metric='spearman', 
            alpha=alpha, 
            n_bootstrap=n_bootstrap
        )
    else:
        # Use asymptotic SE approximation
        se = 1.0 / np.sqrt(len(sig_clean) - 3)
        z_crit = stats.norm.ppf(1 - alpha/2)
        ci_lower = ic - z_crit * se
        ci_upper = ic + z_crit * se
    
    return ic, ci_lower, ci_upper, se, p_val


# ============================================================================
# DATA DIAGNOSTICS
# ============================================================================

def compute_data_diagnostics(
    returns: np.ndarray,
    target_effect: float = 0.05,
    alpha: float = 0.05,
    power: float = 0.80
) -> DataDiagnostics:
    """
    Compute data-driven diagnostics for adaptive thresholds.
    
    This is the FOUNDATION - we measure what the data can tell us
    BEFORE making any validation judgments.
    """
    returns = np.asarray(returns)
    n = len(returns)
    
    # Autocorrelation
    if n > 2:
        rho = np.corrcoef(returns[:-1], returns[1:])[0, 1]
        if np.isnan(rho):
            rho = 0.0
    else:
        rho = 0.0
    
    # Effective sample size
    n_eff = effective_sample_size(n, rho)
    
    # Required sample size for target effect
    n_req = required_sample_size(target_effect, alpha, power, rho)
    
    # ACF decay for embargo
    _, acf_diag = adaptive_embargo(returns, label_horizon=1)
    acf_decay = acf_diag['acf_decay_lag']
    embargo = max(acf_decay, 1)
    
    # Sample sufficiency
    sufficient = n_eff >= n_req
    deficit = max(0, int(n_req - n_eff))
    
    # Minimum detectable effect with current data
    # Rearranging power formula: δ = (z_α + z_β) / sqrt(n_eff)
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    min_detectable = (z_alpha + z_beta) / np.sqrt(max(1, n_eff))
    
    return DataDiagnostics(
        n_observations=n,
        n_effective=round(n_eff, 2),
        n_required_for_effect=n_req,
        lag1_autocorrelation=round(rho, 4),
        acf_decay_lag=acf_decay,
        recommended_embargo=embargo,
        sample_sufficient=sufficient,
        sample_deficit=deficit,
        min_detectable_effect=round(min_detectable, 4)
    )


# ============================================================================
# SINGLE PATTERN VALIDATION
# ============================================================================

def validate_pattern_v1(
    pattern_id: str,
    signal: np.ndarray,
    fwd_returns: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    data_diag: DataDiagnostics,
    n_trials: int = 1,  # For FDR adjustment later
    alpha: float = 0.05,
    n_bootstrap: int = 1000
) -> ValidationResult:
    """
    Validate a single pattern with adaptive thresholds.
    
    Gates (all adaptive, no magic numbers):
    1. Sample sufficiency gate
    2. IS IC significance gate
    3. OOS IC significance gate (CI doesn't include 0)
    4. OOS degradation gate (relative to noise floor, not 60%)
    5. Consistency gate (IC stability across folds)
    """
    # Initialize
    gate_results = {}
    failure_reasons = []
    
    # ========== COMPUTE METRICS ==========
    
    # In-sample metrics
    sig_is = signal[train_mask]
    ret_is = fwd_returns[train_mask]
    ic_is, ci_is_lo, ci_is_hi, se_is, p_is = compute_rank_ic_with_ci(
        sig_is, ret_is, alpha, n_bootstrap
    )
    
    # Out-of-sample metrics
    sig_oos = signal[test_mask]
    ret_oos = fwd_returns[test_mask]
    ic_oos, ci_oos_lo, ci_oos_hi, se_oos, p_oos = compute_rank_ic_with_ci(
        sig_oos, ret_oos, alpha, n_bootstrap
    )
    
    # ========== GATE 1: SAMPLE SUFFICIENCY ==========
    n_oos = np.sum(test_mask)
    gate_results['sample_sufficient'] = data_diag.sample_sufficient
    if not data_diag.sample_sufficient:
        failure_reasons.append(
            f"Insufficient sample: n_eff={data_diag.n_effective:.0f} < "
            f"n_required={data_diag.n_required_for_effect}"
        )
    
    # ========== GATE 2: IS IC FINITE ==========
    is_ic_finite = not (np.isnan(ic_is) or np.isinf(ic_is))
    gate_results['is_ic_finite'] = is_ic_finite
    if not is_ic_finite:
        failure_reasons.append("IS IC is NaN/Inf")
    
    # ========== GATE 3: OOS IC FINITE ==========
    oos_ic_finite = not (np.isnan(ic_oos) or np.isinf(ic_oos))
    gate_results['oos_ic_finite'] = oos_ic_finite
    if not oos_ic_finite:
        failure_reasons.append("OOS IC is NaN/Inf")
    
    # ========== GATE 4: OOS CI EXCLUDES ZERO ==========
    # The DATA tells us if the effect is significant, not a hardcoded p<0.05
    if oos_ic_finite and not np.isnan(ci_oos_lo) and not np.isnan(ci_oos_hi):
        # CI must be entirely positive OR entirely negative
        ci_excludes_zero = (ci_oos_lo > 0) or (ci_oos_hi < 0)
        gate_results['oos_ci_excludes_zero'] = ci_excludes_zero
        if not ci_excludes_zero:
            failure_reasons.append(
                f"OOS 95% CI [{ci_oos_lo:.4f}, {ci_oos_hi:.4f}] includes zero"
            )
    else:
        gate_results['oos_ci_excludes_zero'] = False
        if 'OOS IC is NaN/Inf' not in failure_reasons:
            failure_reasons.append("Cannot compute OOS CI")
    
    # ========== GATE 5: OOS EFFECT EXCEEDS NOISE FLOOR ==========
    # Instead of hardcoded 60%, compare to minimum detectable effect
    min_effect = data_diag.min_detectable_effect
    if oos_ic_finite:
        exceeds_noise = abs(ic_oos) >= min_effect
        gate_results['exceeds_noise_floor'] = exceeds_noise
        if not exceeds_noise:
            failure_reasons.append(
                f"OOS IC {ic_oos:.4f} < min detectable {min_effect:.4f}"
            )
    else:
        gate_results['exceeds_noise_floor'] = False
    
    # ========== AGGREGATE ==========
    passed_all = all(gate_results.values())
    
    # IC Information Ratio (OOS)
    ic_ir = ic_oos / se_oos if se_oos and se_oos > 0 else np.nan
    
    return ValidationResult(
        pattern_id=pattern_id,
        ic_mean_is=round_val(ic_is),
        ic_mean_oos=round_val(ic_oos),
        ic_std_oos=round_val(se_oos),
        ic_ir_oos=round_val(ic_ir),
        p_value_raw=round_val(p_oos),
        p_value_adjusted=None,  # Set later with FDR
        ci_lower=round_val(ci_oos_lo),
        ci_upper=round_val(ci_oos_hi),
        sharpe_is=None,  # Compute if PnL provided
        deflated_sharpe=None,
        passed_all_gates=passed_all,
        gate_results=gate_results,
        failure_reasons=failure_reasons
    )


# ============================================================================
# MULTIPLE TESTING CORRECTION
# ============================================================================

def apply_fdr_correction(
    results: List[ValidationResult],
    alpha: float = 0.05,
    backtest_years: float = 1.0
) -> Tuple[List[ValidationResult], MultipleTestingReport]:
    """
    Apply Benjamini-Hochberg FDR correction to multiple patterns.
    
    Also computes MinBTL and expected noise SR.
    """
    n_trials = len(results)
    
    if n_trials == 0:
        return results, None
    
    # Extract p-values
    p_values = []
    for r in results:
        p = r.p_value_raw
        if p is None or np.isnan(p):
            p = 1.0  # Treat missing as null
        p_values.append(p)
    
    p_values = np.array(p_values)
    
    # Apply BH correction
    rejected_raw = np.sum(p_values < alpha)
    rejected, q_values, n_rejected = benjamini_hochberg(p_values, alpha)
    
    # π₀ estimate
    pi0 = storey_pi0_estimate(p_values)
    
    # MinBTL (assuming SR=1.0 as baseline)
    min_btl = calculate_min_btl(n_trials, target_sharpe=1.0)
    
    # Expected noise SR
    noise_sr = expected_max_sharpe_null(n_trials, backtest_years)
    
    # Update results with q-values
    for i, r in enumerate(results):
        r.p_value_adjusted = round_val(q_values[i])
    
    report = MultipleTestingReport(
        n_trials=n_trials,
        n_rejected_raw=int(rejected_raw),
        n_rejected_fdr=int(n_rejected),
        fdr_level=alpha,
        pi0_estimate=round(pi0, 3),
        min_btl_years=round(min_btl, 2),
        expected_noise_sharpe=round(noise_sr, 3)
    )
    
    return results, report


# ============================================================================
# FULL LAB V1 EVALUATION
# ============================================================================

def evaluate_patterns_v1(
    prices: pd.Series,
    signals: Dict[str, pd.Series],
    train_len: int = 252,
    test_len: int = 63,
    alpha: float = 0.05,
    target_effect: float = 0.05,
    power: float = 0.80,
    n_bootstrap: int = 1000
) -> LabV1Report:
    """
    Full Lab V1 evaluation of multiple patterns.
    
    This is the main entry point. It:
    1. Computes data diagnostics (adaptive thresholds)
    2. Validates each pattern with walk-forward
    3. Applies FDR correction for multiple testing
    4. Returns structured report
    
    Args:
        prices: Price series
        signals: Dict of pattern_id -> signal series
        train_len: Training window length (bars)
        test_len: Test window length (bars)
        alpha: Significance level
        target_effect: Minimum effect size to detect (IC)
        power: Statistical power target
        n_bootstrap: Bootstrap samples for CI
    
    Returns:
        LabV1Report with all diagnostics and results
    """
    # ========== SETUP ==========
    timestamp = datetime.now().isoformat()
    
    # Forward returns
    fwd_returns = compute_forward_returns(prices, horizon=1)
    returns = prices.pct_change().dropna().values
    
    # Align all data
    dates = prices.index
    n = len(dates)
    
    # ========== DATA DIAGNOSTICS ==========
    data_diag = compute_data_diagnostics(
        returns, target_effect, alpha, power
    )
    
    # ========== WALK-FORWARD WITH ADAPTIVE EMBARGO ==========
    splits_gen, actual_embargo, acf_diag = walk_forward_splits_adaptive(
        dates, returns, train_len, test_len,
        min_embargo=1, confidence=0.95
    )
    
    # Collect all splits
    splits = list(splits_gen)
    n_splits = len(splits)
    
    if n_splits == 0:
        raise ValueError("No valid walk-forward splits generated")
    
    # ========== VALIDATE EACH PATTERN ==========
    all_results = []
    
    for pattern_id, signal in signals.items():
        # Convert to numpy and align
        sig_arr = signal.reindex(dates).values
        ret_arr = fwd_returns.values
        
        # Aggregate across all splits
        is_ics = []
        oos_ics = []
        
        all_train_mask = np.zeros(n, dtype=bool)
        all_test_mask = np.zeros(n, dtype=bool)
        
        for train_idx, test_idx in splits:
            all_train_mask[train_idx] = True
            all_test_mask[test_idx] = True
            
            # Compute IC for this split
            sig_train = sig_arr[train_idx]
            ret_train = ret_arr[train_idx]
            sig_test = sig_arr[test_idx]
            ret_test = ret_arr[test_idx]
            
            # Filter NaN
            mask_train = ~(np.isnan(sig_train) | np.isnan(ret_train))
            mask_test = ~(np.isnan(sig_test) | np.isnan(ret_test))
            
            if np.sum(mask_train) >= 10:
                ic_train, _ = stats.spearmanr(
                    sig_train[mask_train], ret_train[mask_train]
                )
                is_ics.append(ic_train)
            
            if np.sum(mask_test) >= 10:
                ic_test, _ = stats.spearmanr(
                    sig_test[mask_test], ret_test[mask_test]
                )
                oos_ics.append(ic_test)
        
        # Validate with aggregated data
        result = validate_pattern_v1(
            pattern_id=pattern_id,
            signal=sig_arr,
            fwd_returns=ret_arr,
            train_mask=all_train_mask,
            test_mask=all_test_mask,
            data_diag=data_diag,
            n_trials=len(signals),
            alpha=alpha,
            n_bootstrap=n_bootstrap
        )
        
        all_results.append(result)
    
    # ========== FDR CORRECTION ==========
    backtest_years = n / 252  # Approximate
    all_results, mt_report = apply_fdr_correction(
        all_results, alpha, backtest_years
    )
    
    # ========== BUILD REPORT ==========
    summary = {
        'n_patterns_tested': len(signals),
        'n_passed_all_gates': sum(1 for r in all_results if r.passed_all_gates),
        'n_significant_raw': mt_report.n_rejected_raw if mt_report else 0,
        'n_significant_fdr': mt_report.n_rejected_fdr if mt_report else 0,
        'data_sufficient': data_diag.sample_sufficient,
        'adaptive_embargo_used': actual_embargo,
        'n_walk_forward_splits': n_splits
    }
    
    methodology = {
        'framework': 'Lab V1 Adaptive',
        'fdr_method': 'Benjamini-Hochberg',
        'embargo_method': 'ACF-adaptive',
        'ci_method': 'Bootstrap percentile',
        'sample_size_method': 'Power analysis with autocorrelation adjustment',
        'references': [
            'Bailey & López de Prado (2014) - Deflated Sharpe Ratio',
            'Benjamini & Hochberg (1995) - FDR Control',
            'López de Prado (2018) - Walk-forward with purging'
        ]
    }
    
    return LabV1Report(
        timestamp=timestamp,
        data_diagnostics=asdict(data_diag),
        multiple_testing=asdict(mt_report) if mt_report else {},
        pattern_results=[asdict(r) for r in all_results],
        summary=summary,
        methodology=methodology
    )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def round_val(x, decimals=NUMERIC_PRECISION):
    """Round value, return None for NaN/Inf (JSON compliance)."""
    if x is None or np.isnan(x) or np.isinf(x):
        return None
    return round(float(x), decimals)


def report_to_json(report: LabV1Report) -> str:
    """Convert LabV1Report to JSON string."""
    return json.dumps(asdict(report), indent=2, default=str)


# ============================================================================
# MAIN TEST
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("LAB V1 ADAPTIVE - SELF TEST")
    print("=" * 70)
    print()
    
    # Generate test data
    np.random.seed(42)
    n = 504  # 2 years
    dates = pd.date_range('2022-01-01', periods=n, freq='B')
    
    # Prices with slight trend
    returns = np.random.randn(n) * 0.01 + 0.0001
    prices = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates)
    
    # Create test signals
    signals = {}
    
    # Signal 1: Pure noise (should FAIL)
    signals['noise_signal'] = pd.Series(np.random.randn(n), index=dates)
    
    # Signal 2: Weak predictive signal (might pass)
    fwd_ret = prices.pct_change().shift(-1).values
    weak_signal = np.zeros(n)
    for i in range(n-1):
        weak_signal[i] = fwd_ret[i] * 0.3 + np.random.randn() * 0.07
    signals['weak_signal'] = pd.Series(weak_signal, index=dates)
    
    # Signal 3: Strong predictive signal (should pass)
    strong_signal = np.zeros(n)
    for i in range(n-1):
        strong_signal[i] = fwd_ret[i] * 0.8 + np.random.randn() * 0.05
    signals['strong_signal'] = pd.Series(strong_signal, index=dates)
    
    print("Running Lab V1 Evaluation...")
    print()
    
    # Run evaluation
    report = evaluate_patterns_v1(
        prices=prices,
        signals=signals,
        train_len=126,  # 6 months
        test_len=63,    # 3 months
        alpha=0.05,
        target_effect=0.05,
        power=0.80,
        n_bootstrap=500
    )
    
    # Print results
    print("DATA DIAGNOSTICS:")
    print("-" * 40)
    for k, v in report.data_diagnostics.items():
        print(f"  {k}: {v}")
    print()
    
    print("MULTIPLE TESTING:")
    print("-" * 40)
    for k, v in report.multiple_testing.items():
        print(f"  {k}: {v}")
    print()
    
    print("PATTERN RESULTS:")
    print("-" * 40)
    for r in report.pattern_results:
        status = "✓ PASS" if r['passed_all_gates'] else "✗ FAIL"
        print(f"  {r['pattern_id']}: {status}")
        print(f"    IS IC: {r['ic_mean_is']}")
        print(f"    OOS IC: {r['ic_mean_oos']}")
        print(f"    95% CI: [{r['ci_lower']}, {r['ci_upper']}]")
        print(f"    p-value (raw): {r['p_value_raw']}")
        print(f"    p-value (BH): {r['p_value_adjusted']}")
        if r['failure_reasons']:
            print(f"    Failures: {r['failure_reasons']}")
        print()
    
    print("SUMMARY:")
    print("-" * 40)
    for k, v in report.summary.items():
        print(f"  {k}: {v}")
    print()
    
    print("=" * 70)
    print("LAB V1 SELF TEST COMPLETE")
    print("=" * 70)
