"""
Pattern Discovery Lab - Runner v0.1
Full implementation with enforcement gates.
"""
import sys
import os
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from pattern_discovery_lab.gates_v0p1 import (
    compute_rho1,
    gate_a_shuffle_control,
    gate_b_spectrum_error,
    generate_phase_surrogate,
    get_detector_class,
    get_expected_decision,
    CERTIFIED_DETECTORS,
    EXPERIMENTAL_DETECTORS,
)

from pattern_discovery_lab.lab_v0 import (
    walk_forward_splits,
    compute_forward_returns,
    compute_rank_ic,
    compute_ic_tstat,
    negative_control_random,
    placebo_time_shift,
    check_oos_degradation_gate,
    check_finite_metrics_gate,
    is_finite,
    round_numeric,
)

from pattern_discovery_lab.detector_v0 import momentum_detector

from pattern_discovery_lab.schema_v0 import (
    write_json,
    SCHEMA_VERSION,
)


# ============================================================================
# DATA GENERATORS
# ============================================================================

def generate_white_noise(n: int, seed: int = None) -> np.ndarray:
    """Generate white noise."""
    rng = np.random.RandomState(seed)
    return rng.randn(n)


def generate_ar1(n: int, phi: float = 0.9, seed: int = None) -> np.ndarray:
    """Generate AR(1) process: x[t] = phi * x[t-1] + eps[t]."""
    rng = np.random.RandomState(seed)
    x = np.zeros(n)
    x[0] = rng.randn()
    for t in range(1, n):
        x[t] = phi * x[t-1] + rng.randn()
    return x


def generate_garch(n: int, seed: int = None) -> np.ndarray:
    """
    Generate GARCH(1,1) process with no linear autocorrelation.
    Returns: x[t] ~ N(0, h[t]) where h[t] = omega + alpha*eps[t-1]^2 + beta*h[t-1]
    """
    rng = np.random.RandomState(seed)
    omega, alpha, beta = 0.1, 0.1, 0.8
    
    h = np.zeros(n)
    eps = np.zeros(n)
    h[0] = omega / (1 - alpha - beta)
    eps[0] = rng.randn() * np.sqrt(h[0])
    
    for t in range(1, n):
        h[t] = omega + alpha * eps[t-1]**2 + beta * h[t-1]
        eps[t] = rng.randn() * np.sqrt(h[t])
    
    return eps


# ============================================================================
# DETECTORS
# ============================================================================

def run_autocorrelation_dependence(data: np.ndarray, n_surrogates: int = 999, 
                                   seed: int = None) -> Dict[str, Any]:
    """
    CERTIFIED detector: Tests for autocorrelation dependence.
    
    Real statistic: abs(rho1(data))
    Null: shuffle surrogates
    """
    rng = np.random.RandomState(seed)
    
    # Real statistic
    real_rho1 = compute_rho1(data)
    real_stat = abs(real_rho1)
    
    # Generate shuffle surrogates and compute stats
    surrogate_stats = []
    for i in range(n_surrogates):
        shuffled = rng.permutation(data)
        surr_rho1 = compute_rho1(shuffled)
        surr_stat = abs(surr_rho1)
        surrogate_stats.append(surr_stat)
    
    surrogate_stats = np.array(surrogate_stats)
    
    # Compute p-value
    k = np.sum(surrogate_stats >= real_stat)
    N = n_surrogates
    p_value = (k + 1) / (N + 1)
    
    return {
        'real_score': float(real_stat),
        'k': int(k),
        'N': int(N),
        'p_value': float(p_value),
        'surrogate_stats': surrogate_stats,
    }


def run_time_reversal_asymmetry(data: np.ndarray, n_surrogates: int = 999,
                                seed: int = None) -> Dict[str, Any]:
    """
    EXPERIMENTAL detector: Time reversal asymmetry.
    Uses third-order statistic: mean((x[t+1] - x[t])^3)
    Null: phase randomization
    """
    rng = np.random.RandomState(seed)
    
    # Real statistic
    diffs = np.diff(data)
    real_stat = abs(np.mean(diffs ** 3))
    
    # Generate phase surrogates and compute stats
    surrogate_stats = []
    for i in range(n_surrogates):
        surrogate = generate_phase_surrogate(data, seed=rng.randint(0, 1e9))
        surr_diffs = np.diff(surrogate)
        surr_stat = abs(np.mean(surr_diffs ** 3))
        surrogate_stats.append(surr_stat)
    
    surrogate_stats = np.array(surrogate_stats)
    
    # Compute p-value
    k = np.sum(surrogate_stats >= real_stat)
    N = n_surrogates
    p_value = (k + 1) / (N + 1)
    
    return {
        'real_score': float(real_stat),
        'k': int(k),
        'N': int(N),
        'p_value': float(p_value),
        'surrogate_stats': surrogate_stats,
    }


# ============================================================================
# GATE C: STABILITY DECISION
# ============================================================================

def run_gate_c_stability(data: np.ndarray, detector_func, n_surrogates: int = 999,
                        seed: int = None, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Gate C: Decision Stability
    Split data into 3 non-overlapping blocks, run detector on each.
    """
    rng = np.random.RandomState(seed)
    N = len(data)
    block_size = N // 3
    
    blocks = [
        data[0:block_size],
        data[block_size:2*block_size],
        data[2*block_size:3*block_size]
    ]
    
    decisions = []
    block_results = []
    
    for i, block in enumerate(blocks):
        result = detector_func(block, n_surrogates=n_surrogates, seed=rng.randint(0, 1e9))
        decision = 'REJECT' if result['p_value'] < alpha else 'ACCEPT'
        decisions.append(decision)
        block_results.append({
            'block': i + 1,
            'p_value': result['p_value'],
            'decision': decision,
        })
    
    # Count agreements
    unique_decisions = list(set(decisions))
    if len(unique_decisions) == 1:
        # All agree
        stability_decision = 1.0
    else:
        # Find majority
        counts = {d: decisions.count(d) for d in unique_decisions}
        majority_count = max(counts.values())
        stability_decision = majority_count / 3.0
    
    return {
        'block_results': block_results,
        'stability_decision': float(stability_decision),
        'decisions': decisions,
    }


# ============================================================================
# TRUTH TESTS
# ============================================================================

def run_truth_tests(n_surrogates: int = 99) -> bool:
    """
    Run truth tests T1-T4.
    Returns True if all pass, False otherwise.
    """
    print("=" * 70)
    print("TRUTH TESTS (T1-T4)")
    print("=" * 70)
    print()
    
    all_passed = True
    
    # T1: White noise should ACCEPT
    print("T1: white_noise → EXPECT ACCEPT")
    data = generate_white_noise(1000, seed=42)
    result = run_autocorrelation_dependence(data, n_surrogates=n_surrogates, seed=42)
    decision = 'REJECT' if result['p_value'] < 0.05 else 'ACCEPT'
    p_formula = f"p = ({result['k']}+1)/({result['N']}+1) = {result['p_value']:.4f}"
    print(f"  {p_formula}")
    print(f"  Decision: {decision}")
    if decision == 'ACCEPT':
        print("  ✓ PASS")
    else:
        print("  ✗ FAIL")
        all_passed = False
    print()
    
    # T2: AR(1) should REJECT
    print("T2: ar1_phi_0p9 → EXPECT REJECT")
    data = generate_ar1(1000, phi=0.9, seed=42)
    result = run_autocorrelation_dependence(data, n_surrogates=n_surrogates, seed=42)
    decision = 'REJECT' if result['p_value'] < 0.05 else 'ACCEPT'
    p_formula = f"p = ({result['k']}+1)/({result['N']}+1) = {result['p_value']:.4f}"
    print(f"  {p_formula}")
    print(f"  Decision: {decision}")
    if decision == 'REJECT':
        print("  ✓ PASS")
    else:
        print("  ✗ FAIL")
        all_passed = False
    print()
    
    # T3: GARCH should ACCEPT (no linear autocorrelation)
    print("T3: garch → EXPECT ACCEPT")
    data = generate_garch(1000, seed=42)
    result = run_autocorrelation_dependence(data, n_surrogates=n_surrogates, seed=42)
    decision = 'REJECT' if result['p_value'] < 0.05 else 'ACCEPT'
    p_formula = f"p = ({result['k']}+1)/({result['N']}+1) = {result['p_value']:.4f}"
    print(f"  {p_formula}")
    print(f"  Decision: {decision}")
    if decision == 'ACCEPT':
        print("  ✓ PASS")
    else:
        print("  ✗ FAIL")
        all_passed = False
    print()
    
    # T4: Shuffle threshold check
    print("T4: Gate A threshold T(N=1000) = 4.5/sqrt(1000)")
    data = generate_white_noise(1000, seed=42)
    gate_a_result = gate_a_shuffle_control(data, M=100)
    expected_threshold = 4.5 / np.sqrt(1000)
    print(f"  Expected: {expected_threshold:.4f}")
    print(f"  Actual: {gate_a_result['threshold']:.4f}")
    print(f"  rho1_shuffled_max: {gate_a_result['rho1_shuffled_max']:.4f}")
    print(f"  Verdict: {gate_a_result['shuffle_check']}")
    if abs(gate_a_result['threshold'] - expected_threshold) < 1e-6 and gate_a_result['shuffle_check'] == 'PASS':
        print("  ✓ PASS")
    else:
        print("  ✗ FAIL")
        all_passed = False
    print()
    
    print("=" * 70)
    if all_passed:
        print("ALL TRUTH TESTS PASSED ✓")
    else:
        print("SOME TRUTH TESTS FAILED ✗")
    print("=" * 70)
    print()
    
    return all_passed


# ============================================================================
# MAIN RUNNER
# ============================================================================

def run_all(seed: int = 42, n_surrogates: int = 999) -> Tuple[str, Dict[str, Any]]:
    """
    Run full enforcement contract.
    
    Returns:
        Tuple of (run_folder, results_dict)
    """
    # Set global seed
    np.random.seed(seed)
    
    # Create run folder
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    run_folder = os.path.join(
        os.path.dirname(__file__),
        'runs',
        timestamp
    )
    os.makedirs(run_folder, exist_ok=True)
    
    # Print header (Block 1)
    cli_command = f"python -m pattern_discovery_lab {' '.join(sys.argv[1:])}"
    print(f"CLI Command: {cli_command}")
    print()
    print(f"Run Folder: {run_folder}")
    print()
    
    # Generate datasets
    N = 1000
    datasets = {
        'white_noise': generate_white_noise(N, seed=seed),
        'ar1_phi_0p9': generate_ar1(N, phi=0.9, seed=seed + 1),
        'garch': generate_garch(N, seed=seed + 2),
    }
    
    # Detectors
    detectors = {
        'autocorrelation_dependence': run_autocorrelation_dependence,
        'time_reversal_asymmetry': run_time_reversal_asymmetry,
    }
    
    # Results storage
    results = {
        'meta': {
            'timestamp': timestamp,
            'seed': seed,
            'n_surrogates': n_surrogates,
            'N': N,
        },
        'stop_rules': {},
        'rows': [],
        'overall_run_status': 'PASS',
    }
    
    halt_reasons = []
    
    # Run gates per dataset
    for dataset_name, data in datasets.items():
        # Gate A: Shuffle Control
        gate_a_result = gate_a_shuffle_control(data, M=100)
        
        # Gate B: Spectrum Preservation (generate ONE phase surrogate)
        phase_surrogate = generate_phase_surrogate(data, seed=seed + 100)
        gate_b_result = gate_b_spectrum_error(data, phase_surrogate)
        
        # Store for this dataset
        results['stop_rules'][dataset_name] = {
            'gate_a': gate_a_result,
            'gate_b': gate_b_result,
        }
        
        # Check Gate A
        if gate_a_result['shuffle_check'] == 'HALT':
            halt_reasons.append(f"Gate A: {dataset_name} rho1_shuffled_max={gate_a_result['rho1_shuffled_max']:.4f} > T({gate_a_result['N']})={gate_a_result['threshold']:.4f}")
        
        # Check Gate B
        if gate_b_result['spectrum_check'] == 'HALT':
            halt_reasons.append(f"Gate B: {dataset_name} spectrum_error_max={gate_b_result['spectrum_error_max']:.4f} > {gate_b_result['threshold']}")
    
    # Run detectors on each dataset
    for dataset_name, data in datasets.items():
        for detector_name, detector_func in detectors.items():
            detector_class = get_detector_class(detector_name)
            is_certified = detector_class == 'CERTIFIED'
            
            # Run detector
            det_result = detector_func(data, n_surrogates=n_surrogates, seed=seed + 200)
            
            # Statistical decision
            stat_decision = 'REJECT' if det_result['p_value'] < 0.05 else 'ACCEPT'
            
            # Gate C: Stability (CERTIFIED only)
            stability_decision = None
            if is_certified:
                gate_c_result = run_gate_c_stability(data, detector_func, 
                                                     n_surrogates=n_surrogates, 
                                                     seed=seed + 300)
                stability_decision = gate_c_result['stability_decision']
                results['stop_rules'].setdefault(dataset_name, {})['gate_c'] = {
                    detector_name: gate_c_result
                }
                
                if stability_decision != 1.0:
                    halt_reasons.append(f"Gate C: {detector_name} on {dataset_name} stability_decision={stability_decision:.3f} != 1.0")
            
            # Gate D: Noise hallucination (CERTIFIED only, check white_noise)
            if is_certified and dataset_name == 'white_noise' and stat_decision == 'REJECT':
                halt_reasons.append(f"Gate D: CERTIFIED detector {detector_name} rejected white_noise")
            
            # Expectation match
            expected = get_expected_decision(detector_name, dataset_name)
            exp_match = 'PASS' if stat_decision == expected else 'FAIL' if expected != 'N/A' else 'N/A'
            
            if is_certified and exp_match == 'FAIL':
                halt_reasons.append(f"Expectation: {detector_name} on {dataset_name} expected {expected} but got {stat_decision}")
            
            # Get gate results for this dataset
            gate_a_result = results['stop_rules'][dataset_name]['gate_a']
            gate_b_result = results['stop_rules'][dataset_name]['gate_b']
            
            # Overall status for this row
            if is_certified:
                row_status = 'CONTRIBUTES'
            else:
                row_status = 'EXCLUDED'
            
            # Store row
            row = {
                'dataset': dataset_name,
                'detector': detector_name,
                'class': detector_class,
                'real_score': det_result['real_score'],
                'k': det_result['k'],
                'N': det_result['N'],
                'p_value': det_result['p_value'],
                'stat_decision': stat_decision,
                'expected': expected,
                'exp_match': exp_match,
                'rho1_shuffled_max': gate_a_result['rho1_shuffled_max'],
                'shuffle_threshold': gate_a_result['threshold'],
                'shuffle_check': gate_a_result['shuffle_check'],
                'spectrum_error_max': gate_b_result['spectrum_error_max'],
                'spectrum_check': gate_b_result['spectrum_check'],
                'stability_decision': stability_decision,
                'row_status': row_status,
            }
            results['rows'].append(row)
    
    # Determine overall status
    if halt_reasons:
        results['overall_run_status'] = 'HALT'
        overall_reason = '; '.join(halt_reasons)
    else:
        overall_reason = 'All gates passed'
    
    # Compute aggregated gate results for contract
    all_shuffle_max = max(results['stop_rules'][dn]['gate_a']['rho1_shuffled_max'] for dn in datasets.keys())
    all_shuffle_threshold = results['stop_rules']['white_noise']['gate_a']['threshold']
    shuffle_verdict = 'PASS' if all(results['stop_rules'][dn]['gate_a']['shuffle_check'] == 'PASS' for dn in datasets.keys()) else 'HALT'
    
    all_spectrum_max = max(results['stop_rules'][dn]['gate_b']['spectrum_error_max'] for dn in datasets.keys())
    spectrum_verdict = 'PASS' if all(results['stop_rules'][dn]['gate_b']['spectrum_check'] == 'PASS' for dn in datasets.keys()) else 'HALT'
    
    # Stability gate
    certified_decisions = {}
    stability_verdict = 'PASS'
    for dataset_name in datasets.keys():
        if 'gate_c' in results['stop_rules'][dataset_name]:
            for det_name, gc_result in results['stop_rules'][dataset_name]['gate_c'].items():
                key = f"{dataset_name}/{det_name}"
                certified_decisions[key] = gc_result['stability_decision']
                if gc_result['stability_decision'] != 1.0:
                    stability_verdict = 'HALT'
    
    # No hallucination check (Gate D)
    white_noise_decision = None
    for row in results['rows']:
        if row['class'] == 'CERTIFIED' and row['dataset'] == 'white_noise':
            white_noise_decision = row['stat_decision']
            break
    
    hallucination_verdict = 'PASS' if white_noise_decision == 'ACCEPT' else 'HALT'
    
    # Build contract-compliant stop_rules
    contract_stop_rules = {
        'shuffle_check': {
            'rho1_shuffled_max': float(all_shuffle_max),
            'threshold_T_N': float(all_shuffle_threshold),
            'threshold_formula': '4.5/√N',
            'N_samples': int(N),
            'verdict': shuffle_verdict,
        },
        'spectrum_check': {
            'spectrum_error_max': float(all_spectrum_max),
            'threshold': 0.05,
            'verdict': spectrum_verdict,
        },
        'stability_gate': {
            'certified_decisions': certified_decisions,
            'verdict': stability_verdict,
        },
        'no_hallucination_check': {
            'dataset': 'white_noise',
            'detector': 'autocorrelation_dependence',
            'statistical_decision': white_noise_decision,
            'verdict': hallucination_verdict,
        }
    }
    
    # Build contract-compliant detector_results
    detector_results = []
    for row in results['rows']:
        # P-value formula with double substitution: p=(k+1)/(N+1)=((k)+1)/((N)+1)=0.XXXX
        k, N_surr = row['k'], row['N']
        p_val = row['p_value']
        surr_formula = f"p=(k+1)/(N+1)=(({k})+1)/(({N_surr})+1)={p_val:.4f}"
        
        dr = {
            'dataset': row['dataset'],
            'detector': row['detector'],
            'class': row['class'],
            'gate_status': row['row_status'],
            'real_score': row['real_score'],
            'surrogate_N': row['N'],
            'surrogate_k': row['k'],
            'surrogate_p': row['p_value'],
            'surrogate_formula': surr_formula,
            'statistical_decision': row['stat_decision'],
            'expected_decision': row['expected'],
            'expectation_match': row['exp_match'],
            'rho1_shuffled': row['rho1_shuffled_max'],
            'spectrum_error': row['spectrum_error_max'],
            'stability_decision': row['stability_decision'],
        }
        detector_results.append(dr)
    
    # Build contract-compliant overall_status
    overall_status_obj = {
        'verdict': results['overall_run_status'],
        'reason': overall_reason,
    }
    
    # ========================================================================
    # STDOUT OUTPUT (5 blocks only)
    # ========================================================================
    
    # Block 1: Header (already printed above)
    # CLI Command and Run Folder already printed
    
    # Block 2: STOP RULES EVALUATED (ENFORCE_NOW)
    print("=" * 70)
    print("STOP RULES EVALUATED (ENFORCE_NOW)")
    print("=" * 70)
    print()
    print(f"Gate A — shuffle_check: {shuffle_verdict}")
    print(f"  N={N}, M=100")
    print(f"  rho1_shuffled_max={all_shuffle_max:.4f}")
    print(f"  threshold T(N)=4.5/sqrt({N})={all_shuffle_threshold:.4f}")
    print()
    print(f"Gate B — spectrum_check: {spectrum_verdict}")
    print(f"  spectrum_error_max={all_spectrum_max:.6f}")
    print(f"  threshold=0.05")
    print()
    print(f"Gate C — stability_gate: {stability_verdict}")
    for key, val in certified_decisions.items():
        print(f"  {key}: stability_decision={val:.3f}")
    print()
    print(f"Gate D — no_hallucination_check: {hallucination_verdict}")
    print(f"  autocorrelation_dependence on white_noise: {white_noise_decision}")
    print()
    
    # Block 3: Results table
    print("=" * 70)
    print("| Dataset         | Detector                  | Class        | Surr Formula              | Stat Dec | Exp Dec  | Exp Match | rho1_shuf | Spec Err   | Stab Dec | Gate Status  |")
    print("=" * 70)
    for dr in detector_results:
        stab_str = f"{dr['stability_decision']:.3f}" if dr['stability_decision'] is not None else "N/A"
        print(f"| {dr['dataset']:<15} | {dr['detector']:<25} | {dr['class']:<12} | {dr['surrogate_formula']:<25} | {dr['statistical_decision']:<8} | {dr['expected_decision']:<8} | {dr['expectation_match']:<9} | {dr['rho1_shuffled']:<9.4f} | {dr['spectrum_error']:<10.6f} | {stab_str:<8} | {dr['gate_status']:<12} |")
    print()
    
    # Block 4: OVERALL STATUS DETERMINATION
    print("=" * 70)
    print("OVERALL STATUS DETERMINATION")
    print("=" * 70)
    print()
    
    print("STOP RULES:")
    print(f"  Gate A (shuffle): {shuffle_verdict}")
    print(f"  Gate B (spectrum): {spectrum_verdict}")
    print(f"  Gate C (stability): {stability_verdict}")
    print(f"  Gate D (no hallucination): {hallucination_verdict}")
    print()
    
    print("CERTIFIED DETECTORS (contribute to gating):")
    for dr in detector_results:
        if dr['class'] == 'CERTIFIED':
            print(f"  {dr['dataset']} × {dr['detector']}: {dr['statistical_decision']} (expected: {dr['expected_decision']}) → {dr['expectation_match']}")
    print()
    
    print("EXPERIMENTAL DETECTORS (excluded from gating):")
    for dr in detector_results:
        if dr['class'] == 'EXPERIMENTAL':
            print(f"  {dr['dataset']} × {dr['detector']}: {dr['statistical_decision']} (info only)")
    print()
    
    print("GATING LOGIC:")
    print("  - All stop rules must PASS")
    print("  - All CERTIFIED expectation matches must PASS")
    print("  - EXPERIMENTAL detectors do not affect overall status")
    print()
    
    print(f"Overall Status: {overall_status_obj['verdict']}")
    print(f"Reason: {overall_status_obj['reason']}")
    print()
    
    # ========================================================================
    # Write results.json (CANONICAL CONTRACT - ONLY 3 KEYS)
    # ========================================================================
    canonical_results = {
        'stop_rules': contract_stop_rules,
        'detector_results': detector_results,
        'overall_status': overall_status_obj,
    }
    
    results_path = os.path.join(run_folder, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(canonical_results, f, indent=2)
    
    # ========================================================================
    # Write results_debug.json (OPTIONAL DEBUG - NOT PART OF CONTRACT)
    # ========================================================================
    debug_results = {
        'meta': results['meta'],
        'rows': results['rows'],
        'overall_run_status': results['overall_run_status'],
        'legacy_stop_rules': results['stop_rules'],
    }
    
    debug_path = os.path.join(run_folder, 'results_debug.json')
    with open(debug_path, 'w') as f:
        json.dump(debug_results, f, indent=2)
    
    print(f"Results written to: {results_path}")
    print(f"Debug info written to: {debug_path}")
    print()
    
    return run_folder, results


# ============================================================================
# LAB V0 RUNNER
# ============================================================================

def run_lab_v0(seed: int = 42) -> Tuple[str, Dict[str, Any]]:
    """
    Run Lab V0: RankIC-based walk-forward evaluation.
    
    Returns:
        Tuple of (run_folder, results_dict)
    """
    # Set global seed
    np.random.seed(seed)
    
    # Create run folder
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    run_folder = os.path.join(
        os.path.dirname(__file__),
        'runs',
        timestamp
    )
    os.makedirs(run_folder, exist_ok=True)
    
    # Print minimal header
    cli_command = f"python -m pattern_discovery_lab {' '.join(sys.argv[1:])}"
    print(f"CLI Command: {cli_command}")
    print()
    print(f"Run Folder: {run_folder}")
    print()
    
    # Generate synthetic price data for demonstration
    # (In production, this would load real data)
    print("Generating synthetic price data...")
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    prices = pd.Series(
        100 * np.exp(np.random.randn(500).cumsum() * 0.02),
        index=dates
    )
    
    # Parameters
    lookback = 20
    horizon = 1
    train_len = 200
    test_len = 50
    embargo = 1
    
    print(f"Parameters: lookback={lookback}, horizon={horizon}, train_len={train_len}, test_len={test_len}, embargo={embargo}")
    print()
    
    # Generate signal
    print("Computing momentum signal...")
    signal = momentum_detector(prices, lookback=lookback)
    
    # Compute forward returns
    fwd_returns = compute_forward_returns(prices, horizon=horizon)
    
    # Walk-forward splits
    print("Performing walk-forward evaluation...")
    splits = list(walk_forward_splits(dates, train_len=train_len, test_len=test_len, embargo=embargo))
    print(f"Number of splits: {len(splits)}")
    print()
    
    # Evaluate each split
    split_results = []
    is_ic_values = []
    oos_ic_values = []
    
    for i, (train_idx, test_idx) in enumerate(splits):
        # In-sample (train)
        train_signal = signal.iloc[train_idx]
        train_fwd_ret = fwd_returns.iloc[train_idx]
        ic_train = compute_rank_ic(train_signal, train_fwd_ret)
        is_ic_values.append(ic_train)
        
        # Out-of-sample (test)
        test_signal = signal.iloc[test_idx]
        test_fwd_ret = fwd_returns.iloc[test_idx]
        ic_test = compute_rank_ic(test_signal, test_fwd_ret)
        oos_ic_values.append(ic_test)
        
        split_results.append({
            "split_id": i + 1,
            "train_start": str(dates[train_idx[0]].date()),
            "train_end": str(dates[train_idx[-1]].date()),
            "test_start": str(dates[test_idx[0]].date()),
            "test_end": str(dates[test_idx[-1]].date()),
            "ic_train": round_numeric(ic_train),
            "ic_test": round_numeric(ic_test)
        })
    
    # Compute statistics
    is_ic_mean = round_numeric(np.nanmean(is_ic_values))
    oos_ic_mean = round_numeric(np.nanmean(oos_ic_values))
    
    is_t_stat, is_p_val = compute_ic_tstat(is_ic_values)
    oos_t_stat, oos_p_val = compute_ic_tstat(oos_ic_values)
    
    ic_results = {
        "in_sample": {
            "ic_mean": is_ic_mean,
            "ic_std": round_numeric(np.nanstd(is_ic_values)),
            "t_stat": round_numeric(is_t_stat),
            "p_value": round_numeric(is_p_val)
        },
        "out_of_sample": {
            "ic_mean": oos_ic_mean,
            "ic_std": round_numeric(np.nanstd(oos_ic_values)),
            "t_stat": round_numeric(oos_t_stat),
            "p_value": round_numeric(oos_p_val)
        }
    }
    
    # Negative controls
    print("Running negative controls...")
    
    # Random signal placebo (use same index as prices for alignment)
    random_signal = negative_control_random(len(prices), seed=seed, index=prices.index)
    ic_random = compute_rank_ic(random_signal, fwd_returns)
    
    # Time-shift placebo
    shifted_signal = placebo_time_shift(signal, shift=10)
    ic_shift = compute_rank_ic(shifted_signal, fwd_returns)
    
    negative_controls = {
        "random_placebo": {
            "ic": round_numeric(ic_random),
            "expected": "near_zero"
        },
        "time_shift_placebo": {
            "ic": round_numeric(ic_shift),
            "shift_bars": 10,
            "expected": "degraded"
        }
    }
    
    # Gates
    print("Evaluating gates...")
    
    # Gate: OOS degradation
    oos_gate_passed, oos_gate_reason = check_oos_degradation_gate(is_ic_mean if is_ic_mean is not None else 0.0, oos_ic_mean if oos_ic_mean is not None else 0.0, threshold=0.60)
    
    # Gate: Finite metrics
    all_metrics = {
        "is_ic_mean": is_ic_mean,
        "oos_ic_mean": oos_ic_mean,
        "ic_random": ic_random,
        "ic_shift": ic_shift,
    }
    finite_gate_passed, finite_gate_reason = check_finite_metrics_gate(all_metrics)
    
    # Gate: Embargo respected (always true by construction)
    embargo_gate_passed = True
    embargo_gate_reason = f"Embargo={embargo} bar enforced in walk_forward_splits"
    
    gates = {
        "embargo_respected": {
            "passed": embargo_gate_passed,
            "reason": embargo_gate_reason
        },
        "finite_metrics": {
            "passed": finite_gate_passed,
            "reason": finite_gate_reason
        },
        "oos_degradation": {
            "passed": oos_gate_passed,
            "reason": oos_gate_reason,
            "threshold": 0.60
        },
        "ascii_only": {
            "passed": True,
            "reason": "All strings ASCII-validated by schema"
        },
        "deterministic_ordering": {
            "passed": True,
            "reason": "JSON sort_keys=True enforced"
        }
    }
    
    # Overall status
    overall_status = "PASS" if (oos_gate_passed and finite_gate_passed) else "FAIL"
    
    # Build results with frozen schema
    run_id = timestamp
    dataset_version = "synthetic_v0"
    universe_version = "demo_universe_v0"
    date_range = {
        "start": str(dates[0].date()),
        "end": str(dates[-1].date())
    }
    split_spec = {
        "method": "walk_forward",
        "train_len": train_len,
        "test_len": test_len,
        "embargo": embargo
    }
    
    meta = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "dataset_version": dataset_version,
        "universe_version": universe_version,
        "date_range": date_range,
        "split_spec": split_spec,
        "seed": seed,
        "n_trials": len(splits)
    }
    
    # Candidates (even if only 1 in V0)
    candidates_ranked = [
        {
            "rank": 1,
            "candidate_id": "momentum_baseline",
            "lookback": lookback,
            "horizon": horizon,
            "in_sample": {
                "ic_mean": is_ic_mean,
                "ic_std": round_numeric(np.nanstd(is_ic_values)),
                "t_stat": round_numeric(is_t_stat),
                "p_value": round_numeric(is_p_val)
            },
            "out_of_sample": {
                "ic_mean": oos_ic_mean,
                "ic_std": round_numeric(np.nanstd(oos_ic_values)),
                "t_stat": round_numeric(oos_t_stat),
                "p_value": round_numeric(oos_p_val)
            }
        }
    ]
    
    results = {
        "schema_version": SCHEMA_VERSION,
        "meta": meta,
        "gates": gates,
        "controls": negative_controls,
        "candidates_ranked": candidates_ranked,
        "overall_status": overall_status
    }
    
    # Print summary
    print("=" * 70)
    print("LAB V0 RESULTS")
    print("=" * 70)
    print()
    
    # Format IC values - never print "nan"
    is_ic_str = f"{is_ic_mean:.6f}" if is_ic_mean is not None else "null"
    oos_ic_str = f"{oos_ic_mean:.6f}" if oos_ic_mean is not None else "null"
    is_t_str = f"{is_t_stat:.4f}" if is_t_stat is not None else "null"
    is_p_str = f"{is_p_val:.4f}" if is_p_val is not None else "null"
    oos_t_str = f"{oos_t_stat:.4f}" if oos_t_stat is not None else "null"
    oos_p_str = f"{oos_p_val:.4f}" if oos_p_val is not None else "null"
    
    print(f"In-Sample IC:  mean={is_ic_str}, t-stat={is_t_str}, p-value={is_p_str}")
    print(f"Out-of-Sample IC: mean={oos_ic_str}, t-stat={oos_t_str}, p-value={oos_p_str}")
    print()
    print("Negative Controls:")
    ic_random_val = ic_random if is_finite(ic_random) else None
    ic_shift_val = ic_shift if is_finite(ic_shift) else None
    ic_random_str = f"{ic_random_val:.6f}" if ic_random_val is not None else "null"
    ic_shift_str = f"{ic_shift_val:.6f}" if ic_shift_val is not None else "null"
    print(f"  Random placebo IC: {ic_random_str} (expected: near 0)")
    print(f"  Time-shift placebo IC: {ic_shift_str} (expected: degraded)")
    print()
    print("Gates:")
    print(f"  Embargo Respected: {'PASS' if embargo_gate_passed else 'FAIL'}")
    print(f"  Finite Metrics: {'PASS' if finite_gate_passed else 'FAIL'}")
    if not finite_gate_passed:
        print(f"    {finite_gate_reason}")
    print(f"  OOS Degradation: {'PASS' if oos_gate_passed else 'FAIL'}")
    print(f"    {oos_gate_reason}")
    print()
    print(f"Overall Status: {overall_status}")
    print()
    
    # Write results
    results_path = os.path.join(run_folder, 'results_lab_v0.json')
    write_json(results_path, results)
    print(f"Results written to: {results_path}")
    
    # Write debug info (full split details)
    debug_info = {
        "meta": meta,
        "splits": split_results,
        "raw_ic_values": {
            "in_sample": [round_numeric(ic) for ic in is_ic_values],
            "out_of_sample": [round_numeric(ic) for ic in oos_ic_values]
        }
    }
    debug_path = os.path.join(run_folder, 'results_lab_v0_debug.json')
    write_json(debug_path, debug_info)
    print(f"Debug info written to: {debug_path}")
    print()
    
    return run_folder, results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Pattern Discovery Lab - Enforcement Contract v0.1')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--n-surrogates', type=int, default=999, help='Number of surrogates (default: 999)')
    parser.add_argument('--truth-tests', action='store_true', help='Run truth tests T1-T4 and exit')
    parser.add_argument('--run-all', action='store_true', help='Run full enforcement contract')
    parser.add_argument('--lab-v0', action='store_true', help='Run Lab V0 (RankIC walk-forward evaluation)')
    
    args = parser.parse_args()
    
    if args.truth_tests:
        # Run truth tests
        all_passed = run_truth_tests(n_surrogates=args.n_surrogates)
        return 0 if all_passed else 1
    
    elif args.run_all:
        # Run full contract
        run_folder, results = run_all(seed=args.seed, n_surrogates=args.n_surrogates)
        return 0 if results['overall_run_status'] == 'PASS' else 1
    
    elif args.lab_v0:
        # Run Lab V0
        run_folder, results = run_lab_v0(seed=args.seed)
        return 0 if results['overall_status'] == 'PASS' else 1
    
    else:
        # Default: print help
        parser.print_help()
        return 0


if __name__ == '__main__':
    sys.exit(main())
