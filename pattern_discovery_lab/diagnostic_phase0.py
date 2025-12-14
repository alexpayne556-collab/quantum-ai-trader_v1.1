#!/usr/bin/env python3
"""
Phase 0 Diagnostics: Measure False Discovery Rate and Current Lab V0 Flaws

Tests DeepSeek's Red Team challenges:
1. Feed pure random data through Lab V0 - does it find "patterns"?
2. Run same pattern with 100 different seeds - is validation seed-dependent?
3. Measure ACF structure - should embargo be longer than 1 bar?
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import sys

# Add lab to path
sys.path.insert(0, str(Path(__file__).parent))

from lab_v0 import (
    walk_forward_splits,
    compute_forward_returns,
    compute_rank_ic,
    check_finite_metrics_gate,
    check_oos_degradation_gate
)


def test_false_discovery_rate(n_trials=100, seed_base=1000):
    """
    DeepSeek Challenge: Feed random data, count false discoveries.
    
    If Lab V0 finds patterns in pure noise, those are false positives.
    """
    print("=" * 70)
    print("TEST 1: FALSE DISCOVERY RATE")
    print("=" * 70)
    print(f"Running {n_trials} trials with pure random data...")
    print()
    
    discoveries = []
    
    for trial in range(n_trials):
        seed = seed_base + trial
        np.random.seed(seed)
        
        # Pure random walk (no signal)
        n_bars = 300
        prices = pd.Series(
            100 * np.exp(np.cumsum(np.random.randn(n_bars) * 0.01)),
            index=pd.date_range('2020-01-01', periods=n_bars, freq='D')
        )
        
        # Random "signal" (pure noise)
        signal = pd.Series(
            np.random.randn(n_bars),
            index=prices.index
        )
        
        # Compute forward returns
        fwd_ret = compute_forward_returns(prices, horizon=1)
        
        # Walk-forward evaluation
        train_len = 200
        test_len = 50
        embargo = 1
        
        is_ics = []
        oos_ics = []
        
        for train_idx, test_idx in walk_forward_splits(
            prices.index, train_len, test_len, embargo
        ):
            # In-sample
            is_sig = signal.iloc[train_idx]
            is_ret = fwd_ret.iloc[train_idx]
            is_ic = compute_rank_ic(is_sig, is_ret)
            if is_ic is not None:
                is_ics.append(is_ic)
            
            # Out-of-sample
            oos_sig = signal.iloc[test_idx]
            oos_ret = fwd_ret.iloc[test_idx]
            oos_ic = compute_rank_ic(oos_sig, oos_ret)
            if oos_ic is not None:
                oos_ics.append(oos_ic)
        
        if not is_ics or not oos_ics:
            continue
        
        is_ic_mean = np.mean(is_ics)
        oos_ic_mean = np.mean(oos_ics)
        
        # Check gates (same as Lab V0)
        metrics = {
            'is_ic': is_ic_mean,
            'oos_ic': oos_ic_mean
        }
        
        finite_pass, _ = check_finite_metrics_gate(metrics)
        oos_pass, _ = check_oos_degradation_gate(is_ic_mean, oos_ic_mean, threshold=0.60)
        
        # Simple significance: |IC| > 0.05 and p < 0.05 (naive)
        from scipy import stats
        if len(is_ics) > 1:
            _, p_val = stats.ttest_1samp(is_ics, 0.0)
            is_significant = p_val < 0.05
        else:
            is_significant = False
        
        # Count as "discovery" if passes gates AND significant
        if finite_pass and oos_pass and is_significant:
            discoveries.append({
                'trial': trial,
                'is_ic': is_ic_mean,
                'oos_ic': oos_ic_mean,
                'p_value': p_val
            })
    
    fdr = len(discoveries) / n_trials
    
    print(f"Results:")
    print(f"  Total trials: {n_trials}")
    print(f"  'Significant' patterns found: {len(discoveries)}")
    print(f"  Empirical FDR: {fdr:.2%}")
    print()
    
    if fdr > 0.05:
        print(f"⚠️  FDR > 5%! Lab V0 discovers patterns in pure noise.")
    else:
        print(f"✓ FDR ≤ 5% - within expected Type I error rate")
    
    print()
    return fdr, discoveries


def test_seed_sensitivity(n_seeds=100):
    """
    DeepSeek Challenge: Does validation outcome depend on seed?
    
    Run same "pattern" with different seeds, measure outcome variance.
    """
    print("=" * 70)
    print("TEST 2: SEED SENSITIVITY")
    print("=" * 70)
    print(f"Testing {n_seeds} different seeds...")
    print()
    
    outcomes = []
    
    for seed in range(42, 42 + n_seeds):
        np.random.seed(seed)
        
        # Fixed pattern: simple momentum
        n_bars = 300
        prices = pd.Series(
            100 * np.exp(np.cumsum(np.random.randn(n_bars) * 0.01)),
            index=pd.date_range('2020-01-01', periods=n_bars, freq='D')
        )
        
        # Momentum signal (10-day rate of change)
        signal = prices.pct_change(10)
        fwd_ret = compute_forward_returns(prices, horizon=1)
        
        # Walk-forward
        train_len = 200
        test_len = 50
        embargo = 1
        
        is_ics = []
        oos_ics = []
        
        for train_idx, test_idx in walk_forward_splits(
            prices.index, train_len, test_len, embargo
        ):
            is_ic = compute_rank_ic(signal.iloc[train_idx], fwd_ret.iloc[train_idx])
            oos_ic = compute_rank_ic(signal.iloc[test_idx], fwd_ret.iloc[test_idx])
            
            if is_ic is not None:
                is_ics.append(is_ic)
            if oos_ic is not None:
                oos_ics.append(oos_ic)
        
        if not is_ics or not oos_ics:
            continue
        
        is_ic_mean = np.mean(is_ics)
        oos_ic_mean = np.mean(oos_ics)
        
        # Check gates
        finite_pass, _ = check_finite_metrics_gate({
            'is_ic': is_ic_mean,
            'oos_ic': oos_ic_mean
        })
        oos_pass, _ = check_oos_degradation_gate(is_ic_mean, oos_ic_mean)
        
        validation_passed = finite_pass and oos_pass
        
        outcomes.append({
            'seed': seed,
            'is_ic': is_ic_mean,
            'oos_ic': oos_ic_mean,
            'passed': validation_passed
        })
    
    # Analyze variance
    pass_rate = np.mean([o['passed'] for o in outcomes])
    is_ic_std = np.std([o['is_ic'] for o in outcomes])
    oos_ic_std = np.std([o['oos_ic'] for o in outcomes])
    
    print(f"Results across {len(outcomes)} seeds:")
    print(f"  Validation pass rate: {pass_rate:.1%}")
    print(f"  IS IC std dev: {is_ic_std:.4f}")
    print(f"  OOS IC std dev: {oos_ic_std:.4f}")
    print()
    
    if pass_rate < 0.8 or pass_rate > 0.99:
        print(f"⚠️  Pass rate varies significantly with seed!")
    else:
        print(f"✓ Validation outcome relatively stable across seeds")
    
    print()
    return outcomes


def measure_acf_structure(n_bars=500):
    """
    DeepSeek Challenge: Is 1-bar embargo sufficient?
    
    Measure autocorrelation decay in typical return series.
    """
    print("=" * 70)
    print("TEST 3: AUTOCORRELATION STRUCTURE")
    print("=" * 70)
    print(f"Measuring ACF decay in synthetic returns...")
    print()
    
    np.random.seed(42)
    
    # Generate realistic return series (GARCH-like)
    returns = []
    vol = 0.01
    for _ in range(n_bars):
        ret = np.random.randn() * vol
        returns.append(ret)
        # Volatility clustering
        vol = 0.9 * vol + 0.1 * abs(ret)
    
    returns = np.array(returns)
    
    # Compute ACF
    from scipy.stats import pearsonr
    max_lag = 50
    acf_values = []
    
    for lag in range(1, max_lag + 1):
        if lag < len(returns):
            corr, _ = pearsonr(returns[:-lag], returns[lag:])
            acf_values.append(corr)
        else:
            acf_values.append(0)
    
    # Find where ACF becomes insignificant
    significance_bound = 1.96 / np.sqrt(n_bars)
    
    decay_lag = None
    for lag, acf in enumerate(acf_values, 1):
        if abs(acf) < significance_bound:
            decay_lag = lag
            break
    
    print(f"ACF Analysis:")
    print(f"  Sample size: {n_bars}")
    print(f"  Significance bound: ±{significance_bound:.4f}")
    print(f"  ACF decays to insignificance at lag: {decay_lag}")
    print()
    
    print(f"First 10 lags:")
    for lag in range(1, min(11, len(acf_values) + 1)):
        sig = "***" if abs(acf_values[lag-1]) > significance_bound else ""
        print(f"  Lag {lag:2d}: {acf_values[lag-1]:7.4f} {sig}")
    print()
    
    if decay_lag and decay_lag > 1:
        print(f"⚠️  ACF persists beyond 1 bar! Embargo should be ~{decay_lag} bars.")
    else:
        print(f"✓ 1-bar embargo appears sufficient for this series")
    
    print()
    return acf_values, decay_lag


def main():
    """Run all Phase 0 diagnostics."""
    print("\n")
    print("=" * 70)
    print("PHASE 0 DIAGNOSTICS: Current Lab V0 State")
    print("=" * 70)
    print()
    
    results = {}
    
    # Test 1: False Discovery Rate
    fdr, discoveries = test_false_discovery_rate(n_trials=100)
    results['fdr'] = {
        'rate': fdr,
        'discoveries': len(discoveries),
        'trials': 100
    }
    
    # Test 2: Seed Sensitivity
    seed_outcomes = test_seed_sensitivity(n_seeds=100)
    results['seed_sensitivity'] = {
        'n_seeds': len(seed_outcomes),
        'pass_rate': np.mean([o['passed'] for o in seed_outcomes]),
        'is_ic_std': np.std([o['is_ic'] for o in seed_outcomes]),
        'oos_ic_std': np.std([o['oos_ic'] for o in seed_outcomes])
    }
    
    # Test 3: ACF Structure
    acf_values, decay_lag = measure_acf_structure()
    results['acf'] = {
        'decay_lag': decay_lag,
        'acf_lag1': acf_values[0] if acf_values else None
    }
    
    # Save results
    output_path = Path(__file__).parent / 'diagnostic_phase0_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"False Discovery Rate: {fdr:.2%}")
    print(f"Seed Sensitivity: {results['seed_sensitivity']['pass_rate']:.1%} pass rate")
    print(f"ACF Decay Lag: {decay_lag}")
    print()
    print(f"Results saved to: {output_path}")
    print()
    
    return results


if __name__ == '__main__':
    main()
