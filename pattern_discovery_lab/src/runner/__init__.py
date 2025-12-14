"""
Runner - orchestrates full calibration pipeline.
"""

import os
import yaml
import argparse
from datetime import datetime
from typing import Dict, Any, List
import numpy as np

from pattern_discovery_lab.src.datasets.pure_noise import WhiteNoiseDataset
from pattern_discovery_lab.src.datasets.known_structure import ARProcessDataset, GARCHProcessDataset
from pattern_discovery_lab.src.detectors.dependence import AutocorrelationDetector
from pattern_discovery_lab.src.detectors.time_reversal import TimeReversalAsymmetryDetector
from pattern_discovery_lab.src.controls import TimeShuffleControl, PhaseRandomizationControl, compute_shuffle_threshold
from pattern_discovery_lab.src.reporting import write_results_json, write_report_md


def generate_run_id() -> str:
    """Generate unique run ID."""
    return datetime.utcnow().strftime('%Y%m%d_%H%M%S')


# Detector classification
CERTIFIED_DETECTORS = {'autocorrelation_dependence'}
EXPERIMENTAL_DETECTORS = {'time_reversal_asymmetry'}

# Expected decisions for CERTIFIED detectors (for expectation match)
# Statistical decision: REJECT if p < alpha, ACCEPT otherwise
# Expectation match: PASS if matches expected, FAIL otherwise
EXPECTED_DECISIONS = {
    'autocorrelation_dependence': {
        'white_noise': 'ACCEPT',  # no linear structure
        'ar1': 'REJECT',          # strong linear structure
        'garch': 'ACCEPT',        # no linear autocorrelation (only volatility clustering)
    }
}


def get_detector_class(detector_name: str) -> str:
    """Return CERTIFIED or EXPERIMENTAL."""
    if detector_name in CERTIFIED_DETECTORS:
        return 'CERTIFIED'
    return 'EXPERIMENTAL'


def run_detector_with_controls(detector, data: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run detector with full control battery.
    
    Returns:
        Dict with scores, p-values, stability, pass/fail
    """
    detector_name = detector.get_name()
    detector_class = get_detector_class(detector_name)
    
    # Real score
    real_score = detector.detect(data)
    
    # Time-shuffle control (use 25 shuffles for gate validation)
    n_shuffles = config.get('n_shuffles', 100)
    shuffle_control = TimeShuffleControl(n_shuffles=n_shuffles, seed=config['seed'])
    shuffle_result = shuffle_control.test(detector, data, alpha=0.01)
    
    # Phase-randomization control
    n_surrogates = config.get('n_surrogates', 100)
    phase_control = PhaseRandomizationControl(n_surrogates=n_surrogates, seed=config['seed'])
    phase_result = phase_control.test(detector, data, alpha=0.01)
    
    # Extract ENFORCE_NOW metrics
    rho1_shuffled_mean = shuffle_result.metadata.get('mean_abs_rho1_shuffled', 0.0)
    rho1_shuffled_max = shuffle_result.metadata.get('max_abs_rho1_shuffled', 0.0)
    spectrum_error_mean = phase_result.metadata.get('mean_spectrum_error', 0.0)
    spectrum_error_max = phase_result.metadata.get('max_spectrum_error', 0.0)
    
    # Surrogate p-value with N, k
    surrogate_N = phase_result.metadata.get('N', n_surrogates)
    surrogate_k = phase_result.metadata.get('k', 0)
    
    # Stability: 3 non-overlapping blocks with DECISION per block
    n_blocks = 3
    block_size = len(data) // n_blocks
    block_scores = []
    block_decisions = []  # REJECT or ACCEPT per block
    alpha = config.get('alpha', 0.05)
    
    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size if i < n_blocks - 1 else len(data)
        block_data = data[start:end]
        if len(block_data) >= 10:
            block_score = detector.detect(block_data)
            block_scores.append(block_score)
            
            # Run shuffle test on block to get p-value for decision
            block_shuffle = TimeShuffleControl(n_shuffles=25, seed=config['seed'] + i)
            block_result = block_shuffle.test(detector, block_data, alpha=alpha)
            block_decision = 'REJECT' if block_result.p_value < alpha else 'ACCEPT'
            block_decisions.append({
                'block': i + 1,
                'score': float(block_score),
                'p_value': float(block_result.p_value),
                'decision': block_decision
            })
    
    # Stability decision: majority agreement / 3
    if len(block_decisions) == 3:
        n_reject = sum(1 for b in block_decisions if b['decision'] == 'REJECT')
        n_accept = 3 - n_reject
        majority_agreement = max(n_reject, n_accept)
        stability_decision = majority_agreement / 3.0
    else:
        stability_decision = 0.0
    
    # Stability CV (distributional, print-only)
    if len(block_scores) > 1:
        mean_block = np.mean(block_scores)
        std_block = np.std(block_scores)
        stability_cv = std_block / abs(mean_block) if abs(mean_block) > 1e-10 else 0.0
    else:
        stability_cv = 0.0
    
    # se:
        stability_decision = 0.0
    
    # Stability CV (distributional, print-only)
    if len(block_scores) > 1:
        mean_block = np.mean(block_scores)
        std_block = np.std(block_scores)
        stability_cv = std_block / abs(mean_block) if abs(mean_block) > 1e-10 else 0.0
    else:
        stability_cv = 0.0
    
    # Get data length for N-based threshold
    N = len(data)
    M = n_shuffles
    
    # ENFORCE_NOW checks with explicit verdicts
    # Gate A: Shuffle rho1 with N-based threshold: T(N) = 4.5/sqrt(N)
    shuffle_threshold = compute_shuffle_threshold(N, M)
    shuffle_check = "PASS" if rho1_shuffled_max <= shuffle_threshold else "HALT"
    
    # Gate B: Spectrum error (ANY surrogate must have error <= 0.05)
    spectrum_threshold = 0.05
    spectrum_check = "PASS" if spectrum_error_max <= spectrum_threshold else "HALT"
    
    # Gate C: Stability decision (CERTIFIED only, must be 1.0)
    stability_check = "PASS" if stability_decision == 1.0 else "WARN"
    
    # Determine row status
    failure_reasons = []
    
    if shuffle_check == "HALT":
        failure_reasons.append(f"shuffle_rho1_max={rho1_shuffled_max:.4f}>T({N})={shuffle_threshold:.4f}")
    
    if spectrum_check == "HALT":
        failure_reasons.append(f"spectrum_error_max={spectrum_error_max:.4f}>0.05")
    
    if not shuffle_result.passes:
        failure_reasons.append("failed_time_shuffle_p_value")
    
    passes = len(failure_reasons) == 0
    
    return {
        'detector_name': detector_name,
        'detector_class': detector_class,
        'real_score': float(real_score),
        'shuffle_p_value': float(shuffle_result.p_value),
        'rho1_shuffled_mean': float(rho1_shuffled_mean),
        'rho1_shuffled_max': float(rho1_shuffled_max),
        'shuffle_threshold': float(shuffle_threshold),
        'shuffle_check': shuffle_check,
        'surrogate_p_value': float(phase_result.p_value),
        'surrogate_N': int(surrogate_N),
        'surrogate_
    """
    run_id = generate_run_id()
    
    results = {
        'run_id': run_id,
        'timestamp': datetime.utcnow().isoformat(),
        'config': config,
        'datasets': []
    }
    
    seed = config['seed']
    length = config['series_length']
    
    # Generate datasets
    datasets = []
    
    # 1. White noise (should show NO structure)
    noise_ds = WhiteNoiseDataset(seed, {'length': length})
    noise_data = noise_ds.generate()
    datasets.append(('white_noise', noise_data, noise_ds.get_true_structure()))
    
    # 2. AR(1) (should show linear structure)
    ar_phi = config.get('ar1_phi', 0.7)
    ar_ds = ARProcessDataset(seed, {'length': length, 'phi': ar_phi})
    ar_data = ar_ds.generate()
    datasets.append(('ar1', ar_data, ar_ds.get_true_structure()))
    
    # 3. GARCH (nonlinear structure)
    garch_params = config.get('garch_params', {'omega': 0.1, 'alpha': 0.1, 'beta': 0.85})
    garch_ds = GARCHProcessDataset(seed, {
        'length': length,
        **garch_params
    })
    garch_data = garch_ds.generate()
    datasets.append(('garch', garch_data, garch_ds.get_true_structure()))
    
    # Initialize detectors
    detectors = [
        AutocorrelationDetector({'max_lag': 20}),
        TimeReversalAsymmetryDetector({'normalize': True})
    ]
    
    # Run detectors on each dataset
    for dataset_name, data, true_structure in datasets:
        dataset_result = {
            'dataset_name': dataset_name,
            'generator_params': {
                'name': dataset_name,
                'seed': seed,
                'length': length,
                'true_structure': true_structure
            },
            'detectors': []
        }
        
        for detector in detectors:
            detector_result = run_detector_with_controls(detector, data, config)
            dataset_result['detectors'].append(detector_result)
        
        results['datasets'].append(dataset_result)
    
    return results


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description='Pattern Discovery Lab Runner')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['single', 'calibrate'], default=None,
                       help='Run mode: single (default) or calibrate (PHASE_2 - not implemented)')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config['name'] = os.path.basename(args.config)
    
    # Mode override from CLI or config
    mode = args.mode or config.get('mode', 'single')
    
    if mode == 'calibrate':
        print("❌ ERROR: --mode calibrate not yet implemented (PHASE_2 placeholder)")
        print("TODO: Loop over n_realizations, aggregate rejection_rate vs shuffle/surrogate, stability_fail_rate")
        return
    
    # Run calibration
    print("=" * 80)
    print("PATTERN DISCOVERY LAB - CALIBRATION RUN")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Mode: {mode}")
    print(f"Seed: {config['seed']}")
    print(f"Series length: {config['series_length']}")
    print("")
    
    results = run_calibration(config)
    
    # Write outputs
    run_id = results['run_id']
    config_dir = os.path.dirname(os.path.abspath(args.config))
    project_root = os.path.dirname(config_dir)
    output_dir = os.path.join(project_root, 'runs', run_id)
    
    json_path = write_results_json(run_id, results, output_dir)
    report_path = write_report_md(run_id, results, output_dir)
    
    # ========================================================================
    # OUTPUT CONTRACT (per user spec)
    # ========================================================================
    
    # Get relative path from project root
    try:
        rel_output_dir = os.path.relpath(output_dir, os.getcwd())
    except ValueError:
        rel_output_dir = output_dir
    
    # Read report for extracting sections
    with open(report_path, 'r') as f:
        report_content = f.read()
    
    # 1. Exact CLI command executed
    cli_cmd = f"python -m pattern_discovery_lab --config {args.config}"
    if args.mode:
        cli_cmd += f" --mode {args.mode}"
    
    print("=" * 80)
    print("RUN COMPLETE")
    print("=" * 80)
    print("")
    print(f"**CLI Command Executed**:")
    print(f"```")
    print(f"{cli_cmd}")
    print(f"```")
    print("")
    
    # 2. Run folder path
    print(f"**Run Folder**: `{rel_output_dir}`")
    print("")
    
    # 3. Stop Rules Evaluated section
    in_stop_rules = False
    stop_rules_lines = []
    for line in report_content.split('\n'):
        if 'STOP RULES EVALUATED' in line:
            in_stop_rules = True
        if in_stop_rules:
            stop_rules_lines.append(line)
            if line.startswith('## Results Summary'):
                break
    
    print('\n'.join(stop_rules_lines[:stop_rules_lines.index('## Results Summary') if '## Results Summary' in stop_rules_lines else len(stop_rules_lines)]))
    print("")
    
    # 4. PASS/FAIL table
    in_table = False
    for line in report_content.split('\n'):
        if '## Results Summary' in line:
            in_table = True
        if in_table:
            print(line)
            if line.startswith('## Detailed'):
                break


if __name__ == '__main__':
    main()
