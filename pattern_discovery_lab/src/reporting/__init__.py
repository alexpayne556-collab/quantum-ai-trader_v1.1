""""""
















































































































































































































































































































































    return output_path            f.write('\n'.join(lines))    with open(output_path, 'w') as f:                lines.append(f"")                lines.append(f"- Block {bd['block']}: score={bd['score']:.4f}, p={bd['p_value']:.4f}, decision={bd['decision']}")            for bd in block_decisions:            lines.append(f"**{dataset_name} x {detector_name}** (stability_decision={stab_dec:.3f})")                        stab_dec = detector_result.get('stability_decision', 0.0)            block_decisions = detector_result.get('block_decisions', [])            detector_name = detector_result['detector_name']        for detector_result in dataset_result['detectors']:                dataset_name = dataset_result['dataset_name']    for dataset_result in results['datasets']:        lines.append(f"")    lines.append(f"## Block Decisions (Stability Detail)")    # Block decisions per detector                lines.append(f"")            lines.append(f"- stability_decision={stab_dec:.3f}")            lines.append(f"- spectrum_error_max={spec_err_max:.4f}, spectrum_check={spectrum_chk}")            lines.append(f"- surrogate: p = (k+1)/(N+1) = ({surr_k}+1)/({surr_N}+1) = {surr_p:.4f}")            lines.append(f"- rho1_shuffled_max={rho1_max:.4f}, T(N={N})={thresh:.4f}, shuffle_check={shuffle_chk}")            lines.append(f"- real_score={real_score:.4f}")            lines.append(f"**{dataset_name} x {detector_name}** ({class_abbr}, {gate_status})")                        class_abbr = "CERT" if is_certified else "EXP"            gate_status = 'CONTRIBUTES' if is_certified else 'EXCLUDED'                        stab_dec = detector_result.get('stability_decision', 0.0)            spectrum_chk = detector_result.get('spectrum_check', 'PASS')            spec_err_max = detector_result.get('spectrum_error_max', 0.0)            surr_p = detector_result.get('surrogate_p_value', 0.0)            surr_k = detector_result.get('surrogate_k', 0)            surr_N = detector_result.get('surrogate_N', 100)            shuffle_chk = detector_result.get('shuffle_check', 'PASS')            M = detector_result.get('M', M_value)            N = detector_result.get('N', N_value)            thresh = detector_result.get('shuffle_threshold', shuffle_threshold)            rho1_max = detector_result.get('rho1_shuffled_max', 0.0)            shuffle_p = detector_result['shuffle_p_value']            real_score = detector_result['real_score']            is_certified = detector_class == 'CERTIFIED'            detector_class = detector_result.get('detector_class', 'EXPERIMENTAL')            detector_name = detector_result['detector_name']        for detector_result in dataset_result['detectors']:                dataset_name = dataset_result['dataset_name']    for dataset_result in results['datasets']:        lines.append(f"")    lines.append(f"## Detailed One-Line Summary")    # One-line per dataset x detector with full details        lines.append(f"")                lines.append(f"| {dataset_name} | {detector_name} | {class_abbr} | {real_score:.4f} | {shuffle_p:.4f} | {stat_dec} | {expected} | {exp_match} | {rho1_max:.4f} | {thresh:.4f} | {shuffle_chk} | {spec_err_max:.4f} | {spectrum_chk} | {stab_dec:.3f} | {gate_status} |")                        class_abbr = "CERT" if is_certified else "EXP"                            gate_status = 'EXCLUDED'                exp_match = 'N/A'                expected = 'N/A'            else:                gate_status = 'CONTRIBUTES'                    exp_match = 'N/A'                else:                    exp_match = 'PASS' if stat_dec == expected else 'FAIL'                if expected != 'N/A':                expected = EXPECTED_DECISIONS.get(detector_name, {}).get(dataset_name, 'N/A')            if is_certified:            # Layer 2: Expectation Match                        stat_dec = 'REJECT' if shuffle_p < 0.05 else 'ACCEPT'            # Layer 1: Statistical Decision                        spectrum_chk = detector_result.get('spectrum_check', 'PASS')            shuffle_chk = detector_result.get('shuffle_check', 'PASS')            stab_dec = detector_result.get('stability_decision', 0.0)            spec_err_max = detector_result.get('spectrum_error_max', 0.0)            thresh = detector_result.get('shuffle_threshold', shuffle_threshold)            rho1_max = detector_result.get('rho1_shuffled_max', 0.0)            shuffle_p = detector_result['shuffle_p_value']            real_score = detector_result['real_score']            is_certified = detector_class == 'CERTIFIED'            detector_class = detector_result.get('detector_class', 'EXPERIMENTAL')            detector_name = detector_result['detector_name']        for detector_result in dataset_result['detectors']:                dataset_name = dataset_result['dataset_name']    for dataset_result in results['datasets']:        lines.append(f"|---------|----------|-------|-------|--------|----------|----------|-----------|----------|------|----------|----------|----------|----------|------|")    lines.append(f"| Dataset | Detector | Class | Score | Shuf p | Stat Dec | Expected | Exp Match | rho1_max | T(N) | Shuf Chk | Spec Err | Spec Chk | Stab Dec | Gate |")    lines.append(f"")    lines.append(f"## Results Summary")    # Results Summary table with THREE-LAYER columns        lines.append(f"")    lines.append(f"**Overall Status**: **{overall_status}**")            lines.append(f"")            lines.append(f"- X {reason}")        for reason in halt_reasons:        lines.append(f"### HALT Reasons")    if halt_reasons:    # Halt reasons summary        lines.append(f"")    lines.append(f"- **Verdict**: **{exp_verdict}**")        lines.append(f"  - {r['dataset']} x {r['detector']}: p={r['shuffle_p']:.4f} -> {r['statistical_decision']} (expected: {r['expected']}) -> {r['expectation_match']}")    for r in expectation_results:    lines.append(f"- Expectation match: PASS if statistical decision matches expected")    lines.append(f"- Statistical decision: REJECT if p < 0.05, else ACCEPT")    lines.append(f"### Expectation Match (CERTIFIED only)")    # Expectation Match (Layer 2)        lines.append(f"")    lines.append(f"- **Verdict**: **{gate_c_verdict}**")        lines.append(f"  - {r['dataset']} x {r['detector']}: stability_decision={r['stability_decision']:.3f} -> {r['check']}")    for r in gate_c_results:    lines.append(f"- **Threshold**: stability_decision == 1.0 (3/3 blocks agree)")    lines.append(f"### Gate C: Decision Stability (CERTIFIED only)")    # Gate C        lines.append(f"")    lines.append(f"- **Verdict**: **{gate_b_verdict}**")    lines.append(f"- **Worst case**: spectrum_error_max = {gate_b_worst:.4f}")        lines.append(f"  - {r['dataset']} x {r['detector']}: spec_err_max={r['spectrum_error_max']:.4f}, spec_err_mean={r['spectrum_error_mean']:.4f} -> {r['check']}")    for r in gate_b_results:    lines.append(f"- **Evaluated on**: CERTIFIED detectors only")    lines.append(f"- **Threshold**: spectrum_error_max <= 0.05 (5%)")    lines.append(f"### Gate B: Spectrum Preservation")    # Gate B        lines.append(f"")    lines.append(f"- **Verdict**: **{gate_a_verdict}**")    lines.append(f"- **Worst case**: rho1_shuffled_max = {gate_a_worst:.4f}")        lines.append(f"  - {r['dataset']} x {r['detector']}: rho1_max={r['rho1_max']:.4f} <= {r['threshold']:.4f} -> {r['check']}")    for r in gate_a_results:    lines.append(f"- **Evaluated on**: CERTIFIED detectors only")    lines.append(f"- **Shuffles**: M = {M_value}")    lines.append(f"- **Threshold**: T(N) = 4.5/sqrt(N) = 4.5/sqrt({N_value}) = {shuffle_threshold:.4f}")    lines.append(f"### Gate A: Shuffle Control (rho1)")    # Gate A        lines.append(f"")    lines.append(f"## STOP RULES EVALUATED (ENFORCE_NOW)")    # STOP RULES section with explicit gate evaluation                halt_reasons.append(f"Expectation: {f['detector']} on {f['dataset']} expected {f['expected']} but got {f['statistical_decision']}")        for f in exp_failures:        overall_status = "HALT"    if exp_verdict == "HALT":    exp_verdict = "PASS" if len(exp_failures) == 0 else "HALT"    exp_failures = [r for r in expectation_results if r['expectation_match'] == 'FAIL']    # Expectation match check (all CERTIFIED detectors must match expected)                halt_reasons.append(f"Gate C: {f['detector']} on {f['dataset']} has stability_decision={f['stability_decision']:.3f} != 1.0")        for f in gate_c_failures:        overall_status = "HALT"    if gate_c_verdict == "HALT":    gate_c_verdict = "PASS" if len(gate_c_failures) == 0 else "HALT"    gate_c_failures = [r for r in gate_c_results if r['stability_decision'] != 1.0]    # Gate C check (CERTIFIED must have stability_decision == 1.0)            halt_reasons.append(f"Gate B: spectrum_error_max={gate_b_worst:.4f} > 0.05")        overall_status = "HALT"    if gate_b_verdict == "HALT":    gate_b_verdict = "PASS" if gate_b_worst <= 0.05 else "HALT"    gate_b_worst = max((r['spectrum_error_max'] for r in gate_b_results), default=0.0)    # Gate B check            halt_reasons.append(f"Gate A: rho1_shuffled_max={gate_a_worst:.4f} > T({N_value})={shuffle_threshold:.4f}")        overall_status = "HALT"    if gate_a_verdict == "HALT":    gate_a_verdict = "PASS" if gate_a_worst <= shuffle_threshold else "HALT"    gate_a_worst = max((r['rho1_max'] for r in gate_a_results), default=0.0)    # Gate A check with N-based threshold        halt_reasons = []    overall_status = "PASS"    # Determine overall status                    })                    'expectation_match': expectation_match                    'expected': expected if expected else 'N/A',                    'statistical_decision': statistical_decision,                    'shuffle_p': shuffle_p,                    'detector': detector_name,                    'dataset': dataset_name,                expectation_results.append({                                    expectation_match = 'N/A'                else:                    expectation_match = 'PASS' if statistical_decision == expected else 'FAIL'                if expected is not None:                expected = EXPECTED_DECISIONS.get(detector_name, {}).get(dataset_name, None)                                statistical_decision = 'REJECT' if shuffle_p < 0.05 else 'ACCEPT'                shuffle_p = detector_result.get('shuffle_p_value', 1.0)                # Expectation Match (Layer 2)                                })                    'check': 'PASS' if stab_dec == 1.0 else 'HALT'                    'stability_decision': stab_dec,                    'detector': detector_name,                    'dataset': dataset_name,                gate_c_results.append({                stab_dec = detector_result.get('stability_decision', 0.0)                # Gate C                                })                    'check': detector_result.get('spectrum_check', 'PASS')                    'spectrum_error_mean': detector_result.get('spectrum_error_mean', 0.0),                    'spectrum_error_max': spec_max,                    'detector': detector_name,                    'dataset': dataset_name,                gate_b_results.append({                spec_max = detector_result.get('spectrum_error_max', 0.0)                # Gate B                                })                    'check': detector_result.get('shuffle_check', 'PASS')                    'threshold': detector_result.get('shuffle_threshold', shuffle_threshold),                    'rho1_max': rho1_max,                    'detector': detector_name,                    'dataset': dataset_name,                gate_a_results.append({                rho1_max = detector_result.get('rho1_shuffled_max', 0.0)                # Gate A            if is_certified:                            shuffle_threshold = detector_result.get('shuffle_threshold', 4.5 / np.sqrt(N_value))                M_value = detector_result.get('M', 100)                N_value = detector_result['N']            if 'N' in detector_result:            # Get N, M, threshold from result if available                        is_certified = detector_class == 'CERTIFIED'            detector_class = detector_result.get('detector_class', 'EXPERIMENTAL')            detector_name = detector_result['detector_name']        for detector_result in dataset_result['detectors']:        dataset_name = dataset_result['dataset_name']    for dataset_result in results['datasets']:        shuffle_threshold = 4.5 / np.sqrt(1000)  # default for N=1000    M_value = 100   # default    N_value = 1000  # default    # Get N and M from first detector result for threshold display        expectation_results = []  # expectation match    gate_c_results = []  # stability decision    gate_b_results = []  # spectrum error    gate_a_results = []  # shuffle rho1    # Collect gate results for CERTIFIED detectors        lines.append(f"")    lines.append(f"**Config**: {results['config']['name']}  ")    lines.append(f"**Timestamp**: {results['timestamp']}  ")    lines.append(f"**Run ID**: `{run_id}`  ")    lines.append(f"")    lines.append(f"# Pattern Discovery Lab - Run Report")    lines = []        output_path = os.path.join(output_dir, 'report.md')        os.makedirs(output_dir, exist_ok=True)    """    Layer 3: Overall Status (PASS/HALT)    Layer 2: Expectation Match (PASS/FAIL)    Layer 1: Statistical Decision (ACCEPT/REJECT)        Write report.md with three-layer model.    """def write_report_md(run_id: str, results: Dict[str, Any], output_dir: str):    return output_path            json.dump(serializable_results, f, indent=2)    with open(output_path, 'w') as f:        serializable_results = convert_to_serializable(results)    # Convert numpy types to native Python types        output_path = os.path.join(output_dir, 'results.json')        os.makedirs(output_dir, exist_ok=True)    """Write results.json."""def write_results_json(run_id: str, results: Dict[str, Any], output_dir: str):    return obj        return [convert_to_serializable(item) for item in obj]    elif isinstance(obj, list):        return {k: convert_to_serializable(v) for k, v in obj.items()}    elif isinstance(obj, dict):        return obj.tolist()    elif isinstance(obj, np.ndarray):        return bool(obj)    elif isinstance(obj, np.bool_):        return float(obj)    if isinstance(obj, (np.integer, np.floating)):    """Convert numpy types to Python native types for JSON serialization."""def convert_to_serializable(obj):}    }        'garch': 'ACCEPT',        # no linear autocorrelation        'ar1': 'REJECT',          # strong linear structure        'white_noise': 'ACCEPT',  # no linear structure    'autocorrelation_dependence': {EXPECTED_DECISIONS = {# Expected decisions for CERTIFIED detectorsfrom typing import Dict, Any, Listfrom datetime import datetimeimport numpy as npimport osimport json"""  Layer 3: Overall Status (system gate based on all checks)  Layer 2: Expectation Match (PASS/FAIL based on expected decision for dataset)  Layer 1: Statistical Decision (ACCEPT/REJECT based on p < alpha)Implements THREE-LAYER MODEL:Reporting module - writes JSON logs and Markdown reports.Reporting module - writes JSON logs and Markdown reports.

Implements THREE-LAYER MODEL:
  Layer 1: Statistical Decision (ACCEPT/REJECT based on p < alpha)
  Layer 2: Expectation Match (PASS/FAIL based on expected decision for dataset)
  Layer 3: Overall Status (system gate based on all checks)
"""

import json
import os
import numpy as np
from datetime import datetime
from typing import Dict, Any, List


# Expected decisions for CERTIFIED detectors
EXPECTED_DECISIONS = {
    'autocorrelation_dependence': {
        'white_noise': 'ACCEPT',  # no linear structure
        'ar1': 'REJECT',          # strong linear structure
        'garch': 'ACCEPT',        # no linear autocorrelation
    }
}


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


def write_results_json(run_id: str, results: Dict[str, Any], output_dir: str):
    """Write results.json."""
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'results.json')
    
    # Convert numpy types to native Python types
    serializable_results = convert_to_serializable(results)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    return output_path


def write_report_md(run_id: str, results: Dict[str, Any], output_dir: str):
    """
    Write report.md with three-layer model.
    
    Layer 1: Statistical Decision (ACCEPT/REJECT)
    Layer 2: Expectation Match (PASS/FAIL)
    Layer 3: Overall Status (PASS/HALT)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'report.md')
    
    lines = []
    lines.append(f"# Pattern Discovery Lab - Run Report")
    lines.append(f"")
    lines.append(f"**Run ID**: `{run_id}`  ")
    lines.append(f"**Timestamp**: {results['timestamp']}  ")
    lines.append(f"**Config**: {results['config']['name']}  ")
    lines.append(f"")
    
    # Collect gate results for CERTIFIED detectors
    gate_a_results = []  # shuffle rho1
    gate_b_results = []  # spectrum error
    gate_c_results = []  # stability decision
    expectation_results = []  # expectation match
    
    # Get N and M from first detector result for threshold display
    N_value = 1000  # default
    M_value = 100   # default
    shuffle_threshold = 4.5 / np.sqrt(1000)  # default for N=1000
    
    for dataset_result in results['datasets']:
        dataset_name = dataset_result['dataset_name']
        for detector_result in dataset_result['detectors']:
            detector_name = detector_result['detector_name']
            detector_class = detector_result.get('detector_class', 'EXPERIMENTAL')
            is_certified = detector_class == 'CERTIFIED'
            
            # Get N, M, threshold from result if available
            if 'N' in detector_result:
                N_value = detector_result['N']
                M_value = detector_result.get('M', 100)
                shuffle_threshold = detector_result.get('shuffle_threshold', 4.5 / np.sqrt(N_value))
            
            if is_certified:
                # Gate A
                rho1_max = detector_result.get('rho1_shuffled_max', 0.0)
                gate_a_results.append({
                    'dataset': dataset_name,
                    'detector': detector_name,
                    'rho1_max': rho1_max,
                    'threshold': detector_result.get('shuffle_threshold', shuffle_threshold),
                    'check': detector_result.get('shuffle_check', 'PASS')
                })
                
                # Gate B
                spec_max = detector_result.get('spectrum_error_max', 0.0)
                gate_b_results.append({
                    'dataset': dataset_name,
                    'detector': detector_name,
                    'spectrum_error_max': spec_max,
                    'spectrum_error_mean': detector_result.get('spectrum_error_mean', 0.0),
                    'check': detector_result.get('spectrum_check', 'PASS')
                })
                
                # Gate C
                stab_dec = detector_result.get('stability_decision', 0.0)
                gate_c_results.append({
                    'dataset': dataset_name,
                    'detector': detector_name,
                    'stability_decision': stab_dec,
                    'check': 'PASS' if stab_dec == 1.0 else 'HALT'
                })
                
                # Expectation Match (Layer 2)
                shuffle_p = detector_result.get('shuffle_p_value', 1.0)
                statistical_decision = 'REJECT' if shuffle_p < 0.05 else 'ACCEPT'
                
                expected = EXPECTED_DECISIONS.get(detector_name, {}).get(dataset_name, None)
                if expected is not None:
                    expectation_match = 'PASS' if statistical_decision == expected else 'FAIL'
                else:
                    expectation_match = 'N/A'
                
                expectation_results.append({
                    'dataset': dataset_name,
                    'detector': detector_name,
                    'shuffle_p': shuffle_p,
                    'statistical_decision': statistical_decision,
                    'expected': expected if expected else 'N/A',
                    'expectation_match': expectation_match
                })
    
    # Determine overall status
    overall_status = "PASS"
    halt_reasons = []
    
    # Gate A check with N-based threshold
    gate_a_worst = max((r['rho1_max'] for r in gate_a_results), default=0.0)
    gate_a_verdict = "PASS" if gate_a_worst <= shuffle_threshold else "HALT"
    if gate_a_verdict == "HALT":
        overall_status = "HALT"
        halt_reasons.append(f"Gate A: rho1_shuffled_max={gate_a_worst:.4f} > T({N_value})={shuffle_threshold:.4f}")
    
    # Gate B check
    gate_b_worst = max((r['spectrum_error_max'] for r in gate_b_results), default=0.0)
    gate_b_verdict = "PASS" if gate_b_worst <= 0.05 else "HALT"
    if gate_b_verdict == "HALT":
        overall_status = "HALT"
        halt_reasons.append(f"Gate B: spectrum_error_max={gate_b_worst:.4f} > 0.05")
    
    # Gate C check (CERTIFIED must have stability_decision == 1.0)
    gate_c_failures = [r for r in gate_c_results if r['stability_decision'] != 1.0]
    gate_c_verdict = "PASS" if len(gate_c_failures) == 0 else "HALT"
    if gate_c_verdict == "HALT":
        overall_status = "HALT"
        for f in gate_c_failures:
            halt_reasons.append(f"Gate C: {f['detector']} on {f['dataset']} has stability_decision={f['stability_decision']:.3f} != 1.0")
    
    # Expectation match check (all CERTIFIED detectors must match expected)
    exp_failures = [r for r in expectation_results if r['expectation_match'] == 'FAIL']
    exp_verdict = "PASS" if len(exp_failures) == 0 else "HALT"
    if exp_verdict == "HALT":
        overall_status = "HALT"
        for f in exp_failures:
            halt_reasons.append(f"Expectation: {f['detector']} on {f['dataset']} expected {f['expected']} but got {f['statistical_decision']}")
    
    # STOP RULES section with explicit gate evaluation
    lines.append(f"## STOP RULES EVALUATED (ENFORCE_NOW)")
    lines.append(f"")
    
    # Gate A
    lines.append(f"### Gate A: Shuffle Control (rho1)")
    lines.append(f"- **Threshold**: T(N) = 4.5/sqrt(N) = 4.5/sqrt({N_value}) = {shuffle_threshold:.4f}")
    lines.append(f"- **Shuffles**: M = {M_value}")
    lines.append(f"- **Evaluated on**: CERTIFIED detectors only")
    for r in gate_a_results:
        lines.append(f"  - {r['dataset']} × {r['detector']}: rho1_max={r['rho1_max']:.4f} <= {r['threshold']:.4f} → {r['check']}")
    lines.append(f"- **Worst case**: rho1_shuffled_max = {gate_a_worst:.4f}")
    lines.append(f"- **Verdict**: **{gate_a_verdict}**")
    lines.append(f"")
    
    # Gate B
    lines.append(f"### Gate B: Spectrum Preservation")
    lines.append(f"- **Threshold**: spectrum_error_max <= 0.05 (5%)")
    lines.append(f"- **Evaluated on**: CERTIFIED detectors only")
    for r in gate_b_results:
        lines.append(f"  - {r['dataset']} × {r['detector']}: spec_err_max={r['spectrum_error_max']:.4f}, spec_err_mean={r['spectrum_error_mean']:.4f} → {r['check']}")
    lines.append(f"- **Worst case**: spectrum_error_max = {gate_b_worst:.4f}")
    lines.append(f"- **Verdict**: **{gate_b_verdict}**")
    lines.append(f"")
    
    # Gate C
    lines.append(f"### Gate C: Decision Stability (CERTIFIED only)")
    lines.append(f"- **Threshold**: stability_decision == 1.0 (3/3 blocks agree)")
    for r in gate_c_results:
        lines.append(f"  - {r['dataset']} × {r['detector']}: stability_decision={r['stability_decision']:.3f} → {r['check']}")
    lines.append(f"- **Verdict**: **{gate_c_verdict}**")
    lines.append(f"")
    
    # Expectation Match (Layer 2)
    lines.append(f"### Expectation Match (CERTIFIED only)")
    lines.append(f"- Statistical decision: REJECT if p < 0.05, else ACCEPT")
    lines.append(f"- Expectation match: PASS if statistical decision matches expected")
    for r in expectation_results:
        lines.append(f"  - {r['dataset']} × {r['detector']}: p={r['shuffle_p']:.4f} → {r['statistical_decision']} (expected: {r['expected']}) → {r['expectation_match']}")
    lines.append(f"- **Verdict**: **{exp_verdict}**")
    lines.append(f"")
    
    # Halt reasons summary
    if halt_reasons:
        lines.append(f"### HALT Reasons")
        for reason in halt_reasons:
            lines.append(f"- ❌ {reason}")
        lines.append(f"")
    
    lines.append(f"**Overall Status**: **{overall_status}**")
    lines.append(f"")
    
    # Results Summary table with THREE-LAYER columns
    lines.append(f"## Results Summary")
    lines.append(f"")
    lines.append(f"| Dataset | Detector | Class | Score | Shuf p | Stat Dec | Expected | Exp Match | rho1_max | T(N) | Shuf Chk | Spec Err | Spec Chk | Stab Dec | Gate |")
    lines.append(f"|---------|----------|-------|-------|--------|----------|----------|-----------|----------|------|----------|----------|----------|----------|------|")
    
    for dataset_result in results['datasets']:
        dataset_name = dataset_result['dataset_name']
        
        for detector_result in dataset_result['detectors']:
            detector_name = detector_result['detector_name']
            detector_class = detector_result.get('detector_class', 'EXPERIMENTAL')
            is_certified = detector_class == 'CERTIFIED'
            real_score = detector_result['real_score']
            shuffle_p = detector_result['shuffle_p_value']
            rho1_max = detector_result.get('rho1_shuffled_max', 0.0)
            thresh = detector_result.get('shuffle_threshold', shuffle_threshold)
            spec_err_max = detector_result.get('spectrum_error_max', 0.0)
            stab_dec = detector_result.get('stability_decision', 0.0)
            shuffle_chk = detector_result.get('shuffle_check', 'PASS')
            spectrum_chk = detector_result.get('spectrum_check', 'PASS')
            
            # Layer 1: Statistical Decision
            stat_dec = 'REJECT' if shuffle_p < 0.05 else 'ACCEPT'
            
            # Layer 2: Expectation Match
            if is_certified:
                expected = EXPECTED_DECISIONS.get(detector_name, {}).get(dataset_name, 'N/A')
                if expected != 'N/A':
                    exp_match = 'PASS' if stat_dec == expected else 'FAIL'
                else:
                    exp_match = 'N/A'
                gate_status = 'CONTRIBUTES'
            else:
                expected = 'N/A'
                exp_match = 'N/A'
                gate_status = 'EXCLUDED'
            
            class_abbr = "CERT" if is_certified else "EXP"
            
            lines.append(f"| {dataset_name} | {detector_name} | {class_abbr} | {real_score:.4f} | {shuffle_p:.4f} | {stat_dec} | {expected} | {exp_match} | {rho1_max:.4f} | {thresh:.4f} | {shuffle_chk} | {spec_err_max:.4f} | {spectrum_chk} | {stab_dec:.3f} | {gate_status} |")
    
    lines.append(f"")
    
    # One-line per dataset×detector with full details
    lines.append(f"## Detailed One-Line Summary")
    lines.append(f"")
    
    for dataset_result in results['datasets']:
        dataset_name = dataset_result['dataset_name']
        
        for detector_result in dataset_result['detectors']:
            detector_name = detector_result['detector_name']
            detector_class = detector_result.get('detector_class', 'EXPERIMENTAL')
            is_certified = detector_class == 'CERTIFIED'
            real_score = detector_result['real_score']
            shuffle_p = detector_result['shuffle_p_value']
            rho1_max = detector_result.get('rho1_shuffled_max', 0.0)
            thresh = detector_result.get('shuffle_threshold', shuffle_threshold)
            N = detector_result.get('N', N_value)
            M = detector_result.get('M', M_value)
            shuffle_chk = detector_result.get('shuffle_check', 'PASS')
            surr_N = detector_result.get('surrogate_N', 100)
            surr_k = detector_result.get('surrogate_k', 0)
            surr_p = detector_result.get('surrogate_p_value', 0.0)
            spec_err_max = detector_result.get('spectrum_error_max', 0.0)
            spectrum_chk = detector_result.get('spectrum_check', 'PASS')
            stab_dec = detector_result.get('stability_decision', 0.0)
            
            gate_status = 'CONTRIBUTES' if is_certified else 'EXCLUDED'
            class_abbr = "CERT" if is_certified else "EXP"
            
            lines.append(f"**{dataset_name} × {detector_name}** ({class_abbr}, {gate_status})")
            lines.append(f"- real_score={real_score:.4f}")
            lines.append(f"- rho1_shuffled_max={rho1_max:.4f}, T(N={N})={thresh:.4f}, shuffle_check={shuffle_chk}")
            lines.append(f"- surrogate: p = (k+1)/(N+1) = ({surr_k}+1)/({surr_N}+1) = {surr_p:.4f}")
            lines.append(f"- spectrum_error_max={spec_err_max:.4f}, spectrum_check={spectrum_chk}")
            lines.append(f"- stability_decision={stab_dec:.3f}")
            lines.append(f"")
    
    # Block decisions per detector
    lines.append(f"## Block Decisions (Stability Detail)")
    lines.append(f"")
    
    for dataset_result in results['datasets']:
        dataset_name = dataset_result['dataset_name']
        
        for detector_result in dataset_result['detectors']:
            detector_name = detector_result['detector_name']
            block_decisions = detector_result.get('block_decisions', [])
            stab_dec = detector_result.get('stability_decision', 0.0)
            
            lines.append(f"**{dataset_name} × {detector_name}** (stability_decision={stab_dec:.3f})")
            for bd in block_decisions:
                lines.append(f"- Block {bd['block']}: score={bd['score']:.4f}, p={bd['p_value']:.4f}, decision={bd['decision']}")
            lines.append(f"")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    return output_path
