#!/usr/bin/env python3
"""
Validator for Pattern Discovery Lab v0.1 Contract
Validates stdout and results.json against enforcement contract.
"""
import sys
import json
import re


def validate_stdout(stdout_path):
    """Validate stdout format."""
    with open(stdout_path) as f:
        content = f.read()
    
    issues = []
    
    # Must have 5 sections
    required_sections = [
        r'CLI Command:',
        r'Run Folder:',
        r'STOP RULES EVALUATED \(ENFORCE_NOW\)',
        r'\| Dataset.*\| Detector',
        r'OVERALL STATUS DETERMINATION'
    ]
    
    for pattern in required_sections:
        if not re.search(pattern, content):
            issues.append(f"Missing section: {pattern}")
    
    # Must have exact field labels
    required_labels = [
        'shuffle_check:',
        'spectrum_check:',
        'stability_gate:',
        'no_hallucination_check:',
        'Overall Status:',
        'Reason:',
    ]
    
    for label in required_labels:
        if label not in content:
            issues.append(f"Missing field label: {label}")
    
    # Must have surrogate formula
    if not re.search(r'p=\(k\+1\)/\(N\+1\)=\(\(\d+\)\+1\)/\(\(\d+\)\+1\)=\d+\.\d+', content):
        issues.append("Missing surrogate formula p=(k+1)/(N+1)=((X)+1)/((Y)+1)=X.XXXX")
    
    # Must NOT have debug logs
    forbidden = ['INFO', 'WARNING', 'DEBUG', 'Traceback']
    for word in forbidden:
        if word in content:
            issues.append(f"Contains forbidden word: {word}")
    
    return issues


def validate_json(json_path):
    """Validate results.json schema."""
    with open(json_path) as f:
        data = json.load(f)
    
    issues = []
    
    # Top-level keys
    required_top = ['stop_rules', 'detector_results', 'overall_status']
    for key in required_top:
        if key not in data:
            issues.append(f"Missing top-level key: {key}")
    
    if 'stop_rules' in data:
        sr = data['stop_rules']
        
        # shuffle_check
        if 'shuffle_check' not in sr:
            issues.append("Missing stop_rules.shuffle_check")
        else:
            sc = sr['shuffle_check']
            for field in ['rho1_shuffled_max', 'threshold_T_N', 'verdict']:
                if field not in sc:
                    issues.append(f"Missing shuffle_check.{field}")
        
        # spectrum_check
        if 'spectrum_check' not in sr:
            issues.append("Missing stop_rules.spectrum_check")
        else:
            sc = sr['spectrum_check']
            for field in ['spectrum_error_max', 'verdict']:
                if field not in sc:
                    issues.append(f"Missing spectrum_check.{field}")
        
        # stability_gate
        if 'stability_gate' not in sr:
            issues.append("Missing stop_rules.stability_gate")
        else:
            sg = sr['stability_gate']
            if 'verdict' not in sg:
                issues.append("Missing stability_gate.verdict")
        
        # no_hallucination_check
        if 'no_hallucination_check' not in sr:
            issues.append("Missing stop_rules.no_hallucination_check")
        else:
            nhc = sr['no_hallucination_check']
            for field in ['dataset', 'detector', 'statistical_decision', 'verdict']:
                if field not in nhc:
                    issues.append(f"Missing no_hallucination_check.{field}")
    
    # detector_results
    if 'detector_results' in data:
        for i, row in enumerate(data['detector_results']):
            required_row_fields = [
                'dataset', 'detector', 'class', 'gate_status',
                'real_score', 'surrogate_N', 'surrogate_k', 'surrogate_p',
                'surrogate_formula', 'statistical_decision', 'expected_decision',
                'expectation_match', 'rho1_shuffled', 'spectrum_error'
            ]
            for field in required_row_fields:
                if field not in row:
                    issues.append(f"detector_results[{i}] missing field: {field}")
    
    # overall_status
    if 'overall_status' in data:
        os = data['overall_status']
        if not isinstance(os, dict):
            issues.append("overall_status must be object, not string")
        else:
            for field in ['verdict', 'reason']:
                if field not in os:
                    issues.append(f"Missing overall_status.{field}")
    
    return issues


def main():
    if len(sys.argv) != 3:
        print("Usage: validate_run_v0p1.py <stdout.txt> <results.json>")
        sys.exit(1)
    
    stdout_path = sys.argv[1]
    json_path = sys.argv[2]
    
    print("=" * 70)
    print("VALIDATION REPORT - Enforcement Contract v0.1")
    print("=" * 70)
    print()
    
    # Validate stdout
    print("STDOUT VALIDATION:")
    stdout_issues = validate_stdout(stdout_path)
    if stdout_issues:
        for issue in stdout_issues:
            print(f"  ✗ {issue}")
    else:
        print("  ✓ All checks passed")
    print()
    
    # Validate JSON
    print("JSON SCHEMA VALIDATION:")
    json_issues = validate_json(json_path)
    if json_issues:
        for issue in json_issues:
            print(f"  ✗ {issue}")
    else:
        print("  ✓ All checks passed")
    print()
    
    # Summary
    total_issues = len(stdout_issues) + len(json_issues)
    total_checks = 11  # Approximate
    passed = total_checks - total_issues
    
    print("=" * 70)
    print(f"VALIDATION RESULT: {passed}/{total_checks} checks passed")
    print("=" * 70)
    
    if total_issues > 0:
        sys.exit(1)
    else:
        print("\n✓ Contract satisfied!")
        sys.exit(0)


if __name__ == '__main__':
    main()
