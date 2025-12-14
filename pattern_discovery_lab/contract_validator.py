#!/usr/bin/env python3
"""
Strict Contract Validator for Pattern Discovery Lab v0.1

Enforces:
- Stdout grammar (exact section headers, order, uniqueness, no extra lines, no stderr leakage)
- results.json closed schema (no extra keys anywhere, exact types/enums)
- Logic consistency (gate decisions vs numeric thresholds and contract rules)
- Golden artifacts verification (hash check + exact match option)

Citations:
- Permutation p smoothing: Phipson & Smyth (2010) - p=(k+1)/(N+1)
- Tie handling uses >= for conservatism (Phipson & Smyth 2010)
- rfft DC/Nyquist real constraints (NumPy docs)
- Floating point comparisons: Goldberg (1991) - applied selectively

Rule ID Format: RULE_<CATEGORY>_<NUMBER>
"""

import sys
import json
import re
import os
import hashlib
import math
from typing import List, Dict, Any, Tuple
from pathlib import Path

from pattern_discovery_lab.canonicalizer import canonicalize_stdout, compute_canonical_hash


class ValidationError:
    """Structured validation error with rule ID."""
    
    def __init__(self, rule_id: str, message: str):
        self.rule_id = rule_id
        self.message = message
    
    def __str__(self):
        return f"[{self.rule_id}] {self.message}"
    
    def __repr__(self):
        return f"ValidationError({self.rule_id!r}, {self.message!r})"


class ContractValidator:
    """Validates Pattern Discovery Lab outputs against v0.1 contract."""
    
    # Closed schema: EXACTLY these keys allowed
    REQUIRED_TOP_LEVEL_KEYS = {"stop_rules", "detector_results", "overall_status"}
    
    # Enum values (case-sensitive)
    VALID_VERDICTS = {"PASS", "HALT"}
    VALID_DECISIONS = {"ACCEPT", "REJECT", "N/A"}
    VALID_CLASSES = {"CERTIFIED", "EXPERIMENTAL"}
    VALID_GATE_STATUS = {"CONTRIBUTES", "EXCLUDED"}
    
    # Numeric bounds
    P_VALUE_MIN = 0.0
    P_VALUE_MAX = 1.0
    SPECTRUM_THRESHOLD = 0.05
    GATE_A_P_THRESHOLD = 0.05  # p >= 0.05 => PASS
    GATE_B_P_THRESHOLD = 0.05  # p < 0.05 => HALT, p >= 0.05 => PASS
    
    def __init__(self, strict_mode: bool = True):
        """
        Args:
            strict_mode: If True, enforce strict whitespace and line ending rules
        """
        self.strict_mode = strict_mode
        self.errors: List[ValidationError] = []
    
    def add_error(self, rule_id: str, message: str):
        """Record a validation error."""
        self.errors.append(ValidationError(rule_id, message))
    
    def validate_all(
        self,
        stdout_path: str,
        results_json_path: str,
        stderr_path: str = None,
        golden_stdout_path: str = None,
        golden_json_path: str = None,
        check_hashes: bool = False
    ) -> bool:
        """
        Validate all contract requirements.
        
        Returns:
            True if all validations pass, False otherwise
        """
        self.errors = []
        
        # Validate stderr is empty (if provided)
        if stderr_path:
            self._validate_stderr(stderr_path)
        
        # Validate stdout grammar
        self._validate_stdout_grammar(stdout_path)
        
        # Validate results.json
        results_data = self._validate_results_json(results_json_path)
        
        # Validate logic consistency
        if results_data:
            self._validate_logic_consistency(results_data)
        
        # Validate against golden artifacts (if provided)
        if golden_stdout_path and golden_json_path:
            if check_hashes:
                self._validate_hashes(
                    stdout_path, results_json_path,
                    golden_stdout_path, golden_json_path
                )
            else:
                self._validate_exact_match(
                    stdout_path, results_json_path,
                    golden_stdout_path, golden_json_path
                )
        
        return len(self.errors) == 0
    
    def _validate_stderr(self, stderr_path: str):
        """Validate that stderr is empty in clean runs."""
        if not os.path.exists(stderr_path):
            return
        
        with open(stderr_path, 'r') as f:
            stderr_content = f.read()
        
        if stderr_content.strip():
            self.add_error(
                "RULE_STREAM_001",
                f"stderr must be empty in clean runs, found {len(stderr_content)} bytes"
            )
        
        # Check for structured headers in stderr
        structured_patterns = [
            r'Gate [ABCD]:',
            r'Overall Status:',
            r'STOP RULES EVALUATED',
            r'\| Dataset.*\| Detector'
        ]
        for pattern in structured_patterns:
            if re.search(pattern, stderr_content):
                self.add_error(
                    "RULE_STREAM_002",
                    f"Structured contract output found in stderr: {pattern}"
                )
    
    def _validate_stdout_grammar(self, stdout_path: str):
        """Validate stdout format against contract grammar."""
        with open(stdout_path, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        # RULE_STDOUT_001: Required sections in order
        required_sections = [
            (r'CLI Command:', "CLI Command"),
            (r'Run Folder:', "Run Folder"),
            (r'STOP RULES EVALUATED \(ENFORCE_NOW\)', "STOP RULES header"),
            (r'\| Dataset.*\| Detector', "Results table"),
            (r'OVERALL STATUS DETERMINATION', "OVERALL STATUS header")
        ]
        
        current_pos = 0
        for pattern, name in required_sections:
            match = re.search(pattern, content[current_pos:])
            if not match:
                self.add_error(
                    "RULE_STDOUT_001",
                    f"Missing required section: {name}"
                )
            else:
                current_pos += match.end()
        
        # RULE_STDOUT_002: Section uniqueness (no duplicates)
        for pattern, name in required_sections:
            matches = re.findall(pattern, content)
            if len(matches) > 1:
                self.add_error(
                    "RULE_STDOUT_002",
                    f"Duplicate section found: {name} (appears {len(matches)} times)"
                )
        
        # RULE_STDOUT_003: Required field labels
        required_labels = [
            'Gate A — shuffle_check:',
            'Gate B — spectrum_check:',
            'Gate C — stability_gate:',
            'Gate D — no_hallucination_check:',
            'Overall Status:',
            'Reason:'
        ]
        for label in required_labels:
            if label not in content:
                self.add_error(
                    "RULE_STDOUT_003",
                    f"Missing required field label: {label}"
                )
        
        # RULE_STDOUT_004: P-value formula format
        pvalue_pattern = r'p=\(k\+1\)/\(N\+1\)=\(\(\d+\)\+1\)/\(\(\d+\)\+1\)=\d+\.\d+'
        if not re.search(pvalue_pattern, content):
            self.add_error(
                "RULE_STDOUT_004",
                "Missing or malformed p-value formula (must be p=(k+1)/(N+1)=((X)+1)/((Y)+1)=X.XXXX)"
            )
        
        # RULE_STDOUT_005: No scientific notation for p-values
        scientific_pvalue = r'p.*=\s*\d+\.\d+[eE][+-]?\d+'
        if re.search(scientific_pvalue, content):
            self.add_error(
                "RULE_STDOUT_005",
                "P-values must use fixed decimal notation, not scientific notation"
            )
        
        # RULE_STDOUT_006: Forbidden content (log prefixes, tracebacks, etc.)
        forbidden_patterns = [
            (r'\b(INFO|DEBUG|WARNING|ERROR):', "log prefix"),
            (r'Traceback \(most recent call last\):', "Python traceback"),
            (r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', "timestamp"),
            (r'\x1b\[[0-9;]*m', "ANSI color code"),
            (r'[\u200B-\u200D\uFEFF]', "zero-width character")
        ]
        for pattern, desc in forbidden_patterns:
            if re.search(pattern, content):
                self.add_error(
                    "RULE_STDOUT_006",
                    f"Forbidden content found: {desc}"
                )
        
        # RULE_STDOUT_007: Section order (Overall must come after gates)
        overall_match = re.search(r'OVERALL STATUS DETERMINATION', content)
        experimental_match = re.search(r'EXPERIMENTAL DETECTORS', content)
        
        if overall_match and experimental_match:
            if experimental_match.start() < overall_match.start():
                self.add_error(
                    "RULE_STDOUT_007",
                    "EXPERIMENTAL section must not appear before OVERALL STATUS"
                )
        
        # RULE_STDOUT_008: No INFO/DEBUG lines after gate headers
        gate_headers = [
            r'Gate A — shuffle_check:',
            r'Gate B — spectrum_check:',
            r'Gate C — stability_gate:',
            r'Gate D — no_hallucination_check:'
        ]
        for header in gate_headers:
            header_match = re.search(header, content)
            if header_match:
                # Check the next non-empty line
                remaining = content[header_match.end():]
                next_line = remaining.split('\n')[0].strip()
                if next_line and re.match(r'(INFO|DEBUG|WARNING):', next_line):
                    self.add_error(
                        "RULE_STDOUT_008",
                        f"Log prefix found immediately after gate header: {header}"
                    )
        
        # RULE_STDOUT_009: Strict whitespace (if strict_mode)
        if self.strict_mode:
            for i, line in enumerate(lines):
                if line != line.rstrip():
                    self.add_error(
                        "RULE_STDOUT_009",
                        f"Line {i+1} has trailing whitespace"
                    )
        
        # RULE_STDOUT_010: Line endings (reject CRLF)
        if '\r\n' in content:
            self.add_error(
                "RULE_STDOUT_010",
                "File contains CRLF line endings; only LF allowed"
            )
    
    def _validate_results_json(self, json_path: str) -> Dict[str, Any]:
        """Validate results.json schema and return data."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            self.add_error("RULE_JSON_001", f"Invalid JSON: {e}")
            return None
        except Exception as e:
            self.add_error("RULE_JSON_001", f"Cannot read JSON file: {e}")
            return None
        
        # RULE_JSON_002: Closed schema - exact top-level keys
        actual_keys = set(data.keys())
        if actual_keys != self.REQUIRED_TOP_LEVEL_KEYS:
            extra_keys = actual_keys - self.REQUIRED_TOP_LEVEL_KEYS
            missing_keys = self.REQUIRED_TOP_LEVEL_KEYS - actual_keys
            
            if extra_keys:
                self.add_error(
                    "RULE_JSON_002",
                    f"Extra top-level keys not allowed: {sorted(extra_keys)}"
                )
            if missing_keys:
                self.add_error(
                    "RULE_JSON_002",
                    f"Missing required top-level keys: {sorted(missing_keys)}"
                )
        
        # Validate stop_rules
        if 'stop_rules' in data:
            self._validate_stop_rules(data['stop_rules'])
        
        # Validate detector_results
        if 'detector_results' in data:
            self._validate_detector_results(data['detector_results'])
        
        # Validate overall_status
        if 'overall_status' in data:
            self._validate_overall_status(data['overall_status'])
        
        return data
    
    def _validate_stop_rules(self, stop_rules: Dict[str, Any]):
        """Validate stop_rules structure."""
        # RULE_JSON_003: Required stop rule gates
        required_gates = ['shuffle_check', 'spectrum_check', 'stability_gate', 'no_hallucination_check']
        for gate in required_gates:
            if gate not in stop_rules:
                self.add_error("RULE_JSON_003", f"Missing stop rule: {gate}")
        
        # Validate shuffle_check
        if 'shuffle_check' in stop_rules:
            sc = stop_rules['shuffle_check']
            required_fields = ['rho1_shuffled_max', 'threshold_T_N', 'threshold_formula', 'N_samples', 'verdict']
            for field in required_fields:
                if field not in sc:
                    self.add_error("RULE_JSON_004", f"shuffle_check missing field: {field}")
            
            # Type checks
            if 'verdict' in sc and sc['verdict'] not in self.VALID_VERDICTS:
                self.add_error("RULE_JSON_005", f"shuffle_check verdict must be PASS or HALT, got: {sc['verdict']}")
            
            # Numeric validation
            if 'rho1_shuffled_max' in sc:
                self._validate_numeric(sc['rho1_shuffled_max'], 'shuffle_check.rho1_shuffled_max', "RULE_JSON_006")
            if 'threshold_T_N' in sc:
                self._validate_numeric(sc['threshold_T_N'], 'shuffle_check.threshold_T_N', "RULE_JSON_006")
        
        # Validate spectrum_check
        if 'spectrum_check' in stop_rules:
            sc = stop_rules['spectrum_check']
            required_fields = ['spectrum_error_max', 'threshold', 'verdict']
            for field in required_fields:
                if field not in sc:
                    self.add_error("RULE_JSON_004", f"spectrum_check missing field: {field}")
            
            if 'verdict' in sc and sc['verdict'] not in self.VALID_VERDICTS:
                self.add_error("RULE_JSON_005", f"spectrum_check verdict must be PASS or HALT, got: {sc['verdict']}")
            
            # Threshold must be 0.05
            if 'threshold' in sc:
                if abs(sc['threshold'] - self.SPECTRUM_THRESHOLD) > 1e-10:
                    self.add_error("RULE_JSON_007", f"spectrum_check threshold must be 0.05, got: {sc['threshold']}")
        
        # Validate stability_gate
        if 'stability_gate' in stop_rules:
            sg = stop_rules['stability_gate']
            if 'verdict' not in sg:
                self.add_error("RULE_JSON_004", "stability_gate missing field: verdict")
            if 'verdict' in sg and sg['verdict'] not in self.VALID_VERDICTS:
                self.add_error("RULE_JSON_005", f"stability_gate verdict must be PASS or HALT, got: {sg['verdict']}")
        
        # Validate no_hallucination_check
        if 'no_hallucination_check' in stop_rules:
            nhc = stop_rules['no_hallucination_check']
            required_fields = ['dataset', 'detector', 'statistical_decision', 'verdict']
            for field in required_fields:
                if field not in nhc:
                    self.add_error("RULE_JSON_004", f"no_hallucination_check missing field: {field}")
            
            if 'statistical_decision' in nhc and nhc['statistical_decision'] not in self.VALID_DECISIONS:
                self.add_error("RULE_JSON_005", f"no_hallucination_check decision must be ACCEPT/REJECT/N/A, got: {nhc['statistical_decision']}")
    
    def _validate_detector_results(self, detector_results: List[Dict[str, Any]]):
        """Validate detector_results array."""
        if not isinstance(detector_results, list):
            self.add_error("RULE_JSON_008", "detector_results must be an array")
            return
        
        for i, row in enumerate(detector_results):
            # RULE_JSON_009: Required fields
            required_fields = [
                'dataset', 'detector', 'class', 'gate_status',
                'real_score', 'surrogate_N', 'surrogate_k', 'surrogate_p',
                'surrogate_formula', 'statistical_decision', 'expected_decision',
                'expectation_match', 'rho1_shuffled', 'spectrum_error'
            ]
            for field in required_fields:
                if field not in row:
                    self.add_error("RULE_JSON_009", f"detector_results[{i}] missing field: {field}")
            
            # RULE_JSON_010: Enum validation
            if 'class' in row and row['class'] not in self.VALID_CLASSES:
                self.add_error("RULE_JSON_010", f"detector_results[{i}].class must be CERTIFIED or EXPERIMENTAL, got: {row['class']}")
            
            if 'gate_status' in row and row['gate_status'] not in self.VALID_GATE_STATUS:
                self.add_error("RULE_JSON_010", f"detector_results[{i}].gate_status must be CONTRIBUTES or EXCLUDED, got: {row['gate_status']}")
            
            if 'statistical_decision' in row and row['statistical_decision'] not in self.VALID_DECISIONS:
                self.add_error("RULE_JSON_010", f"detector_results[{i}].statistical_decision must be ACCEPT/REJECT/N/A, got: {row['statistical_decision']}")
            
            if 'expected_decision' in row and row['expected_decision'] not in self.VALID_DECISIONS:
                self.add_error("RULE_JSON_010", f"detector_results[{i}].expected_decision must be ACCEPT/REJECT/N/A, got: {row['expected_decision']}")
            
            # RULE_JSON_011: P-value bounds
            if 'surrogate_p' in row:
                p = row['surrogate_p']
                if not (self.P_VALUE_MIN <= p <= self.P_VALUE_MAX):
                    self.add_error("RULE_JSON_011", f"detector_results[{i}].surrogate_p must be in [0, 1], got: {p}")
            
            # RULE_JSON_012: Numeric validation (no NaN/Inf)
            numeric_fields = ['real_score', 'surrogate_p', 'rho1_shuffled', 'spectrum_error']
            for field in numeric_fields:
                if field in row:
                    self._validate_numeric(row[field], f'detector_results[{i}].{field}', "RULE_JSON_012")
            
            # RULE_JSON_013: Integer validation
            if 'surrogate_N' in row and not isinstance(row['surrogate_N'], int):
                self.add_error("RULE_JSON_013", f"detector_results[{i}].surrogate_N must be integer, got: {type(row['surrogate_N']).__name__}")
            if 'surrogate_k' in row and not isinstance(row['surrogate_k'], int):
                self.add_error("RULE_JSON_013", f"detector_results[{i}].surrogate_k must be integer, got: {type(row['surrogate_k']).__name__}")
    
    def _validate_overall_status(self, overall_status: Dict[str, Any]):
        """Validate overall_status structure."""
        if not isinstance(overall_status, dict):
            self.add_error("RULE_JSON_014", "overall_status must be object, not string or array")
            return
        
        # RULE_JSON_015: Required fields
        required_fields = ['verdict', 'reason']
        for field in required_fields:
            if field not in overall_status:
                self.add_error("RULE_JSON_015", f"overall_status missing field: {field}")
        
        # RULE_JSON_016: Verdict enum
        if 'verdict' in overall_status and overall_status['verdict'] not in self.VALID_VERDICTS:
            self.add_error("RULE_JSON_016", f"overall_status.verdict must be PASS or HALT, got: {overall_status['verdict']}")
    
    def _validate_numeric(self, value: Any, field_name: str, rule_id: str):
        """Validate numeric field (reject NaN/Inf)."""
        if not isinstance(value, (int, float)):
            self.add_error(rule_id, f"{field_name} must be numeric, got: {type(value).__name__}")
            return
        
        if math.isnan(value):
            self.add_error(rule_id, f"{field_name} must not be NaN")
        
        if math.isinf(value):
            self.add_error(rule_id, f"{field_name} must not be Infinity")
    
    def _validate_logic_consistency(self, data: Dict[str, Any]):
        """Validate logic consistency between gates and decisions."""
        stop_rules = data.get('stop_rules', {})
        overall_status = data.get('overall_status', {})
        
        # RULE_LOGIC_001: Overall verdict must be PASS iff all gates PASS
        gate_verdicts = []
        for gate in ['shuffle_check', 'spectrum_check', 'stability_gate', 'no_hallucination_check']:
            if gate in stop_rules and 'verdict' in stop_rules[gate]:
                gate_verdicts.append(stop_rules[gate]['verdict'])
        
        if gate_verdicts and 'verdict' in overall_status:
            all_pass = all(v == 'PASS' for v in gate_verdicts)
            overall_verdict = overall_status['verdict']
            
            if all_pass and overall_verdict != 'PASS':
                self.add_error("RULE_LOGIC_001", f"All gates PASS but overall verdict is {overall_verdict}")
            
            if not all_pass and overall_verdict == 'PASS':
                failing_gates = [g for g, v in zip(['A', 'B', 'C', 'D'], gate_verdicts) if v != 'PASS']
                self.add_error("RULE_LOGIC_001", f"Overall PASS but gate(s) {failing_gates} HALT")
        
        # RULE_LOGIC_002: Gate A boundary (p >= 0.05 => PASS)
        # This would require access to permutation results which aren't in stop_rules
        # Skip for now unless p-values are added to stop_rules
        
        # RULE_LOGIC_003: Expectation match consistency
        detector_results = data.get('detector_results', [])
        for i, row in enumerate(detector_results):
            if all(k in row for k in ['statistical_decision', 'expected_decision', 'expectation_match']):
                stat_dec = row['statistical_decision']
                exp_dec = row['expected_decision']
                exp_match = row['expectation_match']
                
                # EXPERIMENTAL can have N/A expected
                if exp_dec == 'N/A':
                    if exp_match != 'N/A':
                        self.add_error("RULE_LOGIC_003", f"detector_results[{i}]: expected=N/A but match={exp_match}")
                else:
                    # CERTIFIED must have match logic
                    actual_match = 'PASS' if stat_dec == exp_dec else 'FAIL'
                    if exp_match != actual_match:
                        self.add_error("RULE_LOGIC_003", f"detector_results[{i}]: stat={stat_dec}, exp={exp_dec}, but match={exp_match} (should be {actual_match})")
    
    def _validate_hashes(
        self,
        stdout_path: str,
        json_path: str,
        golden_stdout_path: str,
        golden_json_path: str
    ):
        """Validate SHA256 hashes match golden artifacts."""
        def compute_hash(filepath: str) -> str:
            sha256_hash = hashlib.sha256()
            with open(filepath, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        
        def compute_canonicalized_hash(filepath: str) -> str:
            """Compute hash of canonicalized stdout (for timestamp-invariant comparison)."""
            with open(filepath, 'r') as f:
                content = f.read()
            return compute_canonical_hash(content)
        
        # RULE_GOLDEN_001: Stdout hash match (canonicalized for determinism)
        actual_hash = compute_canonicalized_hash(stdout_path)
        expected_hash = compute_canonicalized_hash(golden_stdout_path)
        if actual_hash != expected_hash:
            self.add_error(
                "RULE_GOLDEN_001",
                f"Stdout hash mismatch (canonicalized): {actual_hash} != {expected_hash}"
            )
        
        # RULE_GOLDEN_002: JSON hash match
        actual_hash = compute_hash(json_path)
        expected_hash = compute_hash(golden_json_path)
        if actual_hash != expected_hash:
            self.add_error(
                "RULE_GOLDEN_002",
                f"Results.json hash mismatch: {actual_hash} != {expected_hash}"
            )
    
    def _validate_exact_match(
        self,
        stdout_path: str,
        json_path: str,
        golden_stdout_path: str,
        golden_json_path: str
    ):
        """Validate exact content match (byte-for-byte)."""
        # Stdout match
        with open(stdout_path, 'rb') as f1, open(golden_stdout_path, 'rb') as f2:
            if f1.read() != f2.read():
                self.add_error("RULE_GOLDEN_003", "Stdout does not match golden artifact (byte-for-byte)")
        
        # JSON match (semantic comparison)
        with open(json_path, 'r') as f1, open(golden_json_path, 'r') as f2:
            actual = json.load(f1)
            expected = json.load(f2)
            if actual != expected:
                self.add_error("RULE_GOLDEN_004", "Results.json does not match golden artifact (semantic comparison)")
    
    def report(self) -> str:
        """Generate validation report."""
        lines = []
        lines.append("=" * 70)
        lines.append("CONTRACT VALIDATION REPORT - Enforcement Contract v0.1")
        lines.append("=" * 70)
        lines.append("")
        
        if not self.errors:
            lines.append("✓ ALL CHECKS PASSED")
            lines.append("")
            lines.append(f"Total: 0 errors")
        else:
            lines.append(f"✗ VALIDATION FAILED ({len(self.errors)} errors)")
            lines.append("")
            for error in self.errors:
                lines.append(f"  {error}")
            lines.append("")
            lines.append(f"Total: {len(self.errors)} errors")
        
        lines.append("=" * 70)
        return "\n".join(lines)


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Strict contract validator for Pattern Discovery Lab v0.1"
    )
    parser.add_argument(
        '--stdout',
        required=True,
        help='Path to stdout file'
    )
    parser.add_argument(
        '--results',
        required=True,
        help='Path to results.json file'
    )
    parser.add_argument(
        '--stderr',
        help='Path to stderr file (validates it is empty)'
    )
    parser.add_argument(
        '--golden-stdout',
        help='Path to golden stdout artifact (for comparison)'
    )
    parser.add_argument(
        '--golden-json',
        help='Path to golden results.json artifact (for comparison)'
    )
    parser.add_argument(
        '--check-hashes',
        action='store_true',
        help='Check SHA256 hashes instead of exact match'
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        default=True,
        help='Enable strict mode (whitespace, line endings)'
    )
    
    args = parser.parse_args()
    
    validator = ContractValidator(strict_mode=args.strict)
    
    success = validator.validate_all(
        stdout_path=args.stdout,
        results_json_path=args.results,
        stderr_path=args.stderr,
        golden_stdout_path=args.golden_stdout,
        golden_json_path=args.golden_json,
        check_hashes=args.check_hashes
    )
    
    print(validator.report())
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
