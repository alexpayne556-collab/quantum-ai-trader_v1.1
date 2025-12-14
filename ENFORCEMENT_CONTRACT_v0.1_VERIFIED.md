# ENFORCEMENT CONTRACT v0.1 — CANONICAL IMPLEMENTATION VERIFIED

**Date:** December 13, 2025  
**Repository:** /workspaces/quantum-ai-trader_v1.1  
**Module:** pattern_discovery_lab  
**Status:** ✅ FULLY COMPLIANT (11/11 validation checks passed)

---

## VERIFICATION SUMMARY

### Truth Tests (T1-T4)
```
✓ T1: white_noise → ACCEPT (p=0.8120)
✓ T2: ar1_phi_0p9 → REJECT (p=0.0010)  
✓ T3: garch → ACCEPT (p=0.8630)
✓ T4: Gate A threshold T(N=1000) = 0.1423 (4.5/√1000)
```

### Validation Results
```
STDOUT VALIDATION:     ✓ All checks passed
JSON SCHEMA VALIDATION: ✓ All checks passed
RESULT: 11/11 checks passed ✓
```

---

## IMPLEMENTATION DETAILS

### Detectors

**CERTIFIED** (contributes to PASS/HALT):
- `autocorrelation_dependence` — uses shuffle surrogates

**EXPERIMENTAL** (information only, excluded from gating):
- `time_reversal_asymmetry` — uses phase surrogates

### Datasets + Expected Decisions

| Dataset | Expected Decision (CERTIFIED only) |
|---------|-----------------------------------|
| white_noise | ACCEPT |
| ar1_phi_0p9 | REJECT |
| garch | ACCEPT |

*EXPERIMENTAL: expected = N/A*

### Three-Layer Decision Model

**Layer 1: Statistical Decision**
- REJECT if p < 0.05
- ACCEPT if p ≥ 0.05

**Layer 2: Expectation Match**
- CERTIFIED: PASS if stat_decision == expected_decision, else FAIL
- EXPERIMENTAL: N/A (never fails)

**Layer 3: Overall Status**
- PASS iff all stop rules PASS AND all CERTIFIED expectation matches PASS
- Otherwise HALT
- EXPERIMENTAL never affects overall

---

## STOP RULES (ENFORCE_NOW)

### Gate A — Shuffle Control (Lag-1 Autocorrelation)

**Implementation:**
```python
x_c = x - mean(x)
rho1 = sum(x_c[1:] * x_c[:-1]) / sum(x_c**2)
```

**Threshold:** `T(N) = 4.5 / sqrt(N)` ✅ (LOCKED)

**Metric:** `rho1_shuffled_max = max_i |rho1(shuffle_i)|`

**Verdict:** PASS if `rho1_shuffled_max <= T(N)` else HALT

**Output:**
```
Gate A — shuffle_check: PASS
  N=1000, M=100
  rho1_shuffled_max=0.0945
  threshold T(N)=4.5/sqrt(1000)=0.1423
```

### Gate B — Spectrum Preservation (Phase Surrogate)

**Implementation:**
- Mean-center both series
- Use `np.fft.rfft` for power spectrum
- Compare power: `abs(FFT)**2`
- Exclude DC/Nyquist via slice `[1:N//2]`
- Epsilon guard: `eps = 1e-15 * max(P_orig)` or 1.0

**Threshold:** 0.05 (5%)

**Verdict:** PASS if `spectrum_error_max <= 0.05` else HALT

**Output:**
```
Gate B — spectrum_check: PASS
  spectrum_error_max=0.000000
  threshold=0.05
```

### Gate C — Decision Stability (CERTIFIED ONLY)

**Implementation:**
- Split series into 3 non-overlapping blocks
- Run detector on each block with same null/control
- Compute decision (ACCEPT/REJECT) at α=0.05
- `stability_decision = majority_agreement/3 ∈ {1.0, 0.667, 0.333}`

**Enforce:** HALT if `stability_decision != 1.0` for CERTIFIED

**Output:**
```
Gate C — stability_gate: PASS
  white_noise/autocorrelation_dependence: stability_decision=1.000
  ar1_phi_0p9/autocorrelation_dependence: stability_decision=1.000
  garch/autocorrelation_dependence: stability_decision=1.000
```

### Gate D — No Hallucination (CERTIFIED ONLY)

**Rule:** If CERTIFIED detector REJECTS on white_noise → HALT

**Output:**
```
Gate D — no_hallucination_check: PASS
  autocorrelation_dependence on white_noise: ACCEPT
```

---

## P-VALUE FORMULA (NON-NEGOTIABLE)

**Formula:** `p = (k+1)/(N+1)`

**Required Format:**
```
p=(k+1)/(N+1)=((81)+1)/((99)+1)=0.8200
```

**Implementation:**
```python
k = count(surr_stat >= real_stat)  # ties count
N = number of surrogates
p_value = (k + 1) / (N + 1)
surr_formula = f"p=(k+1)/(N+1)=(({k})+1)/(({N})+1)={p_value:.4f}"
```

---

## STDOUT CONTRACT (EXACT FORMAT)

### Block 1: Header
```
CLI Command: python -m pattern_discovery_lab --run-all --seed 42 --n-surrogates 99

Run Folder: /workspaces/quantum-ai-trader_v1.1/pattern_discovery_lab/runs/20251213_203137
```

### Block 2: STOP RULES EVALUATED (ENFORCE_NOW)
See Gate A/B/C/D outputs above

### Block 3: Results Table
```
| Dataset         | Detector                  | Class        | Surr Formula              | Stat Dec | Exp Dec  | Exp Match | rho1_shuf | Spec Err   | Stab Dec | Gate Status  |
```

### Block 4: OVERALL STATUS DETERMINATION
```
STOP RULES:
  Gate A (shuffle): PASS
  Gate B (spectrum): PASS
  Gate C (stability): PASS
  Gate D (no hallucination): PASS

CERTIFIED DETECTORS (contribute to gating):
  white_noise × autocorrelation_dependence: ACCEPT (expected: ACCEPT) → PASS
  ar1_phi_0p9 × autocorrelation_dependence: REJECT (expected: REJECT) → PASS
  garch × autocorrelation_dependence: ACCEPT (expected: ACCEPT) → PASS

EXPERIMENTAL DETECTORS (excluded from gating):
  white_noise × time_reversal_asymmetry: ACCEPT (info only)
  ar1_phi_0p9 × time_reversal_asymmetry: ACCEPT (info only)
  garch × time_reversal_asymmetry: ACCEPT (info only)

GATING LOGIC:
  - All stop rules must PASS
  - All CERTIFIED expectation matches must PASS
  - EXPERIMENTAL detectors do not affect overall status

Overall Status: PASS
Reason: All gates passed
```

**Forbidden:** DEBUG/INFO/WARNING/ERROR, timestamps, tracebacks, duplicate headers

---

## RESULTS.JSON CONTRACT

### Top-Level Keys (Required)
```json
{
  "stop_rules": {...},
  "detector_results": [...],
  "overall_status": {...}
}
```

### stop_rules Structure
```json
{
  "shuffle_check": {
    "rho1_shuffled_max": 0.0945,
    "threshold_T_N": 0.1423,
    "threshold_formula": "4.5/√N",
    "N_samples": 1000,
    "verdict": "PASS"
  },
  "spectrum_check": {
    "spectrum_error_max": 3.01e-14,
    "threshold": 0.05,
    "verdict": "PASS"
  },
  "stability_gate": {
    "certified_decisions": {
      "white_noise/autocorrelation_dependence": 1.0,
      "ar1_phi_0p9/autocorrelation_dependence": 1.0,
      "garch/autocorrelation_dependence": 1.0
    },
    "verdict": "PASS"
  },
  "no_hallucination_check": {
    "dataset": "white_noise",
    "detector": "autocorrelation_dependence",
    "statistical_decision": "ACCEPT",
    "verdict": "PASS"
  }
}
```

### detector_results Structure
```json
[
  {
    "dataset": "white_noise",
    "detector": "autocorrelation_dependence",
    "class": "CERTIFIED",
    "gate_status": "CONTRIBUTES",
    "real_score": 0.0075,
    "surrogate_N": 99,
    "surrogate_k": 81,
    "surrogate_p": 0.82,
    "surrogate_formula": "p=(k+1)/(N+1)=((81)+1)/((99)+1)=0.8200",
    "statistical_decision": "ACCEPT",
    "expected_decision": "ACCEPT",
    "expectation_match": "PASS",
    "rho1_shuffled": 0.0893,
    "spectrum_error": 1.83e-14,
    "stability_decision": 1.0
  }
]
```

### overall_status Structure (MUST BE OBJECT)
```json
{
  "verdict": "PASS",
  "reason": "All gates passed"
}
```

---

## FILE INVENTORY

### Core Implementation
- `pattern_discovery_lab/gates_v0p1.py` — Gate A/B implementations, phase surrogate
- `pattern_discovery_lab/runner_v0p1.py` — Main runner with enforcement contract
- `pattern_discovery_lab/__main__.py` — CLI entrypoint

### Validation
- `validate_run_v0p1.py` — Contract validator (11 checks)

### Output
- `pattern_discovery_lab/runs/YYYYMMDD_HHMMSS/results.json` — Per-run results
- `stdout.txt` — Captured stdout for validation

---

## USAGE

### Run Full Contract
```bash
python -m pattern_discovery_lab --run-all --seed 42 --n-surrogates 99
```

### Run Truth Tests
```bash
python -m pattern_discovery_lab --truth-tests
```

### Validate Output
```bash
RUN_DIR="$(ls -1dt pattern_discovery_lab/runs/* | head -n 1)"
python validate_run_v0p1.py stdout.txt "$RUN_DIR/results.json"
```

---

## COMPLIANCE VERIFICATION

✅ Gate A threshold: `4.5/√N` (NOT 1.96/√N)  
✅ Gate B threshold: 0.05  
✅ P-value formula: `p=(k+1)/(N+1)=((k)+1)/((N)+1)=X.XXXX`  
✅ EXPERIMENTAL excluded from gating  
✅ overall_status is object not string  
✅ All required stdout sections present  
✅ All required JSON keys present  
✅ No forbidden debug logs  
✅ Truth tests T1-T4 pass  
✅ Validator passes 11/11 checks  

**CONTRACT STATUS: FULLY SATISFIED ✓**
