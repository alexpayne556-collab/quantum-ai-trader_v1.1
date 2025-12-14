PARALLEL THINK-TANK SYNTHESIS → IMPLEMENTATION REQUIREMENTS (V0.1 + PHASE_2)

A) STABILITY METRIC (ENFORCE_NOW) — MAKE IT AUDITABLE
Problem: Current “Stability” scalar is ambiguous. Redefine stability in report + results.json using explicit block decisions.

Implement TWO stability metrics and PRINT BOTH, but gate ENFORCE_NOW on Decision Stability.

A1) Decision Stability (ENFORCE_NOW gate) — DeepSeek Candidate 2
- For each dataset x detector:
  - Split series into 3 non-overlapping blocks.
  - Compute block effect_size and block p_value using the SAME null/control as the full series (no leakage).
  - Convert each block to decision d_i ∈ {REJECT, ACCEPT} at alpha=0.05 (or configured alpha).
  - stability_decision = majority_agreement / 3
    where majority_agreement = max(#REJECT, #ACCEPT).
Interpretation:
  - PASS: stability_decision == 1.0 (all 3 blocks agree)
  - WARN: stability_decision == 0.667 (2/3 agree)
  - FAIL: stability_decision <= 0.333

ENFORCE_NOW stop rule:
- If any CERTIFIED detector has stability_decision != 1.0 → HALT run (overall FAIL) with reason “STABILITY FAILURE”.

A2) Distributional Stability (print-only, not gating tonight) — DeepSeek Candidate 1
- Compute CV on effect sizes across 3 blocks:
  CV = std(e1,e2,e3) / |mean(e1,e2,e3)| (epsilon if mean≈0)
  stability_distributional = max(0, 1 - CV)
Thresholds (report-only):
  PASS ≥ 0.8, WARN 0.6–0.8, FAIL < 0.6

Report.md MUST include, per dataset x detector:
- Block 1: effect_size=..., p_value=..., decision=REJECT/ACCEPT
- Block 2: ...
- Block 3: ...
- stability_decision=... (PASS/WARN/FAIL)
- stability_distributional=... (PASS/WARN/FAIL)
- Majority decision + how many blocks agree

B) TIME-REVERSAL ASYMMETRY (TRA) POLICY (V0.1)
DeepSeek conclusion: TRA requires IAAFT to be valid and is extremely sensitive to trends/outliers. Therefore:

B1) Status: EXPERIMENTAL UNTIL PHASE_2 (IAAFT exists)
- TRA may run in v0.1, but it MUST NOT:
  - contribute to overall PASS
  - trigger ENFORCE_NOW HALT conditions
- It CAN generate WARNINGS, e.g. “TRA flagged structure but detector is EXPERIMENTAL; requires IAAFT to interpret.”

B2) Default detector set in v0.1 config:
- CERTIFIED detectors: autocorrelation_dependence ONLY
- EXPERIMENTAL detectors: time_reversal_asymmetry (optional; default ON is ok but must not gate PASS/FAIL)

C) SURROGATE TESTING DEFAULTS (DOCS + CONFIG) — Perplexity rules
C1) Null selection decision tree (document in docs/null_models_v0_1.md or similar):
- FT Phase Randomization is valid only under stationary linear Gaussian null.
- If amplitudes are non-Gaussian and we still want a linear null, AAFT/IAAFT are required.
- Gold standard for “nonlinear beyond linear spectrum” claims: IAAFT.

C2) P-values MUST be rank-based (no z-scores):
p = (k + 1) / (N + 1)
- N = number of surrogates
- k = number of surrogates with statistic >= original (or “more extreme” consistently defined)
Report must print N, k, and resulting p.

C3) Surrogate counts:
- Exploratory default (v0.1): N = 199 (min p≈0.005) OR N=99 (min p≈0.01) if runtime tight
- Confirmatory (PHASE_2): N = 999 or 1999

C4) Phase randomization validity diagnostics (ENFORCE_NOW + report):
- Spectrum preservation error must be computed and printed:
  - Target: ≤1% (PASS)
  - Hard FAIL: >5% in any band (HALT)
- Endpoint mismatch warning:
  - If x0 != x_end substantially, warn “FT surrogate may ring / spectral leakage risk”.
(Nonstationarity tests ADF/KPSS can be PHASE_2 unless quick to add.)

D) UPDATE STOP RULES / OVERALL STATUS LOGIC (IMPORTANT FIX)
Current logic “Overall PASS (no noise hallucination)” must be based on CERTIFIED detectors only.

ENFORCE_NOW HALT if ANY occurs (CERTIFIED detectors only):
1) Shuffle control failure: after shuffle |rho1| ≥ 0.05
2) Phase surrogate failure: spectrum error >5% any band
3) Stability failure: stability_decision != 1.0 on any certified detector
4) Certified noise hallucination proxy: certified detector REJECTS null on white_noise in this single run

WARN only (do not HALT):
- Any EXPERIMENTAL detector flags structure on white_noise
- Any EXPERIMENTAL detector has stability_decision < 1.0
- Any endpoint mismatch risk flagged for FT surrogates
- Any nonstationarity diagnostic flagged (if implemented)

E) PHASE_2 --CALIBRATE MODE (SPEC ONLY IF NOT IMPLEMENTING TONIGHT)
Add runner mode=calibrate with exploratory and confirmatory realization counts.

Required datasets:
- Noise: iid Gaussian (1000 confirm / 100 explore)
- Linear: AR(1) phi ∈ {0.3,0.5,0.7,0.9} (200 per phi confirm / 50 per phi explore)
- Nonlinear: logistic map r=3.9, Henon, AR(1)-GARCH(1,1) (100 each confirm / 30 each explore)

Aggregate metrics per detector:
- Noise FPR (target 5% ±1; hard fail >10 or <2)
- AR(1) power phi=0.5 (target ≥95; hard fail <80)
- Stability fail rate (target ≤10; hard fail <70% agreement on stationary)
- p-value uniformity under null (KS p<0.01 => hard fail)
- Surrogate validity (spectrum error thresholds)
- Seed/order sensitivity (>5% => hard fail)

If calibrate mode isn’t implemented tonight, add TODO and mark PHASE_2 in docs; do not claim it exists.

F) OUTPUT CONTRACT (MAKE AGENT PRINT RIGHT THING)
After running:
print ONLY:
1) exact CLI command executed
2) run folder path
3) “Stop Rules Evaluated (ENFORCE_NOW)” section
4) PASS/FAIL table section
5) one-line per dataset: real_score, shuffle_check(|rho1|), surrogate_p (with N,k), spectrum_error, stability_decision

No logs, no code dumps, no secrets.
# Smoke Test Results

**Date**: 2025-12-13  
**Status**: ✅ ALL TESTS PASSED (5/5)

## Test Suite

### 1. Minimal Run ✅
- **Purpose**: Validate complete pipeline executes without errors
- **Tests**:
  - Dataset generation (white_noise, ar1, garch)
  - Detector execution (autocorrelation_dependence, time_reversal_asymmetry)
  - Results structure validation
- **Result**: PASSED

### 2. Output Generation ✅
- **Purpose**: Verify JSON and Markdown outputs are created correctly
- **Tests**:
  - JSON parseable and contains run_id
  - Markdown contains required sections:
    - `## STOP RULES EVALUATED (ENFORCE_NOW)`
    - `## Results Summary`
    - `Stability Metric` definition
    - `Coefficient of Variation` explanation
    - PASS/FAIL table
    - `## Detailed Results`
- **Result**: PASSED

### 3. Noise Rejection ✅
- **Purpose**: Ensure detectors don't hallucinate structure on white noise
- **Tests**:
  - White noise dataset processed
  - Detectors execute on pure noise
  - Control gates evaluated
- **Result**: PASSED

### 4. AR(1) Detection ✅
- **Purpose**: Verify autocorrelation detector finds linear structure
- **Tests**:
  - AR(1) autocorrelation score > 0.3
  - Shuffle p-value < 0.05 (structure survives control)
- **Result**: PASSED
  - Score: 0.7541
  - Shuffle p: 0.0000

### 5. Stability Metric ✅
- **Purpose**: Validate stability computation across blocks
- **Tests**:
  - 3 block scores computed
  - CV (coefficient of variation) ≥ 0
  - Stability metric present for all detectors
- **Result**: PASSED

## Command

```bash
cd /workspaces/quantum-ai-trader_v1.1
python pattern_discovery_lab/tests/test_smoke.py
```

## Output

```
================================================================================
PATTERN DISCOVERY LAB - SMOKE TEST
================================================================================

────────────────────────────────────────────────────────────────────────────────
TEST: Minimal Run
────────────────────────────────────────────────────────────────────────────────
✅ PASSED: Minimal Run

────────────────────────────────────────────────────────────────────────────────
TEST: Output Generation
────────────────────────────────────────────────────────────────────────────────
✅ PASSED: Output Generation

────────────────────────────────────────────────────────────────────────────────
TEST: Noise Rejection
────────────────────────────────────────────────────────────────────────────────
✅ PASSED: Noise Rejection

────────────────────────────────────────────────────────────────────────────────
TEST: AR(1) Detection
────────────────────────────────────────────────────────────────────────────────
✅ PASSED: AR(1) Detection

────────────────────────────────────────────────────────────────────────────────
TEST: Stability Metric
────────────────────────────────────────────────────────────────────────────────
✅ PASSED: Stability Metric

================================================================================
SMOKE TEST SUMMARY
================================================================================
Passed: 5/5
Failed: 0/5

✅ ALL SMOKE TESTS PASSED
```

## Coverage

**Components Tested**:
- ✅ Dataset generators (pure_noise, known_structure)
- ✅ Detectors (autocorrelation_dependence, time_reversal_asymmetry)
- ✅ Control suite (time-shuffle, phase-randomization, stability)
- ✅ Runner orchestration
- ✅ JSON report writer
- ✅ Markdown report writer
- ✅ STOP RULES enforcement
- ✅ PASS/FAIL logic

**Not Tested** (out of scope for smoke test):
- Extra detectors (require `enable_extra_detectors: true`)
- Calibrate mode (PHASE_2 - not implemented)
- Real data adapter (interface only)
- Finance-null generators (GARCH+jumps, stochastic vol, Levy)

## Next Steps

1. ✅ Smoke test passes - system ready for use
2. Run full calibration: `python -m pattern_discovery_lab --config pattern_discovery_lab/configs/default.yaml`
3. Optional: Enable extra detectors in config and re-run
4. PHASE_2: Implement `--mode calibrate` for statistical validation
