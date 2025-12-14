# Enforcement Contract v0.1

## Audit Rubric

### Mandatory Fields Per Dataset × Detector Row

| Field | Type | Description |
|-------|------|-------------|
| `detector_class` | CERTIFIED \| EXPERIMENTAL | Determines if detector gates HALT |
| `real_score` | float | Detector output on original series |
| `rho1_shuffled` | float | Mean \|rho1\| across shuffled surrogates |
| `rho1_shuffled_max` | float | Max \|rho1\| across shuffled surrogates |
| `shuffle_check` | PASS \| HALT | PASS if max \|rho1_shuffled\| < 0.05 |
| `surrogate_N` | int | Number of phase-randomized surrogates |
| `surrogate_k` | int | Count of surrogates with score >= original |
| `surrogate_p` | float | Rank-based p-value: (k+1)/(N+1) |
| `spectrum_error` | float | Mean spectrum preservation error |
| `spectrum_error_max` | float | Max spectrum error across surrogates |
| `spectrum_check` | PASS \| HALT | PASS if max spectrum_error <= 0.05 |
| `stability_decision` | 0.333 \| 0.667 \| 1.0 | Decision agreement across 3 blocks |
| `block_decisions` | [d1, d2, d3] | REJECT/ACCEPT per block at alpha=0.05 |
| `status` | PASS \| FAIL | Row-level verdict |

### 6 Logic Consistency Checks

1. **CHECK 1: P-value math** - p = (k+1)/(N+1), verified by N and k
2. **CHECK 2: Shuffle destroys structure** - rho1_shuffled must be near zero
3. **CHECK 3: Stability decision gating** - stability_decision != 1.0 for CERTIFIED → HALT
4. **CHECK 4: Spectrum preservation** - spectrum_error_max <= 0.05 else HALT
5. **CHECK 5: CERTIFIED vs EXPERIMENTAL** - Only CERTIFIED gates overall PASS/FAIL
6. **CHECK 6: Stop rules actually stop** - If any HALT condition fires, Overall = FAIL

### Red Flags (Auto-Fail Audit)

- 🚩 **Red Flag #1**: CERTIFIED row has status=FAIL but Overall=PASS
- 🚩 **Red Flag #2**: Contradiction tolerance - any logic inconsistency ignored
- 🚩 **Red Flag #3**: Missing causality chain - no explicit list of what gated the run

### Stop Rules Template (ENFORCE_NOW Section)

```
## STOP RULES EVALUATED (ENFORCE_NOW)

### Gate A: Shuffle Control (rho1)
- Threshold: |rho1_shuffled| < 0.05
- Evaluated on: [list certified detectors]
- Worst case: rho1_shuffled_max = X.XXXX
- Verdict: PASS | HALT

### Gate B: Spectrum Preservation
- Threshold: spectrum_error <= 5%
- Evaluated on: [list certified detectors]  
- Worst case: spectrum_error_max = X.XXXX
- Verdict: PASS | HALT

### Gate C: Decision Stability (CERTIFIED only)
- Threshold: stability_decision == 1.0 (3/3 blocks agree)
- Per certified detector: [list with values]
- Verdict: PASS | HALT

### Gate D: Noise Hallucination (CERTIFIED only)
- Check: CERTIFIED detector must NOT reject null on white_noise
- [detector]: decision = ACCEPT | REJECT
- Verdict: PASS | HALT

**Overall Status**: PASS only if ALL gates PASS for CERTIFIED detectors
```

### Detector Classification

| Detector | Class | Gates Overall? |
|----------|-------|----------------|
| autocorrelation_dependence | CERTIFIED | YES |
| time_reversal_asymmetry | EXPERIMENTAL | NO (warns only) |

### Truth Test Requirements

| Test | Purpose | Expected |
|------|---------|----------|
| T1 | AR(1) phi=0.9 detection | rho1_orig high, shuffle/spectrum PASS |
| T2 | White noise rejection | rho1_orig ~0, no hallucination |
| T3 | No-shuffle attack | Shuffle returns original → HALT |
| T4 | Broken surrogate attack | Magnitudes corrupted → spectrum HALT |
