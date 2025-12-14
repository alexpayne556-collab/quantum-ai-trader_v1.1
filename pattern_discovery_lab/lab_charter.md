# Pattern Discovery Lab Charter

## Mission

Detect whether **time-series structure exists** that survives null comparisons and perturbation tests.

**NOT**: Stock picking, return prediction, alpha generation, trading signals, or PnL optimization.

**Success includes negative results**: Proving that detectors hallucinate structure, or proving "no robust structure exists at this resolution" is a VALID and VALUABLE outcome.

---

## Operational Definition: What is a "Pattern"?

A pattern is **structure in time-series data** that:

1. **Survives null comparisons**: Detector output must differ significantly from output on pure-noise control datasets
2. **Survives perturbations**: Structure persists under time-window shifts, resampling, and parameter variations
3. **Dies under structure-killing controls**: Must be destroyed by time-shuffle and phase-randomization surrogates
4. **Is reproducible**: Same seed + config → identical results

**Not a pattern**:
- One-time anomalies or outliers
- Artifacts of window selection or preprocessing
- Statistical flukes that disappear on surrogate data
- In-sample overfits that fail out-of-sample tests

---

## Evidence Criteria

### Gate 1: Calibration Gate
All detectors MUST pass calibration tests before being used on real data:

- **Pure noise world**: Detector must NOT claim robust structure (false positive rate <5%)
- **Known-structure world**: Detector MUST find the planted structure (true positive rate >90%)
- **Finance-like null world**: May detect "texture" (volatility clustering, fat tails) but must still fail robustness tests

### Gate 2: Robustness Gate
Structure claims must survive:

- **Time-shuffle control**: Random permutation of time indices → structure disappears
- **Phase-randomization surrogate**: Fourier transform → randomize phases → inverse transform → structure disappears
- **Window shift**: Split data into non-overlapping windows → structure detected in multiple windows
- **Resampling**: Bootstrap or block-bootstrap resamples → structure stable across resamples

### Gate 3: Null-World Comparison Gate
Structure score on real data must be statistically significant vs:

- Pure noise world (same length, same variance)
- Finance-like null world (same volatility clustering, same fat tails, but no "edge")

**Significance threshold**: p-value < 0.01 (Bonferroni-corrected if multiple comparisons)

### Gate 4: Reproducibility Gate
All results must include:

- Exact random seed
- Full config (detector params, dataset params, control params)
- Versioned dataset (hash or version tag)
- Execution timestamp
- Software versions (numpy, scipy, etc.)

**If results cannot be reproduced from logs → results are INVALID**

---

## Forbidden Moves

### 1. No Outcome Labels
❌ Do NOT use forward returns, "winners/losers" labels, or any supervised learning on future price moves in Lab v0.

**Why**: Creates look-ahead bias and incentivizes overfitting to noise.

**Exception**: May label regimes AFTER the fact for visualization, but not for training.

### 2. No Prediction Language
❌ Forbidden terms in code/docs/reports:
- `predict`, `forecast`, `expected_return`, `alpha`, `signal`, `edge`, `buy`, `sell`, `PnL`, `Sharpe`

✅ Allowed terms:
- `structure_score`, `dependence_score`, `regime_label`, `breakpoint_index`, `stability_score`, `null_comparison`

**Why**: Prevents drift toward trading/prediction system. This is a structure detection lab.

### 3. No Post-Hoc Filtering
❌ Do NOT cherry-pick detectors, windows, or parameters after seeing results.

**Why**: "Researcher degrees of freedom" inflate false positive rates.

**Solution**: Pre-register detector configs in `configs/` before running. Report ALL results, not just "interesting" ones.

### 4. No Feature Engineering Without Controls
❌ Do NOT add features/transforms without running full calibration + control battery.

**Why**: Complex features can fit noise perfectly. Each new transform must pass Gates 1-4.

### 5. Multiplicative Gatekeeping
A detector result is **INVALID** unless:

```
beats_null_worlds AND survives_perturbations AND killed_by_controls
```

**All three conditions are mandatory**. One failure → entire result is rejected.

---

## Negative Result Policy

**Negative results (no robust structure detected) are SUCCESS.**

If the lab concludes:
- "No structure survives controls at this resolution"
- "Detected patterns are artifacts of volatility clustering (finance texture)"
- "Cross-series structure is explainable by common factor (SPY correlation)"

**These are VALID, PUBLISHABLE outcomes.**

Do NOT:
- Add more detectors to "find something"
- Relax thresholds to get "significant" results
- Hide negative results in favor of positive results

**Report everything**: Both null results and structure-detected results.

---

## Workflow

### Phase 1: Calibration (current)
1. Generate calibration datasets (pure noise, known-structure, finance-null)
2. Implement minimal detector set (5 detectors)
3. Implement control suite (time-shuffle, phase-randomization, null-world)
4. Run calibration tests → verify Gates 1-4 pass
5. **STOP**. Do not proceed until calibration is validated.

### Phase 2: Real-Data Exploration (BLOCKED until Phase 1 complete)
1. Load real financial time-series (200-ticker universe)
2. Run same detectors + controls on real data
3. Compare structure scores to null worlds
4. Document where structure survives vs fails
5. Accept negative results as valid outcomes

### Phase 3: Hypothesis Generation (BLOCKED until Phase 2 complete)
1. For patterns that survive Gates 1-4, generate hypotheses about mechanisms
2. Design targeted detectors to test specific hypotheses
3. Re-run full calibration + control battery on new detectors
4. Update charter if new evidence criteria emerge

---

## Key Principles

### 1. Null Hypothesis Primacy
Default assumption: **No structure exists beyond finance texture (GARCH, fat tails, autocorrelation in volatility).**

Burden of proof is on claiming structure BEYOND texture.

### 2. Controls are Mandatory
Every detector run MUST include:
- Time-shuffle control
- Phase-randomization surrogate
- Null-world comparison

No exceptions. If controls aren't run, results are INVALID.

### 3. Calibration is Non-Negotiable
No detector touches real data until it passes calibration tests.

If a detector fails calibration → fix the detector OR remove it. Do NOT adjust calibration to pass tests.

### 4. Reproducibility is Sacred
Every run produces:
- JSON log (full details: seed, config, params, software versions)
- Markdown report (human-readable summary)
- Versioned dataset metadata

If peer cannot reproduce from log → result is VOID.

### 5. Negative Results = Success
Proving "no structure" or "structure is noise" is equally valuable as finding structure.

Do NOT bias toward positive results.

---

## Governance

### Amendment Process
This charter can be amended only when:

1. New evidence suggests current gates are too strict or too loose
2. Calibration tests reveal systematic issues with current criteria
3. ALL previous results are re-validated under new criteria

**No retroactive relaxation of criteria to "save" failed results.**

### Violations
If a run violates this charter (skips controls, cherry-picks, uses forbidden language):

1. Results are VOID
2. Document violation in lab log
3. Re-run correctly OR archive as "invalid run"

---

## Version

- **Version**: 0.1
- **Date**: December 13, 2025
- **Authors**: Pattern Discovery Lab Team
- **Status**: Active

---

## Acknowledgments

This charter is inspired by:
- Registered Reports (pre-registration in psychology)
- Financial Economics reproducibility standards
- NIST Statistical Engineering principles
- "The Garden of Forking Paths" (Gelman & Loken)
