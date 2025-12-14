# Pattern Discovery Lab

**A rigorous research instrument for detecting time-series structure in financial data.**

---

## ⚠️ What This Is NOT

This is **NOT**:
- A trading system
- A stock screener
- A return predictor
- An alpha generator
- A signal generator
- A backtesting engine
- A portfolio optimizer

**No predictions. No trades. No PnL optimization.**

---

## What This IS

A **structure detection laboratory** that answers the question:

> "Does time-series structure exist that survives null comparisons and perturbation tests?"

**Success includes negative results**: Proving that apparent patterns are noise or artifacts is a VALID outcome.

---

## Core Principles

### 1. Null Hypothesis Primacy
Default: **No exploitable structure exists** beyond "finance texture" (volatility clustering, fat tails, autocorrelation).

Burden of proof: Claim structure BEYOND texture.

### 2. Multiplicative Gatekeeping
A pattern is valid ONLY if it:
- Beats null worlds (pure noise, finance-like null)
- Survives perturbations (time-window shifts, resampling)
- Dies under controls (time-shuffle, phase-randomization)

**All three gates are mandatory.**

### 3. Calibration First
No detector touches real data until it passes calibration:
- Pure noise → no structure claimed
- Known-structure → structure detected
- Finance-null → texture detected but fails robustness

### 4. Reproducibility
Every run produces:
- JSON log (seed, config, params, versions)
- Markdown report (summary, results, failures)
- Versioned datasets

Same seed + config → identical results.

### 5. Negative Results = Success
Proving "no robust structure" is as valuable as finding structure.

Do NOT bias toward positive results.

---

## Project Structure

```
pattern_discovery_lab/
├── README.md                 # This file
├── lab_charter.md           # Constitution (evidence criteria, forbidden moves)
├── configs/                 # Run configurations (YAML/JSON)
├── data/
│   └── calibration/         # Generated test datasets + metadata
├── src/
│   ├── datasets/            # Dataset generators + loaders
│   ├── detectors/           # Structure detection telescope
│   ├── controls/            # Null controls (shuffle, surrogate, comparison)
│   ├── runner/              # Orchestration + execution
│   └── reporting/           # JSON log + markdown report writers
├── runs/                    # Timestamped run outputs
│   └── <run_id>/
│       ├── log.json         # Full execution log
│       ├── report.md        # Human-readable summary
│       └── artifacts/       # Plots, intermediate data
└── tests/                   # Calibration tests + unit tests
```

---

## Calibration Datasets (4 Families)

### 1. Pure Noise World
- White noise
- Random walk
- No structure by design
- **Expected**: Detectors claim no robust structure

### 2. Known-Structure World
- AR(1) process with known coefficient
- GARCH with known parameters
- Designed breakpoints at known timestamps
- **Expected**: Detectors find planted structure AND survive controls

### 3. Finance-Like Null World
- GARCH + fat-tailed innovations
- Autocorrelated volatility
- Jump diffusion
- **Has finance texture but no "edge"**
- **Expected**: Detectors may find texture but structure fails robustness gates

### 4. Real-Data Adapter
- Empty interface initially
- Later: Load 200-ticker financial data
- BLOCKED until calibration passes

---

## Detector Families (v0 Minimal Set)

### 1. Compressibility Proxies
- Lempel-Ziv complexity
- Description length (MDL)
- Entropy rate

### 2. Dependence Detectors
- Linear: Autocorrelation decay, Ljung-Box
- Nonlinear: Mutual information, BDS test

### 3. Regime Structure
- Bayesian changepoint detection
- Ruptures library (Pelt, BottomUp)
- Hidden Markov Models

### 4. Cross-Series Structure
- Transfer entropy
- Granger causality
- Dynamic correlation

### 5. Stability Checks
- Window shift (split data → detect in each window)
- Bootstrap resampling (structure stable?)
- Parameter sensitivity (perturb detector params)

**Only 5 detectors in v0. Do NOT add more until calibration passes.**

---

## Controls (Mandatory for Every Run)

### 1. Time-Shuffle Control
Randomly permute time indices → structure should disappear.

**If structure survives time-shuffle → likely spurious.**

### 2. Phase-Randomization Surrogate
- FFT → randomize phases → inverse FFT
- Preserves power spectrum, destroys temporal structure
- **If structure survives surrogate → might be real.**

### 3. Null-World Comparison
Run same detector on:
- Pure noise world (same length, variance)
- Finance-like null world (same texture)

**Real-data structure score must be >> null-world scores (p < 0.01).**

---

## Quick Start

### 1. Install Dependencies
```bash
cd pattern_discovery_lab
pip install -r requirements.txt
```

### 2. Run Calibration
```bash
# Run on pure noise
python -m src.runner --dataset pure_noise --config default

# Run on known-structure
python -m src.runner --dataset known_structure --config default

# Run on finance-null
python -m src.runner --dataset finance_null --config default
```

### 3. Validate Calibration Tests
```bash
pytest tests/test_calibration.py -v
```

**Expected**:
- Pure noise: All detectors report "no robust structure"
- Known-structure: Detectors find structure, survive controls
- Finance-null: Detectors may find texture, but fail robustness

**If ANY test fails → fix detectors/controls. Do NOT proceed to real data.**

### 4. Run on Real Data (ONLY after calibration passes)
```bash
python -m src.runner --dataset real_data --config default
```

---

## Interpreting Reports

Every run produces: `runs/<run_id>/report.md`

### Section 1: Dataset
- Family (pure_noise, known_structure, finance_null, real_data)
- Config (length, seed, parameters)
- Metadata hash (for reproducibility)

### Section 2: Detectors Run
- Detector name + version
- Parameters
- Structure score (numeric)

### Section 3: Controls
- Time-shuffle: Pass/Fail
- Phase-randomization: Pass/Fail
- Null-world comparison: p-value

### Section 4: Results
- Overall verdict: "Robust structure detected" OR "No robust structure" OR "Structure is finance texture"
- Gate status:
  - Calibration Gate: ✅/❌
  - Robustness Gate: ✅/❌
  - Null-Comparison Gate: ✅/❌
  - Reproducibility Gate: ✅/❌

### Section 5: Where It Broke
- If any gate failed: detailed diagnostic
- Suggested next steps

---

## Calibration Requirements (Hard Assertions)

The lab enforces these via `tests/test_calibration.py`:

### Assertion 1: Pure Noise
```python
assert detector.structure_score(pure_noise) < threshold
assert not detector.passes_gates(pure_noise)
```

### Assertion 2: Known-Structure
```python
assert detector.structure_score(known_structure) > threshold
assert detector.passes_gates(known_structure)
assert detector.survives_time_shuffle(known_structure) == False
```

### Assertion 3: Finance-Null
```python
# May detect texture
assert detector.may_detect_texture(finance_null)
# But fails robustness
assert not detector.passes_robustness_gate(finance_null)
```

### Assertion 4: Controls
```python
# Time-shuffle destroys structure in known-structure world
shuffled = time_shuffle(known_structure)
assert detector.structure_score(shuffled) < detector.structure_score(known_structure)
```

**If any assertion fails → STOP. Fix before proceeding.**

---

## Execution Order (What to Build First)

### Phase 1: Foundation (Current)
1. ✅ Write docs (`README.md`, `lab_charter.md`)
2. ⏳ Build calibration dataset generators
3. Build detector interfaces + control wrappers
4. Implement minimal detector set (5 detectors)
5. Implement runner + logging + reporting
6. Write calibration tests
7. **STOP after calibration passes**

### Phase 2: Real Data (BLOCKED)
Only after Phase 1 complete:
1. Implement real-data adapter
2. Run same telescope + controls on 200 tickers
3. Document results (positive OR negative)

### Phase 3: Hypothesis (BLOCKED)
Only after Phase 2 complete:
1. Generate hypotheses from validated patterns
2. Design targeted detectors
3. Re-run full calibration

---

## Naming Conventions

### ✅ Allowed Terms
- `structure_score`
- `dependence_score`
- `stability_score`
- `regime_label`
- `breakpoint_index`
- `null_comparison`
- `perturbation_test`

### ❌ Forbidden Terms
- `predict`, `forecast`
- `alpha`, `signal`, `edge`
- `buy`, `sell`, `trade`
- `PnL`, `Sharpe`, `return`
- `winner`, `loser`

**If forbidden terms appear in code → refactor immediately.**

---

## Common Pitfalls to Avoid

### 1. "But it looks significant!"
Visual patterns can be deceiving. Run controls before claiming structure.

### 2. "Just one more detector..."
Do NOT add detectors to "find something." Calibration must pass first.

### 3. "Let's just peek at real data..."
NO. Calibration gate is non-negotiable.

### 4. "This negative result is boring"
Negative results ARE valuable. Document and publish.

### 5. "I'll just tweak the threshold..."
Post-hoc parameter tuning inflates false positives. Pre-register configs.

---

## FAQ

### Q: Can I use this lab to build a trading strategy?
**A**: No. This lab detects structure, not tradeable edges. If you find robust structure, you'd need a SEPARATE research track (with different methodology) to test trading viability.

### Q: What if I find structure that survives all gates?
**A**: Document thoroughly. Then ask: "What mechanism could produce this structure?" Design targeted tests. Do NOT immediately assume it's exploitable.

### Q: What if calibration tests fail?
**A**: Fix the detectors OR fix the controls. Do NOT relax calibration criteria. Do NOT skip to real data.

### Q: Can I use forward returns for anything?
**A**: NOT in Lab v0. This is a structure detection lab, not a prediction lab. Forward returns create look-ahead bias.

### Q: What's the difference between "finance texture" and "structure"?
**A**: Texture = well-known stylized facts (volatility clustering, fat tails). Structure = exploitable patterns beyond texture. Texture alone is NOT sufficient.

### Q: How do I know if results are reproducible?
**A**: Re-run with same seed + config. Should get identical results. If not → log is incomplete or code is non-deterministic.

---

## Dependencies

```
numpy>=1.24
scipy>=1.10
pandas>=2.0
scikit-learn>=1.3
statsmodels>=0.14
ruptures>=1.1  # Changepoint detection
hmmlearn>=0.3  # Hidden Markov Models
pyyaml>=6.0  # Config files
pytest>=7.4  # Testing
```

---

## License

MIT License (for code)

Data/results: CC-BY-4.0 (must cite if used)

---

## Contact

Pattern Discovery Lab Team  
December 13, 2025

---

## Version

- **Version**: 0.1.0
- **Status**: Calibration Phase
- **Last Updated**: December 13, 2025

---

## Acknowledgments

Methodology inspired by:
- Registered Reports (psychology)
- Lopez de Prado, "Advances in Financial Machine Learning"
- Harvey, Liu, Zhu, "...and the Cross-Section of Expected Returns" (multiple testing)
- NIST Statistical Engineering Division
