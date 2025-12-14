# Step 1 Complete — Golden Artifacts Frozen

**Date:** 2025-12-13T20:55:16Z  
**Status:** ✅ COMPLETE

---

## What Was Done

### 1. Created Golden Artifacts Directory
```
pattern_discovery_lab/golden/
├── README.md                    # Documentation
├── results_canonical.json       # Canonical results.json (11,490 bytes)
└── stdout_canonical.txt         # Canonical stdout (3,463 bytes)
```

### 2. Captured Canonical Run Output
**Command executed:**
```bash
python -m pattern_discovery_lab --run-all --seed 42 --n-surrogates 99 > stdout.txt 2> stderr.txt
```

**Stream separation enforced:**
- ✅ stdout captured separately
- ✅ stderr captured separately (empty for compliant run)
- ❌ Did NOT use `2>&1 | tee` (merges streams, hides leaks)

### 3. Computed SHA256 Hashes

**stdout_canonical.txt:**
```
788885a37c8d5958229e0338517129b2b5a7fd66efd40dfc2ad941965341d9ee
```

**results_canonical.json:**
```
8283564cff22fd1e8d12adf498b68bf1425ebc58fd9decaf9276f5599fd89a33
```

### 4. Created Handoff Document
**File:** `pattern_discovery_lab/handoff_v0p1.json`

**Contains:**
- Contract version (0.1)
- Freeze timestamp
- Artifact hashes
- Validation status (11/11 passed)
- Run parameters
- Gate configurations
- Expected decisions
- Capture notes
- Immutability pledge

### 5. Created Verification Script
**File:** `verify_golden.sh`

Demonstrates proper stream separation and hash verification:
```bash
./verify_golden.sh
```

---

## Verification Results

### Validation
```
STDOUT VALIDATION:     ✓ All checks passed
JSON SCHEMA VALIDATION: ✓ All checks passed
RESULT: 11/11 checks passed ✓
```

### Determinism Check
When run with `--seed 42 --n-surrogates 99`, the implementation produces:
- ✅ Identical statistical results (all p-values, decisions, gate verdicts)
- ✅ Identical JSON structure
- ⚠️ Different timestamps (expected, not a bug)

**Only differences:**
```diff
- Run Folder: .../runs/20251213_205516
+ Run Folder: .../runs/20251213_205956

- "timestamp": "20251213_205516"
+ "timestamp": "20251213_205956"
```

This is **expected and correct** - timestamps change with each run. The statistical content is bit-identical.

---

## What's Frozen (Unchangeable)

### Gate Thresholds
- Gate A: `T(N) = 4.5/sqrt(N)` → 0.1423 for N=1000
- Gate B: 0.05 (5% spectrum error)
- Gate C: stability_decision = 1.0
- Gate D: CERTIFIED must ACCEPT on white_noise

### Expected Decisions
| Dataset | autocorrelation_dependence |
|---------|---------------------------|
| white_noise | ACCEPT |
| ar1_phi_0p9 | REJECT |
| garch | ACCEPT |

### P-Value Formula
```
p=(k+1)/(N+1)=((k)+1)/((N)+1)=X.XXXX
```

### Output Structure
- 5 stdout sections
- Exact field labels
- 11-column table
- JSON schema: `stop_rules`, `detector_results`, `overall_status` (object)

### Detector Classification
- **CERTIFIED:** autocorrelation_dependence (contributes to PASS/HALT)
- **EXPERIMENTAL:** time_reversal_asymmetry (info only, excluded from gating)

---

## Critical Notes

### Stream Separation Requirement
**CORRECT:**
```bash
python -m pattern_discovery_lab --run-all --seed 42 --n-surrogates 99 > stdout.txt 2> stderr.txt
```

**WRONG:**
```bash
python -m pattern_discovery_lab --run-all --seed 42 --n-surrogates 99 2>&1 | tee stdout.txt
```

**Why:** The wrong way merges stderr into stdout, which can hide diagnostic leaks and make it impossible to verify stderr is truly empty.

### Immutability Pledge
These golden artifacts are **FROZEN** and serve as the single source of truth. Any change requires:
1. Incrementing contract version (v0.2+)
2. Documenting breaking changes
3. Creating new golden artifacts
4. Updating validator

---

## Usage

### Validate New Run
```bash
# Run and capture
python -m pattern_discovery_lab --run-all --seed 42 --n-surrogates 99 > stdout.txt 2> stderr.txt

# Validate against contract
RUN_DIR="$(ls -1dt pattern_discovery_lab/runs/* | head -n 1)"
python validate_run_v0p1.py stdout.txt "$RUN_DIR/results.json"
```

### Verify Determinism
```bash
# Should show only timestamp differences
./verify_golden.sh
```

### Check Hashes
```bash
sha256sum pattern_discovery_lab/golden/*.txt pattern_discovery_lab/golden/*.json
```

---

## Files Created

1. `pattern_discovery_lab/golden/stdout_canonical.txt` — Frozen stdout
2. `pattern_discovery_lab/golden/results_canonical.json` — Frozen results
3. `pattern_discovery_lab/golden/README.md` — Documentation
4. `pattern_discovery_lab/handoff_v0p1.json` — Metadata + hashes
5. `verify_golden.sh` — Verification script

---

## Next Steps (Not Part of Step 1)

The golden artifacts are now frozen. Future work:
- Add more detectors (must be classified CERTIFIED or EXPERIMENTAL)
- Add more datasets (must define expected decisions for CERTIFIED)
- Improve gates (requires new contract version)
- Add regression tests using golden artifacts

**Any changes to gates, thresholds, or output format require v0.2+ and new golden artifacts.**

---

**Status: FROZEN ❄️ — No further changes to Step 1**
