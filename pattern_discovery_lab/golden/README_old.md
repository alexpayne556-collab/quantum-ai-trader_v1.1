# Golden Artifacts — Enforcement Contract v0.1

**Status:** 🔒 FROZEN (2025-12-13T20:55:16Z)

This directory contains the canonical, immutable reference outputs for Enforcement Contract v0.1. These artifacts serve as the single source of truth for what a compliant run looks like.

## Contents

### `stdout_canonical.txt`
- **SHA256:** `788885a37c8d5958229e0338517129b2b5a7fd66efd40dfc2ad941965341d9ee`
- **Size:** 3,463 bytes
- **Description:** Complete stdout from a passing run with `--seed 42 --n-surrogates 99`

### `results_canonical.json`
- **SHA256:** `8283564cff22fd1e8d12adf498b68bf1425ebc58fd9decaf9276f5599fd89a33`
- **Size:** 11,490 bytes
- **Description:** Complete results.json from the same passing run

## Validation

Both artifacts passed **11/11** checks from `validate_run_v0p1.py`:

```bash
$ python validate_run_v0p1.py golden/stdout_canonical.txt golden/results_canonical.json
======================================================================
VALIDATION RESULT: 11/11 checks passed
======================================================================
✓ Contract satisfied!
```

## Reproducing Golden Run

To reproduce these exact outputs (bit-for-bit identical):

```bash
python -m pattern_discovery_lab --run-all --seed 42 --n-surrogates 99 \
  > stdout.txt 2> stderr.txt

# Verify bit-identical match
sha256sum stdout.txt
# Should output: 788885a37c8d5958229e0338517129b2b5a7fd66efd40dfc2ad941965341d9ee

# Get latest run folder
RUN_DIR="$(ls -1dt pattern_discovery_lab/runs/* | head -n 1)"
sha256sum "$RUN_DIR/results.json"
# Should output: 8283564cff22fd1e8d12adf498b68bf1425ebc58fd9decaf9276f5599fd89a33
```

## Critical Notes

### Stream Separation
**DO NOT** use `2>&1 | tee stdout.txt` for compliance checks. This merges stderr into stdout and can hide diagnostic leaks.

**CORRECT:**
```bash
python -m pattern_discovery_lab --run-all --seed 42 --n-surrogates 99 > stdout.txt 2> stderr.txt
```

**WRONG:**
```bash
python -m pattern_discovery_lab --run-all --seed 42 --n-surrogates 99 2>&1 | tee stdout.txt
```

### Determinism
With `--seed 42`, the implementation **must** be bit-identical to these golden artifacts. Any deviation indicates:
- Bug in implementation
- Platform-specific floating point differences (should not happen with numpy)
- Unintended non-determinism in the code

### Immutability Pledge
These artifacts are **FROZEN**. They cannot be changed without:
1. Incrementing contract version (v0.2+)
2. Recording breaking changes
3. Creating new golden artifacts
4. Updating validation suite

## What's Frozen

### Gates
- **Gate A threshold:** `4.5/sqrt(N)` = 0.1423 for N=1000
- **Gate B threshold:** 0.05
- **Gate C requirement:** stability_decision = 1.0
- **Gate D rule:** CERTIFIED must ACCEPT on white_noise

### Expected Decisions
| Dataset | autocorrelation_dependence |
|---------|---------------------------|
| white_noise | ACCEPT |
| ar1_phi_0p9 | REJECT |
| garch | ACCEPT |

### P-Value Formula
```
p=(k+1)/(N+1)=((81)+1)/((99)+1)=0.8200
```

### Output Structure
- 5 stdout sections in exact order
- Exact field labels (shuffle_check:, spectrum_check:, etc.)
- Table with 11 columns
- JSON schema with stop_rules, detector_results, overall_status

## Usage in CI/Testing

```bash
# Regression test: verify current implementation matches golden
python -m pattern_discovery_lab --run-all --seed 42 --n-surrogates 99 > /tmp/stdout.txt 2> /tmp/stderr.txt
diff golden/stdout_canonical.txt /tmp/stdout.txt || echo "REGRESSION: stdout differs!"

RUN_DIR="$(ls -1dt pattern_discovery_lab/runs/* | head -n 1)"
diff golden/results_canonical.json "$RUN_DIR/results.json" || echo "REGRESSION: results.json differs!"
```

## Metadata

See `handoff_v0p1.json` for complete metadata including:
- Contract version
- Freeze timestamp
- Validation status
- Run parameters
- Gate configurations
- Expected decisions

---

**Remember:** These artifacts are the unchanging reference point. Everything else must match them.
