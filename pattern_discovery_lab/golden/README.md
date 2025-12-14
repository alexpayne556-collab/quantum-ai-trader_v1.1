# Golden Artifacts - Pattern Discovery Lab v0.1

## Contract-Compliant Reference Outputs

This directory contains frozen, contract-compliant reference outputs that serve as the canonical specification for Pattern Discovery Lab v0.1.

### Artifacts

| File | SHA256 | Size | Contract Status |
|------|--------|------|-----------------|
| `stdout_canonical.txt` | `f74691c94834f1eba6825f564890cd6d5c41f08b0913ada65724bcb8dee9d55b` | 3.4 KB | ✅ COMPLIANT |
| `results_canonical.json` | `e1a3629c872dd669aea9d2fe63f87cb4c6db1cc3d13a2c5a27df39cf3af18a80` | 4.4 KB | ✅ COMPLIANT |

### Generation Command

```bash
python -m pattern_discovery_lab --run-all --seed 42 --n-surrogates 99 \
  > stdout_canonical.txt 2> stderr_canonical.txt
```

**Deterministic Parameters**:
- Seed: 42
- N surrogates: 99
- Sample size: 1000
- Datasets: white_noise, ar1_phi_0p9, garch
- Detectors: autocorrelation_dependence (CERTIFIED), time_reversal_asymmetry (EXPERIMENTAL)

### Contract Requirements

#### stdout_canonical.txt
- ✅ Required sections in order (CLI, Run Folder, STOP RULES, Table, OVERALL)
- ✅ Exact field labels (Gate A-D, Overall Status, Reason)
- ✅ P-value formula: `p=(k+1)/(N+1)=((X)+1)/((Y)+1)=X.XXXX`
- ✅ No log prefixes (INFO, DEBUG, WARNING, ERROR)
- ✅ No scientific notation in p-values
- ✅ LF line endings (no CRLF)
- ✅ No trailing whitespace (strict mode)
- ⚠️ Contains timestamps (Run Folder, Results written to) - prevents bit-identical hash match

#### results_canonical.json
- ✅ **Closed schema**: Exactly 3 top-level keys
  - `stop_rules`
  - `detector_results`
  - `overall_status`
- ✅ No extra keys (meta, rows, legacy_stop_rules, etc.)
- ✅ All numeric values finite (no NaN, no Infinity)
- ✅ P-values in [0, 1]
- ✅ Verdict enums: PASS or HALT
- ✅ Decision enums: ACCEPT, REJECT, or N/A
- ✅ Class enums: CERTIFIED or EXPERIMENTAL
- ✅ Logic consistent (all gates PASS → overall PASS)
- ✅ No timestamps (deterministic, hash-stable)

### Debug Output (Not Contract-Validated)

`results_debug.json` contains additional diagnostic information:
- `meta`: Timestamp, seed, parameters
- `rows`: Row-level diagnostics
- `legacy_stop_rules`: Detailed gate results

**This file is NOT part of the contract and is NOT validated.**

### Validation

#### Quick Validation
```bash
# Verify against this golden
./verify_golden.sh
```

#### Manual Validation
```bash
# Create empty stderr
touch /tmp/empty.txt

# Run contract validator
python -m pattern_discovery_lab.contract_validator \
  --stdout pattern_discovery_lab/golden/stdout_canonical.txt \
  --results pattern_discovery_lab/golden/results_canonical.json \
  --stderr /tmp/empty.txt \
  --strict
```

#### Test Suite
```bash
# Happy path tests
pytest tests/test_contract_validator.py -v

# Mutation tests
pytest tests/test_contract_mutations.py -v
```

### Regeneration

**WARNING**: Only regenerate if contract requirements change or implementation is fixed.

```bash
# 1. Run deterministic test
python -m pattern_discovery_lab --run-all --seed 42 --n-surrogates 99 \
  > /tmp/stdout_new.txt 2> /tmp/stderr_new.txt

# 2. Get latest run
RUN_DIR=$(ls -1dt pattern_discovery_lab/runs/* | head -n 1)

# 3. Validate new artifacts
python -m pattern_discovery_lab.contract_validator \
  --stdout /tmp/stdout_new.txt \
  --results "$RUN_DIR/results.json" \
  --stderr /tmp/stderr_new.txt \
  --strict

# 4. If validation passes, update golden
cp /tmp/stdout_new.txt pattern_discovery_lab/golden/stdout_canonical.txt
cp "$RUN_DIR/results.json" pattern_discovery_lab/golden/results_canonical.json

# 5. Compute new hashes
sha256sum pattern_discovery_lab/golden/*.{txt,json}

# 6. Update this README with new hashes

# 7. Run full test suite
pytest tests/test_contract_*.py -v
```

### Contract Lock Reference

See [CONTRACT_LOCK_v0p1.md](../../CONTRACT_LOCK_v0p1.md) for immutable contract specification.

### Mutation Test Coverage

27+ mutation test cases validate that minimal changes to these artifacts trigger specific validation failures:

- Stdout grammar violations (11 cases)
- JSON schema violations (14 cases)  
- Stream separation violations (2 cases)

See [tests/test_contract_mutations.py](../../tests/test_contract_mutations.py) for details.

### Version History

| Date | Version | Change | SHA256 (results.json) |
|------|---------|--------|----------------------|
| 2025-12-13 | v0.1.1 | Regenerated with closed schema (removed extra keys) | `e1a3629c872d...` |
| 2025-12-13 | v0.1.0 | Initial freeze | `8283564c...` (violated contract - 7 keys) |

---

**Status**: ✅ FROZEN - Contract-compliant reference artifacts for v0.1
