#!/bin/bash
# verify_golden.sh - Verify current implementation matches golden artifacts
#
# Usage: ./verify_golden.sh
#
# Exit codes:
#   0 - Perfect match (bit-identical)
#   1 - Validation failed
#   2 - Hash mismatch

set -e

REPO_ROOT="/workspaces/quantum-ai-trader_v1.1"
GOLDEN_DIR="$REPO_ROOT/pattern_discovery_lab/golden"

echo "======================================================================="
echo "GOLDEN ARTIFACT VERIFICATION"
echo "======================================================================="
echo ""

# Clean up temp files
rm -f /tmp/stdout_test.txt /tmp/stderr_test.txt

# Run with stream separation (CORRECT METHOD)
echo "Running: python -m pattern_discovery_lab --run-all --seed 42 --n-surrogates 99"
echo "Capturing stdout and stderr separately..."
echo ""

cd "$REPO_ROOT"
python -m pattern_discovery_lab --run-all --seed 42 --n-surrogates 99 \
  > /tmp/stdout_test.txt 2> /tmp/stderr_test.txt

echo "Exit code: $?"
echo ""

# Check stderr
if [ -s /tmp/stderr_test.txt ]; then
    echo "⚠️  WARNING: stderr is not empty!"
    echo "Contents:"
    cat /tmp/stderr_test.txt
    echo ""
fi

# Compute hashes
echo "Computing SHA256 hashes..."

# Use canonicalizer module for deterministic stdout hashing
STDOUT_HASH=$(python3 -m pattern_discovery_lab.canonicalizer --hash < /tmp/stdout_test.txt)
echo "stdout hash (canonicalized):  $STDOUT_HASH"

GOLDEN_STDOUT_HASH=$(python3 -m pattern_discovery_lab.canonicalizer --hash < "$GOLDEN_DIR/stdout_canonical.txt")
echo "golden hash (canonicalized):  $GOLDEN_STDOUT_HASH"
echo ""

# Get latest run
RUN_DIR=$(ls -1dt "$REPO_ROOT/pattern_discovery_lab/runs/"* | head -n 1)
echo "Latest run: $RUN_DIR"

JSON_HASH=$(sha256sum "$RUN_DIR/results.json" | awk '{print $1}')
echo "results hash: $JSON_HASH"

GOLDEN_JSON_HASH=$(sha256sum "$GOLDEN_DIR/results_canonical.json" | awk '{print $1}')
echo "golden hash:  $GOLDEN_JSON_HASH"
echo ""

# Run contract validator
echo "======================================================================="
echo "Running Contract Validator..."
echo "======================================================================="
python -m pattern_discovery_lab.contract_validator \
  --stdout /tmp/stdout_test.txt \
  --results "$RUN_DIR/results.json" \
  --stderr /tmp/stderr_test.txt \
  --golden-stdout "$GOLDEN_DIR/stdout_canonical.txt" \
  --golden-json "$GOLDEN_DIR/results_canonical.json" \
  --check-hashes \
  --strict

VALIDATOR_EXIT=$?
echo ""

# Compare
if [ "$STDOUT_HASH" == "$GOLDEN_STDOUT_HASH" ] && [ "$JSON_HASH" == "$GOLDEN_JSON_HASH" ]; then
    echo "======================================================================="
    echo "✅ PERFECT MATCH - Bit-identical to golden artifacts!"
    echo "======================================================================="
    if [ $VALIDATOR_EXIT -eq 0 ]; then
        echo "✅ Contract validation: PASSED"
    else
        echo "⚠️  Contract validation: FAILED (but hashes match)"
    fi
    exit 0
else
    echo "======================================================================="
    echo "❌ HASH MISMATCH - Output differs from golden artifacts!"
    echo "======================================================================="
    
    if [ "$STDOUT_HASH" != "$GOLDEN_STDOUT_HASH" ]; then
        echo ""
        echo "stdout differs. Running diff:"
        diff -u "$GOLDEN_DIR/stdout_canonical.txt" /tmp/stdout_test.txt || true
    fi
    
    if [ "$JSON_HASH" != "$GOLDEN_JSON_HASH" ]; then
        echo ""
        echo "results.json differs. Running diff:"
        diff -u "$GOLDEN_DIR/results_canonical.json" "$RUN_DIR/results.json" || true
    fi
    
    exit 2
fi
