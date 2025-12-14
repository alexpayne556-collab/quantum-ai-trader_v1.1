# CONTRACT LOCK v0.1 (IMMUTABLE)

## Contract Artifacts (ONLY)
Engines MUST produce exactly two contract artifacts:
1) stdout (structured; contract-validated)
2) results.json (machine output; contract-validated)
Any other outputs are NON-CONTRACT.

## results.json CLOSED SCHEMA (Top-level keys EXACTLY)
results.json MUST have exactly these top-level keys (no more, no fewer):
- stop_rules
- detector_results
- overall_status
If any other top-level key exists, the run is NON-COMPLIANT.

## Debug Escape Hatch
Optional: results_debug.json may exist.
- It is NOT part of the contract.
- It is NOT read by downstream systems.
- It may contain any extra fields (meta, rows, legacy_stop_rules, etc.)

## Stream Separation (Mandatory)
Contract stdout MUST be captured from stdout ONLY.
Do NOT merge stderr into stdout. (No `2>&1 | tee` for canonical capture.)

## Contract Version
This lock applies to ENFORCEMENT CONTRACT v0.1.
Any schema change requires v0.2 and a new lock file.
