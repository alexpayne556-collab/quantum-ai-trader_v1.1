#!/usr/bin/env python3
"""
Stdout Canonicalizer for Pattern Discovery Lab

Canonicalizes ONLY the 3 volatile run-path lines for deterministic hashing:
1) Run Folder: .../pattern_discovery_lab/runs/<run_id>
2) Results written to: .../pattern_discovery_lab/runs/<run_id>/results.json
3) Debug info written to: .../pattern_discovery_lab/runs/<run_id>/results_debug.json

All other stdout content remains byte-strict and hash-sensitive.
"""

import re
import hashlib
import sys


def canonicalize_stdout(raw_stdout: str) -> str:
    """
    Canonicalize stdout for deterministic hashing.
    
    Replaces ONLY the <run_id> segment (timestamp folder) with <RUN_ID>
    on the 3 volatile lines. Uses strict anchored regex to avoid overmatching.
    
    Args:
        raw_stdout: Raw stdout content
    
    Returns:
        Canonicalized stdout with volatile timestamps replaced
    """
    lines = raw_stdout.split('\n')
    result = []
    
    for line in lines:
        # Pattern 1: Run Folder line
        # e.g., "Run Folder: /workspaces/.../pattern_discovery_lab/runs/20251213_221509"
        match = re.match(
            r'^(Run Folder: .*/pattern_discovery_lab/runs/)(\d{8}_\d{6})(\s*)$',
            line
        )
        if match:
            result.append(f"{match.group(1)}<RUN_ID>{match.group(3)}")
            continue
        
        # Pattern 2: Results written to line
        # e.g., "Results written to: .../pattern_discovery_lab/runs/20251213_221509/results.json"
        match = re.match(
            r'^(Results written to: .*/pattern_discovery_lab/runs/)(\d{8}_\d{6})(/results\.json)(\s*)$',
            line
        )
        if match:
            result.append(f"{match.group(1)}<RUN_ID>{match.group(3)}{match.group(4)}")
            continue
        
        # Pattern 3: Debug info written to line
        # e.g., "Debug info written to: .../pattern_discovery_lab/runs/20251213_221509/results_debug.json"
        match = re.match(
            r'^(Debug info written to: .*/pattern_discovery_lab/runs/)(\d{8}_\d{6})(/results_debug\.json)(\s*)$',
            line
        )
        if match:
            result.append(f"{match.group(1)}<RUN_ID>{match.group(3)}{match.group(4)}")
            continue
        
        # No match - keep line unchanged
        result.append(line)
    
    return '\n'.join(result)


def compute_canonical_hash(raw_stdout: str) -> str:
    """
    Compute SHA256 hash of canonicalized stdout.
    
    Args:
        raw_stdout: Raw stdout content
    
    Returns:
        Hexadecimal SHA256 hash of canonicalized content
    """
    canonicalized = canonicalize_stdout(raw_stdout)
    return hashlib.sha256(canonicalized.encode('utf-8')).hexdigest()


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Canonicalize stdout for deterministic hashing'
    )
    parser.add_argument(
        '--hash',
        action='store_true',
        help='Output SHA256 hash of canonicalized stdin'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Output canonicalized stdin'
    )
    
    args = parser.parse_args()
    
    raw_stdin = sys.stdin.read()
    
    if args.hash:
        print(compute_canonical_hash(raw_stdin))
    elif args.show:
        print(canonicalize_stdout(raw_stdin), end='')
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
