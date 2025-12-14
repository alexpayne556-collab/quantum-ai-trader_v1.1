#!/usr/bin/env python3
"""
Schema definitions and JSON serialization for Lab V0.

Ensures deterministic output with sort_keys=True and fixed numeric rounding.
"""
import json
import re
from typing import Dict, Any
from pathlib import Path


SCHEMA_VERSION = "lab_v0_0.1"


def sanitize_string(s: str) -> str:
    """
    Check string for non-ASCII/control characters.
    
    Raises ValueError if restricted characters found.
    
    Args:
        s: String to check
    
    Returns:
        Original string if valid
    
    Raises:
        ValueError: If non-ASCII or control characters found
    """
    # Check for non-ASCII
    try:
        s.encode('ascii')
    except UnicodeEncodeError:
        raise ValueError(f"String contains non-ASCII characters: {s[:50]}")
    
    # Check for control characters (except \n, \r, \t)
    control_chars = re.findall(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', s)
    if control_chars:
        raise ValueError(f"String contains control characters: {control_chars}")
    
    return s


def build_results_dict(
    meta: Dict[str, Any],
    splits: list,
    ic_results: Dict[str, Any],
    gates: Dict[str, Any],
    negative_controls: Dict[str, Any],
    overall_status: str
) -> Dict[str, Any]:
    """
    Build results dictionary with lab_v0 schema.
    
    Args:
        meta: Metadata (timestamp, seed, parameters)
        splits: List of split results
        ic_results: IC statistics
        gates: Gate results
        negative_controls: Negative control results
        overall_status: Overall run status
    
    Returns:
        Results dictionary
    """
    results = {
        "schema_version": SCHEMA_VERSION,
        "meta": meta,
        "splits": splits,
        "ic_results": ic_results,
        "gates": gates,
        "negative_controls": negative_controls,
        "overall_status": overall_status
    }
    
    return results


def write_json(path: str, obj: Dict[str, Any]) -> None:
    """
    Write JSON with deterministic serialization.
    
    Uses:
    - sort_keys=True for key ordering
    - indent=2 for readability
    - ensure_ascii=True for determinism
    
    Args:
        path: Output file path
        obj: Object to serialize
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(
            obj,
            f,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False
        )


def read_json(path: str) -> Dict[str, Any]:
    """
    Read JSON file.
    
    Args:
        path: Input file path
    
    Returns:
        Parsed JSON object
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
