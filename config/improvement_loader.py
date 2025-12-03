"""
Improvement Loader
------------------
Loads optimized weights and detected patterns from Colab auto-improve output.
File expected: `improvement_recommendations.json` at workspace root.
Safe to import even if file missing (returns empty defaults).
"""
from __future__ import annotations

import json
import os
from typing import Dict, Any, Tuple

DEFAULT_PATH = os.path.join(os.getcwd(), 'improvement_recommendations.json')


def load_improvements(path: str | None = None) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Return (weights, patterns) loaded from JSON, or empty defaults."""
    p = path or DEFAULT_PATH
    if not os.path.exists(p):
        return {}, {}
    try:
        with open(p, 'r') as f:
            data = json.load(f)
        weights = data.get('optimized_weights', {}) or {}
        patterns = data.get('patterns', {}) or {}
        # Normalize keys to strings
        weights = {str(k): float(v) for k, v in weights.items() if isinstance(v, (int, float))}
        return weights, patterns
    except Exception:
        return {}, {}


def get_weight(key: str, default: float = 0.0, path: str | None = None) -> float:
    weights, _ = load_improvements(path)
    return float(weights.get(key, default))
