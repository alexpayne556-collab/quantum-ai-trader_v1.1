#!/usr/bin/env python3
"""
QUICK VERIFICATION TEST
=======================
Tests the PRODUCTION_SYSTEM_2026.py imports and basic functionality.

Run this BEFORE pushing to Shadow PC to catch any errors.
"""

import sys
print("=" * 60)
print("PRODUCTION SYSTEM 2026 - VERIFICATION TEST")
print("=" * 60)

# Test 1: Import the module
print("\n[TEST 1] Importing PRODUCTION_SYSTEM_2026...")
try:
    from PRODUCTION_SYSTEM_2026 import (
        ValidationConfig,
        MarketRegime,
        load_watchlist_data,
        calculate_features,
        detect_regime_lagged,
        get_strategy_definitions,
        benjamini_hochberg_correction,
        generate_walk_forward_windows,
        calculate_t_statistic,
        kelly_position_size,
        InstitutionalActivityDetector,
        FiveFilterDipDetector,
    )
    print("  ✅ All imports successful")
except Exception as e:
    print(f"  ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Create configuration
print("\n[TEST 2] Creating ValidationConfig...")
try:
    config = ValidationConfig()
    print(f"  ✅ Config created:")
    print(f"      min_sample_size = {config.min_sample_size}")
    print(f"      t_threshold = {config.t_threshold}")
    print(f"      cost_round_trip = {config.cost_round_trip}")
    print(f"      return_cap = {config.return_cap}")
except Exception as e:
    print(f"  ❌ Config failed: {e}")
    sys.exit(1)

# Test 3: Get strategies
print("\n[TEST 3] Loading strategies...")
try:
    strategies = get_strategy_definitions()
    print(f"  ✅ {len(strategies)} strategies loaded:")
    for name in list(strategies.keys())[:5]:
        print(f"      - {name}")
    if len(strategies) > 5:
        print(f"      ... and {len(strategies) - 5} more")
except Exception as e:
    print(f"  ❌ Strategies failed: {e}")
    sys.exit(1)

# Test 4: BH correction
print("\n[TEST 4] Testing Benjamini-Hochberg correction...")
try:
    import numpy as np
    p_values = np.array([0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    significant = benjamini_hochberg_correction(p_values, alpha=0.05)
    n_sig = significant.sum()
    print(f"  ✅ BH correction works: {n_sig}/{len(p_values)} significant at α=0.05")
except Exception as e:
    print(f"  ❌ BH correction failed: {e}")
    sys.exit(1)

# Test 5: Kelly criterion
print("\n[TEST 5] Testing Kelly criterion...")
try:
    size = kelly_position_size(
        win_rate=0.60, 
        avg_win=0.05, 
        avg_loss=0.03, 
        config=config
    )
    print(f"  ✅ Kelly size = {size:.2%} (60% WR, 5% win, 3% loss)")
except Exception as e:
    print(f"  ❌ Kelly failed: {e}")
    sys.exit(1)

# Test 6: T-statistic
print("\n[TEST 6] Testing t-statistic calculation...")
try:
    import pandas as pd
    returns = pd.Series(np.random.randn(150) * 0.02)
    mean, n, t = calculate_t_statistic(returns, min_n=100)
    print(f"  ✅ T-stat = {t:.2f} (n={n}, mean={mean:.4f})")
except Exception as e:
    print(f"  ❌ T-stat failed: {e}")
    sys.exit(1)

# Test 7: Classes instantiation
print("\n[TEST 7] Testing class instantiation...")
try:
    detector = InstitutionalActivityDetector()
    dip_system = FiveFilterDipDetector()
    print("  ✅ InstitutionalActivityDetector created")
    print("  ✅ FiveFilterDipDetector created")
except Exception as e:
    print(f"  ❌ Class instantiation failed: {e}")
    sys.exit(1)

# Test 8: Enums
print("\n[TEST 8] Testing MarketRegime enum...")
try:
    assert MarketRegime.BULL.value == "BULL"
    assert MarketRegime.BEAR.value == "BEAR"
    assert MarketRegime.RANGE.value == "RANGE"
    print("  ✅ MarketRegime enum works")
except Exception as e:
    print(f"  ❌ Enum failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✅")
print("=" * 60)
print("""
NEXT STEPS:
1. Push both files to Shadow PC:
   - PRODUCTION_SYSTEM_2026.py
   - MASTER_RESEARCH_SYNTHESIS_2026.md

2. On Shadow PC, run:
   python PRODUCTION_SYSTEM_2026.py

3. Compare results to previous validation runs

4. Review MASTER_RESEARCH_SYNTHESIS_2026.md for external AI review
""")
