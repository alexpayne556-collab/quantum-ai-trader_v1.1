"""
Quick smoke test for dataset generators.
Validates that all datasets can be generated and have expected properties.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.datasets.pure_noise import get_pure_noise_dataset
from src.datasets.known_structure import get_known_structure_dataset
from src.datasets.finance_null import get_finance_null_dataset


def test_dataset_generation():
    """Test that all datasets can be generated without errors."""
    
    print("=" * 80)
    print("DATASET GENERATOR SMOKE TEST")
    print("=" * 80)
    
    seed = 42
    config = {'length': 500}
    
    # Test Pure Noise datasets
    print("\n📊 Testing Pure Noise Datasets...")
    for name in ['white_noise', 'random_walk', 'iid_uniform']:
        ds = get_pure_noise_dataset(name, seed, config)
        data = ds.generate()
        structure = ds.get_true_structure()
        
        assert len(data) == 500, f"{name}: Wrong length"
        assert not structure['has_structure'], f"{name}: Should have no structure"
        assert np.isfinite(data).all(), f"{name}: Contains inf/nan"
        
        print(f"  ✅ {name}: Generated {len(data)} points")
    
    # Test Known-Structure datasets
    print("\n🎯 Testing Known-Structure Datasets...")
    for name in ['ar1', 'garch', 'regime_shift', 'ma1']:
        if name == 'ar1':
            config['phi'] = 0.7
        elif name == 'garch':
            config.update({'omega': 0.1, 'alpha': 0.1, 'beta': 0.85})
        elif name == 'regime_shift':
            config['breakpoint'] = 250
        elif name == 'ma1':
            config['theta'] = 0.5
        
        ds = get_known_structure_dataset(name, seed, config)
        data = ds.generate()
        structure = ds.get_true_structure()
        
        assert len(data) == 500, f"{name}: Wrong length"
        assert structure['has_structure'], f"{name}: Should have structure"
        assert np.isfinite(data).all(), f"{name}: Contains inf/nan"
        
        print(f"  ✅ {name}: Generated {len(data)} points, type={structure['type']}")
    
    # Test Finance-Null datasets
    print("\n💹 Testing Finance-Null Datasets...")
    for name in ['garch_jumps', 'stochastic_vol', 'levy_jumps']:
        if name == 'garch_jumps':
            config.update({'omega': 0.1, 'alpha': 0.1, 'beta': 0.85, 'dof': 5})
        elif name == 'stochastic_vol':
            config.update({'mu': 0.0, 'phi': 0.95, 'sigma_eta': 0.2})
        elif name == 'levy_jumps':
            config.update({'lambda_jumps': 0.05, 'diffusion_std': 1.0})
        
        ds = get_finance_null_dataset(name, seed, config)
        data = ds.generate()
        structure = ds.get_true_structure()
        
        assert len(data) == 500, f"{name}: Wrong length"
        assert not structure['has_exploitable_structure'], f"{name}: Should have no exploitable structure"
        assert structure['has_finance_texture'], f"{name}: Should have finance texture"
        assert np.isfinite(data).all(), f"{name}: Contains inf/nan"
        
        print(f"  ✅ {name}: Generated {len(data)} points, texture={structure['has_finance_texture']}")
    
    print("\n" + "=" * 80)
    print("✅ ALL DATASETS PASSED SMOKE TEST")
    print("=" * 80)


def test_reproducibility():
    """Test that same seed produces same data."""
    
    print("\n🔄 Testing Reproducibility...")
    
    seed = 42
    config = {'length': 100, 'phi': 0.7}
    
    # Generate twice with same seed
    ds1 = get_known_structure_dataset('ar1', seed, config)
    data1 = ds1.generate()
    
    ds2 = get_known_structure_dataset('ar1', seed, config)
    data2 = ds2.generate()
    
    assert np.allclose(data1, data2), "Same seed should produce identical data"
    print("  ✅ Reproducibility: Same seed → identical data")
    
    # Generate with different seed
    ds3 = get_known_structure_dataset('ar1', seed + 1, config)
    data3 = ds3.generate()
    
    assert not np.allclose(data1, data3), "Different seed should produce different data"
    print("  ✅ Reproducibility: Different seed → different data")


def test_stationarity():
    """Quick check for stationarity in AR(1)."""
    
    print("\n📈 Testing AR(1) Stationarity...")
    
    seed = 42
    config = {'length': 5000, 'phi': 0.7}
    
    ds = get_known_structure_dataset('ar1', seed, config)
    data = ds.generate()
    
    # Check mean is roughly 0
    mean = np.mean(data)
    assert abs(mean) < 0.1, f"AR(1) mean should be ~0, got {mean:.3f}"
    
    # Check variance is roughly stationary
    first_half_var = np.var(data[:2500])
    second_half_var = np.var(data[2500:])
    ratio = max(first_half_var, second_half_var) / min(first_half_var, second_half_var)
    assert ratio < 1.5, f"Variance ratio too large: {ratio:.2f}"
    
    print(f"  ✅ AR(1) stationarity: mean={mean:.3f}, var_ratio={ratio:.2f}")


if __name__ == "__main__":
    try:
        test_dataset_generation()
        test_reproducibility()
        test_stationarity()
        
        print("\n🎉 ALL TESTS PASSED - Dataset generators are working correctly!")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
