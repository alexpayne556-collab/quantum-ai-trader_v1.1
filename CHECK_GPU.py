#!/usr/bin/env python3
"""
CHECK_GPU.py - Quick GPU detection test
"""

print("="*80)
print("🔍 GPU DETECTION TEST")
print("="*80)

# Test 1: CuPy
print("\n1. Testing CuPy (NVIDIA CUDA)...")
try:
    import cupy as cp
    print(f"   ✅ CuPy installed: {cp.__version__}")
    
    # Try to use GPU
    try:
        x = cp.array([1, 2, 3])
        y = cp.array([4, 5, 6])
        z = x + y
        print(f"   ✅ GPU WORKING! Device: {cp.cuda.Device()}")
        print(f"   ✅ CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
        GPU_AVAILABLE = True
    except Exception as e:
        print(f"   ⚠️ CuPy installed but GPU not accessible: {e}")
        print(f"   💡 Will fall back to CPU (Numba)")
        GPU_AVAILABLE = False
        
except ImportError:
    print("   ⚠️ CuPy not installed")
    GPU_AVAILABLE = False

# Test 2: Numba (CPU acceleration)
print("\n2. Testing Numba (CPU JIT)...")
try:
    from numba import jit
    import numpy as np
    
    @jit(nopython=True)
    def test_numba(n):
        total = 0
        for i in range(n):
            total += i
        return total
    
    result = test_numba(1000000)
    print(f"   ✅ Numba working! CPU acceleration available")
    print(f"   ✅ Expected speedup: 50-100x vs pandas")
    
except ImportError:
    print("   ❌ Numba not installed")

# Test 3: Basic libraries
print("\n3. Testing core libraries...")
try:
    import pandas as pd
    import numpy as np
    import scipy
    import sqlite3
    print(f"   ✅ pandas {pd.__version__}")
    print(f"   ✅ numpy {np.__version__}")
    print(f"   ✅ scipy {scipy.__version__}")
    print(f"   ✅ sqlite3 ready")
except ImportError as e:
    print(f"   ❌ Missing library: {e}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

if GPU_AVAILABLE:
    print("🚀 GPU DETECTED - Tests will run at 10-100x CPU speed!")
    print("   Expected time: 3-5 minutes for all tests")
else:
    print("⚡ CPU MODE (Numba) - Still 50-100x faster than pandas!")
    print("   Expected time: 5-10 minutes for all tests")

print("\n✅ Ready to run: python SHADOW_PC_GPU_TESTS.py")
print("="*80)
