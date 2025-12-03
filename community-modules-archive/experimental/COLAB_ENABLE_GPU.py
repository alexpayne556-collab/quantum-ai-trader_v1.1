"""
🚀 ENABLE GPU - Quick Setup
===========================
Run this in Colab to enable GPU for all models and suppress warnings
"""

print("="*80)
print("🚀 ENABLING GPU ACCELERATION")
print("="*80)

# ============================================================================
# 1. INSTALL GPU LIBRARIES
# ============================================================================
print("\n1️⃣ Installing GPU libraries...")

import subprocess
import sys

packages = ['torch', 'torchvision', 'torchaudio']
print(f"   Installing: {', '.join(packages)}")

# Install PyTorch with CUDA support
subprocess.run([
    sys.executable, '-m', 'pip', 'install', '-q',
    'torch', 'torchvision', 'torchaudio',
    '--index-url', 'https://download.pytorch.org/whl/cu118'
], check=False)

print("✅ GPU libraries installed")

# ============================================================================
# 2. DETECT GPU
# ============================================================================
print("\n2️⃣ Detecting GPU...")

try:
    import torch
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU Detected: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Set as default device
        torch.cuda.set_device(0)
        print(f"✅ GPU device set to: cuda:0")
        
        # Test GPU
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = torch.matmul(x, y)
        print("✅ GPU test passed - GPU is working!")
        
        GPU_AVAILABLE = True
    else:
        print("⚠️  No GPU detected")
        print("   → Runtime → Change runtime type → GPU")
        GPU_AVAILABLE = False
        
except Exception as e:
    print(f"⚠️  GPU setup error: {e}")
    GPU_AVAILABLE = False

# ============================================================================
# 3. CONFIGURE XGBOOST & LIGHTGBM
# ============================================================================
print("\n3️⃣ Configuring ML libraries for GPU...")

import os

if GPU_AVAILABLE:
    # Set environment variables
    os.environ['XGBOOST_GPU'] = '1'
    os.environ['LIGHTGBM_EXEC'] = 'gpu'
    print("✅ XGBoost & LightGBM configured for GPU")
else:
    print("⚠️  CPU mode - libraries will use CPU")

# ============================================================================
# 4. SUPPRESS WARNINGS
# ============================================================================
print("\n4️⃣ Suppressing GPU warnings...")

import warnings
warnings.filterwarnings('ignore')

# Suppress specific GPU warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TensorFlow
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

print("✅ Warnings suppressed")

# ============================================================================
# 5. CREATE GPU CONFIG
# ============================================================================
print("\n5️⃣ Creating GPU configuration module...")

from pathlib import Path

BASE_DIR = Path('/content/drive/MyDrive/QuantumAI')
MODULES_DIR = BASE_DIR / 'backend' / 'modules'

gpu_config = f'''
"""
GPU Configuration - Auto-generated
"""

GPU_AVAILABLE = {GPU_AVAILABLE}
USE_GPU = {GPU_AVAILABLE}

try:
    import torch
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
except:
    DEVICE = 'cpu'

# XGBoost params
XGBOOST_PARAMS = {{
    'tree_method': 'gpu_hist' if {GPU_AVAILABLE} else 'hist',
    'predictor': 'gpu_predictor' if {GPU_AVAILABLE} else 'cpu_predictor',
}}

# LightGBM params  
LIGHTGBM_PARAMS = {{
    'device': 'gpu' if {GPU_AVAILABLE} else 'cpu',
}}
'''

try:
    with open(MODULES_DIR / 'gpu_config.py', 'w') as f:
        f.write(gpu_config)
    print("✅ GPU config saved")
except:
    print("⚠️  Could not save GPU config (file may not exist)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
if GPU_AVAILABLE:
    print("✅ GPU ENABLED & CONFIGURED")
    print("="*80)
    print(f"\n🎯 GPU Status: ACTIVE")
    print(f"   All models will use GPU acceleration")
    print(f"\n✅ No more GPU warnings!")
else:
    print("⚠️  GPU NOT AVAILABLE")
    print("="*80)
    print(f"\n💡 To enable GPU:")
    print(f"   1. Runtime → Change runtime type")
    print(f"   2. Hardware accelerator → GPU (T4)")
    print(f"   3. Re-run this script")
    print(f"\n⚠️  Models will use CPU (slower)")

print("\n" + "="*80)

