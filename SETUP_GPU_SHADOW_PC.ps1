# GPU Setup Script for Shadow PC
# Run this in PowerShell on Shadow PC to set up RAPIDS GPU acceleration
# ============================================================================

Write-Host "🚀 GPU Setup for Quantum AI Trader - Shadow PC" -ForegroundColor Cyan
Write-Host "============================================`n" -ForegroundColor Cyan

# Step 1: Check if conda is installed
Write-Host "Step 1: Checking for Miniconda/Anaconda..." -ForegroundColor Yellow
$condaPath = Get-Command conda -ErrorAction SilentlyContinue

if (-not $condaPath) {
    Write-Host "❌ Conda not found. Installing Miniconda..." -ForegroundColor Red
    
    $minicondaUrl = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
    $installerPath = "$env:TEMP\miniconda-installer.exe"
    
    Write-Host "Downloading Miniconda..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $minicondaUrl -OutFile $installerPath
    
    Write-Host "Installing Miniconda (this may take a few minutes)..." -ForegroundColor Yellow
    Start-Process -FilePath $installerPath -ArgumentList "/S /AddToPath=1 /RegisterPython=0" -Wait
    
    Write-Host "✅ Miniconda installed. Please RESTART PowerShell and run this script again." -ForegroundColor Green
    Write-Host "   (Conda needs to be added to PATH)" -ForegroundColor Yellow
    exit
} else {
    Write-Host "✅ Conda found at: $($condaPath.Source)`n" -ForegroundColor Green
}

# Step 2: Check CUDA/GPU availability
Write-Host "Step 2: Checking GPU/CUDA..." -ForegroundColor Yellow
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue

if ($nvidiaSmi) {
    Write-Host "✅ NVIDIA GPU detected:" -ForegroundColor Green
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    Write-Host ""
} else {
    Write-Host "⚠️  WARNING: nvidia-smi not found. GPU acceleration may not work." -ForegroundColor Red
    Write-Host "   Make sure NVIDIA drivers are installed on Shadow PC." -ForegroundColor Yellow
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne "y") { exit }
}

# Step 3: Create conda environment with RAPIDS
Write-Host "Step 3: Creating conda environment 'quant-gpu'..." -ForegroundColor Yellow
Write-Host "   This includes: CuPy, cuDF, cuML, scikit-cuda" -ForegroundColor Cyan
Write-Host "   (This will take 5-10 minutes)...`n" -ForegroundColor Cyan

# Check if environment already exists
$envExists = conda env list | Select-String "quant-gpu"

if ($envExists) {
    Write-Host "⚠️  Environment 'quant-gpu' already exists." -ForegroundColor Yellow
    $recreate = Read-Host "Delete and recreate? (y/n)"
    if ($recreate -eq "y") {
        conda env remove -n quant-gpu -y
    } else {
        Write-Host "Using existing environment..." -ForegroundColor Cyan
        conda activate quant-gpu
        Write-Host "✅ Environment activated`n" -ForegroundColor Green
        exit
    }
}

# Create environment (Python 3.10 for compatibility with RAPIDS)
conda create -n quant-gpu python=3.10 -y

# Activate environment
conda activate quant-gpu

# Step 4: Install RAPIDS (CuPy, cuDF, cuML)
Write-Host "`nStep 4: Installing RAPIDS libraries..." -ForegroundColor Yellow
Write-Host "   This is the BIG install - may take 10-15 minutes" -ForegroundColor Cyan

# Install RAPIDS using conda (CUDA 12 version)
conda install -c rapidsai -c conda-forge -c nvidia `
    cudf=24.12 cuml=24.12 cugraph=24.12 cuspatial=24.12 `
    cupy cudatoolkit=12.0 -y

# Step 5: Install additional Python packages
Write-Host "`nStep 5: Installing additional packages..." -ForegroundColor Yellow
conda install -c conda-forge pandas numpy scipy matplotlib seaborn yfinance scikit-learn -y

# Pip packages
pip install --upgrade pip
pip install ta-lib alpaca-trade-api hmmlearn

Write-Host "`n✅ GPU environment setup complete!`n" -ForegroundColor Green

# Step 6: Test GPU
Write-Host "Step 6: Testing GPU acceleration..." -ForegroundColor Yellow

$testScript = @"
import cupy as cp
import cudf
import pandas as pd
import numpy as np
from datetime import datetime

print('🔍 GPU Test Starting...\n')

# Test 1: CuPy (NumPy on GPU)
print('Test 1: CuPy Matrix Multiplication')
size = 5000
cpu_a = np.random.rand(size, size).astype(np.float32)
cpu_b = np.random.rand(size, size).astype(np.float32)

# CPU timing
start = datetime.now()
cpu_result = np.dot(cpu_a, cpu_b)
cpu_time = (datetime.now() - start).total_seconds()
print(f'   CPU: {cpu_time:.2f}s')

# GPU timing
gpu_a = cp.array(cpu_a)
gpu_b = cp.array(cpu_b)
start = datetime.now()
gpu_result = cp.dot(gpu_a, gpu_b)
cp.cuda.Stream.null.synchronize()  # Wait for GPU to finish
gpu_time = (datetime.now() - start).total_seconds()
print(f'   GPU: {gpu_time:.2f}s')
print(f'   ⚡ Speedup: {cpu_time/gpu_time:.1f}x\n')

# Test 2: cuDF (Pandas on GPU)
print('Test 2: cuDF DataFrame Operations')
n = 10_000_000
cpu_df = pd.DataFrame({
    'A': np.random.rand(n),
    'B': np.random.rand(n),
    'C': np.random.rand(n)
})

start = datetime.now()
cpu_result = cpu_df.groupby(pd.cut(cpu_df['A'], 100))['B'].mean()
cpu_time = (datetime.now() - start).total_seconds()
print(f'   CPU: {cpu_time:.2f}s')

gpu_df = cudf.DataFrame.from_pandas(cpu_df)
start = datetime.now()
gpu_result = gpu_df.groupby(cudf.cut(gpu_df['A'], 100))['B'].mean()
gpu_time = (datetime.now() - start).total_seconds()
print(f'   GPU: {gpu_time:.2f}s')
print(f'   ⚡ Speedup: {cpu_time/gpu_time:.1f}x\n')

print('✅ GPU acceleration is working!')
print(f'   GPU Memory: {cp.cuda.Device().mem_info[1] / 1e9:.1f} GB total')
print(f'   GPU Free: {cp.cuda.Device().mem_info[0] / 1e9:.1f} GB')
"@

$testScript | python

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "🎯 SETUP COMPLETE!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "`nTo use GPU environment:" -ForegroundColor Yellow
Write-Host "   conda activate quant-gpu" -ForegroundColor Cyan
Write-Host "`nTo deactivate:" -ForegroundColor Yellow
Write-Host "   conda deactivate" -ForegroundColor Cyan
Write-Host ""
