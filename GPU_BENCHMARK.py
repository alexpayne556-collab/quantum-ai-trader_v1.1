"""
GPU BENCHMARK - Test your RTX speed
"""
import torch
import time
import numpy as np

def benchmark_gpu():
    print("="*60)
    print("GPU BENCHMARK")
    print("="*60)
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("NO GPU - Running on CPU only")
        return
    
    device = torch.device('cuda')
    
    # Test 1: Matrix multiplication (core ML operation)
    print("\n[TEST 1] Matrix Multiplication (10000x10000)")
    sizes = [1000, 5000, 10000]
    
    for size in sizes:
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        # Warmup
        torch.cuda.synchronize()
        c = torch.mm(a, b)
        torch.cuda.synchronize()
        
        # Timed run
        start = time.time()
        for _ in range(10):
            c = torch.mm(a, b)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / 10
        
        gflops = (2 * size**3) / (elapsed * 1e9)
        print(f"  {size}x{size}: {elapsed*1000:.2f}ms ({gflops:.0f} GFLOPS)")
    
    # Test 2: Convolution (deep learning)
    print("\n[TEST 2] 2D Convolution (batch=64, channels=256)")
    x = torch.randn(64, 256, 56, 56, device=device)
    conv = torch.nn.Conv2d(256, 256, 3, padding=1).to(device)
    
    # Warmup
    torch.cuda.synchronize()
    y = conv(x)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(50):
        y = conv(x)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / 50
    print(f"  Time: {elapsed*1000:.2f}ms per forward pass")
    
    # Test 3: Memory bandwidth
    print("\n[TEST 3] Memory Bandwidth")
    size = 100_000_000  # 100M floats = 400MB
    a = torch.randn(size, device=device)
    b = torch.randn(size, device=device)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        c = a + b
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / 100
    
    bandwidth = (3 * size * 4) / (elapsed * 1e9)  # 3 arrays * 4 bytes
    print(f"  Bandwidth: {bandwidth:.0f} GB/s")
    
    # Test 4: Trading simulation speed
    print("\n[TEST 4] Trading Simulation (1M price bars)")
    prices = torch.randn(1_000_000, device=device)
    
    start = time.time()
    for _ in range(100):
        # Calculate returns
        returns = prices[1:] / prices[:-1] - 1
        # Rolling mean (20-day)
        kernel = torch.ones(20, device=device) / 20
        ma = torch.nn.functional.conv1d(
            prices.view(1, 1, -1), 
            kernel.view(1, 1, -1), 
            padding=10
        )
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / 100
    
    bars_per_sec = 1_000_000 / elapsed
    print(f"  Speed: {bars_per_sec/1e6:.1f}M bars/second")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"GPU: {gpu_name}")
    print(f"Matrix mult (10k): {gflops:.0f} GFLOPS")
    print(f"Memory bandwidth: {bandwidth:.0f} GB/s")
    print(f"Trading sim: {bars_per_sec/1e6:.1f}M bars/sec")
    print("="*60)

if __name__ == "__main__":
    benchmark_gpu()
