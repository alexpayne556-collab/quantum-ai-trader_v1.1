#!/usr/bin/env python3
"""
GPU Training Quick Start
Run this on Shadow PC with CUDA GPU
"""
import os
import pandas as pd
import json

# Load config
with open('training_config.json', 'r') as f:
    config = json.load(f)

# Load top tickers
with open('elite_20.txt', 'r') as f:
    elite_20 = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(elite_20)} elite tickers")
print(f"Validated edges: {list(config['validated_edges'].keys())}")
print("\nReady for GPU training!")
print("Run: python train_gpu.py")
