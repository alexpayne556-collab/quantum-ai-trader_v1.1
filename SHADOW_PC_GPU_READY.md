# SHADOW PC GPU SETUP - READY TO EXPAND TO 10K STRATEGIES

**Date:** December 22, 2025, 10:30 PM  
**Mission:** Use Shadow PC RTX 3070 to test 3,000+ new strategies  
**Current:** 6,859 strategies → Target: 10,000+  

---

## STEP 1: SHADOW PC - GIT PULL & SETUP

```powershell
# Open Anaconda Prompt on Shadow PC

# Navigate to project
cd C:\Users\YOUR_USERNAME\quantum-ai-trader_v1.1

# Pull latest code
git pull origin main

# Create/activate conda environment with GPU support
conda create -n quant_gpu python=3.12 -y
conda activate quant_gpu

# Install GPU-enabled packages
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install xgboost lightgbm pandas numpy scipy numba
pip install ta-lib-bin yfinance pandas-ta

# Verify GPU
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

**Expected output:**
```
CUDA Available: True
GPU: NVIDIA GeForce RTX 3070
```

---

## STEP 2: WHAT WE'RE TESTING TONIGHT

### BATCH 1: Advanced Technical (1,000 strategies)
- Ichimoku Cloud (162 strategies)
- Fibonacci retracements (90 strategies)
- Chart patterns (90 strategies)
- Advanced momentum (200 strategies)
- Multi-timeframe confluence (200 strategies)
- Volatility regime shifts (200 strategies)

### BATCH 2: Machine Learning Features (1,000 strategies)
- XGBoost on top 50 indicators
- LightGBM ensemble predictions
- Neural network price patterns
- Feature importance analysis
- Regime-specific models

### BATCH 3: 4-Factor & 5-Factor Fusions (1,000 strategies)
- FUSION_4F: 400 combinations
- FUSION_5F: 200 combinations
- Sector + Momentum + Value: 200 combinations
- Volume + Volatility + Trend: 200 combinations

**Total: 3,000 new strategies = 9,859 total**

---

## STEP 3: RUN THE GPU EXPANSION

```powershell
# On Shadow PC, run:
python SHADOW_GPU_EXPANSION_PART1.py

# This will:
# 1. Load data/market_data.db (496 MB)
# 2. Test 1,000 advanced technical strategies
# 3. Use GPU for parallel calculations
# 4. Save results to data/GPU_EXPANSION_PART1.csv
# 5. Estimated time: 2-3 hours (GPU accelerated)
```

---

## STEP 4: PUSH RESULTS BACK

```powershell
# After script completes
git add data/GPU_EXPANSION_PART1.csv
git commit -m "Shadow PC GPU: Part 1 - 1000 advanced technical strategies"
git push origin main
```

---

## STEP 5: CODESPACES - PULL & CONSOLIDATE

```bash
# Back in Codespaces
git pull
python consolidate_results.py
```

---

## GPU ACCELERATION EXPLAINED

**Why GPU helps:**
- **Parallel calculations:** Test 1,000 stocks simultaneously
- **Matrix operations:** NumPy/Pandas operations can use CUDA
- **XGBoost/LightGBM:** GPU tree building is 10-50x faster
- **PyTorch:** Neural network training on price sequences

**What gets faster:**
- Rolling calculations (RSI, MACD, etc.) across 9,501 tickers
- Correlation matrices
- Machine learning training
- Monte Carlo simulations

**Estimated speedups:**
- CPU (Codespaces): 6,859 strategies in ~8 hours
- GPU (Shadow PC): 3,000 strategies in ~2-3 hours

---

## TONIGHT'S SCRIPT: SHADOW_GPU_EXPANSION_PART1.py

This file will be created once you're ready. It will contain:

1. **Ichimoku Cloud Detection:**
   - Tenkan-sen (9-period conversion line)
   - Kijun-sen (26-period base line)
   - Senkou Span A & B (cloud)
   - Chikou Span (lagging span)
   - 18 signals × 9 hold periods = 162 strategies

2. **Fibonacci Retracements:**
   - Detect swing highs/lows
   - Calculate 23.6%, 38.2%, 50%, 61.8%, 78.6% levels
   - Test bounce vs breakdown at each level
   - 10 setups × 9 hold periods = 90 strategies

3. **Advanced Momentum:**
   - Multi-timeframe RSI alignment
   - MACD histogram expansion/contraction
   - Rate of change acceleration
   - Momentum persistence
   - 200 strategies

4. **GPU-Optimized Calculations:**
   - Uses PyTorch tensors for parallel processing
   - Batch processing 100 tickers at a time
   - CUDA memory management
   - Progress bar with ETA

---

## SYSTEM SPECS

**Shadow PC (Your GPU Machine):**
- GPU: NVIDIA GeForce RTX 3070
- VRAM: 8 GB
- CUDA Cores: 5,888
- Tensor Cores: 184
- Perfect for: XGBoost, LightGBM, PyTorch training

**Codespaces (Current Location):**
- CPU: Intel Xeon (2-4 cores)
- RAM: 8-16 GB
- No GPU
- Good for: Analysis, visualization, consolidation

---

## WHAT HAPPENS AFTER TONIGHT

**Tomorrow (Dec 23):**
- Review 1,000 new strategies from Part 1
- Consolidate with existing 6,859
- Analyze which new indicators found significance

**Tuesday (Dec 24):**
- Run Part 2: Machine Learning features (1,000 strategies)
- XGBoost on top 50 technical indicators
- Feature importance analysis

**Wednesday (Dec 25):**
- Run Part 3: 4-Factor & 5-Factor fusions (1,000 strategies)
- Reach 10,000 total strategies tested

**After 10K:**
- Out-of-sample validation
- Walk-forward analysis
- Build the companion that applies these discoveries

---

## READY TO START

Once you open Anaconda Prompt on Shadow PC, paste:

```powershell
cd C:\Users\YOUR_USERNAME\quantum-ai-trader_v1.1
git pull
conda activate quant_gpu
python -c "import torch; print('GPU Ready:', torch.cuda.is_available())"
```

Then say "GPU ready" and I'll create the expansion script.

---

**LET'S EXPAND.**
