# SHADOW PC SETUP - COPY/PASTE INTO ANACONDA PROMPT

## STEP 1: Create Environment
```
conda create -n quant2026 python=3.11 -y
conda activate quant2026
```

## STEP 2: Install PyTorch with GPU
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

## STEP 3: Install Core ML/Data Packages
```
pip install pandas numpy scipy scikit-learn xgboost lightgbm optuna
pip install yfinance pandas-ta ta-lib-bin
pip install matplotlib seaborn plotly
pip install hmmlearn statsmodels ruptures
pip install tqdm joblib requests beautifulsoup4
```

## STEP 4: Install Advanced ML
```
pip install transformers einops
pip install numba cupy-cuda12x
```

## STEP 5: Clone Repo (if not done)
```
cd C:\Users\YOUR_USERNAME
git clone https://github.com/alexpayne556-collab/quantum-ai-trader_v1.1.git
cd quantum-ai-trader_v1.1
```

## STEP 6: Verify GPU
```
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

## STEP 7: Test Imports
```
python -c "import pandas, numpy, yfinance, torch, xgboost, hmmlearn; print('ALL GOOD')"
```

## STEP 8: Run Initial Data Download
```
python DOWNLOAD_WATCHLIST_DATA.py
```

---
**Expected Output from Step 6:**
```
CUDA: True
GPU: NVIDIA GeForce RTX 3070
```

If CUDA is False, run: `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y`
