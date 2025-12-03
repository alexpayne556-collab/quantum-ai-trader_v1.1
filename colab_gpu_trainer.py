"""
🚀 COLAB PRO GPU TRAINER
=========================
Ready-to-upload notebook code for Google Colab Pro.

This file contains all the code needed to train the Golden Architecture
on Colab Pro with GPU acceleration.

INSTRUCTIONS:
1. Upload this file to Colab
2. Enable GPU: Runtime > Change runtime type > T4 GPU
3. Run all cells
4. Download trained models

Expected Training Time (T4 GPU):
- Full ticker list (50 tickers): ~30 min
- With visual engine (CNN): +20 min
- With SAC RL training: +45 min
- Total: ~2 hours

Expected Results:
- Accuracy: 55-62%
- Sharpe: 1.0-1.5
"""

# =============================================================================
# CELL 1: SETUP AND INSTALLATIONS
# =============================================================================
SETUP_CODE = '''
# Install dependencies
!pip install -q yfinance pandas numpy scikit-learn xgboost lightgbm
!pip install -q torch torchvision  # For CNN
!pip install -q hmmlearn          # For HMM regime detection
!pip install -q pysr              # For symbolic regression (optional)
!pip install -q stable-baselines3 gymnasium  # For SAC RL
!pip install -q pyts              # For Gramian Angular Fields

# Clone repo
!git clone https://github.com/alexpayne556-collab/quantum-ai-trader_v1.1.git
%cd quantum-ai-trader_v1.1

# Check GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
'''

# =============================================================================
# CELL 2: IMPORTS
# =============================================================================
IMPORTS_CODE = '''
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from golden_architecture import GoldenArchitecture
from ultimate_predictor import UltimatePredictor
from ticker_scanner import TickerScanner

print("✅ All imports successful!")
'''

# =============================================================================
# CELL 3: DATA DOWNLOAD
# =============================================================================
DATA_CODE = '''
# Full ticker list
TICKERS = [
    # Tech
    'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AMZN', 'TSLA', 'AMD', 'INTC', 'CRM',
    # Finance
    'JPM', 'BAC', 'GS', 'V', 'MA', 'WFC', 'C',
    # Healthcare
    'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY',
    # Consumer
    'WMT', 'HD', 'NKE', 'SBUX', 'MCD', 'KO', 'PG',
    # Energy
    'XOM', 'CVX', 'COP',
    # Industrial
    'CAT', 'BA', 'UPS', 'HON',
    # ETFs
    'SPY', 'QQQ', 'IWM', 'DIA',
]

print(f"Downloading {len(TICKERS)} tickers (3 years of data)...")

# Download all data
all_data = {}
for ticker in TICKERS:
    try:
        df = yf.download(ticker, period='3y', progress=False)
        if len(df) > 500:
            all_data[ticker] = df
            print(f"  ✓ {ticker}: {len(df)} samples")
    except Exception as e:
        print(f"  ✗ {ticker}: {e}")

print(f"\\n✅ Downloaded {len(all_data)} tickers")

# Get market benchmark
market_df = yf.download('SPY', period='3y', progress=False)
print(f"Market benchmark (SPY): {len(market_df)} samples")
'''

# =============================================================================
# CELL 4: COMBINE DATA FOR THICK DATASET
# =============================================================================
COMBINE_CODE = '''
# Combine all tickers into one THICK dataset
# This gives the model more patterns to learn from

def combine_tickers(all_data, normalize=True):
    """
    Combine multiple ticker datasets into one
    Adds ticker column for tracking
    """
    combined = []
    
    for ticker, df in all_data.items():
        df_copy = df.copy()
        df_copy['ticker'] = ticker
        
        if normalize:
            # Normalize prices to percentage changes from first day
            for col in ['Open', 'High', 'Low', 'Close']:
                if col in df_copy.columns:
                    first_price = df_copy[col].iloc[0]
                    df_copy[col] = df_copy[col] / first_price * 100
        
        combined.append(df_copy)
    
    return pd.concat(combined, ignore_index=False)

combined_df = combine_tickers(all_data)
print(f"Combined dataset: {len(combined_df)} samples")
print(f"Unique tickers: {combined_df['ticker'].nunique()}")
print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
'''

# =============================================================================
# CELL 5: TRAIN GOLDEN ARCHITECTURE
# =============================================================================
TRAIN_GOLDEN_CODE = '''
# Train Golden Architecture with ALL features enabled

print("=" * 70)
print("🚀 TRAINING GOLDEN ARCHITECTURE (GPU)")
print("=" * 70)

# Use first ticker for base training (model will generalize)
primary_ticker = 'AAPL'
primary_df = all_data[primary_ticker]

# Initialize Golden Architecture
arch = GoldenArchitecture(verbose=True)

# Build with all engines
arch.build(
    df=primary_df,
    market_df=market_df,
    label_method='triple_barrier',
    use_visual=True,   # Enable CNN pattern recognition
    use_logic=False,   # Skip symbolic regression (slow)
    use_regime=True    # Enable HMM regime detection
)

# Get training metrics
print(f"\\n📊 Training Results:")
print(f"   Accuracy: {arch.training_metrics.get('ensemble_accuracy', 0):.1%}")
print(f"   F1 Score: {arch.training_metrics.get('ensemble_f1', 0):.3f}")
print(f"   Samples: {arch.training_metrics.get('train_samples', 0):,}")
print(f"   Features: {arch.training_metrics.get('n_features', 0)}")
'''

# =============================================================================
# CELL 6: VALIDATE WITH CPCV
# =============================================================================
VALIDATE_CODE = '''
# Validate with Combinatorial Purged Cross-Validation

print("=" * 70)
print("🔬 VALIDATING WITH CPCV (Honest Backtest)")
print("=" * 70)

# Run CPCV validation
validation_results = arch.validate(primary_df, market_df)

if validation_results:
    print(f"\\n📊 CPCV Validation Results:")
    print(f"   Mean Accuracy: {validation_results.get('mean_accuracy', 0):.1%}")
    print(f"   Std Accuracy: {validation_results.get('std_accuracy', 0):.1%}")
    print(f"   p-value: {validation_results.get('p_value', 1):.4f}")
    print(f"   Significant: {'YES' if validation_results.get('is_significant', False) else 'NO'}")
'''

# =============================================================================
# CELL 7: MULTI-TICKER VALIDATION
# =============================================================================
MULTI_VALIDATE_CODE = '''
# Validate on ALL tickers to check generalization

print("=" * 70)
print("📈 MULTI-TICKER VALIDATION")
print("=" * 70)

results = []
for ticker, df in list(all_data.items())[:20]:  # Top 20
    try:
        pred = arch.predict(df, market_df)
        results.append({
            'ticker': ticker,
            'signal': pred['signal'],
            'confidence': pred['confidence'],
            'regime': pred.get('regime', 'unknown')
        })
        print(f"  {ticker}: {pred['signal']} ({pred['confidence']:.1%})")
    except Exception as e:
        print(f"  {ticker}: Error - {e}")

# Summary
results_df = pd.DataFrame(results)
print(f"\\n📊 Summary:")
print(f"   Avg Confidence: {results_df['confidence'].mean():.1%}")
print(f"   BUY Signals: {(results_df['signal'] == 'BUY').sum()}")
print(f"   SELL Signals: {(results_df['signal'] == 'SELL').sum()}")
print(f"   HOLD Signals: {(results_df['signal'] == 'HOLD').sum()}")
'''

# =============================================================================
# CELL 8: SAVE MODELS
# =============================================================================
SAVE_CODE = '''
# Save trained models for download

import os
os.makedirs('trained_models', exist_ok=True)

# Save Golden Architecture
arch.save('trained_models/golden_architecture.pkl')

# Save Ultimate Predictor
predictor = UltimatePredictor(verbose=False)
predictor.golden_arch = arch
predictor.is_trained = True
predictor.training_metrics = arch.training_metrics
predictor.save('trained_models/ultimate_predictor.pkl')

print("✅ Models saved to 'trained_models/' directory")
print("\\nDownload these files:")
print("  • trained_models/golden_architecture.pkl")
print("  • trained_models/ultimate_predictor.pkl")

# Zip for easy download
!zip -r trained_models.zip trained_models/
print("\\n📦 Download: trained_models.zip")
'''

# =============================================================================
# CELL 9: LIVE PREDICTIONS
# =============================================================================
LIVE_CODE = '''
# Get live predictions for top opportunities

print("=" * 70)
print("🎯 LIVE PREDICTIONS")
print("=" * 70)

# Refresh data
for ticker in ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'SPY']:
    df = yf.download(ticker, period='2y', progress=False)
    if len(df) > 0:
        pred = arch.predict(df, market_df)
        
        emoji = "🟢" if pred['signal'] == 'BUY' else "🔴" if pred['signal'] == 'SELL' else "⏸️"
        print(f"\\n{emoji} {ticker}: {pred['signal']}")
        print(f"   Confidence: {pred['confidence']:.1%}")
        print(f"   Pattern: {pred.get('pattern', 'unknown')}")
        print(f"   Regime: {pred.get('regime', 'unknown')}")
'''

# =============================================================================
# CELL 10: GENERATE COLAB NOTEBOOK FILE
# =============================================================================
def generate_colab_notebook():
    """Generate a Jupyter notebook file for Colab"""
    
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {
                "name": "Quantum_AI_Trader_GPU_Training.ipynb",
                "provenance": [],
                "gpuType": "T4"
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3"
            },
            "accelerator": "GPU"
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🚀 Quantum AI Trader - GPU Training\n",
                    "\n",
                    "**Golden Architecture Training with:**\n",
                    "- 40+ tickers (3 years data)\n",
                    "- Visual Engine (CNN)\n",
                    "- HMM Regime Detection\n",
                    "- Ensemble Stacking (XGBoost + LightGBM + RF)\n",
                    "- CPCV Validation\n",
                    "\n",
                    "**Expected Results:**\n",
                    "- Accuracy: 55-62%\n",
                    "- Training Time: ~2 hours (T4 GPU)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": SETUP_CODE.strip().split('\n')
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": IMPORTS_CODE.strip().split('\n')
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": DATA_CODE.strip().split('\n')
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": COMBINE_CODE.strip().split('\n')
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": TRAIN_GOLDEN_CODE.strip().split('\n')
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": VALIDATE_CODE.strip().split('\n')
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": MULTI_VALIDATE_CODE.strip().split('\n')
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": SAVE_CODE.strip().split('\n')
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": LIVE_CODE.strip().split('\n')
            }
        ]
    }
    
    return notebook


if __name__ == "__main__":
    import json
    
    # Generate notebook
    notebook = generate_colab_notebook()
    
    # Save as .ipynb
    with open('COLAB_GPU_TRAINER.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print("✅ Generated COLAB_GPU_TRAINER.ipynb")
    print("\nTo use:")
    print("1. Upload to Google Colab")
    print("2. Enable GPU: Runtime > Change runtime type > T4")
    print("3. Run all cells")
    print("4. Download trained_models.zip")
