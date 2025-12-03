"""
🚀 COLAB PRO QUICK START GUIDE
Step-by-step instructions to train your swing trading models in Google Colab Pro.

REQUIREMENTS:
- Google Colab Pro (for faster GPU training)
- Google Drive (to save trained models)
"""

# ============================================================================
# STEP 1: SETUP COLAB NOTEBOOK
# ============================================================================

# Create new notebook in Colab: File > New Notebook
# Enable GPU: Runtime > Change runtime type > T4 GPU > Save

# ============================================================================
# STEP 2: MOUNT GOOGLE DRIVE (for persistent storage)
# ============================================================================

from google.colab import drive
drive.mount('/content/drive')

# ============================================================================
# STEP 3: INSTALL DEPENDENCIES
# ============================================================================

!pip install -q yfinance talib-binary xgboost scikit-learn pandas numpy joblib

# ============================================================================
# STEP 4: UPLOAD CORE FILES
# ============================================================================

# Create directory structure
!mkdir -p /content/quantum-trader/core
!mkdir -p /content/quantum-trader/training
!mkdir -p /content/drive/MyDrive/quantum-trader/models
!mkdir -p /content/drive/MyDrive/quantum-trader/data

# Upload these files to /content/quantum-trader/:
# - colab_pro_trainer.py
# - core/institutional_feature_engineer.py
# - core/pattern_stats_engine.py
# - core/quantile_forecaster.py
# - core/confluence_engine.py
# - training/training_logger.py

# Or clone from GitHub (if you've pushed):
# !git clone https://github.com/YOUR_USERNAME/quantum-ai-trader.git /content/quantum-trader

# ============================================================================
# STEP 5: CONFIGURE YOUR TRAINING
# ============================================================================

import sys
sys.path.insert(0, '/content/quantum-trader')

from colab_pro_trainer import ColabProTrainer

# Configuration for swing trading
TICKERS = [
    # Large cap (safer, more liquid)
    'SPY', 'QQQ', 'IWM',  # Indexes
    'AAPL', 'MSFT', 'GOOGL', 'AMZN',  # Tech giants
    
    # Growth stocks (higher volatility, better for swings)
    'NVDA', 'AMD', 'TSLA', 'META',
    
    # Optional: Add your favorite swing trading stocks
    'B', 'FSM', 'APLD', 'HOOD', 'BABA', 'WSH', 'IONQ',
    # 'COIN', 'RIOT', 'MARA', 'PLTR', 'SOFI','MARVL','MU',

]

# Swing horizon options:
# '1bar' = 1-day swings (day trading style)
# '3bar' = 3-day swings (typical swing trading)
# '5bar' = 5-day swings (week-long positions)
# '10bar' = 10-day swings (2-week positions)
# '21bar' = 21-day swings (month-long positions)

SWING_HORIZON = '5bar'  # 5-day swings recommended for $1K-$5K capital

# ============================================================================
# STEP 6: INITIALIZE TRAINER
# ============================================================================

trainer = ColabProTrainer(
    tickers=TICKERS,
    swing_horizon=SWING_HORIZON,
    lookback_days=730,  # 2 years of data
    model_dir='/content/drive/MyDrive/quantum-trader/models',  # Saves to Google Drive
    use_gpu=True
)

# ============================================================================
# STEP 7: RUN TRAINING PIPELINE
# ============================================================================

# This will:
# 1. Download 2 years of data for all tickers
# 2. Engineer 50+ institutional features
# 3. Create swing labels (±2% thresholds)
# 4. Run walk-forward validation (5 folds, 10-day embargo)
# 5. Train final model with XGBoost/HistGradientBoosting
# 6. Save models to Google Drive

trainer.run_full_training_pipeline()

# Expected time: 5-10 minutes with GPU
# Expected accuracy: 65-70% overall, 70-75% directional

# ============================================================================
# STEP 8: REVIEW RESULTS
# ============================================================================

import json
from pathlib import Path

# Load training results
results_path = Path('/content/drive/MyDrive/quantum-trader/models') / f'training_results_{SWING_HORIZON}.json'

with open(results_path, 'r') as f:
    results = json.load(f)

print("=" * 60)
print("📊 TRAINING RESULTS")
print("=" * 60)
print(f"Swing Horizon: {results['swing_horizon']} ({results['horizon_bars']} bars)")
print(f"Total Samples: {results['total_samples']}")
print(f"Feature Count: {results['feature_count']}")
print(f"\nAverage Accuracy: {results['avg_accuracy']:.3f}")
print(f"Average Directional Accuracy: {results['avg_directional_accuracy']:.3f}")
print(f"Final Model Accuracy: {results['final_metrics']['accuracy']:.3f}")
print("=" * 60)

# ============================================================================
# STEP 9: TEST TRAINED MODEL (Optional)
# ============================================================================

import joblib
import pandas as pd
import yfinance as yf

# Load trained model
model_path = Path('/content/drive/MyDrive/quantum-trader/models') / f'swing_model_{SWING_HORIZON}.pkl'
scaler_path = Path('/content/drive/MyDrive/quantum-trader/models') / f'scaler_{SWING_HORIZON}.pkl'

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

print("✅ Model loaded successfully!")
print(f"Model type: {type(model).__name__}")

# Test prediction on recent data
test_ticker = 'AAPL'
df_test = yf.download(test_ticker, period='3mo', progress=False)

print(f"\n🔍 Testing prediction on {test_ticker}...")
print(f"Latest price: ${df_test['Close'].iloc[-1]:.2f}")

# Engineer features (simplified for demo)
from core.institutional_feature_engineer import InstitutionalFeatureEngineer

fe = InstitutionalFeatureEngineer()
spy_df = yf.download('SPY', period='3mo', progress=False)
vix_df = yf.download('^VIX', period='3mo', progress=False)

features = fe.engineer(df_test, spy_df=spy_df, vix_series=vix_df['Close'])
features = features.dropna()

if len(features) > 0:
    X_test = scaler.transform(features.iloc[[-1]])  # Latest observation
    
    prediction = model.predict(X_test)[0]
    proba = model.predict_proba(X_test)[0]
    
    signal_map = {0: 'SELL/SHORT', 1: 'HOLD', 2: 'BUY'}
    
    print(f"\n🎯 Prediction: {signal_map[prediction]}")
    print(f"Probabilities:")
    print(f"  SELL: {proba[0]:.1%}")
    print(f"  HOLD: {proba[1]:.1%}")
    print(f"  BUY: {proba[2]:.1%}")

# ============================================================================
# STEP 10: EXPORT FOR LOCAL USE
# ============================================================================

# Models are already saved to Google Drive
# Download from Drive to your local machine:
# 1. Go to Google Drive
# 2. Navigate to MyDrive/quantum-trader/models/
# 3. Download:
#    - swing_model_5bar.pkl
#    - scaler_5bar.pkl
#    - training_results_5bar.json

print("\n✅ Training complete!")
print("\n📥 Download models from Google Drive:")
print("   MyDrive/quantum-trader/models/")
print("   - swing_model_5bar.pkl")
print("   - scaler_5bar.pkl")
print("   - training_results_5bar.json")

# ============================================================================
# OPTIONAL: TRAIN MULTIPLE HORIZONS
# ============================================================================

# Train models for different swing durations
horizons_to_train = ['3bar', '5bar', '10bar']

for horizon in horizons_to_train:
    print(f"\n{'='*60}")
    print(f"Training {horizon} model...")
    print(f"{'='*60}")
    
    trainer_multi = ColabProTrainer(
        tickers=TICKERS,
        swing_horizon=horizon,
        lookback_days=730,
        model_dir='/content/drive/MyDrive/quantum-trader/models',
        use_gpu=True
    )
    
    trainer_multi.run_full_training_pipeline()

print("\n🎉 All models trained successfully!")

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

"""
Common Issues:

1. "No module named 'talib'"
   Solution: !pip install talib-binary (NOT python-talib)

2. "No data downloaded"
   Solution: Check internet connection, try different tickers

3. "Insufficient data"
   Solution: Reduce lookback_days or use more liquid tickers

4. "Out of memory"
   Solution: Reduce number of tickers or use shorter lookback_days

5. "GPU not available"
   Solution: Runtime > Change runtime type > GPU > T4 GPU

6. "Training too slow"
   Solution: Ensure GPU is enabled, reduce n_splits in walk_forward

7. "Low accuracy (<60%)"
   Solution: Train on more tickers, increase lookback_days, check label thresholds
"""

# ============================================================================
# NEXT STEPS
# ============================================================================

"""
After training:

1. BACKTEST
   - Load models
   - Run backtest with realistic slippage
   - Validate Sharpe > 1.5, Max DD < 10%

2. PAPER TRADE (30 days)
   - Generate signals daily
   - Track performance (no real money)
   - Monitor vs training metrics

3. LIVE DEPLOY (if paper trading successful)
   - Start with $1K
   - Max 3 positions concurrent
   - 10% stop losses
   - Scale to $5K after 90 days profitable

4. CONTINUOUS IMPROVEMENT
   - Retrain monthly with new data
   - Monitor pattern edge decay
   - Review training_logger recommendations
"""

# ============================================================================
# 🔥 YOU'RE READY TO BUILD YOUR TRADING EDGE! 🔥
# ============================================================================
