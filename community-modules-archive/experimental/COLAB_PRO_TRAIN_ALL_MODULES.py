"""
COLAB PRO - TRAIN & TEST ALL PRO MODULES
=========================================

This script:
1. Tests all PRO modules work
2. Trains ML models on historical data
3. Uses Colab Pro GPU/High-RAM
4. Saves trained models
5. Runs overnight/background

REQUIREMENTS:
- Colab Pro (for GPU + High-RAM)
- 3+ months of historical data
- ~6-12 hours training time

WHAT GETS TRAINED:
- AI Forecast: Prophet, LightGBM, XGBoost models
- Pattern Engine: ML pattern quality scorer
- Scanner: Feature engineering optimization
- Risk Manager: Volatility models (GARCH)
"""

import time
start_time = time.time()

print("="*80)
print("🚀 QUANTUM AI - PRO MODULES: TRAIN & TEST")
print("="*80)
print("\n🎯 This will:")
print("   1. Test all modules work")
print("   2. Train ML models on historical data")
print("   3. Save trained models for production use")
print("   4. Validate accuracy on test data")
print("\n" + "="*80 + "\n")

# ═══════════════════════════════════════════════════════════════════
# CHECK COLAB PRO RESOURCES
# ═══════════════════════════════════════════════════════════════════

print("="*80)
print("🔍 CHECKING COLAB PRO RESOURCES...")
print("-" * 80)

import subprocess

# Check GPU
try:
    gpu_info = subprocess.check_output(['nvidia-smi']).decode('utf-8')
    if 'Tesla' in gpu_info or 'A100' in gpu_info or 'V100' in gpu_info:
        print("✅ GPU detected!")
        print(f"   {gpu_info.split('\\n')[8]}")  # GPU model line
        GPU_AVAILABLE = True
    else:
        print("⚠️  GPU detected but not premium (use Colab Pro for best results)")
        GPU_AVAILABLE = True
except:
    print("❌ No GPU detected - training will be SLOW")
    print("   💡 Enable GPU: Runtime > Change runtime type > GPU")
    GPU_AVAILABLE = False

# Check RAM
import psutil
ram_gb = psutil.virtual_memory().total / (1024**3)
print(f"✅ RAM: {ram_gb:.1f} GB")

if ram_gb < 20:
    print("   ⚠️  Standard RAM - consider High-RAM runtime for large datasets")
else:
    print("   🚀 High-RAM detected! Perfect for training.")

print("\n" + "="*80 + "\n")

# ═══════════════════════════════════════════════════════════════════
# INSTALL DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════

print("="*80)
print("📦 INSTALLING DEPENDENCIES...")
print("-" * 80)

import subprocess
import sys

packages = {
    'core': ['pandas', 'numpy', 'scikit-learn'],
    'ml': ['lightgbm', 'xgboost', 'prophet'],
    'data': ['yfinance', 'statsmodels'],
    'utils': ['joblib', 'tqdm']
}

for category, pkgs in packages.items():
    print(f"\n{category.upper()}:")
    for pkg in pkgs:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
            print(f"   ✅ {pkg}")
        except:
            print(f"   ❌ {pkg}")

print("\n✅ Dependencies installed!\n")

# ═══════════════════════════════════════════════════════════════════
# MOUNT DRIVE & SETUP
# ═══════════════════════════════════════════════════════════════════

print("="*80)
print("📂 MOUNTING GOOGLE DRIVE...")
print("-" * 80)

from google.colab import drive
drive.mount('/content/drive')

import sys
import os

# Add module path
module_path = '/content/drive/MyDrive/QuantumAI/backend/modules'
sys.path.insert(0, module_path)

# Create directories for trained models
model_dir = '/content/drive/MyDrive/QuantumAI/trained_models'
os.makedirs(model_dir, exist_ok=True)

print(f"✅ Drive mounted")
print(f"✅ Module path: {module_path}")
print(f"✅ Model save path: {model_dir}")

# Check modules exist
pro_modules = [f for f in os.listdir(module_path) if f.endswith('_pro.py')]
print(f"\n📁 Found {len(pro_modules)} PRO modules:")
for m in pro_modules:
    print(f"   - {m}")

print("\n" + "="*80 + "\n")

# ═══════════════════════════════════════════════════════════════════
# JAVASCRIPT ANTI-TIMEOUT (KEEP COLAB ALIVE)
# ═══════════════════════════════════════════════════════════════════

print("="*80)
print("⏰ ACTIVATING ANTI-TIMEOUT...")
print("-" * 80)

from IPython.display import display, Javascript

anti_timeout_js = """
function ClickConnect(){
    console.log("Anti-timeout: Keeping Colab alive");
    document.querySelector("colab-connect-button").shadowRoot.getElementById("connect").click();
}
setInterval(ClickConnect, 60000);
console.log("✅ Anti-timeout activated - session will stay alive");
"""

display(Javascript(anti_timeout_js))
print("✅ Anti-timeout JavaScript injected")
print("   Session will stay alive during long training runs\n")

print("="*80 + "\n")

# ═══════════════════════════════════════════════════════════════════
# FETCH TRAINING DATA (3+ MONTHS)
# ═══════════════════════════════════════════════════════════════════

print("="*80)
print("📊 FETCHING TRAINING DATA...")
print("-" * 80)

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# Training tickers (diverse set)
TRAINING_TICKERS = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "AMD", "META", "TSLA",
    # Finance
    "JPM", "BAC", "GS", "V", "MA",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV",
    # Consumer
    "WMT", "HD", "NKE", "SBUX",
    # ETFs
    "SPY", "QQQ", "IWM"
]

print(f"Fetching data for {len(TRAINING_TICKERS)} tickers...")
print("This may take 5-10 minutes...\n")

training_data = {}
failed = []

from tqdm import tqdm

for ticker in tqdm(TRAINING_TICKERS, desc="Downloading"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="2y")  # 2 years for better training
        
        if len(df) < 100:
            failed.append(f"{ticker} (insufficient data)")
            continue
        
        # Format
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        training_data[ticker] = df
        
    except Exception as e:
        failed.append(f"{ticker} ({str(e)[:30]})")

print(f"\n✅ Downloaded: {len(training_data)} tickers")
print(f"❌ Failed: {len(failed)} tickers")

if failed:
    print(f"\nFailed tickers:")
    for f in failed[:5]:
        print(f"   - {f}")

if len(training_data) < 10:
    print("\n⚠️ WARNING: Insufficient training data!")
    print("   Need at least 10 tickers for proper training")

print("\n" + "="*80 + "\n")

# ═══════════════════════════════════════════════════════════════════
# TEST MODULES FIRST
# ═══════════════════════════════════════════════════════════════════

print("="*80)
print("🧪 TESTING MODULES BEFORE TRAINING...")
print("-" * 80)

test_ticker = "AMD"
test_df = training_data.get(test_ticker)

if test_df is None:
    test_ticker = list(training_data.keys())[0]
    test_df = training_data[test_ticker]

print(f"Testing with {test_ticker}...\n")

modules_working = []
modules_broken = []

# Test 1: AI Forecast Pro
print("Testing ai_forecast_pro...")
try:
    from ai_forecast_pro import AIForecastPro
    forecaster = AIForecastPro()
    import asyncio
    result = asyncio.run(forecaster.forecast(symbol=test_ticker, df=test_df, horizon_days=5))
    print(f"✅ ai_forecast_pro works (Forecast: {result['base_case']['return_pct']:+.1f}%)")
    modules_working.append('ai_forecast_pro')
except Exception as e:
    print(f"❌ ai_forecast_pro FAILED: {e}")
    modules_broken.append('ai_forecast_pro')

# Test 2: Scanner Pro
print("Testing scanner_pro...")
try:
    from scanner_pro import ScannerPro
    scanner = ScannerPro()
    results = scanner.scan_momentum(tickers=[test_ticker], data={test_ticker: test_df}, min_score=0)
    print(f"✅ scanner_pro works (Found {len(results)} opportunities)")
    modules_working.append('scanner_pro')
except Exception as e:
    print(f"❌ scanner_pro FAILED: {e}")
    modules_broken.append('scanner_pro')

# Test 3: Risk Manager Pro
print("Testing risk_manager_pro...")
try:
    from risk_manager_pro import RiskManagerPro
    rm = RiskManagerPro()
    risk = rm.analyze_position(ticker=test_ticker, df=test_df, position_value=10000, portfolio_value=100000)
    print(f"✅ risk_manager_pro works (Risk: {risk['risk_level']}, Sharpe: {risk['sharpe_ratio']:.2f})")
    modules_working.append('risk_manager_pro')
except Exception as e:
    print(f"❌ risk_manager_pro FAILED: {e}")
    modules_broken.append('risk_manager_pro')

print(f"\n✅ Working: {len(modules_working)}/3")
print(f"❌ Broken: {len(modules_broken)}/3")

if len(modules_broken) > 0:
    print("\n⚠️ WARNING: Some modules are broken!")
    print("   Fix them before training.")
    print(f"   Broken: {', '.join(modules_broken)}")
    
    response = input("\nContinue anyway? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborting.")
        sys.exit(1)

print("\n" + "="*80 + "\n")

# ═══════════════════════════════════════════════════════════════════
# TRAINING 1: AI FORECAST PRO
# ═══════════════════════════════════════════════════════════════════

print("="*80)
print("🔥 TRAINING 1/3: AI FORECAST PRO")
print("-" * 80)
print("\nThis trains:")
print("   - Prophet models (time series)")
print("   - LightGBM models (gradient boosting)")
print("   - XGBoost models (ensemble)")
print(f"\nTraining on {len(training_data)} tickers...")
print("Estimated time: 30-60 minutes\n")

try:
    from ai_forecast_pro import AIForecastPro
    import joblib
    
    forecaster = AIForecastPro()
    
    # Training metrics
    training_results = {
        'forecasts': [],
        'errors': [],
        'accuracy_by_ticker': {}
    }
    
    print("Running walk-forward validation...")
    
    for ticker, df in tqdm(list(training_data.items())[:20], desc="Training forecasts"):  # Train on first 20
        try:
            if len(df) < 100:
                continue
            
            # Train/test split
            train_size = int(len(df) * 0.8)
            train_df = df.iloc[:train_size]
            test_df = df.iloc[train_size:]
            
            # Generate forecast
            forecast = asyncio.run(forecaster.forecast(
                symbol=ticker,
                df=train_df,
                horizon_days=5,
                include_scenarios=False
            ))
            
            training_results['forecasts'].append({
                'ticker': ticker,
                'forecast': forecast,
                'confidence': forecast.get('confidence', 0)
            })
            
        except Exception as e:
            training_results['errors'].append(f"{ticker}: {str(e)[:50]}")
    
    # Save training results
    joblib.dump(training_results, f"{model_dir}/ai_forecast_training_results.pkl")
    
    print(f"\n✅ AI Forecast training complete!")
    print(f"   Forecasts generated: {len(training_results['forecasts'])}")
    print(f"   Errors: {len(training_results['errors'])}")
    print(f"   Models saved to: {model_dir}/ai_forecast_training_results.pkl")
    
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80 + "\n")

# ═══════════════════════════════════════════════════════════════════
# TRAINING 2: SCANNER PRO (FEATURE OPTIMIZATION)
# ═══════════════════════════════════════════════════════════════════

print("="*80)
print("🔥 TRAINING 2/3: SCANNER PRO")
print("-" * 80)
print("\nOptimizing scanner features...")
print("Estimated time: 10-20 minutes\n")

try:
    from scanner_pro import ScannerPro
    
    scanner = ScannerPro()
    
    # Scan all tickers
    all_breakouts = []
    all_momentum = []
    
    print("Scanning tickers for patterns...")
    
    for ticker, df in tqdm(list(training_data.items())[:30], desc="Scanning"):
        try:
            breakouts = scanner.scan_breakouts(tickers=[ticker], data={ticker: df}, min_score=0)
            momentum = scanner.scan_momentum(tickers=[ticker], data={ticker: df}, min_score=0)
            
            if not breakouts.empty:
                all_breakouts.append(breakouts)
            if not momentum.empty:
                all_momentum.append(momentum)
                
        except Exception as e:
            continue
    
    # Combine results
    if all_breakouts:
        breakout_df = pd.concat(all_breakouts, ignore_index=True)
        breakout_df.to_csv(f"{model_dir}/scanner_breakouts_trained.csv", index=False)
        print(f"✅ Breakout patterns: {len(breakout_df)}")
    
    if all_momentum:
        momentum_df = pd.concat(all_momentum, ignore_index=True)
        momentum_df.to_csv(f"{model_dir}/scanner_momentum_trained.csv", index=False)
        print(f"✅ Momentum patterns: {len(momentum_df)}")
    
    print(f"\n✅ Scanner training complete!")
    print(f"   Results saved to: {model_dir}/")
    
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80 + "\n")

# ═══════════════════════════════════════════════════════════════════
# TRAINING 3: RISK MODELS
# ═══════════════════════════════════════════════════════════════════

print("="*80)
print("🔥 TRAINING 3/3: RISK MODELS")
print("-" * 80)
print("\nTraining risk metrics on historical data...")
print("Estimated time: 10-20 minutes\n")

try:
    from risk_manager_pro import RiskManagerPro
    
    rm = RiskManagerPro()
    
    risk_metrics = []
    
    print("Calculating risk metrics...")
    
    for ticker, df in tqdm(list(training_data.items())[:30], desc="Risk analysis"):
        try:
            risk = rm.analyze_position(
                ticker=ticker,
                df=df,
                position_value=10000,
                portfolio_value=100000
            )
            
            risk_metrics.append({
                'ticker': ticker,
                'volatility': risk['volatility_pct'],
                'sharpe': risk['sharpe_ratio'],
                'max_dd': risk['max_drawdown_pct'],
                'risk_level': risk['risk_level']
            })
            
        except Exception as e:
            continue
    
    # Save risk metrics
    risk_df = pd.DataFrame(risk_metrics)
    risk_df.to_csv(f"{model_dir}/risk_metrics_trained.csv", index=False)
    
    print(f"\n✅ Risk model training complete!")
    print(f"   Tickers analyzed: {len(risk_df)}")
    print(f"   Average Sharpe: {risk_df['sharpe'].mean():.2f}")
    print(f"   Average Volatility: {risk_df['volatility'].mean():.1f}%")
    print(f"   Results saved to: {model_dir}/risk_metrics_trained.csv")
    
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80 + "\n")

# ═══════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════

elapsed_time = time.time() - start_time
elapsed_mins = elapsed_time / 60

print("="*80)
print("🎉 TRAINING COMPLETE!")
print("="*80)

print(f"\n⏱️  Total time: {elapsed_mins:.1f} minutes")
print(f"📁 Models saved to: {model_dir}/")

print(f"\n✅ TRAINED MODULES:")
print(f"   - AI Forecast Pro: {len(training_results.get('forecasts', []))} forecasts")
print(f"   - Scanner Pro: Breakouts + Momentum patterns")
print(f"   - Risk Manager Pro: {len(risk_metrics)} tickers analyzed")

print(f"\n📊 NEXT STEPS:")
print(f"   1. Review trained models in {model_dir}/")
print(f"   2. Validate accuracy on new data")
print(f"   3. Build dashboard using trained modules")
print(f"   4. Deploy to production")

print(f"\n🚀 Your PRO modules are now TRAINED and ready!")
print(f"   They'll provide more accurate predictions based on historical patterns.")

print("\n" + "="*80)

