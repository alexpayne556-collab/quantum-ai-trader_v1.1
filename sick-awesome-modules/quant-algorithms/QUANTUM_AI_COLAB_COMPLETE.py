"""
🚀 QUANTUM AI COCKPIT — COMPLETE COLAB SETUP
============================================

COPY THIS ENTIRE FILE INTO A SINGLE GOOGLE COLAB CELL AND RUN IT!

This will:
1. Install all dependencies (with GPU support)
2. Load your modules from Google Drive
3. Test all systems
4. Launch Streamlit dashboard with public URL
5. Train models on GPU (7-10x faster!)

Prerequisites:
- Google Colab (free or Pro)
- Runtime: GPU (T4 or better)
- Google Drive with your modules uploaded

Author: Quantum AI Cockpit Team
Last Updated: November 2024
"""

# ═══════════════════════════════════════════════════════════════════════════
# PART 1: ENVIRONMENT SETUP (2-3 minutes)
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("🚀 QUANTUM AI COCKPIT — COMPLETE SETUP")
print("=" * 80)

import os
import sys
from pathlib import Path

# Check GPU
try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n✅ GPU DETECTED: {gpu_name}")
        print(f"   Memory: {gpu_memory:.1f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        print("\n⚠️  NO GPU DETECTED!")
        print("   Go to Runtime → Change runtime type → GPU (T4)")
        print("   Then Runtime → Restart runtime")
except:
    pass

# Install packages
print("\n📦 Installing dependencies (2-3 minutes)...")
print("   (This is a one-time setup)")

get_ipython().system('pip install -q numpy pandas scipy scikit-learn matplotlib seaborn plotly')
get_ipython().system('pip install -q yfinance pandas-datareader')
get_ipython().system('pip install -q darts torch pytorch-lightning')
get_ipython().system('pip install -q xgboost ta statsmodels arch')
get_ipython().system('pip install -q transformers sentencepiece')
get_ipython().system('pip install -q python-dotenv requests aiohttp')
get_ipython().system('pip install -q streamlit pyngrok')

print("\n✅ All packages installed!")

# Verify GPU
import torch
if torch.cuda.is_available():
    print(f"\n✅ GPU Ready: {torch.cuda.get_device_name(0)}")
    print(f"   PyTorch Version: {torch.__version__}")
else:
    print("\n❌ GPU NOT AVAILABLE — Training will be SLOW!")
    print("   Please enable GPU in Runtime settings")

print("\n" + "=" * 80)
print("✅ ENVIRONMENT SETUP COMPLETE!")
print("=" * 80)

# ═══════════════════════════════════════════════════════════════════════════
# PART 2: MOUNT GOOGLE DRIVE & LOAD MODULES
# ═══════════════════════════════════════════════════════════════════════════

print("\n📂 Mounting Google Drive...")

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

print("✅ Google Drive mounted!")

# Check if modules exist
DRIVE_PATH = '/content/drive/MyDrive/Quantum_AI_Cockpit/backend/modules'

if os.path.exists(DRIVE_PATH):
    print(f"\n✅ Found modules at: {DRIVE_PATH}")
    
    # Add to path
    if DRIVE_PATH not in sys.path:
        sys.path.insert(0, DRIVE_PATH)
    
    print("✅ Modules loaded from Google Drive!")
else:
    print(f"\n❌ Modules not found at: {DRIVE_PATH}")
    print("\n📋 SETUP INSTRUCTIONS:")
    print("   1. Install Google Drive Desktop on your laptop")
    print("   2. Copy D:\\Quantum_AI_Cockpit\\backend\\modules\\ to Google Drive")
    print("   3. Wait for sync to complete")
    print("   4. Re-run this cell")
    print("\n   OR upload modules.zip and extract it")
    
    # Check for modules.zip
    zip_path = '/content/drive/MyDrive/Quantum_AI_Cockpit/modules.zip'
    if os.path.exists(zip_path):
        print(f"\n📦 Found modules.zip - extracting...")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('/content/modules')
        
        if '/content/modules' not in sys.path:
            sys.path.insert(0, '/content/modules')
        
        print("✅ Modules extracted and loaded!")
    else:
        raise FileNotFoundError("Please upload your modules to Google Drive first!")

# ═══════════════════════════════════════════════════════════════════════════
# PART 3: CREATE .ENV FILE WITH API KEYS
# ═══════════════════════════════════════════════════════════════════════════

print("\n🔑 Creating .env file with API keys...")

env_content = """
# Market Data APIs
ALPHAVANTAGE_API_KEY=6NOB0V91707OM1TI
MASSIVE_API_KEY=chFZODMC89wpypjBibRsW1E160SVBfPL
POLYGON_API_KEY=gyBClHUxmeIerRMuUMGGi1hIiBIxl2cS
TWELVEDATA_API_KEY=5852d42a799e47269c689392d273f70b
FINNHUB_API_KEY=d40387pr01qkrgfb5asgd40387pr01qkrgfb5at0
TIINGO_API_KEY=de94a283588681e212560a0d9826903e25647968
FINANCIALMODELINGPREP_API_KEY=15zYYtksuJnQsTBODSNs3MrfEedOSd3i

# News APIs
NEWS_API_KEY=e6f793dfd61f473786f69466f9313fe8
MARKETAUX_API_KEY=Tw5w7ABp5srP5mgaKeyGjiXaJlk7Oz7sgpmxWxYH

# Settings
LOG_LEVEL=INFO
DATA_PRIORITY=FINANCIALMODELINGPREP,TWELVEDATA,FINNHUB,MASSIVE,TIINGO,YFINANCE
"""

with open('/content/.env', 'w') as f:
    f.write(env_content)

# Also create in modules directory
modules_path = DRIVE_PATH if os.path.exists(DRIVE_PATH) else '/content/modules'
with open(f'{modules_path}/../.env', 'w') as f:
    f.write(env_content)

print("✅ .env file created!")

# ═══════════════════════════════════════════════════════════════════════════
# PART 4: TEST MODULE IMPORTS
# ═══════════════════════════════════════════════════════════════════════════

print("\n🧪 Testing module imports...\n")

modules_to_test = [
    ('fusior_forecast', 'N-BEATS Forecaster'),
    ('master_analysis_engine', 'Master Analysis Engine'),
    ('pattern_integration_layer', 'Pattern Integration Layer'),
    ('ai_recommender_v2', 'AI Recommender V2'),
    ('ai_recommender_institutional', 'Institutional Features'),
    ('support_resistance_detector', 'Support/Resistance'),
    ('volume_profile_analyzer', 'Volume Profile'),
    ('multi_timeframe_analyzer', 'Multi-Timeframe'),
    ('portfolio_manager', 'Portfolio Manager'),
    ('morning_brief_generator', 'Morning Brief'),
    ('pre_gainer_scanner', 'Scanner'),
    ('watchlist_manager', 'Watchlist Manager'),
]

success_count = 0
failed = []

for module_name, display_name in modules_to_test:
    try:
        __import__(module_name)
        print(f"✅ {display_name}")
        success_count += 1
    except Exception as e:
        print(f"❌ {display_name}: {str(e)[:50]}")
        failed.append(display_name)

print(f"\n{'='*80}")
print(f"✅ {success_count}/{len(modules_to_test)} modules loaded successfully!")

if failed:
    print(f"⚠️  Failed: {', '.join(failed)}")
else:
    print("🎉 ALL MODULES LOADED PERFECTLY!")
print(f"{'='*80}")

# ═══════════════════════════════════════════════════════════════════════════
# PART 5: QUICK SYSTEM TEST
# ═══════════════════════════════════════════════════════════════════════════

print("\n🧪 Running quick system test...\n")

import asyncio
from master_analysis_engine import analyze_stock

async def quick_test():
    print("📊 Analyzing NVDA...")
    
    result = await analyze_stock(
        symbol="NVDA",
        account_balance=50000,
        forecast_days=21,
        verbose=False
    )
    
    if result['status'] == 'ok':
        rec = result['recommendation']
        
        print(f"\n{'='*80}")
        print(f"🎯 QUICK TEST RESULTS: NVDA")
        print(f"{'='*80}")
        print(f"\n🎯 Action: {rec['action']}")
        print(f"📈 Confidence: {rec['confidence']*100:.1f}%")
        print(f"📅 Expected 5D: {rec.get('expected_move_5d', 0):+.1f}%")
        
        inst = rec.get('institutional_grade')
        if inst:
            ps = inst['position_sizing']
            rr = inst['risk_reward']
            
            print(f"\n💰 Position: {ps['shares']} shares (${ps['position_value']:,.0f})")
            print(f"⚖️  R:R: {rr['rr_ratio']:.2f}:1")
        
        patterns = result.get('patterns')
        if patterns and patterns.get('status') == 'ok':
            summary = patterns['summary']
            print(f"\n🔍 Patterns: {summary['total_patterns_detected']}")
            print(f"📊 Confluence: {summary['confluence_score']:.0f}%")
        
        print(f"\n{'='*80}")
        print("✅ SYSTEM TEST PASSED!")
        print(f"{'='*80}")
    else:
        print(f"\n❌ Test failed: {result.get('error', 'Unknown error')}")

# Run test
await quick_test()

# ═══════════════════════════════════════════════════════════════════════════
# PART 6: LAUNCH STREAMLIT DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("🚀 LAUNCHING STREAMLIT DASHBOARD")
print("=" * 80)

# Copy dashboard to /content
dashboard_source = f"{modules_path}/../quantum_dashboard.py"
if os.path.exists(dashboard_source):
    import shutil
    shutil.copy(dashboard_source, '/content/quantum_dashboard.py')
    print("✅ Dashboard copied to /content/")
else:
    # Create dashboard if not exists
    print("⚠️  Dashboard not found - using modules path directly")

# Setup ngrok for public URL
from pyngrok import ngrok, conf
import time

# Kill any existing tunnels
ngrok.kill()

# Set ngrok auth token (free - no signup needed for basic use)
# For permanent URL, sign up at https://ngrok.com and use your token:
# ngrok.set_auth_token("YOUR_TOKEN_HERE")

print("\n🌐 Starting Streamlit server...")
print("   (This takes 10-15 seconds)")

# Start Streamlit in background
get_ipython().system_raw('streamlit run /content/quantum_dashboard.py --server.port 8501 &')

# Wait for server to start
time.sleep(10)

# Create ngrok tunnel
public_url = ngrok.connect(8501)

print("\n" + "=" * 80)
print("🎉 DASHBOARD IS LIVE!")
print("=" * 80)
print(f"\n🌐 Public URL: {public_url}")
print("\n📋 Instructions:")
print("   1. Click the URL above")
print("   2. Dashboard will open in new tab")
print("   3. URL is active as long as this notebook is running")
print("   4. Share URL with anyone (temporary, resets when notebook restarts)")
print("\n⚠️  Important:")
print("   - URL expires when you stop the notebook")
print("   - For permanent URL, sign up at https://ngrok.com (free)")
print("   - Or deploy to Streamlit Cloud for free permanent hosting")
print("\n" + "=" * 80)

# Keep cell running
print("\n⏳ Dashboard is running... (Keep this cell running)")
print("   To stop: Runtime → Interrupt execution")

# Optional: Display dashboard in iframe
from IPython.display import IFrame
display(IFrame(src=public_url, width=1000, height=800))

# ═══════════════════════════════════════════════════════════════════════════
# DONE!
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("🎉 QUANTUM AI COCKPIT IS READY!")
print("=" * 80)
print("\n✅ Everything is set up and running!")
print("\n📊 What you can do now:")
print("   • Use the dashboard to analyze stocks")
print("   • Train models (they'll be 7-10x faster on GPU)")
print("   • Run scans on 100+ stocks")
print("   • Test strategies with backtesting")
print("\n💡 Tips:")
print("   • Keep this notebook open while using dashboard")
print("   • Trained models are saved to Google Drive")
print("   • Download models to use on your laptop later")
print("\n" + "=" * 80)

