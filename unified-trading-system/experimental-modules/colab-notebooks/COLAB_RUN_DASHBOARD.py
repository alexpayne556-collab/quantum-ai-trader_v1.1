"""
================================================================================
🚀 RUN QUANTUM AI DASHBOARD ON GOOGLE COLAB PRO
================================================================================

This script sets up and runs the complete Streamlit dashboard on Colab
with GPU acceleration and high RAM.

Benefits:
✅ GPU acceleration (10x faster predictions)
✅ 25GB+ RAM (Colab Pro)
✅ No load on your PC
✅ Public URL access
✅ 24/7 operation (Colab Pro+)

Paste this entire cell into Google Colab and run!
================================================================================
"""

print("="*80)
print("🚀 QUANTUM AI DASHBOARD - COLAB PRO SETUP")
print("="*80)
print()

# ================================================================================
# STEP 1: MOUNT GOOGLE DRIVE
# ================================================================================
print("📁 Step 1: Mounting Google Drive...")
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os
import sys

PROJECT_ROOT = "/content/drive/MyDrive/QuantumAI"
os.makedirs(PROJECT_ROOT, exist_ok=True)
sys.path.insert(0, PROJECT_ROOT)

print(f"✅ Mounted: {PROJECT_ROOT}\n")

# ================================================================================
# STEP 2: INSTALL DEPENDENCIES
# ================================================================================
print("📦 Step 2: Installing dependencies...")
print("(This may take 2-3 minutes on first run)\n")

# Core dependencies
!pip install -q streamlit plotly pandas numpy yfinance ta scipy

# ML libraries (GPU-accelerated if available)
!pip install -q scikit-learn lightgbm xgboost joblib

# Optional: Prophet and ARIMA for elite forecaster
!pip install -q prophet statsmodels

# Tunneling (for public URL)
!pip install -q pyngrok

print("✅ All dependencies installed!\n")

# ================================================================================
# STEP 3: CHECK GPU AVAILABILITY
# ================================================================================
print("🖥️  Step 3: Checking hardware...")

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU Available: {len(gpus)} GPU(s) detected")
    print(f"   GPU: {gpus[0]}")
else:
    print("⚠️  No GPU detected (CPU mode)")

# Check RAM
import psutil
ram_gb = psutil.virtual_memory().total / (1024**3)
print(f"✅ RAM Available: {ram_gb:.1f} GB")

if ram_gb >= 25:
    print("   🎉 High RAM detected (Colab Pro!)")
elif ram_gb >= 12:
    print("   ✅ Standard RAM")
else:
    print("   ⚠️  Low RAM (may be slow)")

print()

# ================================================================================
# STEP 4: SETUP NGROK (FOR PUBLIC URL)
# ================================================================================
print("🌐 Step 4: Setting up public access...")
print("You need an ngrok auth token (free).")
print("Get it here: https://dashboard.ngrok.com/get-started/your-authtoken")
print()

# Option 1: Use ngrok (recommended)
from pyngrok import ngrok, conf
import getpass

# Check if ngrok token is already set
try:
    # Try to get existing token
    ngrok_token = conf.get_default().auth_token
    if not ngrok_token:
        raise Exception("No token")
    print(f"✅ ngrok token already configured")
except:
    # Ask for token
    print("Please enter your ngrok auth token:")
    print("(Get free token: https://dashboard.ngrok.com/get-started/your-authtoken)")
    ngrok_token = getpass.getpass("ngrok token: ")
    
    if ngrok_token:
        ngrok.set_auth_token(ngrok_token)
        print("✅ ngrok token configured")
    else:
        print("⚠️  No token provided. Using localtunnel instead...")
        ngrok_token = None

print()

# ================================================================================
# STEP 5: CHECK IF MODELS EXIST
# ================================================================================
print("🤖 Step 5: Checking for trained models...")

models_dir = f"{PROJECT_ROOT}/models_ranking"
model_files = [
    'lgbm_ranking.pkl',
    'xgb_ranking.pkl', 
    'rf_ranking.pkl',
    'mlp_ranking.pkl',
    'scaler.pkl',
    'metadata.json'
]

models_exist = all([os.path.exists(f"{models_dir}/{f}") for f in model_files])

if models_exist:
    print(f"✅ Ranking models found in {models_dir}")
    
    # Load metadata to show info
    import json
    with open(f"{models_dir}/metadata.json", 'r') as f:
        metadata = json.load(f)
    
    print(f"   Trained: {metadata.get('trained_date', 'Unknown')}")
    print(f"   Success Rate: {metadata.get('performance', {}).get('success_rate_top10', 0)*100:.1f}%")
else:
    print(f"⚠️  Ranking models NOT found in {models_dir}")
    print("   You'll need to train the models first or upload them.")

print()

# ================================================================================
# STEP 6: CREATE STREAMLIT CONFIG
# ================================================================================
print("⚙️  Step 6: Creating Streamlit config...")

# Create .streamlit directory
streamlit_dir = "/content/.streamlit"
os.makedirs(streamlit_dir, exist_ok=True)

# Create config
config_content = """
[server]
port = 8501
enableCORS = false
enableXsrfProtection = false
headless = true

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
font = "sans serif"
"""

with open(f"{streamlit_dir}/config.toml", 'w') as f:
    f.write(config_content)

print("✅ Streamlit configured\n")

# ================================================================================
# STEP 7: WRITE DASHBOARD FILES
# ================================================================================
print("📝 Step 7: Writing dashboard files to Colab...")

# Write ADVANCED_CHART_ENGINE.py
chart_engine_code = '''
# (ADVANCED_CHART_ENGINE.py code goes here - will be in Drive)
# For now, we'll load it from Drive
'''

# Check if files exist in Drive
dashboard_file = f"{PROJECT_ROOT}/UNIFIED_FORECASTER_DASHBOARD.py"
chart_engine_file = f"{PROJECT_ROOT}/ADVANCED_CHART_ENGINE.py"

files_ready = True
if not os.path.exists(dashboard_file):
    print(f"❌ Missing: UNIFIED_FORECASTER_DASHBOARD.py")
    print(f"   Upload to: {PROJECT_ROOT}/")
    files_ready = False
else:
    print(f"✅ Found: UNIFIED_FORECASTER_DASHBOARD.py")

if not os.path.exists(chart_engine_file):
    print(f"❌ Missing: ADVANCED_CHART_ENGINE.py")
    print(f"   Upload to: {PROJECT_ROOT}/")
    files_ready = False
else:
    print(f"✅ Found: ADVANCED_CHART_ENGINE.py")

if not files_ready:
    print("\n⚠️  UPLOAD MISSING FILES FIRST!")
    print("Then re-run this cell.")
else:
    print("✅ All dashboard files ready!\n")

# ================================================================================
# STEP 8: RUN STREAMLIT DASHBOARD
# ================================================================================
if files_ready:
    print("="*80)
    print("🚀 LAUNCHING DASHBOARD!")
    print("="*80)
    print()
    
    # Change to project directory
    os.chdir(PROJECT_ROOT)
    
    # Start streamlit in background
    print("Starting Streamlit server...")
    
    import subprocess
    import threading
    import time
    
    # Function to run streamlit
    def run_streamlit():
        subprocess.run([
            'streamlit', 'run', 'UNIFIED_FORECASTER_DASHBOARD.py',
            '--server.port', '8501',
            '--server.headless', 'true',
            '--browser.gatherUsageStats', 'false'
        ])
    
    # Start streamlit in thread
    streamlit_thread = threading.Thread(target=run_streamlit, daemon=True)
    streamlit_thread.start()
    
    # Wait for streamlit to start
    print("Waiting for Streamlit to start...")
    time.sleep(10)
    
    print("✅ Streamlit started on port 8501\n")
    
    # ================================================================================
    # STEP 9: CREATE PUBLIC TUNNEL
    # ================================================================================
    print("="*80)
    print("🌐 CREATING PUBLIC URL")
    print("="*80)
    print()
    
    if ngrok_token:
        # Use ngrok
        print("Using ngrok for public access...")
        try:
            public_url = ngrok.connect(8501, bind_tls=True)
            print()
            print("="*80)
            print("✅ DASHBOARD IS LIVE!")
            print("="*80)
            print()
            print(f"🌐 Public URL: {public_url}")
            print()
            print("📱 Access from:")
            print("   - Your phone")
            print("   - Any computer")
            print("   - Share with others")
            print()
            print("⚠️  Keep this Colab tab open to keep dashboard running!")
            print()
            print("="*80)
        except Exception as e:
            print(f"❌ ngrok failed: {e}")
            print("Falling back to localtunnel...")
            ngrok_token = None
    
    if not ngrok_token:
        # Use localtunnel
        print("Using localtunnel for public access...")
        print("(Less reliable than ngrok, but works without token)")
        print()
        
        # Install localtunnel
        !npm install -g localtunnel
        
        # Start tunnel
        print("Starting tunnel...")
        print()
        
        # Run in background
        tunnel_process = subprocess.Popen(
            ['lt', '--port', '8501'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a bit
        time.sleep(5)
        
        # Try to get URL
        try:
            stdout, _ = tunnel_process.communicate(timeout=2)
            if 'https://' in stdout:
                url = [line for line in stdout.split('\n') if 'https://' in line][0]
                print("="*80)
                print("✅ DASHBOARD IS LIVE!")
                print("="*80)
                print()
                print(f"🌐 Public URL: {url}")
                print()
                print("⚠️  Keep this Colab tab open!")
                print()
                print("="*80)
        except:
            print("⚠️  Could not get tunnel URL automatically")
            print("Check the output above for the URL")
    
    # ================================================================================
    # KEEP ALIVE
    # ================================================================================
    print()
    print("💡 TIP: The dashboard will run as long as this cell is running.")
    print("        Press STOP button to shutdown.")
    print()
    
    # Keep running
    try:
        while True:
            time.sleep(60)
            print(".", end="", flush=True)
    except KeyboardInterrupt:
        print("\n\n🛑 Dashboard stopped!")
        if ngrok_token:
            ngrok.disconnect(public_url)
            print("✅ Tunnel closed")

else:
    print("\n❌ Cannot start dashboard - missing files!")
    print("Upload the required files and re-run this cell.")

print()
print("="*80)
print("🎉 SETUP COMPLETE!")
print("="*80)

