"""
🚀 QUANTUM AI DASHBOARD LAUNCHER
Uses: COMPLETE_INTEGRATED_DASHBOARD.py
Path: /content/drive/MyDrive/QuantumAI/
"""

import os
import sys
import subprocess
import time
import threading

print("="*80)
print("🚀 QUANTUM AI DASHBOARD LAUNCHER")
print("="*80)

# ============================================================================
# 1. MOUNT DRIVE
# ============================================================================
print("\n📁 Checking Google Drive...")
from google.colab import drive
import os

if os.path.exists('/content/drive/MyDrive'):
    print("✅ Drive already mounted")
else:
    print("⏳ Mounting drive...")
    drive.mount('/content/drive')
    print("✅ Drive mounted")

# ============================================================================
# 2. SETUP PATHS
# ============================================================================
ROOT_DIR = '/content/drive/MyDrive/QuantumAI'
MODULES_DIR = os.path.join(ROOT_DIR, 'backend', 'modules')
DASHBOARD_FILE = os.path.join(ROOT_DIR, 'COMPLETE_INTEGRATED_DASHBOARD.py')

print(f"\n📂 Root directory: {ROOT_DIR}")
print(f"📂 Modules directory: {MODULES_DIR}")
print(f"📄 Dashboard: {os.path.basename(DASHBOARD_FILE)}")

# Check files exist
if not os.path.exists(DASHBOARD_FILE):
    print(f"\n❌ ERROR: Dashboard not found at {DASHBOARD_FILE}")
    print("\n📂 Files in QuantumAI folder:")
    for f in sorted(os.listdir(ROOT_DIR)):
        print(f"   {'📁' if os.path.isdir(os.path.join(ROOT_DIR, f)) else '📄'} {f}")
    sys.exit(1)

print("✅ Dashboard found!")

# Add both directories to path
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, MODULES_DIR)

print("✅ Paths configured")

# ============================================================================
# 3. INSTALL DEPENDENCIES
# ============================================================================
print("\n📦 Installing dependencies...")
packages = [
    'streamlit',
    'plotly', 
    'yfinance',
    'pyngrok',
    'nest-asyncio',
    'pandas',
    'numpy',
    'scikit-learn',
    'lightgbm',
    'xgboost',
    'prophet',
    'ta',
    'requests',
]

subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q', '--upgrade'] + packages,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)
print("✅ Dependencies installed")

# ============================================================================
# 4. START STREAMLIT
# ============================================================================
print("\n🚀 Starting Streamlit server...")

def run_streamlit():
    """Run Streamlit in background"""
    subprocess.run([
        sys.executable, '-m', 'streamlit', 'run',
        DASHBOARD_FILE,
        '--server.port=8501',
        '--server.headless=true',
        '--server.enableCORS=false',
        '--server.enableXsrfProtection=false',
        '--browser.serverAddress=localhost',
        '--logger.level=error'
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

streamlit_thread = threading.Thread(target=run_streamlit, daemon=True)
streamlit_thread.start()

print("✅ Streamlit starting...")
print("⏳ Waiting for server (10 seconds)...")
time.sleep(10)

# ============================================================================
# 5. CREATE NGROK TUNNEL
# ============================================================================
print("\n🌐 Creating public URL with ngrok...")

try:
    from pyngrok import ngrok
    
    # Load token from file
    token_file = os.path.join(ROOT_DIR, 'ngrok_token.txt')
    with open(token_file, 'r') as f:
        NGROK_TOKEN = f.read().strip()
    
    ngrok.set_auth_token(NGROK_TOKEN)
    
    # Create tunnel
    public_url = ngrok.connect(8501)
    
    print("\n" + "="*80)
    print("✅✅✅ DASHBOARD IS LIVE! ✅✅✅")
    print("="*80)
    print(f"\n🌍 PUBLIC URL: {public_url}")
    print("\n📱 👆 CLICK THE LINK ABOVE! 👆")
    print("\n🎯 Features:")
    print("   • Institutional Ensemble Engine")
    print("   • Real-time signals & scoring")
    print("   • Paper trading portfolio")
    print("   • Advanced charting")
    print("   • ML-powered scanners")
    print("   • 5-day & 21-day forecasters")
    print("\n💡 To stop: Runtime → Interrupt execution")
    print("="*80)
    
    # Keep running
    print("\n🟢 Dashboard running...\n")
    
    while True:
        time.sleep(2)
        if not streamlit_thread.is_alive():
            print("\n⚠️  Streamlit stopped!")
            break
        
except KeyboardInterrupt:
    print("\n\n🛑 Stopped by user")
    
except Exception as e:
    print(f"\n❌ Ngrok error: {e}")
    print(f"\n💡 Streamlit is running on: http://localhost:8501")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopped")

print("\n" + "="*80)
print("Session ended")
print("="*80)

