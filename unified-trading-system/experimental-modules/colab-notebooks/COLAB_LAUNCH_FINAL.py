"""
🚀 QUANTUM AI DASHBOARD LAUNCHER
Path: /content/drive/MyDrive/QuantumAI
"""

import os
import sys
import subprocess
import time
import threading
from pathlib import Path

print("="*80)
print("🚀 QUANTUM AI DASHBOARD LAUNCHER")
print("="*80)

# ============================================================================
# 1. MOUNT DRIVE
# ============================================================================
print("\n📁 Mounting Google Drive...")
from google.colab import drive
drive.mount('/content/drive', force_remount=False)
print("✅ Drive mounted")

# ============================================================================
# 2. FIND YOUR FILES
# ============================================================================
print("\n🔍 Searching for dashboard...")

# Possible locations (in order of likelihood)
possible_paths = [
    '/content/drive/MyDrive/QuantumAI/backend/modules',
    '/content/drive/MyDrive/QuantumAI/modules',
    '/content/drive/MyDrive/QuantumAI',
]

found_dashboard = None
modules_dir = None

for base_path in possible_paths:
    if os.path.exists(base_path):
        print(f"   ✅ Checking: {base_path}")
        
        dashboard_file = os.path.join(base_path, 'ULTIMATE_DASHBOARD_INTEGRATED.py')
        if os.path.exists(dashboard_file):
            found_dashboard = dashboard_file
            modules_dir = base_path
            print(f"   🎯 FOUND DASHBOARD!")
            break
    else:
        print(f"   ❌ Not found: {base_path}")

# If not found, do a deeper search
if not found_dashboard and os.path.exists('/content/drive/MyDrive/QuantumAI'):
    print("\n   🔍 Searching subdirectories...")
    for root, dirs, files in os.walk('/content/drive/MyDrive/QuantumAI'):
        if 'ULTIMATE_DASHBOARD_INTEGRATED.py' in files:
            found_dashboard = os.path.join(root, 'ULTIMATE_DASHBOARD_INTEGRATED.py')
            modules_dir = root
            print(f"   🎯 FOUND: {modules_dir}")
            break

if not found_dashboard:
    print("\n❌ ERROR: Could not find ULTIMATE_DASHBOARD_INTEGRATED.py")
    print("\n📂 Let's see what's in /content/drive/MyDrive/QuantumAI:")
    
    if os.path.exists('/content/drive/MyDrive/QuantumAI'):
        contents = os.listdir('/content/drive/MyDrive/QuantumAI')
        print(f"\n   Found {len(contents)} items:")
        for item in sorted(contents)[:20]:
            item_path = os.path.join('/content/drive/MyDrive/QuantumAI', item)
            if os.path.isdir(item_path):
                print(f"   📁 {item}/")
                # Check subdirectories
                try:
                    sub_items = os.listdir(item_path)
                    if 'ULTIMATE_DASHBOARD_INTEGRATED.py' in sub_items:
                        print(f"      🎯 DASHBOARD IS HERE!")
                        found_dashboard = os.path.join(item_path, 'ULTIMATE_DASHBOARD_INTEGRATED.py')
                        modules_dir = item_path
                except:
                    pass
            else:
                print(f"   📄 {item}")
    else:
        print("\n❌ /content/drive/MyDrive/QuantumAI does not exist!")
        print("\n💡 Available folders in MyDrive:")
        for item in sorted(os.listdir('/content/drive/MyDrive'))[:20]:
            if os.path.isdir(os.path.join('/content/drive/MyDrive', item)):
                print(f"   📁 {item}")
    
    if not found_dashboard:
        print("\n❌ Please upload ULTIMATE_DASHBOARD_INTEGRATED.py to Google Drive")
        sys.exit(1)

# ============================================================================
# 3. SETUP PATHS
# ============================================================================
print(f"\n✅ Dashboard: {os.path.basename(found_dashboard)}")
print(f"✅ Location: {modules_dir}")

os.chdir(modules_dir)
sys.path.insert(0, modules_dir)

# Show available modules
print("\n📄 Available modules:")
py_files = sorted([f for f in os.listdir(modules_dir) if f.endswith('.py')])
for f in py_files[:10]:
    size = os.path.getsize(os.path.join(modules_dir, f)) / 1024
    print(f"   ✅ {f} ({size:.1f} KB)")
if len(py_files) > 10:
    print(f"   ... and {len(py_files) - 10} more")

# ============================================================================
# 4. INSTALL DEPENDENCIES
# ============================================================================
print("\n📦 Installing dependencies (this may take 1-2 minutes)...")
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
]

subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q', '--upgrade'] + packages,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)
print("✅ Dependencies installed")

# ============================================================================
# 5. START STREAMLIT
# ============================================================================
print("\n🚀 Starting Streamlit server...")

def run_streamlit():
    """Run Streamlit in background"""
    subprocess.run([
        sys.executable, '-m', 'streamlit', 'run',
        found_dashboard,
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
print("⏳ Waiting for server to initialize...")
time.sleep(10)  # Give it more time to start

# ============================================================================
# 6. CREATE NGROK TUNNEL
# ============================================================================
print("\n🌐 Creating public URL with ngrok...")

try:
    from pyngrok import ngrok
    
    # Your ngrok token
    NGROK_TOKEN = '35jIAcsNyWEBfkEE5BOs9CKjzUH_5TtS5hwfvk3XurHKMKFLC'
    ngrok.set_auth_token(NGROK_TOKEN)
    
    # Create tunnel
    public_url = ngrok.connect(8501)
    
    print("\n" + "="*80)
    print("✅✅✅ DASHBOARD IS LIVE! ✅✅✅")
    print("="*80)
    print(f"\n🌍 PUBLIC URL: {public_url}")
    print("\n📱 CLICK THE LINK ABOVE! ☝️")
    print("\n💡 Features available:")
    print("   • 🧠 Institutional Ensemble Engine")
    print("   • 📊 Real-time signals & confidence scoring")
    print("   • 💼 Paper trading portfolio")
    print("   • 📈 Advanced charting (all indicators)")
    print("   • 🔍 All ML-powered scanners")
    print("   • 🎯 Top 10 rankings")
    print("   • 🔮 5-day & 21-day forecasters")
    print("\n⏱️  Dashboard will run until you stop this cell")
    print("   To stop: Runtime → Interrupt execution")
    print("\n" + "="*80)
    
    # Keep running and monitor
    print("\n🟢 Dashboard is running...\n")
    
    while True:
        time.sleep(2)
        
        # Check if streamlit is still alive
        if not streamlit_thread.is_alive():
            print("\n⚠️  Streamlit thread stopped unexpectedly!")
            break
        
except KeyboardInterrupt:
    print("\n\n🛑 Dashboard stopped by user")
    
except Exception as e:
    print(f"\n❌ Ngrok error: {e}")
    print("\n💡 FALLBACK: Streamlit is running on port 8501")
    print("   Local URL: http://localhost:8501")
    print("   You can use Colab's port forwarding feature")
    
    # Keep running
    try:
        print("\n⏳ Streamlit still running...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopped")

print("\n" + "="*80)
print("Session ended")
print("="*80)

