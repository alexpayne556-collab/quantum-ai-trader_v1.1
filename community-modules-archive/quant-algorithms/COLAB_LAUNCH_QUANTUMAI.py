"""
🚀 QUANTUM AI DASHBOARD LAUNCHER
Exact path: /content/drive/MyDrive/quantumai
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
print("\n🔍 Searching for dashboard in /content/drive/MyDrive/quantumai...")

# Possible locations
possible_paths = [
    '/content/drive/MyDrive/quantumai',
    '/content/drive/MyDrive/quantumai/backend/modules',
    '/content/drive/MyDrive/quantumai/modules',
]

found_dashboard = None
modules_dir = None

for base_path in possible_paths:
    if os.path.exists(base_path):
        print(f"   ✅ Checking: {base_path}")
        
        # Look for dashboard file
        dashboard_file = os.path.join(base_path, 'ULTIMATE_DASHBOARD_INTEGRATED.py')
        if os.path.exists(dashboard_file):
            found_dashboard = dashboard_file
            modules_dir = base_path
            print(f"   🎯 FOUND DASHBOARD!")
            break
        
        # Also check subdirectories
        for root, dirs, files in os.walk(base_path, maxdepth=3):
            if 'ULTIMATE_DASHBOARD_INTEGRATED.py' in files:
                found_dashboard = os.path.join(root, 'ULTIMATE_DASHBOARD_INTEGRATED.py')
                modules_dir = root
                print(f"   🎯 FOUND DASHBOARD: {modules_dir}")
                break
        
        if found_dashboard:
            break
    else:
        print(f"   ❌ Not found: {base_path}")

if not found_dashboard:
    print("\n❌ ERROR: Could not find ULTIMATE_DASHBOARD_INTEGRATED.py")
    print("\n📂 Let's see what's in /content/drive/MyDrive/quantumai:")
    
    if os.path.exists('/content/drive/MyDrive/quantumai'):
        contents = os.listdir('/content/drive/MyDrive/quantumai')
        print(f"\n   Found {len(contents)} items:")
        for item in sorted(contents)[:20]:
            item_path = os.path.join('/content/drive/MyDrive/quantumai', item)
            if os.path.isdir(item_path):
                print(f"   📁 {item}")
                # Check what's inside
                try:
                    sub_items = os.listdir(item_path)
                    if 'ULTIMATE_DASHBOARD_INTEGRATED.py' in sub_items:
                        print(f"      🎯 Dashboard is here!")
                except:
                    pass
            else:
                print(f"   📄 {item}")
    else:
        print("\n❌ /content/drive/MyDrive/quantumai does not exist!")
        print("\n💡 Please tell me the exact folder name (case-sensitive)")
    
    sys.exit(1)

# ============================================================================
# 3. SETUP PATHS
# ============================================================================
print(f"\n✅ Dashboard found: {found_dashboard}")
print(f"✅ Modules directory: {modules_dir}")

os.chdir(modules_dir)
sys.path.insert(0, modules_dir)

# Show what's available
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
]

subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q'] + packages,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)
print("✅ Dependencies installed")

# ============================================================================
# 5. START STREAMLIT
# ============================================================================
print("\n🚀 Starting Streamlit in background...")

def run_streamlit():
    """Run Streamlit server"""
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

# Start in background thread
streamlit_thread = threading.Thread(target=run_streamlit, daemon=True)
streamlit_thread.start()

print("✅ Streamlit starting...")
print("⏳ Waiting for server to initialize...")
time.sleep(8)

# ============================================================================
# 6. CREATE NGROK TUNNEL
# ============================================================================
print("\n🌐 Creating public URL with ngrok...")

try:
    from pyngrok import ngrok
    
    # Use your saved token
    NGROK_TOKEN = '35jIAcsNyWEBfkEE5BOs9CKjzUH_5TtS5hwfvk3XurHKMKFLC'
    ngrok.set_auth_token(NGROK_TOKEN)
    
    # Create tunnel
    public_url = ngrok.connect(8501)
    
    print("\n" + "="*80)
    print("✅ DASHBOARD IS LIVE!")
    print("="*80)
    print(f"\n🌍 PUBLIC URL: {public_url}")
    print("\n📱 Click the link above to access your dashboard!")
    print("\n💡 Features:")
    print("   • Institutional Ensemble Engine")
    print("   • Real-time signals & scoring")
    print("   • Paper trading portfolio")
    print("   • Advanced charting")
    print("   • All ML scanners")
    print("\n💡 Dashboard will run until you stop this cell")
    print("   To stop: Runtime → Interrupt execution")
    print("\n" + "="*80)
    
    # Keep running
    print("\n⏳ Dashboard running... (monitoring for errors)")
    
    error_count = 0
    while True:
        time.sleep(1)
        
        # Check if streamlit thread is still alive
        if not streamlit_thread.is_alive():
            error_count += 1
            if error_count > 5:
                print("\n⚠️  Streamlit appears to have stopped!")
                print("   Check the dashboard for any errors")
                break
        
except KeyboardInterrupt:
    print("\n\n🛑 Dashboard stopped by user")
    
except Exception as e:
    print(f"\n❌ Error creating ngrok tunnel: {e}")
    print("\n💡 FALLBACK: Streamlit is running locally on port 8501")
    print("   You can use Colab's built-in port forwarding")
    
    # Keep Streamlit running
    try:
        print("\n⏳ Streamlit still running locally...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopped")

print("\n" + "="*80)
print("Dashboard session ended")
print("="*80)

