"""
🚀 STREAMLIT + NGROK LAUNCHER
Shows the public URL immediately!
"""

import subprocess
import sys
import os
import time
from pathlib import Path
import threading

print("="*80)
print("🚀 STREAMLIT + NGROK LAUNCHER")
print("="*80)

# ============================================================================
# 1. SETUP
# ============================================================================
print("\n📦 Installing packages...")
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 
                'streamlit', 'plotly', 'yfinance', 'pyngrok', 'nest-asyncio'], 
               check=False)
print("✅ Packages installed")

# ============================================================================
# 2. PATHS
# ============================================================================
MODULES_DIR = Path('/content/drive/MyDrive/QuantumAI/backend/modules')
DASHBOARD_FILE = MODULES_DIR / 'ULTIMATE_DASHBOARD_INTEGRATED.py'

os.chdir(MODULES_DIR)
sys.path.insert(0, str(MODULES_DIR))

print(f"\n📂 Dashboard: {DASHBOARD_FILE.name}")
print(f"   Size: {DASHBOARD_FILE.stat().st_size / 1024:.1f} KB")

# ============================================================================
# 3. CONFIGURE NGROK
# ============================================================================
print("\n🌐 Setting up ngrok...")

ngrok_token_file = Path('/content/drive/MyDrive/QuantumAI/ngrok_token.txt')

if ngrok_token_file.exists():
    with open(ngrok_token_file, 'r') as f:
        ngrok_token = f.read().strip()
    
    from pyngrok import ngrok
    ngrok.set_auth_token(ngrok_token)
    print("✅ Ngrok configured")
else:
    print("⚠️  No ngrok token - dashboard will be Colab-local only")
    print(f"   Save token to: {ngrok_token_file}")

# ============================================================================
# 4. START STREAMLIT IN BACKGROUND
# ============================================================================
print("\n🚀 Starting Streamlit in background...")

def run_streamlit():
    """Run streamlit in a separate thread"""
    subprocess.run([
        sys.executable, '-m', 'streamlit', 'run',
        str(DASHBOARD_FILE),
        '--server.port=8501',
        '--server.headless=true',
        '--server.enableCORS=false',
        '--server.enableXsrfProtection=false',
        '--browser.serverAddress=localhost',
        '--logger.level=error'
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Start streamlit in background thread
streamlit_thread = threading.Thread(target=run_streamlit, daemon=True)
streamlit_thread.start()

print("✅ Streamlit starting...")
print("⏳ Waiting for server to be ready...")

# Wait for streamlit to start
time.sleep(5)

# ============================================================================
# 5. CREATE NGROK TUNNEL
# ============================================================================
print("\n🌐 Creating public tunnel...")

try:
    from pyngrok import ngrok
    
    # Create tunnel
    public_url = ngrok.connect(8501)
    
    print("\n" + "="*80)
    print("✅ DASHBOARD IS LIVE!")
    print("="*80)
    print(f"\n🌍 PUBLIC URL: {public_url}")
    print("\n📱 Click the link above to access your dashboard!")
    print("\n💡 Tips:")
    print("   • Dashboard will stay running until you stop this cell")
    print("   • To stop: Runtime → Interrupt execution")
    print("   • URL is accessible from any device")
    print("\n" + "="*80)
    
    # Keep running
    print("\n⏳ Dashboard is running... (Press Ctrl+C to stop)")
    
    while True:
        time.sleep(1)
        
except KeyboardInterrupt:
    print("\n\n🛑 Dashboard stopped")
except Exception as e:
    print(f"\n❌ Ngrok error: {e}")
    
    # Fallback: show Colab URL
    print("\n💡 FALLBACK: Use Colab's local proxy")
    print("\n📍 URL: Click the 'Web' button in the Colab output")
    print("   Or try: https://localhost:8501")
    
    # Keep streamlit running
    print("\n⏳ Streamlit is still running locally...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopped")


