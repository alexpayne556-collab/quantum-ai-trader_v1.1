"""
🚀 LAUNCH QUANTUM DASHBOARD - REAL MODULES
===========================================
Runs the dashboard with YOUR actual working modules in Colab
"""

import subprocess
import os
from google.colab import drive

print("🚀 LAUNCHING QUANTUM AI DASHBOARD")
print("="*80)

# Mount Drive
print("\n1️⃣ Mounting Google Drive...")
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')
    print("✅ Drive mounted")
else:
    print("✅ Drive already mounted")

# Install dependencies
print("\n2️⃣ Installing dependencies...")
subprocess.run(['pip', 'install', '-q', 'streamlit', 'plotly', 'yfinance', 'pandas', 'numpy', 'ta'], 
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("✅ Dependencies installed")

# Install ngrok
print("\n3️⃣ Setting up ngrok...")
subprocess.run(['pip', 'install', '-q', 'pyngrok'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("✅ ngrok ready")

# Get ngrok token
from pyngrok import ngrok

print("\n4️⃣ Starting ngrok tunnel...")
try:
    # Try to read token from file
    token_file = '/content/drive/MyDrive/QuantumAI/ngrok_token.txt'
    if os.path.exists(token_file):
        with open(token_file, 'r') as f:
            token = f.read().strip()
        ngrok.set_auth_token(token)
        print("✅ ngrok authenticated")
    else:
        print("⚠️  No ngrok token found")
        print("   Get your token from: https://dashboard.ngrok.com/get-started/your-authtoken")
        print(f"   Save it to: {token_file}")
except Exception as e:
    print(f"⚠️  ngrok setup: {e}")

# Change to modules directory
MODULES_DIR = '/content/drive/MyDrive/QuantumAI/backend/modules'
os.chdir(MODULES_DIR)
print(f"\n5️⃣ Changed to: {MODULES_DIR}")

# Check dashboard file exists
dashboard_file = 'QUANTUM_DASHBOARD_REAL_MODULES.py'
if os.path.exists(dashboard_file):
    print(f"✅ Found {dashboard_file}")
else:
    print(f"❌ {dashboard_file} not found!")
    print("   Make sure you uploaded it to the modules directory")
    exit(1)

print("\n" + "="*80)
print("🎯 LAUNCHING STREAMLIT DASHBOARD")
print("="*80)
print("\n💡 The dashboard will open in a new window")
print("💡 Use the ngrok URL to access it publicly")
print("\n" + "="*80)

# Launch Streamlit with ngrok tunnel
try:
    public_url = ngrok.connect(8501)
    print(f"\n🌐 PUBLIC URL: {public_url}")
    print("\n✅ Click the link above to access your dashboard!")
    print("="*80)
except:
    print("\n⚠️  Running without ngrok (local only)")

# Run Streamlit
subprocess.run(['streamlit', 'run', dashboard_file, '--server.port=8501', '--server.headless=true'])

