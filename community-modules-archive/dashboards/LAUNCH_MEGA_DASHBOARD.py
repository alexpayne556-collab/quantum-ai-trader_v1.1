"""
🚀 LAUNCH QUANTUM MEGA DASHBOARD
=================================
Launches the enhanced dashboard with 8+ real modules
"""

import subprocess
import os
from google.colab import drive

print("🚀 LAUNCHING QUANTUM AI MEGA DASHBOARD")
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

# Change to modules directory
MODULES_DIR = '/content/drive/MyDrive/QuantumAI/backend/modules'
os.chdir(MODULES_DIR)
print(f"\n4️⃣ Changed to: {MODULES_DIR}")

# Check dashboard file exists
dashboard_file = 'QUANTUM_MEGA_DASHBOARD.py'
if os.path.exists(dashboard_file):
    print(f"✅ Found {dashboard_file}")
else:
    print(f"❌ {dashboard_file} not found!")
    print("   Falling back to QUANTUM_DASHBOARD_REAL_MODULES.py")
    dashboard_file = 'QUANTUM_DASHBOARD_REAL_MODULES.py'

print("\n" + "="*80)
print("🎯 LAUNCHING MEGA DASHBOARD")
print("="*80)
print("\n💡 Dashboard Features:")
print("   ✅ 3 ML-Powered Scanners")
print("   ✅ 3 Goldmine Scanners (Dark Pool, Insider, Squeeze)")
print("   ✅ 2 Function-Based Forecasters")
print("   ✅ Regime Detection")
print("   ✅ AI Sentiment (if available)")
print("   ✅ Paper Trading")
print("   ✅ Full Analytics")
print("\n" + "="*80)

# Launch Streamlit with ngrok tunnel
try:
    from pyngrok import ngrok
    
    # Try to read token
    token_file = '/content/drive/MyDrive/QuantumAI/ngrok_token.txt'
    if os.path.exists(token_file):
        with open(token_file, 'r') as f:
            token = f.read().strip()
        ngrok.set_auth_token(token)
        public_url = ngrok.connect(8501)
        print(f"\n🌐 PUBLIC URL: {public_url}")
        print("\n✅ Click the link above to access your MEGA dashboard!")
    else:
        print("\n⚠️  No ngrok token - running locally only")
        print(f"   Get token: https://dashboard.ngrok.com/get-started/your-authtoken")
        print(f"   Save to: {token_file}")
except Exception as e:
    print(f"\n⚠️  ngrok: {str(e)}")

print("="*80)

# Run Streamlit
subprocess.run(['streamlit', 'run', dashboard_file, '--server.port=8501', '--server.headless=true'])

