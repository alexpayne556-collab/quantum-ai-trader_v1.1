"""
🏆 COLAB LAUNCHER - QUANTUM AI ULTIMATE PRO DASHBOARD
=====================================================
Launches the dashboard with ngrok tunnel using your stored API key

PASTE THIS IN COLAB AND RUN!
"""

# ==============================================================================
# CELL 1: Mount Drive & Setup
# ==============================================================================

from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.insert(0, '/content/drive/MyDrive/QuantumAI/backend/modules')

print("✅ Drive mounted and path configured!")

# ==============================================================================
# CELL 2: Install Dependencies
# ==============================================================================

print("📦 Installing dependencies...")
import subprocess
subprocess.run(['pip', 'install', 'streamlit', 'yfinance', 'plotly', 'nest-asyncio', 'pyngrok', 'python-dotenv', '-q'], check=True)
print("✅ All dependencies installed!")

# ==============================================================================
# CELL 3: Load ngrok Token from Your ENV File
# ==============================================================================

import os
from dotenv import load_dotenv

# Load your API keys from env file
env_path = '/content/drive/MyDrive/QuantumAI/backend/modules/api_keys.env'
load_dotenv(env_path)

# Get ngrok token
ngrok_token = os.getenv('NGROK_AUTH_TOKEN')

if not ngrok_token or ngrok_token == 'YOUR_NGROK_TOKEN_HERE':
    print("⚠️ WARNING: ngrok token not found in api_keys.env!")
    print("Please add your ngrok token to:")
    print(env_path)
    print("\nGet free token from: https://ngrok.com")
    
    # Prompt for manual entry
    ngrok_token = input("Or paste your ngrok token here: ").strip()
else:
    print("✅ ngrok token loaded from api_keys.env!")

# ==============================================================================
# CELL 4: Launch Dashboard with ngrok
# ==============================================================================

from pyngrok import ngrok
import subprocess
import time

# Set ngrok auth token
ngrok.set_auth_token(ngrok_token)

print("🚀 Starting Streamlit dashboard...")

# Start Streamlit in background
proc = subprocess.Popen([
    'streamlit', 'run',
    '/content/drive/MyDrive/QuantumAI/backend/modules/QUANTUM_AI_ULTIMATE_PRO_DASHBOARD.py',
    '--server.port', '8501',
    '--server.headless', 'true',
    '--server.enableCORS', 'false',
    '--server.enableXsrfProtection', 'false'
])

# Wait for Streamlit to start
print("⏳ Waiting for Streamlit to start...")
time.sleep(10)

# Create ngrok tunnel
print("🌐 Creating public tunnel with ngrok...")
public_url = ngrok.connect(8501)

print("\n" + "="*80)
print("🎉 DASHBOARD IS LIVE!")
print("="*80)
print(f"\n🌐 PUBLIC URL: {public_url}")
print("\n✅ Click the link above to access your Quantum AI Ultimate Pro Dashboard!")
print("\n📊 Features available:")
print("  - 🔍 Stock Lookup (AI Score 0-10)")
print("  - 🤖 AI Recommender (Plain English explanations)")
print("  - 📊 Market Scanner (Top opportunities)")
print("  - 🔮 21-Day Forecast (Elite ensemble)")
print("  - 📈 Performance Tracking")
print("\n⚠️ Keep this cell running! Don't stop it or the dashboard will go offline.")
print("="*80)

# Keep running
try:
    proc.wait()
except KeyboardInterrupt:
    print("\n🛑 Shutting down dashboard...")
    proc.terminate()
    ngrok.kill()


