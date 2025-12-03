"""
🏆 COLAB LAUNCHER - QUANTUM AI ELITE QUANT DASHBOARD v2
======================================================
Launches the enhanced dashboard with pattern intelligence & live learning

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
subprocess.run(['pip', 'install', 'streamlit', 'yfinance', 'plotly', 'nest-asyncio', 'pyngrok', 'python-dotenv', 'scipy', 'transformers', 'torch', 'textblob', 'vaderSentiment', '-q'], check=True)
print("✅ All dependencies installed!")

# ==============================================================================
# CELL 3: Launch Dashboard with ngrok (reads your token from api_keys.env)
# ==============================================================================

import os
from dotenv import load_dotenv
from pyngrok import ngrok
import subprocess
import time

# Load your ngrok token
load_dotenv('/content/drive/MyDrive/QuantumAI/backend/modules/api_keys.env')
ngrok_token = os.getenv('NGROK_AUTH_TOKEN')

if ngrok_token:
    ngrok.set_auth_token(ngrok_token)
    print("✅ ngrok token loaded!")
else:
    print("⚠️ ngrok token not found in api_keys.env")

# Launch Streamlit
proc = subprocess.Popen([
    'streamlit', 'run',
    '/content/drive/MyDrive/QuantumAI/backend/modules/QUANTUM_AI_ELITE_QUANT_DASHBOARD_V2_PATTERN_INTELLIGENCE.py',
    '--server.port', '8501',
    '--server.headless', 'true',
    '--server.enableCORS', 'false',
    '--server.enableXsrfProtection', 'false'
])

print("⏳ Starting elite quant dashboard v2...")
time.sleep(15)

# Create public URL
public_url = ngrok.connect(8501)

print("\n" + "="*80)
print("🎉 QUANTUM AI ELITE QUANT DASHBOARD v2 IS LIVE!")
print("="*80)
print(f"\n🌐 PUBLIC URL: {public_url}")
print("\n✅ Click the link above to access your institutional-grade dashboard!")
print("\n🚀 NEW FEATURES:")
print("  - 🎨 Pattern Intelligence (Intellectia-style)")
print("  - 🤖 Live Learning & NSD Training")
print("  - 📊 10-Factor Quant Scoring")
print("  - 📰 Sentiment Analysis")
print("  - 💡 Institutional AI Explanations")
print("  - 📈 Real-time Backtesting")
print("\n⚠️ Keep this cell running! Don't stop it or the dashboard will go offline.")
print("⚠️ The AI learns from every analysis - it gets smarter over time!")
print("="*80)

# Keep running
try:
    proc.wait()
except KeyboardInterrupt:
    print("\n🛑 Shutting down dashboard...")
    proc.terminate()
    ngrok.kill()
