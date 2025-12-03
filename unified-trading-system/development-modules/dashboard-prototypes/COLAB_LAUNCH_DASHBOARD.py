"""
🚀 QUANTUM AI COCKPIT - COLAB STREAMLIT LAUNCHER
Run your complete dashboard in Google Colab with a public URL!

Colab Pro gives you:
- Longer runtime (24 hours vs 12)
- Better GPU/CPU
- Background execution
- Public URL for Streamlit!
"""

# ============================================================================
# CELL 1: Mount Drive & Install Dependencies
# ============================================================================
print("="*80)
print("🚀 LAUNCHING QUANTUM AI COCKPIT IN COLAB")
print("="*80 + "\n")

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
print("✅ Drive mounted\n")

# Install dependencies
print("📦 Installing dependencies (2-3 minutes)...")
!pip install -q streamlit plotly yfinance beautifulsoup4 requests pandas numpy scipy scikit-learn lightgbm xgboost statsmodels duckduckgo-search python-dotenv pyngrok

print("✅ All dependencies installed\n")

# ============================================================================
# CELL 2: Setup Project Path
# ============================================================================
import sys
import os

# Your Google Drive path
PROJECT_ROOT = '/content/drive/MyDrive/Quantum_AI_Cockpit'
MODULES_DIR = f'{PROJECT_ROOT}/backend/modules'

# Add to Python path
if MODULES_DIR not in sys.path:
    sys.path.insert(0, MODULES_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Change to project directory
os.chdir(PROJECT_ROOT)

print(f"📁 Working directory: {os.getcwd()}")
print(f"📦 Modules path: {MODULES_DIR}")
print("✅ Paths configured\n")

# ============================================================================
# CELL 3: Setup Streamlit for Colab (ngrok tunnel)
# ============================================================================
print("🌐 Setting up public URL for Streamlit...\n")

# Install pyngrok for public URL
!pip install -q pyngrok

from pyngrok import ngrok
import subprocess
import time

# Kill any existing Streamlit processes
!pkill -f streamlit

# Configure ngrok (optional: add your auth token for longer sessions)
# ngrok.set_auth_token("YOUR_NGROK_TOKEN")  # Get free token at ngrok.com

print("✅ Tunnel setup ready\n")

# ============================================================================
# CELL 4: Launch Dashboard with Public URL
# ============================================================================
print("="*80)
print("🚀 LAUNCHING STREAMLIT DASHBOARD")
print("="*80 + "\n")

# Start Streamlit in background
streamlit_process = subprocess.Popen([
    'streamlit', 'run', 
    'FINAL_PROFIT_DASHBOARD.py',
    '--server.port', '8501',
    '--server.headless', 'true',
    '--server.enableCORS', 'false',
    '--server.enableXsrfProtection', 'false'
])

# Wait for Streamlit to start
print("⏳ Starting Streamlit server...")
time.sleep(10)

# Create ngrok tunnel
public_url = ngrok.connect(8501)

print("\n" + "="*80)
print("✅ DASHBOARD IS LIVE!")
print("="*80)
print(f"\n🌐 PUBLIC URL: {public_url}")
print("\n📱 Click the link above to access your dashboard!")
print("\n⚠️  IMPORTANT:")
print("   • Keep this Colab tab open")
print("   • Colab Pro: 24-hour runtime")
print("   • Dashboard will stop if Colab disconnects")
print("\n" + "="*80)

# Keep the process running
print("\n⏸️  Dashboard is running... (press Stop to end)\n")

# Monitor process
try:
    streamlit_process.wait()
except KeyboardInterrupt:
    print("\n🛑 Stopping dashboard...")
    streamlit_process.kill()
    ngrok.disconnect(public_url)
    print("✅ Dashboard stopped")

# ============================================================================
# ALTERNATIVE: Streamlit Cloud Deployment (Recommended for 24/7)
# ============================================================================
"""
FOR PERMANENT DEPLOYMENT (runs 24/7, even when Colab is off):

1. Create GitHub repo with your code
2. Go to share.streamlit.io
3. Connect your GitHub repo
4. Deploy!

Advantages:
✅ Runs 24/7 (not just when Colab is on)
✅ Free tier available
✅ Better performance
✅ Automatic updates from GitHub

Colab is great for testing, Streamlit Cloud is great for production!
"""

