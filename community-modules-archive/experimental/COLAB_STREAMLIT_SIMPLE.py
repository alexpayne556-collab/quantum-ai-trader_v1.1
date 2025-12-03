"""
🚀 SIMPLE STREAMLIT IN COLAB - NO SETUP NEEDED!
Just run these cells and get a public URL!
"""

# ============================================================================
# CELL 1: Install Everything
# ============================================================================
!pip install -q streamlit plotly yfinance beautifulsoup4 requests pandas numpy scipy scikit-learn lightgbm xgboost statsmodels duckduckgo-search python-dotenv
!npm install -g localtunnel

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# ============================================================================
# CELL 2: Setup & Launch
# ============================================================================
import sys
import os

# Setup paths
PROJECT_ROOT = '/content/drive/MyDrive/Quantum_AI_Cockpit'
sys.path.insert(0, f'{PROJECT_ROOT}/backend/modules')
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

print(f"✅ Working in: {os.getcwd()}\n")

# ============================================================================
# CELL 3: Get Public URL
# ============================================================================
import subprocess
from threading import Thread

def run_streamlit():
    """Run Streamlit in background"""
    subprocess.run([
        'streamlit', 'run',
        'FINAL_PROFIT_DASHBOARD.py',
        '--server.port', '8501',
        '--server.headless', 'true'
    ])

def run_tunnel():
    """Create public tunnel"""
    import time
    time.sleep(10)  # Wait for Streamlit to start
    subprocess.run(['lt', '--port', '8501'])

# Start Streamlit
streamlit_thread = Thread(target=run_streamlit)
streamlit_thread.start()

# Start tunnel (this will show your public URL)
run_tunnel()

# ============================================================================
# EVEN SIMPLER: Streamlit Community Cloud (FREE 24/7!)
# ============================================================================
"""
BEST OPTION - Run 24/7 without Colab:

1. Upload your code to GitHub
2. Go to: share.streamlit.io
3. Click "Deploy an app"
4. Connect GitHub repo
5. Select FINAL_PROFIT_DASHBOARD.py
6. Click Deploy!

Result: 
✅ Free forever
✅ Runs 24/7 (not just when Colab is on)
✅ Fast & reliable
✅ Your own URL: https://yourapp.streamlit.app

Takes 2 minutes to setup!
"""

