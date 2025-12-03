"""
🚀 QUANTUM AI COCKPIT - Colab Deployment Script

This script launches the Streamlit dashboard in Google Colab with ngrok tunneling.

Usage:
    !python LAUNCH_DASHBOARD_COLAB.py
"""

import os
import sys
import subprocess
import time

print("=" * 80)
print("🏆 QUANTUM AI COCKPIT v2.0 - Dashboard Launcher")
print("=" * 80)

# Step 1: Install dependencies
print("\n📦 Installing dependencies...")
subprocess.run([
    "pip", "install", "-q",
    "streamlit", "plotly", "pandas", "numpy",
    "lightgbm", "xgboost", "scikit-learn",
    "yfinance", "pyngrok"
], check=False)

print("✅ Dependencies installed")

# Step 2: Set ngrok token
NGROK_TOKEN = "35jIAcsNyWEBfkEE5BOs9CKjzUH_5TtS5hwfvk3XurHKMKFLC"
os.environ['NGROK_AUTH_TOKEN'] = NGROK_TOKEN

print(f"✅ Ngrok token configured")

# Step 3: Start Streamlit in background
print("\n🚀 Starting Streamlit dashboard...")

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
dashboard_path = os.path.join(script_dir, "QUANTUM_AI_ULTIMATE_DASHBOARD_V2.py")

# Start Streamlit in background
streamlit_process = subprocess.Popen([
    "streamlit", "run", dashboard_path,
    "--server.port", "8501",
    "--server.headless", "true",
    "--server.enableCORS", "false",
    "--server.enableXsrfProtection", "false"
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("✅ Streamlit server started")

# Wait for Streamlit to start
print("⏳ Waiting for Streamlit to initialize...")
time.sleep(5)

# Step 4: Create ngrok tunnel
print("\n🌍 Creating ngrok tunnel...")

from pyngrok import ngrok

try:
    ngrok.set_auth_token(NGROK_TOKEN)
    public_url = ngrok.connect(8501)
    
    print("\n" + "=" * 80)
    print("✅ DASHBOARD READY!")
    print("=" * 80)
    print(f"\n🌍 Public URL: {public_url}")
    print(f"\n📱 Access your dashboard at the URL above")
    print("\n💡 Tips:")
    print("   - Dashboard auto-refreshes every 30 seconds (if enabled)")
    print("   - All AI modules load on first access (may take 10-20 seconds)")
    print("   - Use 'AI Alerts' tab for high-priority signals")
    print("   - Use 'Stock Lookup' for detailed analysis")
    print("\n⚠️  Keep this cell running to maintain the tunnel!")
    print("=" * 80)
    
    # Keep the script running
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down dashboard...")
        ngrok.disconnect(public_url)
        streamlit_process.terminate()
        print("✅ Dashboard stopped")

except Exception as e:
    print(f"\n❌ Error creating tunnel: {e}")
    print("\n🔧 Troubleshooting:")
    print("   1. Verify ngrok token is correct")
    print("   2. Check if port 8501 is available")
    print("   3. Try restarting the Colab runtime")
    streamlit_process.terminate()
    sys.exit(1)

