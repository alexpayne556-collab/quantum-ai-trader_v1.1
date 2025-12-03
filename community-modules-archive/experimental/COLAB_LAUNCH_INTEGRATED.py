"""
🚀 COLAB LAUNCHER - INTEGRATED DASHBOARD
=========================================
Launches the bulletproof dashboard that uses YOUR data infrastructure.
No mocks, no placeholders, production-ready!
"""

import subprocess
import sys
import os
from pathlib import Path

print("="*80)
print("🚀 QUANTUM AI INTEGRATED DASHBOARD LAUNCHER")
print("="*80)

# ============================================================================
# 1. MOUNT GOOGLE DRIVE
# ============================================================================
print("\n📁 Step 1: Mounting Google Drive...")
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("✅ Drive mounted")
except:
    print("⚠️  Not running in Colab or drive already mounted")

# ============================================================================
# 2. SETUP PATHS
# ============================================================================
print("\n📂 Step 2: Setting up paths...")

QUANTUM_DIR = Path('/content/drive/MyDrive/QuantumAI')
MODULES_DIR = QUANTUM_DIR / 'backend' / 'modules'

# Create directories if needed
QUANTUM_DIR.mkdir(parents=True, exist_ok=True)
MODULES_DIR.mkdir(parents=True, exist_ok=True)

print(f"✅ Quantum AI directory: {QUANTUM_DIR}")
print(f"✅ Modules directory: {MODULES_DIR}")

# ============================================================================
# 3. CHECK FOR REQUIRED FILES
# ============================================================================
print("\n🔍 Step 3: Checking for required files...")

required_files = {
    'Dashboard': MODULES_DIR / 'ULTIMATE_DASHBOARD_INTEGRATED.py',
    'Ensemble Engine': MODULES_DIR / 'INSTITUTIONAL_ENSEMBLE_ENGINE.py',
    'Data Orchestrator': MODULES_DIR / 'data_orchestrator.py',
    'Data Router': MODULES_DIR / 'data_router.py',
}

missing_files = []
for name, path in required_files.items():
    if path.exists():
        print(f"✅ {name}: Found")
    else:
        print(f"❌ {name}: MISSING")
        missing_files.append((name, path))

if missing_files:
    print("\n⚠️  MISSING FILES DETECTED!")
    print("\nPlease upload these files to your Google Drive:")
    for name, path in missing_files:
        print(f"   • {name} → {path}")
    print("\nUpload location: /content/drive/MyDrive/QuantumAI/backend/modules/")
    
    # Continue anyway if only optional files missing
    critical_missing = [name for name, _ in missing_files 
                       if name in ['Dashboard', 'Ensemble Engine']]
    
    if critical_missing:
        print("\n❌ Critical files missing. Cannot continue.")
        sys.exit(1)
    else:
        print("\n⚠️  Optional files missing, continuing with fallback mode...")

# ============================================================================
# 4. INSTALL DEPENDENCIES
# ============================================================================
print("\n📦 Step 4: Installing dependencies...")

dependencies = [
    'streamlit',
    'plotly',
    'yfinance',
    'ta',
    'lightgbm',
    'xgboost',
    'pyngrok',
    'nest-asyncio',
]

print("Installing packages (this may take 1-2 minutes)...")
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + dependencies, 
               check=False)
print("✅ Dependencies installed")

# ============================================================================
# 5. SETUP NGROK (OPTIONAL)
# ============================================================================
print("\n🌐 Step 5: Setting up ngrok tunnel...")

ngrok_token_file = QUANTUM_DIR / 'ngrok_token.txt'

if ngrok_token_file.exists():
    with open(ngrok_token_file, 'r') as f:
        ngrok_token = f.read().strip()
    
    try:
        from pyngrok import ngrok
        ngrok.set_auth_token(ngrok_token)
        print("✅ Ngrok configured")
    except Exception as e:
        print(f"⚠️  Ngrok setup failed: {e}")
        print("   Dashboard will run locally only")
else:
    print("⚠️  No ngrok token found")
    print("   Get token: https://dashboard.ngrok.com/get-started/your-authtoken")
    print(f"   Save to: {ngrok_token_file}")
    print("   Dashboard will run locally only (Colab internal)")

# ============================================================================
# 6. LAUNCH DASHBOARD
# ============================================================================
print("\n" + "="*80)
print("🎯 LAUNCHING INTEGRATED DASHBOARD")
print("="*80)

print("\n💡 Dashboard Features:")
print("   ✅ Uses YOUR data_orchestrator.py (production-grade)")
print("   ✅ Uses YOUR data_router.py (multi-provider)")
print("   ✅ Bulletproof pandas handling (NO Series errors)")
print("   ✅ Institutional ensemble integration")
print("   ✅ Real-time signals with confidence scoring")
print("   ✅ Paper trading portfolio")
print("   ✅ Auto-learning from outcomes")

print("\n🚀 Starting Streamlit...")
print("="*80)

# Change to modules directory
os.chdir(MODULES_DIR)

# Launch Streamlit
dashboard_file = MODULES_DIR / 'ULTIMATE_DASHBOARD_INTEGRATED.py'

if dashboard_file.exists():
    # Run streamlit
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run',
            str(dashboard_file),
            '--server.port=8501',
            '--server.headless=true',
            '--server.enableCORS=false',
            '--server.enableXsrfProtection=false',
            '--browser.serverAddress=0.0.0.0'
        ])
    except KeyboardInterrupt:
        print("\n\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Dashboard error: {e}")
else:
    print(f"❌ Dashboard file not found: {dashboard_file}")
    print("\nPlease ensure ULTIMATE_DASHBOARD_INTEGRATED.py is uploaded to:")
    print(f"   {MODULES_DIR}")

print("\n" + "="*80)
print("✅ SESSION COMPLETE")
print("="*80)


