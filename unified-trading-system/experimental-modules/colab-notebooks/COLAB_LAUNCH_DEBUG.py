"""
🐛 DEBUG LAUNCHER - Shows exactly what's happening
"""

import subprocess
import sys
import os
from pathlib import Path

print("="*80)
print("🐛 DEBUG LAUNCHER - STREAMLIT DIAGNOSTICS")
print("="*80)

# Setup paths
MODULES_DIR = Path('/content/drive/MyDrive/QuantumAI/backend/modules')
DASHBOARD_FILE = MODULES_DIR / 'ULTIMATE_DASHBOARD_INTEGRATED.py'

print(f"\n📂 Modules directory: {MODULES_DIR}")
print(f"📄 Dashboard file: {DASHBOARD_FILE}")
print(f"   Exists: {DASHBOARD_FILE.exists()}")

if DASHBOARD_FILE.exists():
    size = DASHBOARD_FILE.stat().st_size / 1024
    print(f"   Size: {size:.1f} KB")

# Change to modules directory
os.chdir(MODULES_DIR)
print(f"\n📁 Changed to: {os.getcwd()}")

# Check streamlit installation
print("\n🔍 Checking Streamlit...")
result = subprocess.run([sys.executable, '-m', 'streamlit', '--version'], 
                       capture_output=True, text=True)
if result.returncode == 0:
    print(f"✅ Streamlit installed: {result.stdout.strip()}")
else:
    print(f"❌ Streamlit check failed: {result.stderr}")
    print("\n📦 Reinstalling streamlit...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'streamlit'], check=True)

# Try to import the dashboard (check for syntax errors)
print("\n🔍 Testing dashboard import...")
sys.path.insert(0, str(MODULES_DIR))

try:
    with open(DASHBOARD_FILE, 'r') as f:
        code = f.read()
    
    # Try to compile it
    compile(code, str(DASHBOARD_FILE), 'exec')
    print("✅ Dashboard code is valid (no syntax errors)")
except SyntaxError as e:
    print(f"❌ Syntax error in dashboard: {e}")
    print(f"   Line {e.lineno}: {e.text}")
    sys.exit(1)
except Exception as e:
    print(f"⚠️  Could not test import: {e}")

# Check for required imports
print("\n🔍 Checking required packages...")
required_packages = [
    'streamlit',
    'pandas',
    'numpy',
    'plotly',
    'yfinance',
]

for package in required_packages:
    try:
        __import__(package)
        print(f"   ✅ {package}")
    except ImportError:
        print(f"   ❌ {package} - MISSING!")

# Check if another streamlit is running
print("\n🔍 Checking for running Streamlit instances...")
result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
streamlit_processes = [line for line in result.stdout.split('\n') if 'streamlit' in line.lower()]
if streamlit_processes:
    print(f"⚠️  Found {len(streamlit_processes)} Streamlit process(es):")
    for proc in streamlit_processes[:3]:
        print(f"   {proc[:100]}")
else:
    print("✅ No other Streamlit processes running")

# Launch with verbose output
print("\n" + "="*80)
print("🚀 LAUNCHING STREAMLIT (with full output)")
print("="*80)
print("\n💡 If you see a URL, click it!")
print("💡 If it hangs, press Ctrl+C and we'll try a different approach")
print("\n")

try:
    # Run streamlit with output shown
    cmd = [
        sys.executable, '-m', 'streamlit', 'run',
        str(DASHBOARD_FILE),
        '--server.port=8501',
        '--server.headless=true',
        '--server.enableCORS=false',
        '--server.enableXsrfProtection=false',
        '--browser.serverAddress=0.0.0.0',
        '--logger.level=info'
    ]
    
    print(f"🔧 Command: {' '.join(cmd)}\n")
    
    # Run without capturing output (so we see everything)
    subprocess.run(cmd)
    
except KeyboardInterrupt:
    print("\n\n🛑 Stopped by user")
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\n💡 Let's try a simpler approach...")

print("\n" + "="*80)


