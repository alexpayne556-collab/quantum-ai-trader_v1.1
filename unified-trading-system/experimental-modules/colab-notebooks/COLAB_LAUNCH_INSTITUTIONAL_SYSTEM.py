"""
🚀 COLAB LAUNCHER - INSTITUTIONAL SYSTEM
========================================
One-click setup for the complete institutional ensemble system

This will:
1. Mount Google Drive
2. Check for required files
3. Install dependencies
4. Let you choose: Backtest OR Dashboard OR Both
"""

import os
import subprocess
import sys
from pathlib import Path

print("="*80)
print("🏆 INSTITUTIONAL ENSEMBLE SYSTEM - COLAB LAUNCHER")
print("="*80)

# Check Python version
import sys
python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
print(f"\n🐍 Python {python_version} (Colab default: Python 3.12)")

# ============================================================================
# STEP 1: MOUNT DRIVE
# ============================================================================
print("\n1️⃣ Mounting Google Drive...")
try:
    from google.colab import drive
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')
    print("✅ Drive mounted")
except:
    print("⚠️  Not in Colab environment")

# ============================================================================
# STEP 2: SETUP PATHS
# ============================================================================
print("\n2️⃣ Setting up paths...")

BASE_DIR = Path('/content/drive/MyDrive/QuantumAI')
MODULES_DIR = BASE_DIR / 'backend' / 'modules'

# Create directories if they don't exist
BASE_DIR.mkdir(parents=True, exist_ok=True)
MODULES_DIR.mkdir(parents=True, exist_ok=True)

print(f"✅ Base directory: {BASE_DIR}")
print(f"✅ Modules directory: {MODULES_DIR}")

# Add to path
sys.path.insert(0, str(MODULES_DIR))

# ============================================================================
# STEP 3: CHECK FOR REQUIRED FILES
# ============================================================================
print("\n3️⃣ Checking for required files...")

required_files = {
    'INSTITUTIONAL_ENSEMBLE_ENGINE.py': 'Core ensemble engine (1000+ lines)',
    'BACKTEST_INSTITUTIONAL_ENSEMBLE.py': 'Backtest framework (600+ lines)',
    'ULTIMATE_INSTITUTIONAL_DASHBOARD.py': 'Complete dashboard',
}

missing_files = []
found_files = []

os.chdir(MODULES_DIR)

for filename, description in required_files.items():
    filepath = MODULES_DIR / filename
    if filepath.exists():
        print(f"  ✅ {filename}")
        found_files.append(filename)
    else:
        print(f"  ❌ {filename} - {description}")
        missing_files.append(filename)

if missing_files:
    print(f"\n⚠️  MISSING {len(missing_files)} FILE(S)!")
    print("\n📋 UPLOAD INSTRUCTIONS:")
    print(f"\n1. Open Google Drive in browser")
    print(f"2. Navigate to: {MODULES_DIR}")
    print(f"3. Upload these files from your local machine:")
    for f in missing_files:
        print(f"   - {f}")
    print(f"\n4. Then re-run this script!")
    print("\n" + "="*80)
    sys.exit(1)

print(f"\n✅ All required files found!")

# ============================================================================
# STEP 4: GPU DETECTION & SETUP
# ============================================================================
print("\n4️⃣ Setting up GPU acceleration...")

# Run GPU setup script if available
gpu_setup_file = MODULES_DIR / 'COLAB_GPU_SETUP.py'
if gpu_setup_file.exists():
    print("   Running GPU setup script...")
    try:
        exec(open(gpu_setup_file).read())
        print("✅ GPU setup complete")
    except Exception as e:
        print(f"⚠️  GPU setup error: {e}")
        print("   Continuing without GPU optimization...")
else:
    # Fallback: basic GPU detection
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU Detected: {gpu_name} ({gpu_memory:.1f} GB)")
            print(f"   GPU will be used for model training and inference")
            USE_GPU = True
        else:
            print("⚠️  No GPU detected - using CPU")
            USE_GPU = False
    except:
        print("⚠️  PyTorch not installed - will install with GPU support")
        USE_GPU = None

# ============================================================================
# STEP 5: INSTALL DEPENDENCIES
# ============================================================================
print("\n5️⃣ Installing dependencies...")

dependencies = [
    'streamlit',
    'plotly',
    'yfinance',
    'pandas',
    'numpy',
    'ta',
    'tqdm',
    'pyngrok'
]

# Add GPU-accelerated libraries if GPU available
if USE_GPU or USE_GPU is None:
    # Always install PyTorch (will use GPU if available)
    dependencies.extend(['torch', 'torchvision', 'torchaudio'])
    print("   Installing GPU-accelerated libraries (PyTorch)...")

print("Installing:", ', '.join(dependencies))
subprocess.run(['pip', 'install', '-q'] + dependencies, 
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("✅ Dependencies installed")

# ============================================================================
# STEP 6: VERIFY DATA_ORCHESTRATOR (CRITICAL!)
# ============================================================================
print("\n6️⃣ Verifying data_orchestrator.py...")

data_orch_file = MODULES_DIR / 'data_orchestrator.py'
if data_orch_file.exists():
    # Check if file has the required methods
    with open(data_orch_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    required_methods = ['get_returns', 'get_ma', 'to_scalar', 'ScalarExtractor']
    missing_methods = []
    
    for method in required_methods:
        if method not in content:
            missing_methods.append(method)
    
    if missing_methods:
        print(f"❌ data_orchestrator.py is OUTDATED!")
        print(f"   Missing methods: {', '.join(missing_methods)}")
        print(f"\n📋 ACTION REQUIRED:")
        print(f"   1. Upload the UPDATED data_orchestrator.py from your local machine")
        print(f"   2. File location: backend/modules/data_orchestrator.py")
        print(f"   3. Upload to: {MODULES_DIR}")
        print(f"   4. Then RESTART RUNTIME and re-run this script")
        print("\n" + "="*80)
        sys.exit(1)
    else:
        print("✅ data_orchestrator.py verified (has all required methods)")
else:
    print("❌ data_orchestrator.py NOT FOUND!")
    print(f"   Upload to: {MODULES_DIR}")
    sys.exit(1)

# ============================================================================
# STEP 7: CLEAR CACHE (IMPORTANT!)
# ============================================================================
print("\n7️⃣ Clearing cached modules...")

# Remove cached modules to ensure fresh imports
modules_to_clear = [
    'INSTITUTIONAL_ENSEMBLE_ENGINE',
    'BACKTEST_INSTITUTIONAL_ENSEMBLE',
    'ULTIMATE_INSTITUTIONAL_DASHBOARD',
]

for module_name in modules_to_clear:
    keys_to_remove = [k for k in sys.modules.keys() if module_name.lower() in k.lower()]
    for key in keys_to_remove:
        del sys.modules[key]

print("✅ Cache cleared")

# ============================================================================
# STEP 8: QUICK IMPORT TEST
# ============================================================================
print("\n8️⃣ Testing imports...")

try:
    # Test DataOrchestrator first (most critical)
    from data_orchestrator import DataOrchestrator
    orch = DataOrchestrator()
    
    # Verify methods exist
    if not hasattr(orch, 'get_returns'):
        print("❌ DataOrchestrator missing get_returns method!")
        print("   File needs to be re-uploaded. RESTART RUNTIME after upload.")
        sys.exit(1)
    
    print("✅ DataOrchestrator imported and verified")
    
    # Test ensemble engine
    from INSTITUTIONAL_ENSEMBLE_ENGINE import InstitutionalEnsembleEngine, Signal
    print("✅ Ensemble engine imported successfully")
    
    # Quick test
    engine = InstitutionalEnsembleEngine()
    print(f"✅ Engine initialized (Weights: {len(engine.current_weights)} modules)")
    
except Exception as e:
    print(f"❌ Import error: {str(e)}")
    print("\nThis might be a file issue. Try re-uploading the files.")
    print("   IMPORTANT: After uploading, RESTART RUNTIME before re-running!")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 9: CHOOSE WHAT TO RUN
# ============================================================================
print("\n" + "="*80)
print("🎯 SYSTEM READY!")
print("="*80)
print("\n💡 TIP: Terminal is now free for all users!")
print("   Use it for debugging: Runtime → Terminal")
print("\nWhat would you like to do?\n")
print("Option 1: 📊 Run Backtest (5-10 minutes)")
print("          - Tests ensemble on 6 months of data")
print("          - Validates 60-70% win rate target")
print("          - Generates performance report")
print("")
print("Option 2: 🚀 Launch Dashboard (Instant)")
print("          - Live trading interface")
print("          - Real-time signals")
print("          - Paper trading")
print("          - Performance tracking")
print("")
print("Option 3: 🎯 Both (Recommended)")
print("          - Run backtest first (saves learned weights)")
print("          - Then launch dashboard with optimized weights")
print("")
print("="*80)

choice = input("\nEnter your choice (1, 2, or 3): ").strip()

print("\n" + "="*80)

# ============================================================================
# OPTION 1: BACKTEST
# ============================================================================
if choice == "1" or choice == "3":
    print("📊 RUNNING BACKTEST")
    print("="*80)
    print("\n⏱️  This will take 5-10 minutes...")
    print("💡 You'll see progress bars and real-time updates\n")
    
    try:
        # Clear cache again before importing backtest
        keys_to_remove = [k for k in sys.modules.keys() if 'backtest' in k.lower()]
        for key in keys_to_remove:
            del sys.modules[key]
        
        # Import and run backtest
        from BACKTEST_INSTITUTIONAL_ENSEMBLE import BacktestEngine
        
        backtest = BacktestEngine()
        results = backtest.run_backtest()
        
        if results:
            backtest.print_results(results)
            backtest.save_results(results)
            backtest.ensemble.save_weights('ensemble_weights.json')
            
            print("\n✅ BACKTEST COMPLETE!")
            print(f"\n📁 Results saved:")
            print(f"   - backtest_results.json")
            print(f"   - backtest_trades.csv")
            print(f"   - ensemble_weights.json (learned weights!)")
            
            # Quick summary
            if results['win_rate'] >= 0.60:
                print(f"\n🎉 TARGET MET! Win rate: {results['win_rate']:.1%} (Target: 60-70%)")
            else:
                print(f"\n🎯 Building... Win rate: {results['win_rate']:.1%} (Target: 60-70%)")
                print(f"   System will improve as it learns from more trades!")
        
    except Exception as e:
        print(f"\n❌ Backtest error: {str(e)}")
        import traceback
        traceback.print_exc()

# ============================================================================
# OPTION 2: DASHBOARD
# ============================================================================
if choice == "2" or choice == "3":
    if choice == "3":
        print("\n" + "="*80)
    
    print("🚀 LAUNCHING DASHBOARD")
    print("="*80)
    
    # Setup ngrok
    try:
        from pyngrok import ngrok
        
        token_file = BASE_DIR / 'ngrok_token.txt'
        if token_file.exists():
            with open(token_file, 'r') as f:
                token = f.read().strip()
            ngrok.set_auth_token(token)
            print("✅ ngrok authenticated")
            
            # Start tunnel
            public_url = ngrok.connect(8501)
            print(f"\n🌐 PUBLIC URL: {public_url}")
            print("\n✅ Click the link above to access your dashboard!")
        else:
            print("\n⚠️  No ngrok token found")
            print(f"   Get token: https://dashboard.ngrok.com/get-started/your-authtoken")
            print(f"   Save to: {token_file}")
            print("\n   Dashboard will run locally only (Colab internal)")
    except Exception as e:
        print(f"⚠️  ngrok setup: {str(e)}")
        print("   Dashboard will run without public URL")
    
    print("\n" + "="*80)
    print("🎯 STARTING STREAMLIT DASHBOARD")
    print("="*80)
    print("\n💡 The dashboard is now launching...")
    print("💡 You'll see a URL appear - click it to open the dashboard")
    print("\n" + "="*80)
    
    # Launch Streamlit
    dashboard_file = MODULES_DIR / 'ULTIMATE_INSTITUTIONAL_DASHBOARD.py'
    subprocess.run(['streamlit', 'run', str(dashboard_file), 
                   '--server.port=8501', '--server.headless=true'])

# ============================================================================
# INVALID CHOICE
# ============================================================================
if choice not in ["1", "2", "3"]:
    print(f"\n❌ Invalid choice: {choice}")
    print("Please run the script again and choose 1, 2, or 3")

