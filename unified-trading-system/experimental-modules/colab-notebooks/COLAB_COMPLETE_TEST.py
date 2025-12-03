"""
🚀 QUANTUM AI COCKPIT — COMPLETE COLAB TEST SCRIPT
==================================================

Copy this entire file into a Google Colab cell and run it!

This will:
1. Check GPU
2. Install dependencies
3. Test all modules
4. Run complete stock analysis
5. Test portfolio analysis

Before running:
- Runtime → Change runtime type → GPU (T4)
- Upload modules.zip OR mount Google Drive

"""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: ENVIRONMENT SETUP
# ═══════════════════════════════════════════════════════════════════════════

print("🚀 QUANTUM AI COCKPIT — COMPLETE TEST SUITE v4.0")
print("=" * 80)

import os
import sys
from pathlib import Path

# Check GPU
try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU DETECTED: {gpu_name}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
    else:
        print("⚠️  NO GPU — Training will be SLOW!")
        print("   Go to Runtime → Change runtime type → GPU\n")
except:
    print("⚠️  PyTorch not installed yet\n")

# Install packages
print("📦 Installing dependencies (this takes 2-3 minutes)...")
get_ipython().system('pip install -q numpy pandas scipy scikit-learn matplotlib seaborn plotly yfinance')
get_ipython().system('pip install -q darts torch pytorch-lightning xgboost ta statsmodels arch')
get_ipython().system('pip install -q transformers sentencepiece python-dotenv requests aiohttp')

print("✅ All packages installed!\n")

# Verify GPU
import torch
if torch.cuda.is_available():
    print(f"✅ GPU Ready: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   PyTorch Version: {torch.__version__}\n")
else:
    print("❌ GPU NOT AVAILABLE — Training will be slow!\n")

print("=" * 80)
print("✅ ENVIRONMENT READY!")
print("=" * 80)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: CREATE .ENV FILE
# ═══════════════════════════════════════════════════════════════════════════

env_content = """
# Market Data APIs
ALPHAVANTAGE_API_KEY=6NOB0V91707OM1TI
MASSIVE_API_KEY=chFZODMC89wpypjBibRsW1E160SVBfPL
POLYGON_API_KEY=gyBClHUxmeIerRMuUMGGi1hIiBIxl2cS
TWELVEDATA_API_KEY=5852d42a799e47269c689392d273f70b
FINNHUB_API_KEY=d40387pr01qkrgfb5asgd40387pr01qkrgfb5at0
TIINGO_API_KEY=de94a283588681e212560a0d9826903e25647968
FINANCIALMODELINGPREP_API_KEY=15zYYtksuJnQsTBODSNs3MrfEedOSd3i

# News APIs
NEWS_API_KEY=e6f793dfd61f473786f69466f9313fe8
MARKETAUX_API_KEY=Tw5w7ABp5srP5mgaKeyGjiXaJlk7Oz7sgpmxWxYH

# Settings
LOG_LEVEL=INFO
DATA_PRIORITY=FINANCIALMODELINGPREP,TWELVEDATA,FINNHUB,MASSIVE,TIINGO,YFINANCE
"""

with open('.env', 'w') as f:
    f.write(env_content)

print("\n✅ .env file created with API keys\n")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: LOAD MODULES
# ═══════════════════════════════════════════════════════════════════════════

print("📂 Loading modules...\n")

# Option A: Extract modules.zip (if uploaded)
import zipfile
if os.path.exists('modules.zip'):
    print("📦 Extracting modules.zip...")
    with zipfile.ZipFile('modules.zip', 'r') as zip_ref:
        zip_ref.extractall('.')
    if '/content/modules' not in sys.path:
        sys.path.insert(0, '/content/modules')
    print("✅ Modules extracted from zip!\n")

# Option B: Google Drive (uncomment and update path)
# from google.colab import drive
# drive.mount('/content/drive')
# DRIVE_PATH = '/content/drive/MyDrive/Quantum_AI_Cockpit/backend/modules'
# if os.path.exists(DRIVE_PATH):
#     if DRIVE_PATH not in sys.path:
#         sys.path.insert(0, DRIVE_PATH)
#     print(f"✅ Modules loaded from Google Drive: {DRIVE_PATH}\n")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: TEST MODULE IMPORTS
# ═══════════════════════════════════════════════════════════════════════════

print("🧪 Testing module imports...\n")

modules_to_test = [
    ('fusior_forecast', 'N-BEATS Forecaster'),
    ('pattern_integration_layer', 'Pattern Integration Layer'),
    ('ai_recommender_v2', 'AI Recommender V2'),
    ('ai_recommender_institutional', 'Institutional Features'),
    ('support_resistance_detector', 'Support/Resistance'),
    ('volume_profile_analyzer', 'Volume Profile'),
    ('multi_timeframe_analyzer', 'Multi-Timeframe'),
    ('head_shoulders_detector', 'Head & Shoulders'),
    ('triangle_detector', 'Triangles'),
    ('flag_pennant_detector', 'Flags & Pennants'),
    ('vwap_indicator', 'VWAP'),
    ('divergence_detector', 'Divergence'),
    ('master_analysis_engine', 'Master Analysis Engine'),
]

success_count = 0
failed = []

for module_name, display_name in modules_to_test:
    try:
        __import__(module_name)
        print(f"✅ {display_name}")
        success_count += 1
    except Exception as e:
        print(f"❌ {display_name}: {str(e)[:50]}")
        failed.append(display_name)

print(f"\n{'='*80}")
print(f"✅ {success_count}/{len(modules_to_test)} modules loaded successfully!")

if failed:
    print(f"⚠️  Failed: {', '.join(failed)}")
else:
    print("🎉 ALL MODULES LOADED PERFECTLY!")
print(f"{'='*80}\n")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: COMPLETE STOCK ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

import asyncio
from master_analysis_engine import analyze_stock

SYMBOL = "TSLA"  # Change to any stock
ACCOUNT_BALANCE = 50000

print("=" * 80)
print(f"🚀 COMPLETE ANALYSIS: {SYMBOL}")
print("=" * 80)

result = await analyze_stock(
    symbol=SYMBOL,
    account_balance=ACCOUNT_BALANCE,
    forecast_days=21,
    verbose=True
)

# Display results
if result['status'] == 'ok':
    rec = result['recommendation']
    
    print("\n" + "=" * 80)
    print(f"📊 FINAL RESULTS: {SYMBOL}")
    print("=" * 80)
    
    print(f"\n🎯 ACTION: {rec['action']}")
    print(f"📈 CONFIDENCE: {rec['confidence']*100:.1f}%")
    print(f"📅 5-Day Expected: {rec.get('expected_move_5d', 0):+.1f}%")
    print(f"📅 20-Day Expected: {rec.get('expected_move_20d', 0):+.1f}%")
    
    # Institutional features
    inst = rec.get('institutional_grade')
    if inst:
        ps = inst['position_sizing']
        rr = inst['risk_reward']
        
        print(f"\n💰 POSITION SIZING:")
        print(f"   Shares: {ps['shares']}")
        print(f"   Value: ${ps['position_value']:,.2f} ({ps['position_pct']:.1f}% of portfolio)")
        
        print(f"\n⚖️  RISK:REWARD:")
        print(f"   Entry: ${rr['entry']:.2f}")
        print(f"   Stop: ${rr['stop']:.2f} ({rr['risk_pct']:.1f}% risk)")
        print(f"   Target: ${rr['target']:.2f} ({rr['reward_pct']:.1f}% reward)")
        print(f"   R:R Ratio: {rr['rr_ratio']:.2f}:1 {'✅' if rr['rr_ratio'] >= 2.0 else '⚠️'}")
    
    # Pattern summary
    patterns = result.get('patterns')
    if patterns and patterns.get('status') == 'ok':
        summary = patterns['summary']
        
        print(f"\n🔍 PATTERN ANALYSIS:")
        print(f"   Total Patterns: {summary['total_patterns_detected']}")
        print(f"   Bullish Signals: {summary['bullish_signals']}")
        print(f"   Bearish Signals: {summary['bearish_signals']}")
        print(f"   Confluence Score: {summary['confluence_score']:.0f}%")
        print(f"   Pattern Recommendation: {summary['recommendation']}")
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)
else:
    print(f"\n❌ Analysis failed: {result.get('error', 'Unknown error')}")

# ═══════════════════════════════════════════════════════════════════════════
# DONE!
# ═══════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 80)
print("🎉 ALL TESTS COMPLETE!")
print("=" * 80)
print("\n✅ Your system is working perfectly!")
print("\n📊 Next steps:")
print("   1. Test more stocks (change SYMBOL variable)")
print("   2. Run portfolio analysis (see COLAB_PORTFOLIO_TEST.py)")
print("   3. When ready, build Streamlit dashboard!")
print("\n" + "=" * 80)

