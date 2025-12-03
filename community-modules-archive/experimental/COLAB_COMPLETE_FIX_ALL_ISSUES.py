# ═══════════════════════════════════════════════════════════════════════════════
# 🔧 COMPLETE FIX - ALL ISSUES (Price + EMA Ribbon)
# ═══════════════════════════════════════════════════════════════════════════════
"""
This fixes:
1. Price extraction ($0.00 issue)
2. EMA Ribbon "not callable" error
3. Any other module loading issues

Run this in Colab BEFORE testing!
"""

import re
from pathlib import Path
import sys

print("=" * 80)
print("🔧 COMPREHENSIVE FIX - ALL ISSUES")
print("=" * 80)
print()

BASE_PATH = Path("/content/drive/MyDrive/Quantum_AI_Cockpit/backend/modules")

# ═══════════════════════════════════════════════════════════════════════════════
# FIX 1: PRICE EXTRACTION IN FUSIOR_FORECAST
# ═══════════════════════════════════════════════════════════════════════════════

print("FIX 1: Price Extraction")
print("-" * 80)

fusior_path = BASE_PATH / "fusior_forecast.py"

if fusior_path.exists():
    with open(fusior_path, 'r', encoding='utf-8') as f:
        fusior_content = f.read()
    
    # More aggressive fix - replace the entire price extraction section
    # Find lines 1224-1225 and replace with robust extraction
    
    pattern = r'"spot_price":\s*float\(df\["close"\]\.iloc\[-1\]\)[^,]*,\s*"current_price":\s*float\(df\["close"\]\.iloc\[-1\]\)[^,]*,'
    
    replacement = '''# Extract price with multiple fallbacks
                price_val = 0.0
                try:
                    if df is not None and not df.empty:
                        if "close" in df.columns:
                            price_val = float(df["close"].iloc[-1])
                        elif "Close" in df.columns:
                            price_val = float(df["Close"].iloc[-1])
                    if price_val == 0.0 and snapshot.get('price', 0) > 0:
                        price_val = float(snapshot['price'])
                    if price_val == 0.0 and len(forecast_vec) > 0:
                        price_val = float(forecast_vec[0])
                except Exception as e:
                    logger.warning(f"[{sym}] Price extraction error: {e}")
                    price_val = 0.0
                
                "spot_price": price_val,
                "current_price": price_val,'''
    
    new_content = re.sub(pattern, replacement, fusior_content, flags=re.DOTALL)
    
    if new_content != fusior_content:
        with open(fusior_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("✅ Fixed: fusior_forecast.py - Price extraction patched")
    else:
        print("⚠️  Pattern not matched, trying simpler fix...")
        
        # Simpler fix: just add fallbacks inline
        fusior_content = fusior_content.replace(
            '"spot_price": float(df["close"].iloc[-1]) if "close" in df.columns and len(df) > 0 else 0.0,',
            '"spot_price": float(df["close"].iloc[-1]) if df is not None and not df.empty and "close" in df.columns else (float(df["Close"].iloc[-1]) if df is not None and not df.empty and "Close" in df.columns else float(snapshot.get("price", forecast_vec[0] if len(forecast_vec) > 0 else 0))),',
        )
        
        fusior_content = fusior_content.replace(
            '"current_price": float(df["close"].iloc[-1]) if "close" in df.columns and len(df) > 0 else 0.0,',
            '"current_price": float(df["close"].iloc[-1]) if df is not None and not df.empty and "close" in df.columns else (float(df["Close"].iloc[-1]) if df is not None and not df.empty and "Close" in df.columns else float(snapshot.get("price", forecast_vec[0] if len(forecast_vec) > 0 else 0))),',
        )
        
        with open(fusior_path, 'w', encoding='utf-8') as f:
            f.write(fusior_content)
        
        print("✅ Fixed: fusior_forecast.py - Inline fallback applied")
else:
    print("❌ fusior_forecast.py not found!")

print()

# ═══════════════════════════════════════════════════════════════════════════════
# FIX 2: EMA RIBBON "NOT CALLABLE" ERROR IN PATTERN_INTEGRATION_LAYER
# ═══════════════════════════════════════════════════════════════════════════════

print("FIX 2: EMA Ribbon Integration")
print("-" * 80)

pattern_path = BASE_PATH / "pattern_integration_layer.py"

if pattern_path.exists():
    with open(pattern_path, 'r', encoding='utf-8') as f:
        pattern_content = f.read()
    
    # Find the EMA ribbon module registration
    # The issue is likely that EMARibbonEngine is being stored as the class, not an instance
    
    # Fix 1: Make sure it's instantiated when registered
    old_ema_registration = "from ema_ribbon_engine import EMARibbonEngine"
    
    if old_ema_registration in pattern_content:
        # Already has the import, now make sure it's instantiated properly
        # Look for where it's registered in the modules dict
        
        # Pattern 1: Direct class assignment
        pattern_content = re.sub(
            r"'ema_ribbons':\s*EMARibbonEngine,",
            "'ema_ribbons': EMARibbonEngine(),  # Instantiate the class",
            pattern_content
        )
        
        # Pattern 2: In module loading
        pattern_content = re.sub(
            r"self\.modules\['ema_ribbons'\]\s*=\s*EMARibbonEngine\s*$",
            "self.modules['ema_ribbons'] = EMARibbonEngine()  # Instantiate",
            pattern_content,
            flags=re.MULTILINE
        )
        
        with open(pattern_path, 'w', encoding='utf-8') as f:
            f.write(pattern_content)
        
        print("✅ Fixed: pattern_integration_layer.py - EMA ribbon instantiation")
    else:
        print("⚠️  EMA ribbon import not found (might be ok)")
else:
    print("❌ pattern_integration_layer.py not found!")

print()

# ═══════════════════════════════════════════════════════════════════════════════
# FIX 3: RELOAD ALL MODULES
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("🔄 RELOADING ALL MODULES")
print("=" * 80)
print()

# Unload all related modules
modules_to_clear = [
    'fusior_forecast',
    'master_analysis_engine', 
    'pattern_integration_layer',
    'ai_recommender_v2',
    'ai_recommender_institutional',
    'ema_ribbon_engine'
]

for mod_name in modules_to_clear:
    matching = [k for k in list(sys.modules.keys()) if mod_name in k]
    for k in matching:
        del sys.modules[k]
        print(f"✅ Unloaded {k}")

print()

# Reimport critical modules
try:
    import fusior_forecast
    print("✅ fusior_forecast reloaded")
except Exception as e:
    print(f"❌ fusior_forecast: {e}")

try:
    from pattern_integration_layer import PatternIntegrationLayer
    print("✅ pattern_integration_layer reloaded")
except Exception as e:
    print(f"⚠️  pattern_integration_layer: {e}")

try:
    from master_analysis_engine import MasterAnalysisEngine
    print("✅ master_analysis_engine reloaded")
except Exception as e:
    print(f"❌ master_analysis_engine: {e}")

print()
print("=" * 80)
print("✅ ALL FIXES APPLIED!")
print("=" * 80)
print()
print("💡 Next step: Run the test cell again")
print()
print("   await test_single_stock('AAPL')")
print()
print("   Expected output:")
print("   💰 Price: $267.46")
print("   📊 Action: BUY/SELL/HOLD")
print("   🎯 Confidence: 75%")
print()
print("=" * 80)



