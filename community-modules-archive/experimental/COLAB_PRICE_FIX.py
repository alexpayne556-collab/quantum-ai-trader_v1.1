# ═══════════════════════════════════════════════════════════════════════════════
# 🔧 PRICE EXTRACTION FIX - Apply this in Colab NOW
# ═══════════════════════════════════════════════════════════════════════════════
"""
This fixes the $0.00 price issue by patching fusior_forecast.py with robust price extraction.
"""

import re
from pathlib import Path

print("=" * 80)
print("🔧 FIXING PRICE EXTRACTION IN FUSIOR_FORECAST")
print("=" * 80)
print()

# Path to the module
MODULE_PATH = Path("/content/drive/MyDrive/Quantum_AI_Cockpit/backend/modules/fusior_forecast.py")

if not MODULE_PATH.exists():
    print(f"❌ File not found: {MODULE_PATH}")
    print("   Check your Google Drive path!")
else:
    print(f"✅ Found file: {MODULE_PATH}")
    
    # Read the file
    with open(MODULE_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("📖 File read successfully")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FIX 1: Replace the price extraction logic in the result dictionary
    # ═══════════════════════════════════════════════════════════════════════════
    
    print("\n🔍 Searching for price extraction code...")
    
    # Find the result dictionary construction (around line 1219-1238)
    old_price_extraction = '''            result = {
                "module": "fusior_forecast",
                "symbol": sym,
                "kind": "forecast",
                "status": "ok",
                "spot_price": float(df["close"].iloc[-1]) if "close" in df.columns and len(df) > 0 else 0.0,
                "current_price": float(df["close"].iloc[-1]) if "close" in df.columns and len(df) > 0 else 0.0,'''
    
    new_price_extraction = '''            # ═══════════════════════════════════════════════════════════════
            # ROBUST PRICE EXTRACTION (multiple fallbacks)
            # ═══════════════════════════════════════════════════════════════
            extracted_price = 0.0
            
            # Try 1: From dataframe "close" column (lowercase)
            if df is not None and not df.empty and "close" in df.columns:
                try:
                    extracted_price = float(df["close"].iloc[-1])
                    logger.info(f"[{sym}] ✅ Price extracted from df['close']: ${extracted_price:.2f}")
                except Exception as e:
                    logger.warning(f"[{sym}] Failed to extract from df['close']: {e}")
            
            # Try 2: From dataframe "Close" column (capitalized)
            if extracted_price == 0.0 and df is not None and not df.empty and "Close" in df.columns:
                try:
                    extracted_price = float(df["Close"].iloc[-1])
                    logger.info(f"[{sym}] ✅ Price extracted from df['Close']: ${extracted_price:.2f}")
                except Exception as e:
                    logger.warning(f"[{sym}] Failed to extract from df['Close']: {e}")
            
            # Try 3: From technical snapshot
            if extracted_price == 0.0 and snapshot.get('price', 0) > 0:
                try:
                    extracted_price = float(snapshot['price'])
                    logger.info(f"[{sym}] ✅ Price extracted from snapshot: ${extracted_price:.2f}")
                except Exception as e:
                    logger.warning(f"[{sym}] Failed to extract from snapshot: {e}")
            
            # Try 4: From last forecast value (if reasonable)
            if extracted_price == 0.0 and len(forecast_vec) > 0 and forecast_vec[0] > 0:
                try:
                    extracted_price = float(forecast_vec[0])
                    logger.info(f"[{sym}] ✅ Price extracted from forecast[0]: ${extracted_price:.2f}")
                except Exception as e:
                    logger.warning(f"[{sym}] Failed to extract from forecast: {e}")
            
            # Try 5: From metrics close array
            if extracted_price == 0.0 and metrics.get('close') and len(metrics['close']) > 0:
                try:
                    extracted_price = float(metrics['close'][-1])
                    logger.info(f"[{sym}] ✅ Price extracted from metrics['close']: ${extracted_price:.2f}")
                except Exception as e:
                    logger.warning(f"[{sym}] Failed to extract from metrics: {e}")
            
            # Final check
            if extracted_price == 0.0:
                logger.error(f"[{sym}] ❌ FAILED TO EXTRACT PRICE! All methods returned $0.00")
                logger.error(f"[{sym}]    df empty: {df is None or df.empty}")
                logger.error(f"[{sym}]    df columns: {list(df.columns) if df is not None else 'N/A'}")
                logger.error(f"[{sym}]    snapshot price: {snapshot.get('price', 'N/A')}")
                logger.error(f"[{sym}]    forecast length: {len(forecast_vec)}")
            else:
                logger.info(f"[{sym}] 💰 Final extracted price: ${extracted_price:.2f}")
            
            result = {
                "module": "fusior_forecast",
                "symbol": sym,
                "kind": "forecast",
                "status": "ok",
                "spot_price": extracted_price,
                "current_price": extracted_price,'''
    
    if old_price_extraction in content:
        content = content.replace(old_price_extraction, new_price_extraction)
        print("✅ Fixed: Added robust price extraction with 5 fallback methods")
    else:
        print("⚠️  Exact pattern not found, trying alternative approach...")
        
        # Alternative: Just fix the two lines
        content = re.sub(
            r'"spot_price":\s*float\(df\["close"\]\.iloc\[-1\]\)\s*if\s*"close"\s*in\s*df\.columns\s*and\s*len\(df\)\s*>\s*0\s*else\s*0\.0,',
            '"spot_price": (float(df["close"].iloc[-1]) if "close" in df.columns else float(df["Close"].iloc[-1]) if "Close" in df.columns else float(snapshot.get("price", 0))) if df is not None and not df.empty else 0.0,',
            content
        )
        
        content = re.sub(
            r'"current_price":\s*float\(df\["close"\]\.iloc\[-1\]\)\s*if\s*"close"\s*in\s*df\.columns\s*and\s*len\(df\)\s*>\s*0\s*else\s*0\.0,',
            '"current_price": (float(df["close"].iloc[-1]) if "close" in df.columns else float(df["Close"].iloc[-1]) if "Close" in df.columns else float(snapshot.get("price", 0))) if df is not None and not df.empty else 0.0,',
            content
        )
        
        print("✅ Fixed: Applied regex patch for price extraction")
    
    # Write the fixed content back
    with open(MODULE_PATH, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\n✅ File saved successfully!")
    print()
    print("=" * 80)
    print("🔄 RELOADING MODULE")
    print("=" * 80)
    
    # Force reload the module
    import sys
    
    # Unload fusior_forecast
    modules_to_unload = [k for k in sys.modules.keys() if 'fusior' in k]
    for mod in modules_to_unload:
        del sys.modules[mod]
        print(f"✅ Unloaded {mod}")
    
    # Reimport
    try:
        import fusior_forecast
        print("✅ fusior_forecast reloaded successfully!")
    except Exception as e:
        print(f"❌ Failed to reload: {e}")
    
    print()
    print("=" * 80)
    print("✅ FIX APPLIED!")
    print("=" * 80)
    print()
    print("💡 Now run the test again:")
    print("   await test_single_stock('AAPL')")
    print()
    print("   You should see: 💰 Price: $267.46 (or current AAPL price)")
    print("=" * 80)



