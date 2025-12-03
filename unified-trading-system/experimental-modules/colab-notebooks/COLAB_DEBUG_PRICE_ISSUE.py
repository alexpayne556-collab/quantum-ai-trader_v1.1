# ═══════════════════════════════════════════════════════════════════════════════
# 🔍 DEEP DEBUG - TRACE THE PRICE ISSUE
# ═══════════════════════════════════════════════════════════════════════════════
"""
This will trace the entire flow to see where the close price is getting lost.
Run this in Colab to diagnose the exact problem.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 80)
print("🔍 DEEP DEBUG - TRACING PRICE EXTRACTION")
print("=" * 80)
print()

# Test with a simple stock
TEST_SYMBOL = "AAPL"

print(f"Testing with: {TEST_SYMBOL}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Test data fetching
# ═══════════════════════════════════════════════════════════════════════════════

print("STEP 1: Testing data fetching...")
print("-" * 80)

async def test_data_fetch(symbol):
    """Test data fetching and column inspection."""
    
    # Try data_orchestrator first
    try:
        from data_orchestrator import get_cached_symbol_data
        print(f"✅ data_orchestrator module imported")
        
        df = await get_cached_symbol_data(symbol)
        
        if df is None or df.empty:
            print(f"❌ data_orchestrator returned empty/None")
            raise ValueError("No data")
        
        print(f"✅ data_orchestrator fetched {len(df)} rows")
        print(f"   Columns (raw): {list(df.columns)}")
        print(f"   Index type: {type(df.index)}")
        print(f"   First 3 rows:")
        print(df.head(3))
        
        # Check for close column (case insensitive)
        df_lower = df.copy()
        df_lower.columns = [c.lower() for c in df_lower.columns]
        
        print(f"\n   Columns (lowercase): {list(df_lower.columns)}")
        
        if 'close' in df_lower.columns:
            close_vals = df_lower['close'].tail(5)
            print(f"\n   ✅ 'close' column found!")
            print(f"   Last 5 close prices:")
            print(close_vals)
            print(f"\n   Most recent close: ${float(close_vals.iloc[-1]):.2f}")
            return df_lower, float(close_vals.iloc[-1])
        else:
            print(f"   ❌ 'close' column NOT found!")
            return df_lower, 0.0
            
    except Exception as e:
        print(f"❌ data_orchestrator failed: {type(e).__name__}: {e}")
        
        # Fallback to yfinance
        try:
            import yfinance as yf
            print(f"\n🔄 Falling back to yfinance...")
            
            df = yf.download(symbol, period="1y", interval="1d", progress=False)
            
            if df is None or df.empty:
                print(f"❌ yfinance also returned empty")
                return pd.DataFrame(), 0.0
            
            print(f"✅ yfinance fetched {len(df)} rows")
            print(f"   Columns (raw): {list(df.columns)}")
            
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            
            print(f"   Columns (lowercase): {list(df.columns)}")
            
            if 'close' in df.columns:
                close_vals = df['close'].tail(5)
                print(f"\n   ✅ 'close' column found!")
                print(f"   Last 5 close prices:")
                print(close_vals)
                print(f"\n   Most recent close: ${float(close_vals.iloc[-1]):.2f}")
                return df, float(close_vals.iloc[-1])
            else:
                print(f"   ❌ 'close' column NOT found!")
                return df, 0.0
                
        except Exception as e2:
            print(f"❌ yfinance also failed: {type(e2).__name__}: {e2}")
            return pd.DataFrame(), 0.0

# Run data fetch test
df_test, close_price = await test_data_fetch(TEST_SYMBOL)

print()
print("=" * 80)
print(f"DATA FETCH RESULT: Close price = ${close_price:.2f}")
print("=" * 80)
print()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Test fusior_forecast directly
# ═══════════════════════════════════════════════════════════════════════════════

if close_price > 0:
    print("STEP 2: Testing fusior_forecast module...")
    print("-" * 80)
    
    try:
        from fusior_forecast import run as fusior_run
        print("✅ fusior_forecast imported")
        
        # Run forecast
        result = await fusior_run(
            symbol=TEST_SYMBOL,
            visualize=False,
            horizon_days=14
        )
        
        print(f"\n✅ fusior_forecast completed")
        print(f"   Status: {result.get('status', 'unknown')}")
        print(f"   Symbol: {result.get('symbol', 'unknown')}")
        
        # Check all price fields
        spot_price = result.get('spot_price', 0)
        current_price = result.get('current_price', 0)
        
        print(f"\n   Price fields in result:")
        print(f"   • spot_price: ${spot_price:.2f}")
        print(f"   • current_price: ${current_price:.2f}")
        
        # Check technical snapshot
        snapshot = result.get('technical_snapshot', {})
        if snapshot:
            snap_price = snapshot.get('price', 0)
            print(f"   • technical_snapshot['price']: ${snap_price:.2f}")
        
        # Check metrics
        metrics = result.get('metrics', {})
        if metrics and 'close' in metrics:
            metrics_close = metrics.get('close', [])
            if metrics_close:
                print(f"   • metrics['close'][-1]: ${float(metrics_close[-1]):.2f}")
        
        # Summary
        print()
        if current_price > 0:
            print("   ✅ SUCCESS: Current price extracted correctly!")
        else:
            print("   ❌ PROBLEM: current_price is $0.00")
            print()
            print("   Debugging info:")
            print(f"   • Result keys: {list(result.keys())}")
            if result.get('error'):
                print(f"   • Error message: {result['error']}")
                
    except Exception as e:
        print(f"❌ fusior_forecast failed: {type(e).__name__}: {e}")
        import traceback
        print("\nFull traceback:")
        print(traceback.format_exc())

else:
    print("⚠️  SKIPPING STEP 2: Data fetch returned $0.00")

print()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Test master_analysis_engine
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("STEP 3: Testing master_analysis_engine...")
print("-" * 80)

try:
    from master_analysis_engine import MasterAnalysisEngine
    print("✅ master_analysis_engine imported")
    
    engine = MasterAnalysisEngine()
    print("✅ Engine initialized")
    
    # Run analysis
    result = await engine.analyze_stock(
        symbol=TEST_SYMBOL,
        account_balance=500,
        forecast_days=14,
        verbose=False
    )
    
    print(f"\n✅ Analysis completed")
    print(f"   Status: {result.get('status', 'unknown')}")
    
    # Check price fields
    current_price = result.get('current_price', 0)
    print(f"   Current price: ${current_price:.2f}")
    
    # Check recommendation
    rec = result.get('recommendation', {})
    if rec:
        action = rec.get('action', 'unknown')
        confidence = rec.get('confidence', 0)
        print(f"   Recommendation: {action} ({confidence:.0f}% confidence)")
    
    # Check forecast
    forecast = result.get('forecast', {})
    if forecast:
        forecast_data = forecast.get('data', {})
        if forecast_data:
            fcast_price = forecast_data.get('current_price', 0)
            print(f"   Forecast current_price: ${fcast_price:.2f}")
    
    print()
    if current_price > 0:
        print("   ✅ SUCCESS: Master engine working correctly!")
    else:
        print("   ❌ PROBLEM: current_price is $0.00 in master engine")
        print()
        print("   Result structure:")
        print(f"   • Keys: {list(result.keys())}")
        if result.get('error'):
            print(f"   • Error: {result['error']}")
        
except Exception as e:
    print(f"❌ master_analysis_engine failed: {type(e).__name__}: {e}")
    import traceback
    print("\nFull traceback:")
    print(traceback.format_exc())

print()
print("=" * 80)
print("🔍 DEBUG COMPLETE")
print("=" * 80)
print()
print("💡 SUMMARY:")
print()
print("If STEP 1 shows $0.00:")
print("  → Data fetching is broken (data_orchestrator or yfinance issue)")
print()
print("If STEP 1 works but STEP 2 shows $0.00:")
print("  → fusior_forecast has a bug in price extraction")
print()
print("If STEP 2 works but STEP 3 shows $0.00:")
print("  → master_analysis_engine has a bug in result packaging")
print()
print("=" * 80)



