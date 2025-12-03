"""
DEBUG: Check how much data we're getting
"""

print("=" * 80)
print("🔍 DEBUGGING DATA FETCH")
print("=" * 80)

import sys
sys.path.insert(0, '/content/drive/MyDrive/Quantum_AI_Cockpit/backend/modules')

from data_orchestrator import DataOrchestrator
import asyncio

async def check_data():
    symbol = 'AMD'
    data_orch = DataOrchestrator()
    
    print(f"\n📊 Fetching data for {symbol}...")
    
    # Try different day counts
    for days in [60, 90, 120, 150, 200]:
        try:
            df = await data_orch.fetch_symbol_data(symbol, days=days)
            if df is not None:
                print(f"\n✅ {days} days requested → Got {len(df)} rows")
                print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
            else:
                print(f"\n❌ {days} days requested → Got None")
        except Exception as e:
            print(f"\n❌ {days} days failed: {e}")
    
    # Check what 90 days gives us (what institutional code uses)
    print("\n" + "=" * 80)
    print("🎯 INSTITUTIONAL CODE USES 90 DAYS")
    print("=" * 80)
    
    df = await data_orch.fetch_symbol_data(symbol, days=90)
    if df is not None:
        rows = len(df)
        print(f"✅ Got {rows} rows")
        print(f"📊 Prophet needs: 100+ rows")
        print(f"📊 LightGBM needs: 60+ rows")
        
        if rows < 100:
            print(f"\n⚠️  PROBLEM FOUND!")
            print(f"   We have: {rows} rows")
            print(f"   Prophet needs: 100+ rows")
            print(f"   Solution: Request more days (150+)")
        else:
            print(f"\n✅ Enough data for all models!")
    
    return df

# Run
df = await check_data()

print("\n" + "=" * 80)
print("💡 RECOMMENDED FIX")
print("=" * 80)
print("\nInstead of:")
print("  df = await data_orch.fetch_symbol_data(symbol, days=90)")
print("\nUse:")
print("  df = await data_orch.fetch_symbol_data(symbol, days=150)")
print("\nThis gives Prophet the 100+ rows it needs.")

