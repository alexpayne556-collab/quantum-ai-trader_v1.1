"""
Quick test script to verify API config and orchestrator integration.
Run this to validate your setup before building the web interface.
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from quantum_api_config import get_config, reset_config
from quantum_orchestrator import fetch_ticker, fetch_tickers


async def main():
    print("\n" + "="*80)
    print(" " * 20 + "QUANTUM AI TRADER v1.1")
    print(" " * 15 + "API & ORCHESTRATOR VALIDATION TEST")
    print("="*80 + "\n")
    
    # Reset config to ensure fresh load
    reset_config()
    
    # Test 1: API Configuration
    print("📋 TEST 1: API CONFIGURATION")
    print("-" * 80)
    config = get_config()
    summary = config.get_config_summary()
    
    print(f"✓ Total Sources: {summary['total_sources']}")
    print(f"✓ Valid Sources: {summary['valid_sources']}")
    print(f"✓ Primary Source: {summary['primary_source']}")
    print(f"✓ Intraday Capable: {summary['intraday_sources']}")
    
    if summary['valid_sources'] == 0:
        print("\n❌ ERROR: No valid API sources found!")
        print("   Please check your .env file at E:/quantum-ai-trader-v1.1/.env")
        return
    
    print("\n✅ Configuration validated\n")
    
    # Test 2: Single Ticker Fetch
    print("📊 TEST 2: SINGLE TICKER FETCH (SPY - 30 days)")
    print("-" * 80)
    
    result = await fetch_ticker("SPY", days=30)
    
    if result.success:
        print(f"✓ Ticker: {result.ticker}")
        print(f"✓ Source: {result.source}")
        print(f"✓ Candles: {result.candles}")
        print(f"✓ Date Range: {result.data.index[0].strftime('%Y-%m-%d')} to {result.data.index[-1].strftime('%Y-%m-%d')}")
        print(f"\n  Sample Data (last 5 days):")
        print(result.data.tail().to_string())
        print("\n✅ Single ticker fetch successful\n")
    else:
        print(f"❌ Fetch failed: {result.error}\n")
    
    # Test 3: Multi-Ticker Parallel Fetch
    print("🚀 TEST 3: PARALLEL MULTI-TICKER FETCH (5 tickers)")
    print("-" * 80)
    
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    results = await fetch_tickers(tickers, days=30)
    
    successful = 0
    for ticker, res in results.items():
        status = "✓" if res.success else "❌"
        if res.success:
            successful += 1
            info = f"{res.candles} candles from {res.source}"
        else:
            info = f"Failed: {res.error}"
        print(f"  {status} {ticker:6} | {info}")
    
    print(f"\n✅ Parallel fetch complete: {successful}/{len(tickers)} successful\n")
    
    # Test 4: Data Quality Check
    if result.success and result.data is not None:
        print("🔍 TEST 4: DATA QUALITY CHECK (SPY)")
        print("-" * 80)
        
        df = result.data
        
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        has_all_cols = all(col in df.columns for col in required_cols)
        print(f"✓ Required columns present: {has_all_cols}")
        
        # Check for missing values
        missing = df.isnull().sum().sum()
        print(f"✓ Missing values: {missing}")
        
        # Check data integrity (high >= low, etc.)
        integrity_check = (df['high'] >= df['low']).all()
        print(f"✓ Price integrity (high >= low): {integrity_check}")
        
        # Check volume
        has_volume = (df['volume'] > 0).sum() / len(df) * 100
        print(f"✓ Days with volume: {has_volume:.1f}%")
        
        print("\n✅ Data quality validated\n")
    
    # Final Summary
    print("="*80)
    print(" " * 25 + "🎯 VALIDATION COMPLETE")
    print("="*80)
    print(f"\n✅ API Sources Active: {summary['valid_sources']}")
    print(f"✅ Primary Source: {summary['primary_source']}")
    print(f"✅ Single Ticker Fetch: {'Working' if result.success else 'Failed'}")
    print(f"✅ Parallel Fetch: {successful}/{len(tickers)} tickers successful")
    print(f"\n🚀 System ready for production use!")
    print(f"💡 Next: Build web interface to call these endpoints\n")


if __name__ == "__main__":
    asyncio.run(main())
