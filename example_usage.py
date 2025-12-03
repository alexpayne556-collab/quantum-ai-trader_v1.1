"""
Simple usage examples for Quantum Data Orchestrator.
Copy and run these examples to see the system in action.
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

from quantum_orchestrator import fetch_ticker, fetch_tickers


# ============================================================================
# EXAMPLE 1: Fetch a Single Stock
# ============================================================================

async def example_single_ticker():
    """Fetch data for a single ticker."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Fetch Single Stock (AAPL)")
    print("="*70 + "\n")
    
    result = await fetch_ticker("AAPL", days=30)
    
    if result.success:
        print(f"✅ Successfully fetched {result.ticker}")
        print(f"   Source: {result.source}")
        print(f"   Candles: {result.candles}")
        print(f"   Date Range: {result.data.index[0].strftime('%Y-%m-%d')} to {result.data.index[-1].strftime('%Y-%m-%d')}")
        print(f"\n   Last 5 Days:")
        print(result.data.tail().to_string())
    else:
        print(f"❌ Failed to fetch {result.ticker}")
        print(f"   Error: {result.error}")


# ============================================================================
# EXAMPLE 2: Fetch Multiple Stocks in Parallel
# ============================================================================

async def example_multiple_tickers():
    """Fetch data for multiple tickers in parallel."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Fetch Multiple Stocks (Parallel)")
    print("="*70 + "\n")
    
    watchlist = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    results = await fetch_tickers(watchlist, days=30)
    
    print("📊 Watchlist Results:\n")
    for ticker, result in results.items():
        if result.success:
            latest_close = result.data.iloc[-1]['close']
            print(f"  ✅ {ticker:6} | {result.source:15} | {result.candles:2} candles | Latest: ${latest_close:.2f}")
        else:
            print(f"  ❌ {ticker:6} | Failed: {result.error}")


# ============================================================================
# EXAMPLE 3: Calculate Simple Statistics
# ============================================================================

async def example_statistics():
    """Fetch data and calculate simple statistics."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Calculate Statistics (SPY)")
    print("="*70 + "\n")
    
    result = await fetch_ticker("SPY", days=90)
    
    if result.success:
        df = result.data
        
        print(f"📊 Statistics for {result.ticker} ({result.candles} days):")
        print(f"   Source: {result.source}")
        print(f"   Period: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"\n   Price Statistics:")
        print(f"   • Current Price: ${df.iloc[-1]['close']:.2f}")
        print(f"   • 90-Day High:   ${df['high'].max():.2f}")
        print(f"   • 90-Day Low:    ${df['low'].min():.2f}")
        print(f"   • Average Volume: {df['volume'].mean():,.0f}")
        print(f"   • 90-Day Return:  {((df.iloc[-1]['close'] / df.iloc[0]['close']) - 1) * 100:.2f}%")
    else:
        print(f"❌ Failed: {result.error}")


# ============================================================================
# EXAMPLE 4: Build a Simple Screener
# ============================================================================

async def example_screener():
    """Screen multiple stocks for specific criteria."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Simple Momentum Screener")
    print("="*70 + "\n")
    
    universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
    results = await fetch_tickers(universe, days=30)
    
    momentum_stocks = []
    
    for ticker, result in results.items():
        if result.success:
            df = result.data
            
            # Calculate 30-day return
            returns = ((df.iloc[-1]['close'] / df.iloc[0]['close']) - 1) * 100
            
            # Calculate average volume
            avg_volume = df['volume'].mean()
            
            # Simple momentum criteria: >5% return, >10M avg volume
            if returns > 5 and avg_volume > 10_000_000:
                momentum_stocks.append({
                    'ticker': ticker,
                    'return': returns,
                    'volume': avg_volume,
                    'price': df.iloc[-1]['close']
                })
    
    # Sort by return
    momentum_stocks.sort(key=lambda x: x['return'], reverse=True)
    
    print("🚀 Momentum Stocks (30-day return > 5%, avg volume > 10M):\n")
    if momentum_stocks:
        for stock in momentum_stocks:
            print(f"  ✅ {stock['ticker']:6} | Return: {stock['return']:+.2f}% | Price: ${stock['price']:.2f} | Vol: {stock['volume']:,.0f}")
    else:
        print("  No stocks matched criteria")


# ============================================================================
# EXAMPLE 5: Export Data to CSV
# ============================================================================

async def example_export_csv():
    """Fetch data and export to CSV."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Export Data to CSV")
    print("="*70 + "\n")
    
    result = await fetch_ticker("SPY", days=90)
    
    if result.success:
        filename = f"{result.ticker}_data.csv"
        result.data.to_csv(filename)
        print(f"✅ Exported {result.candles} candles to {filename}")
        print(f"   Source: {result.source}")
        print(f"   Period: {result.data.index[0].strftime('%Y-%m-%d')} to {result.data.index[-1].strftime('%Y-%m-%d')}")
    else:
        print(f"❌ Failed: {result.error}")


# ============================================================================
# MAIN: Run All Examples
# ============================================================================

async def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("   QUANTUM DATA ORCHESTRATOR - USAGE EXAMPLES")
    print("="*70)
    
    await example_single_ticker()
    await example_multiple_tickers()
    await example_statistics()
    await example_screener()
    await example_export_csv()
    
    print("\n" + "="*70)
    print("✅ ALL EXAMPLES COMPLETE")
    print("="*70 + "\n")
    print("💡 Tip: Copy these examples and modify for your needs!")
    print("📚 Read full docs: backend/README_ORCHESTRATOR.md\n")


if __name__ == "__main__":
    asyncio.run(main())
