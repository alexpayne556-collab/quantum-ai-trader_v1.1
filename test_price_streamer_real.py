"""
Real-world test for Price Streamer with Finnhub WebSocket
"""
import asyncio
from price_streamer import PriceStreamer2024

async def price_callback(data):
    """Callback for price updates"""
    print(f"🔔 CALLBACK: {data['symbol']} @ ${data['price']:.2f} | Vol: {data['volume']} | {data['datetime']}")

async def test_real_streaming():
    streamer = PriceStreamer2024()
    
    # Test 1: Get real-time quote
    print("\n" + "="*60)
    print("TEST 1: Real-time REST API Quote")
    print("="*60)
    quote = await streamer.get_real_time_quote('AAPL')
    if quote:
        print(f"✅ AAPL Quote:")
        print(f"   Current: ${quote['current_price']:.2f}")
        print(f"   High: ${quote['high']:.2f}")
        print(f"   Low: ${quote['low']:.2f}")
        print(f"   Open: ${quote['open']:.2f}")
        print(f"   Prev Close: ${quote['previous_close']:.2f}")
    
    # Test 2: WebSocket streaming
    print("\n" + "="*60)
    print("TEST 2: Live WebSocket Streaming (10 seconds)")
    print("="*60)
    
    # Register callback
    streamer.register_callback('AAPL', price_callback)
    
    # Start streaming
    await streamer.start_streaming(['AAPL', 'TSLA', 'NVDA'])
    
    # Listen for 10 seconds
    await asyncio.sleep(10)
    
    # Show cached prices
    print("\n" + "="*60)
    print("Cached Prices:")
    print("="*60)
    for symbol in ['AAPL', 'TSLA', 'NVDA']:
        price_data = streamer.get_latest_price(symbol)
        if price_data:
            print(f"📊 {symbol}: ${price_data['price']:.2f} @ {price_data['datetime']}")
    
    # Stop streaming
    await streamer.stop_streaming()
    print("\n✅ Real streaming test completed!")

if __name__ == "__main__":
    print("🚀 Starting REAL Price Streamer Test with Finnhub WebSocket...")
    asyncio.run(test_real_streaming())
