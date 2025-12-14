"""
Smoke test for Price Streamer module
"""
import asyncio
from price_streamer import PriceStreamer2024, MockWebSocket

def test_price_streamer_smoke():
    streamer = PriceStreamer2024()
    ws = MockWebSocket()
    async def run_test():
        await streamer.websocket_endpoint('AAPL', ws)
        await streamer.simulate_market_feed('AAPL')
    asyncio.run(run_test())
    print('WebSocket messages:', ws.messages)
    print('Price Streamer smoke test passed!')

if __name__ == "__main__":
    test_price_streamer_smoke()
