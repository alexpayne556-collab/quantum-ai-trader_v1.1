"""
Price Streamer Module - REAL WEBSOCKET IMPLEMENTATION
Research-backed: Finnhub WebSocket, connection pooling, real-time broadcast
2024-2025 institutional best practices with LIVE market data
"""
from typing import Dict, Set, Optional, Callable
import asyncio
import json
import websockets
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Use environment variable for API key (secure approach)
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', '')

class PriceStreamer2024:
    """Production-grade WebSocket price streamer using Finnhub real-time API"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or FINNHUB_API_KEY
        self.ws_url = f"wss://ws.finnhub.io?token={self.api_key}"
        self.connections: Dict[str, Set] = {}
        self.subscribed_symbols: Set[str] = set()
        self.price_cache: Dict[str, Dict] = {}
        self.callbacks: Dict[str, list] = {}
        self._ws = None
        self._running = False
        self.max_cache_size = 10000  # Limit to 10k entries to prevent memory issues
        
    async def connect(self):
        """Establish WebSocket connection to Finnhub"""
        try:
            self._ws = await websockets.connect(self.ws_url)
            self._running = True
            print(f"✅ Connected to Finnhub WebSocket at {datetime.now()}")
            return True
        except Exception as e:
            print(f"❌ WebSocket connection failed: {e}")
            return False
    
    async def subscribe(self, symbol: str):
        """Subscribe to real-time trades for a symbol"""
        if not self._ws:
            await self.connect()
        
        try:
            subscribe_msg = {"type": "subscribe", "symbol": symbol}
            await self._ws.send(json.dumps(subscribe_msg))
            self.subscribed_symbols.add(symbol)
            print(f"📊 Subscribed to {symbol} real-time feed")
            return True
        except Exception as e:
            print(f"❌ Subscription failed for {symbol}: {e}")
            return False
    
    async def unsubscribe(self, symbol: str):
        """Unsubscribe from symbol"""
        if self._ws and symbol in self.subscribed_symbols:
            try:
                unsubscribe_msg = {"type": "unsubscribe", "symbol": symbol}
                await self._ws.send(json.dumps(unsubscribe_msg))
                self.subscribed_symbols.remove(symbol)
                print(f"🔕 Unsubscribed from {symbol}")
            except Exception as e:
                print(f"❌ Unsubscribe failed: {e}")
    
    async def listen(self):
        """Listen for incoming price updates"""
        if not self._ws:
            await self.connect()
        
        try:
            async for message in self._ws:
                data = json.loads(message)
                
                if data.get('type') == 'trade':
                    # Real-time trade data from Finnhub
                    for trade in data.get('data', []):
                        symbol = trade['s']
                        price = trade['p']
                        volume = trade['v']
                        timestamp = trade['t']
                        
                        # Update cache
                        self.price_cache[symbol] = {
                            'symbol': symbol,
                            'price': price,
                            'volume': volume,
                            'timestamp': timestamp,
                            'datetime': datetime.fromtimestamp(timestamp/1000)
                        }
                        
                        # Prevent memory issues by limiting cache size
                        if len(self.price_cache) > self.max_cache_size:
                            # Remove oldest entry (first in dict iteration order)
                            oldest_key = next(iter(self.price_cache))
                            del self.price_cache[oldest_key]
                        
                        # Execute callbacks
                        if symbol in self.callbacks:
                            for callback in self.callbacks[symbol]:
                                await callback(self.price_cache[symbol])
                        
                        # Broadcast to connected clients
                        await self.broadcast_price_update(symbol, self.price_cache[symbol])
                        
        except websockets.exceptions.ConnectionClosed:
            print("⚠️ WebSocket connection closed, reconnecting...")
            self._running = False
            await asyncio.sleep(5)
            await self.connect()
        except Exception as e:
            print(f"❌ Error in listen loop: {e}")
    
    async def broadcast_price_update(self, symbol: str, data: Dict):
        """Broadcast price update to all connected clients"""
        message = json.dumps({
            'type': 'price_update',
            'symbol': symbol,
            'price': data['price'],
            'volume': data['volume'],
            'timestamp': data['timestamp'],
            'datetime': str(data['datetime'])
        })
        
        for ws in self.connections.get(symbol, set()):
            try:
                await ws.send(message)
            except Exception as e:
                print(f"⚠️ Failed to broadcast to client: {e}")
    
    def register_callback(self, symbol: str, callback: Callable):
        """Register callback function for symbol price updates"""
        if symbol not in self.callbacks:
            self.callbacks[symbol] = []
        self.callbacks[symbol].append(callback)
    
    def get_latest_price(self, symbol: str) -> Optional[Dict]:
        """Get latest cached price for symbol"""
        return self.price_cache.get(symbol)
    
    async def start_streaming(self, symbols: list):
        """Start streaming for multiple symbols"""
        await self.connect()
        
        for symbol in symbols:
            await self.subscribe(symbol)
        
        # Start listening in background
        asyncio.create_task(self.listen())
        print(f"🚀 Streaming started for {len(symbols)} symbols")
    
    async def stop_streaming(self):
        """Stop all streaming and close connection"""
        self._running = False
        
        for symbol in list(self.subscribed_symbols):
            await self.unsubscribe(symbol)
        
        if self._ws:
            await self._ws.close()
            print("🛑 WebSocket connection closed")
    
    async def get_real_time_quote(self, symbol: str) -> Dict:
        """Get real-time quote using REST API (for immediate data)"""
        import aiohttp
        
        url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={self.api_key}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'symbol': symbol,
                        'current_price': data['c'],
                        'high': data['h'],
                        'low': data['l'],
                        'open': data['o'],
                        'previous_close': data['pc'],
                        'timestamp': data['t']
                    }
        return None
