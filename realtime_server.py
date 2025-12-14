from typing import Dict, Any, List
import asyncio
import json
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Quantum AI Trader Realtime Server")


class SimulateTradeRequest(BaseModel):
    ticker: str
    side: str  # BUY or SELL
    entry_price: float
    atr_14: float | None = None
    stop_multiple: float = 1.5
    targets_r: List[float] = [1.0, 2.0, 3.0]


@app.get("/health")
async def health():
    return {"status": "ok", "ts": datetime.utcnow().isoformat()}


@app.post("/simulate_trade")
async def simulate_trade(req: SimulateTradeRequest):
    price = req.entry_price
    atr = req.atr_14 or max(price * 0.01, 0.1)
    stop = price - req.stop_multiple * atr if req.side.upper() == "BUY" else price + req.stop_multiple * atr
    targets = [price + r * atr if req.side.upper() == "BUY" else price - r * atr for r in req.targets_r]
    return JSONResponse({
        "ticker": req.ticker,
        "side": req.side.upper(),
        "entry": price,
        "stop": stop,
        "targets": targets,
        "params": {"atr_14": atr, "stop_multiple": req.stop_multiple}
    })


@app.get("/backfill/{ticker}")
async def backfill(ticker: str):
    # Skeleton backfill endpoint; frontend can call this to seed initial state
    return JSONResponse({
        "ticker": ticker,
        "ohlc": [],
        "overlays": [],
        "meta": {"note": "Backfill not yet implemented"}
    })


class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active:
            self.active.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        payload = json.dumps(message)
        for ws in list(self.active):
            try:
                await ws.send_text(payload)
            except Exception:
                self.disconnect(ws)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send initial hello with schema version
        await websocket.send_json({
            "type": "hello",
            "schema": "v1",
            "ts": datetime.utcnow().isoformat(),
        })
        # Basic loop handling subscribe and simulate_trade messages
        while True:
            msg = await websocket.receive_text()
            try:
                data = json.loads(msg)
            except Exception:
                await websocket.send_json({"type": "error", "message": "invalid_json"})
                continue

            mtype = data.get("type")
            if mtype == "subscribe":
                # Acknowledge subscriptions
                await websocket.send_json({
                    "type": "subscribed",
                    "tickers": data.get("tickers", []),
                    "interval": data.get("interval", "5m"),
                })
            elif mtype == "simulate_trade":
                try:
                    req = SimulateTradeRequest(**data.get("payload", {}))
                    result = await simulate_trade(req)
                    await websocket.send_json({"type": "simulate_trade/result", "payload": json.loads(result.body)})
                except Exception as e:
                    await websocket.send_json({"type": "error", "message": f"simulate_trade_failed: {e}"})
            else:
                # Echo unknown types for now
                await websocket.send_json({"type": "echo", "payload": data})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


# Optional: periodic heartbeat task
async def _heartbeat():
    while True:
        await asyncio.sleep(15)
        try:
            await manager.broadcast({"type": "heartbeat", "ts": datetime.utcnow().isoformat()})
        except Exception:
            pass


# Periodic demo overlay + ambient theme broadcaster
async def _demo_stream():
    import math
    tickers = ["AAPL", "MSFT", "NVDA", "SPY"]
    t = 0
    while True:
        await asyncio.sleep(3)
        try:
            # Ambient theme event (tempo reacts to volatility-like signal)
            tempo = 0.4 + 0.4 * max(0.0, math.sin(t / 8.0))
            await manager.broadcast({
                "type": "ambient/theme",
                "payload": {
                    "intensity": tempo,
                    "color": "bull" if math.sin(t / 13.0) > 0 else "bear",
                    "ts": datetime.utcnow().isoformat(),
                }
            })

            # Overlay delta (mock forecast cone and a pattern rectangle)
            for tk in tickers:
                await manager.broadcast({
                    "type": "overlay_delta",
                    "ticker": tk,
                    "payload": {
                        "ops": [
                            {
                                "op": "upsert",
                                "kind": "forecast_cone",
                                "id": f"cone_{tk}",
                                "data": {
                                    "horizon_days": 24,
                                    "conf": 0.6 + 0.3 * max(0.0, math.cos((t % 60)/10.0)),
                                    "ts": datetime.utcnow().isoformat()
                                }
                            },
                            {
                                "op": "upsert",
                                "kind": "pattern_box",
                                "id": f"pat_{tk}_{t%5}",
                                "data": {
                                    "type": "BULLISH" if (t % 2)==0 else "BEARISH",
                                    "confidence": 0.55 + 0.4 * max(0.0, math.sin((t % 50)/7.0)),
                                    "window_bars": 8
                                }
                            }
                        ]
                    }
                })
            t += 1
        except Exception:
            pass


@app.on_event("startup")
async def _on_start():
    asyncio.create_task(_heartbeat())
    asyncio.create_task(_demo_stream())
