"""
🔌 PRODUCTION BACKEND API
==========================
FastAPI backend serving the Golden Architecture predictions.

Endpoints:
- GET  /health          - Health check
- GET  /predict/{ticker} - Get prediction for ticker
- GET  /scan            - Scan watchlist
- GET  /report          - Daily trading report
- POST /train           - Trigger model retrain
- GET  /status          - System status
- WS   /ws/live         - WebSocket for live updates

This API is designed for:
- Local development (uvicorn)
- Cloud deployment (Docker, GCP, AWS)
- Frontend integration (React/Next.js)

Run: uvicorn production_api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import json
import os

# Import our systems
try:
    from ultimate_predictor import UltimatePredictor
    PREDICTOR_AVAILABLE = True
except ImportError:
    PREDICTOR_AVAILABLE = False

try:
    from ticker_scanner import TickerScanner
    SCANNER_AVAILABLE = True
except ImportError:
    SCANNER_AVAILABLE = False


# =============================================================================
# API MODELS
# =============================================================================
class PredictionResponse(BaseModel):
    ticker: str
    signal: str
    confidence: float
    position_size: float
    current_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_score: int
    time_horizon: str
    pattern: Optional[str] = None
    regime: Optional[str] = None
    probabilities: Dict[str, float] = {}
    explanation: List[str] = []
    timestamp: str


class ScanResponse(BaseModel):
    timestamp: str
    n_signals: int
    market_regime: str
    signals: List[Dict]


class TrainRequest(BaseModel):
    tickers: Optional[List[str]] = None
    period: str = "2y"


class StatusResponse(BaseModel):
    version: str
    is_trained: bool
    last_scan: Optional[str] = None
    training_metrics: Dict
    available_engines: Dict[str, bool]


# =============================================================================
# FASTAPI APP
# =============================================================================
app = FastAPI(
    title="Quantum AI Trader API",
    description="Golden Architecture - Ultimate Trading Predictor",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
predictor: Optional[UltimatePredictor] = None
scanner: Optional[TickerScanner] = None
connected_websockets: List[WebSocket] = []


# =============================================================================
# STARTUP / SHUTDOWN
# =============================================================================
@app.on_event("startup")
async def startup():
    global predictor, scanner
    
    print("🚀 Starting Quantum AI Trader API...")
    
    if PREDICTOR_AVAILABLE:
        predictor = UltimatePredictor(verbose=False)
        print("  ✅ UltimatePredictor loaded")
        
        # Auto-train if not trained
        if not predictor.is_trained:
            print("  ⏳ Training model (first run)...")
            predictor.train(['AAPL', 'MSFT', 'GOOGL'], period='2y')
    else:
        print("  ❌ UltimatePredictor not available")
    
    if SCANNER_AVAILABLE:
        scanner = TickerScanner(verbose=False)
        print("  ✅ TickerScanner loaded")
    else:
        print("  ❌ TickerScanner not available")
    
    print("✅ API Ready!")


@app.on_event("shutdown")
async def shutdown():
    print("Shutting down Quantum AI Trader API...")
    
    # Save state
    if predictor and predictor.is_trained:
        predictor.save()
    
    # Close websockets
    for ws in connected_websockets:
        await ws.close()


# =============================================================================
# ENDPOINTS
# =============================================================================
@app.get("/")
async def root():
    return {
        "name": "Quantum AI Trader API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "predictor_available": PREDICTOR_AVAILABLE,
        "scanner_available": SCANNER_AVAILABLE
    }


@app.get("/status", response_model=StatusResponse)
async def status():
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    return StatusResponse(
        version="1.0.0",
        is_trained=predictor.is_trained,
        last_scan=scanner.last_scan_time.isoformat() if scanner and scanner.last_scan_time else None,
        training_metrics=predictor.training_metrics,
        available_engines={
            'predictor': PREDICTOR_AVAILABLE,
            'scanner': SCANNER_AVAILABLE,
            'golden_architecture': predictor.golden_arch is not None if predictor else False
        }
    )


@app.get("/predict/{ticker}", response_model=PredictionResponse)
async def predict(ticker: str):
    """Get prediction for a single ticker"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    if not predictor.is_trained:
        raise HTTPException(status_code=503, detail="Model not trained")
    
    try:
        result = predictor.predict(ticker.upper())
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/scan", response_model=ScanResponse)
async def scan(min_confidence: float = 0.55, sectors: Optional[str] = None):
    """Scan watchlist for opportunities"""
    if not scanner:
        raise HTTPException(status_code=503, detail="Scanner not initialized")
    
    try:
        sector_list = sectors.split(',') if sectors else None
        results = scanner.run_scan(
            scan_type="api",
            sectors=sector_list,
            min_confidence=min_confidence
        )
        
        # Get market regime
        spy_pred = predictor.predict('SPY') if predictor else {}
        market_regime = spy_pred.get('regime', 'unknown')
        
        return ScanResponse(
            timestamp=datetime.now().isoformat(),
            n_signals=len(results),
            market_regime=market_regime,
            signals=[r.to_dict() for r in results]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/report")
async def daily_report():
    """Generate daily trading report"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    try:
        report = predictor.generate_daily_report()
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
async def train(request: TrainRequest, background_tasks: BackgroundTasks):
    """Trigger model retraining"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    def do_train():
        predictor.train(request.tickers, request.period)
    
    background_tasks.add_task(do_train)
    
    return {
        "status": "training_started",
        "tickers": request.tickers or "default",
        "period": request.period
    }


@app.get("/watchlist")
async def get_watchlist():
    """Get configured watchlist"""
    if scanner:
        return {
            "watchlist": scanner.WATCHLIST,
            "total_tickers": len(scanner.get_flat_watchlist())
        }
    elif predictor:
        return {
            "watchlist": predictor.DEFAULT_WATCHLIST,
            "total_tickers": len(predictor.DEFAULT_WATCHLIST)
        }
    else:
        return {"watchlist": [], "total_tickers": 0}


@app.get("/recent")
async def get_recent_signals(hours: int = 24):
    """Get recent signals from database"""
    if not scanner:
        raise HTTPException(status_code=503, detail="Scanner not initialized")
    
    signals = scanner.get_recent_signals(hours)
    return {"signals": signals, "hours": hours}


@app.get("/stats")
async def get_stats():
    """Get performance statistics"""
    if not scanner:
        raise HTTPException(status_code=503, detail="Scanner not initialized")
    
    return scanner.get_performance_stats()


# =============================================================================
# WEBSOCKET FOR LIVE UPDATES
# =============================================================================
@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    connected_websockets.append(websocket)
    
    try:
        while True:
            # Send periodic updates
            if predictor and predictor.is_trained:
                # Get SPY as market indicator
                try:
                    spy_pred = predictor.predict('SPY')
                    await websocket.send_json({
                        "type": "market_update",
                        "data": {
                            "spy_signal": spy_pred['signal'],
                            "spy_confidence": spy_pred['confidence'],
                            "regime": spy_pred.get('regime', 'unknown'),
                            "timestamp": datetime.now().isoformat()
                        }
                    })
                except:
                    pass
            
            # Wait before next update
            await asyncio.sleep(60)  # Every minute
            
    except WebSocketDisconnect:
        connected_websockets.remove(websocket)


async def broadcast_signal(signal: Dict):
    """Broadcast signal to all connected WebSocket clients"""
    for ws in connected_websockets:
        try:
            await ws.send_json({
                "type": "new_signal",
                "data": signal
            })
        except:
            pass


# =============================================================================
# RUN SERVER
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "production_api:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
