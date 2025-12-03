"""
PRODUCTION BACKEND API
======================
FastAPI backend for the Quantum AI Trader.
Serves predictions, manages background scans, and integrates with frontend.

Run with:
    uvicorn backend_api:app --reload --host 0.0.0.0 --port 8000
"""

import os
import json
import asyncio
from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime

# Import our engines
from ultimate_predictor import UltimatePredictor
from ticker_scanner import TickerScanner

app = FastAPI(
    title="Quantum AI Trader API",
    description="Backend for the Ultimate AI Trading System",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class PredictionResponse(BaseModel):
    ticker: str
    signal: str
    confidence: float
    position_size: float
    regime: str
    pattern: str
    probabilities: dict
    explanation: List[str]
    timestamp: str

class ScanStatus(BaseModel):
    last_scan: str
    total_tickers: int
    status: str

# Global State
SCAN_RESULTS_FILE = "scan_results.json"
is_scanning = False

def load_results():
    if os.path.exists(SCAN_RESULTS_FILE):
        with open(SCAN_RESULTS_FILE, 'r') as f:
            return json.load(f)
    return []

@app.get("/")
async def root():
    return {"message": "Quantum AI Trader API is running 🚀"}

@app.get("/api/predictions", response_model=List[PredictionResponse])
async def get_predictions(signal: Optional[str] = None):
    """Get all latest predictions, optionally filtered by signal"""
    results = load_results()
    if signal:
        results = [r for r in results if r['signal'] == signal]
    return results

@app.get("/api/predict/{ticker}", response_model=PredictionResponse)
async def predict_ticker(ticker: str):
    """Get live prediction for a specific ticker"""
    try:
        predictor = UltimatePredictor()
        result = predictor.predict(ticker.upper())
        result['ticker'] = ticker.upper()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/scan")
async def trigger_scan(background_tasks: BackgroundTasks):
    """Trigger a full market scan in the background"""
    global is_scanning
    if is_scanning:
        raise HTTPException(status_code=400, detail="Scan already in progress")
    
    background_tasks.add_task(run_scan)
    return {"message": "Scan started in background"}

@app.get("/api/status", response_model=ScanStatus)
async def get_status():
    """Get system status"""
    results = load_results()
    last_modified = datetime.fromtimestamp(os.path.getmtime(SCAN_RESULTS_FILE)).isoformat() if os.path.exists(SCAN_RESULTS_FILE) else "Never"
    
    return {
        "last_scan": last_modified,
        "total_tickers": len(results),
        "status": "scanning" if is_scanning else "idle"
    }

def run_scan():
    global is_scanning
    is_scanning = True
    try:
        scanner = TickerScanner()
        scanner.scan()
    finally:
        is_scanning = False

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
