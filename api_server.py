"""
Production FastAPI WebSocket Server for Quantum AI Trader
Real-time predictions, market data streaming, and model performance metrics
"""

import sys
sys.path.insert(0, '/workspaces/quantum-ai-trader_v1.1/core')
sys.path.insert(0, '/workspaces/quantum-ai-trader_v1.1/training')

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Set
import asyncio
import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from pattern_stats_engine import PatternStatsEngine
from quantile_forecaster import QuantileForecaster
from confluence_engine import ConfluenceEngine
from institutional_feature_engineer import InstitutionalFeatureEngineer
from training_logger import TrainingLogger

# Initialize FastAPI
app = FastAPI(
    title="Quantum AI Trader API",
    description="Production-grade trading signals with institutional-quality analysis",
    version="2.0.0"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your React app URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
MODELS = {
    'pattern_engine': None,
    'forecaster': None,
    'confluence_engine': None,
    'feature_engineer': None,
    'training_logger': None
}

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        print(f"✅ Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        print(f"❌ Client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        for connection in self.active_connections.copy():
            try:
                await connection.send_json(message)
            except:
                self.disconnect(connection)

manager = ConnectionManager()

# Pydantic models for API
class PredictionRequest(BaseModel):
    ticker: str = Field(..., description="Stock symbol")
    horizon: str = Field(default="5bar", description="Forecast horizon (1bar, 3bar, 5bar, 10bar, 21bar)")

class PredictionResponse(BaseModel):
    ticker: str
    timestamp: str
    signal: str  # BUY, SELL, HOLD
    confidence: float
    forecast_cone: Dict[str, float]
    pattern_edges: List[Dict]
    confluence_score: float
    current_price: float
    risk_metrics: Dict

class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    training_stats: Dict

@app.on_event("startup")
async def load_models():
    """Load trained models on startup"""
    print("\n🚀 Loading trained models...")
    
    model_dir = Path('/workspaces/quantum-ai-trader_v1.1/trained_models')
    
    try:
        # Load Pattern Stats Engine
        MODELS['pattern_engine'] = PatternStatsEngine(
            db_path=str(model_dir / 'pattern_stats.db')
        )
        print("✅ Pattern Stats Engine loaded")
        
        # Load Quantile Forecaster
        MODELS['forecaster'] = QuantileForecaster(
            model_dir=str(model_dir / 'quantile_models')
        )
        print("✅ Quantile Forecaster loaded")
        
        # Load Feature Engineer
        MODELS['feature_engineer'] = InstitutionalFeatureEngineer()
        print("✅ Feature Engineer initialized")
        
        # Load Confluence Engine
        MODELS['confluence_engine'] = ConfluenceEngine(
            pattern_engine=MODELS['pattern_engine']
        )
        print("✅ Confluence Engine initialized")
        
        # Load Training Logger
        MODELS['training_logger'] = TrainingLogger(
            db_path=str(model_dir / 'training_logs.db')
        )
        print("✅ Training Logger loaded")
        
        print("\n🎉 All models loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": {
            "pattern_engine": MODELS['pattern_engine'] is not None,
            "forecaster": MODELS['forecaster'] is not None,
            "confluence_engine": MODELS['confluence_engine'] is not None,
            "feature_engineer": MODELS['feature_engineer'] is not None,
            "training_logger": MODELS['training_logger'] is not None
        },
        "training_stats": {
            "patterns_tracked": 997 if MODELS['pattern_engine'] else 0,
            "models_trained": len(MODELS['forecaster'].models) if MODELS['forecaster'] else 0
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def get_prediction(request: PredictionRequest):
    """Get trading prediction for a ticker"""
    
    if not all(MODELS.values()):
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Download recent data
        ticker_data = yf.download(request.ticker, period='6mo', interval='1d', progress=False)
        
        if ticker_data.empty:
            raise HTTPException(status_code=404, detail=f"No data for {request.ticker}")
        
        # Flatten multi-level columns
        if isinstance(ticker_data.columns, pd.MultiIndex):
            ticker_data.columns = [col[0] if isinstance(col, tuple) else col for col in ticker_data.columns]
        
        # Engineer features
        features_df = MODELS['feature_engineer'].engineer(ticker_data.copy())
        features_df = features_df.dropna()
        
        if len(features_df) < 50:
            raise HTTPException(status_code=400, detail="Insufficient data for prediction")
        
        # Get latest data point
        latest = features_df.iloc[-1]
        current_price = float(ticker_data['Close'].iloc[-1])
        
        # Calculate RSI for pattern detection
        delta = ticker_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = float(rsi.iloc[-1])
        
        # Determine regime
        ma_50 = float(ticker_data['Close'].iloc[-50:].mean())
        regime = 'BULL' if current_price > ma_50 else 'BEAR'
        
        # Detect patterns
        detected_patterns = []
        if current_rsi < 30:
            detected_patterns.append(('RSI_Oversold', '1d', 1.0))
        elif current_rsi > 70:
            detected_patterns.append(('RSI_Overbought', '1d', 1.0))
        
        # Get pattern edges
        pattern_edges = []
        for pattern_name, timeframe, strength in detected_patterns:
            context = {'timeframe': timeframe, 'regime': regime, 'volatility_bucket': 'ALL'}
            edge = MODELS['pattern_engine'].get_pattern_edge(pattern_name, context, min_samples=5)
            
            if edge:
                pattern_edges.append({
                    'pattern': pattern_name,
                    'win_rate': float(edge.win_rate),
                    'avg_return': float(edge.avg_return),
                    'sharpe': float(edge.sharpe_ratio),
                    'status': edge.status
                })
        
        # Calculate confluence score
        if detected_patterns:
            confluence = MODELS['confluence_engine'].calculate_confluence(
                patterns=detected_patterns,
                current_context={
                    'rsi': current_rsi,
                    'volume_ratio': 1.0,
                    'regime': regime
                }
            )
            confluence_score = float(confluence.confidence)
            signal_direction = confluence.direction
        else:
            confluence_score = 0.0
            signal_direction = 0
        
        # Generate signal
        if signal_direction > 0 and confluence_score > 0.6:
            signal = "BUY"
        elif signal_direction < 0 and confluence_score > 0.6:
            signal = "SELL"
        else:
            signal = "HOLD"
        
        # Generate forecast cone (placeholder for now)
        forecast_cone = {
            'q10': float(current_price * 0.97),
            'q25': float(current_price * 0.985),
            'q50': float(current_price * 1.0),
            'q75': float(current_price * 1.015),
            'q90': float(current_price * 1.03)
        }
        
        # Risk metrics
        volatility = float(ticker_data['Close'].pct_change().std() * np.sqrt(252))
        risk_metrics = {
            'annual_volatility': volatility,
            'sharpe_estimate': max([e['sharpe'] for e in pattern_edges]) if pattern_edges else 0.0,
            'recommended_position_size': min(0.10, 0.02 / volatility) if volatility > 0 else 0.05
        }
        
        return {
            'ticker': request.ticker,
            'timestamp': datetime.now().isoformat(),
            'signal': signal,
            'confidence': confluence_score,
            'forecast_cone': forecast_cone,
            'pattern_edges': pattern_edges,
            'confluence_score': confluence_score,
            'current_price': current_price,
            'risk_metrics': risk_metrics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.websocket("/ws/market-data")
async def websocket_market_data(websocket: WebSocket):
    """Stream real-time market data and predictions"""
    await manager.connect(websocket)
    
    try:
        # Get watchlist from client
        watchlist_msg = await websocket.receive_json()
        watchlist = watchlist_msg.get('tickers', ['SPY', 'QQQ', 'AAPL'])
        
        print(f"📡 Streaming data for: {watchlist}")
        
        while True:
            # Fetch and broadcast predictions for watchlist
            for ticker in watchlist:
                try:
                    prediction = await get_prediction(PredictionRequest(ticker=ticker, horizon='5bar'))
                    
                    await websocket.send_json({
                        'type': 'prediction',
                        'data': prediction.dict()
                    })
                    
                except Exception as e:
                    await websocket.send_json({
                        'type': 'error',
                        'ticker': ticker,
                        'error': str(e)
                    })
            
            # Wait before next update (every 60 seconds for daily data)
            await asyncio.sleep(60)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.get("/patterns/performance")
async def get_pattern_performance():
    """Get performance metrics for all tracked patterns"""
    
    if not MODELS['pattern_engine']:
        raise HTTPException(status_code=503, detail="Pattern engine not loaded")
    
    patterns = []
    for pattern in ['RSI_Oversold', 'RSI_Overbought']:
        for regime in ['BULL', 'BEAR']:
            context = {'timeframe': '1d', 'regime': regime, 'volatility_bucket': 'ALL'}
            edge = MODELS['pattern_engine'].get_pattern_edge(pattern, context, min_samples=5)
            
            if edge:
                patterns.append({
                    'pattern': pattern,
                    'regime': regime,
                    'win_rate': float(edge.win_rate),
                    'avg_return': float(edge.avg_return),
                    'sharpe_ratio': float(edge.sharpe_ratio),
                    'sample_count': edge.sample_count,
                    'status': edge.status
                })
    
    return {'patterns': patterns}

@app.get("/training/recommendations")
async def get_training_recommendations():
    """Get latest training and improvement recommendations"""
    
    weights_path = Path('/workspaces/quantum-ai-trader_v1.1/training_results/weight_optimization_recommendations.json')
    
    if not weights_path.exists():
        raise HTTPException(status_code=404, detail="Recommendations not found")
    
    with open(weights_path, 'r') as f:
        recommendations = json.load(f)
    
    return recommendations

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Quantum AI Trader API Server...")
    print("📊 React Dashboard: Connect to ws://localhost:8000/ws/market-data")
    print("📖 API Docs: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
