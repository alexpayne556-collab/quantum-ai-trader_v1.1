"""
Quantum_AI_Cockpit — Real-Time Mission Control API
Protocol 14.2
Integrates all analytical modules into one live data gateway with REST + WebSocket.
OPTIMIZED: 2025-11-13 - Added uvloop, endpoint timing, and JSON caching for performance.
"""

import os
import json
import logging
import asyncio
import time
import uvicorn
from functools import wraps
from typing import Optional, Callable, Any, Dict, List
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import asdict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body, Query, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from modules.utils.json_sanitize import ensure_json_safe
from modules.realtime.ws_manager import ConnectionManager
from modules.prediction_endpoint import router as predict_router
from modules.module_registry import build_registry, get_registry
from modules.unified_executor import run_module
from modules.plotly_formatter import coerce_visual_context
from modules.ticker_search import search_tickers

# Configure structured JSON logging
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","lvl":"%(levelname)s","name":"%(name)s","msg":"%(message)s"}'
)

# --- Global JSON Sanitizer (SAFE & NON-RECURSIVE) ---
def qai_sanitize(obj):
    import numpy as _np
    import pandas as _pd
    from plotly.utils import PlotlyJSONEncoder

    if obj is None or isinstance(obj, (int, float, str, bool)):
        return obj

    if isinstance(obj, list):
        return [qai_sanitize(x) for x in obj]

    if isinstance(obj, dict):
        return {str(k): qai_sanitize(v) for k, v in obj.items()}

    try:
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    try:
        if isinstance(obj, _pd.Series):
            return obj.tolist()
        if isinstance(obj, _pd.DataFrame):
            return obj.to_dict(orient='records')
    except Exception:
        pass

    try:
        if hasattr(obj, 'to_plotly_json'):
            return obj.to_plotly_json()
    except Exception:
        pass

    try:
        return json.loads(json.dumps(obj, cls=PlotlyJSONEncoder))
    except Exception:
        return str(obj)


# OPTIMIZED: Use uvloop for high-performance async event loop (2-4x faster)
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    _UVLOOP_AVAILABLE = True
except ImportError:
    _UVLOOP_AVAILABLE = False

# OPTIMIZED: Use orjson for faster JSON serialization if available
try:
    import orjson
    _ORJSON_AVAILABLE = True
except ImportError:
    _ORJSON_AVAILABLE = False

# OPTIMIZED: Import performance utilities
try:
    from modules.utils.performance import timed_endpoint
    _PERFORMANCE_UTILS_AVAILABLE = True
except ImportError:
    _PERFORMANCE_UTILS_AVAILABLE = False
    # Fallback decorator if performance utils not available
    def timed_endpoint(log_level=logging.INFO):
        def decorator(func):
            return func
        return decorator

# Logger
logger = logging.getLogger("QuantumAICockpit")
logger.setLevel(logging.INFO)

# OPTIMIZED: Log performance optimizations
if _UVLOOP_AVAILABLE:
    logger.info("✅ uvloop enabled (2-4x faster async I/O)")
else:
    logger.debug("⚠️ uvloop not available, using default event loop")

if _ORJSON_AVAILABLE:
    logger.info("✅ orjson available (faster JSON serialization)")
else:
    logger.debug("⚠️ orjson not available, using built-in json")

# Load .env for live API keys and configuration
# OS-agnostic: uses env_bootstrap for cross-platform compatibility
try:
    from backend.core.env_bootstrap import load_environment, setup_paths
    # Setup paths first (ensures sys.path is configured)
    backend_root, modules_dir = setup_paths()
    # Load environment
    env_path, env_mirrored = load_environment()
    if env_path:
        logger.debug(f"✅ Environment loaded from: {env_path}")
    else:
        logger.debug("⚠️ No .env file found, using default or existing env vars")
except ImportError:
    # Fallback if env_bootstrap not available (shouldn't happen, but be safe)
    from pathlib import Path
    import sys
    
    # Setup paths manually
    backend_root = Path(__file__).resolve().parent
    modules_dir = backend_root / "modules"
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))
    if str(modules_dir) not in sys.path:
        sys.path.insert(0, str(modules_dir))
    
    # Try standard locations
    base_paths = [
        backend_root / ".env",
        backend_root.parent / ".env",
    ]
    env_loaded = False
    loaded_path = None
    for env_path in base_paths:
        if env_path.exists():
            load_dotenv(env_path)
            loaded_path = str(env_path)
            logger.info(f"✅ Loaded .env from: {env_path}")
            env_loaded = True
            break
    if not env_loaded:
        load_dotenv()  # Fallback to default behavior
        loaded_path = None
        if not (os.getenv("FINANCIALMODELINGPREP_API_KEY") or os.getenv("DATA_PRIORITY")):
            logger.warning("⚠️ No .env file found in known paths")
    
    # Log env tip and key presence (masked)
    env_logger = logging.getLogger("Env")
    env_logger.info("env loaded from: %s", loaded_path or "<auto>")
    env_logger.info("Tip: do NOT 'source' .env; python-dotenv loads it automatically.")
    for k in ("FMP_API_KEY", "FINANCIALMODELINGPREP_API_KEY", "TWELVEDATA_API_KEY", 
              "FINNHUB_API_KEY", "TIINGO_API_KEY", "MASSIVE_API_KEY"):
        v = os.getenv(k)
        if v:
            env_logger.info("%s=***%s", k, v[-4:] if len(v) >= 4 else "****")
        else:
            env_logger.debug("%s=<missing>", k)

# Normalize any Windows-style paths in environment variables
def normalize_env_path(var_name: str) -> Optional[str]:
    """Normalize environment variable paths for cross-platform compatibility."""
    value = os.getenv(var_name)
    if value:
        try:
            return str(Path(value).resolve())
        except Exception:
            return value
    return None

# -------------------------------------------------------
# INITIALIZE APP
# -------------------------------------------------------
app = FastAPI(
    title="Quantum AI Cockpit — Mission Control",
    description="Unified real-time trading intelligence backend",
    version="14.2"
)

# Import modules dynamically (real-world engines)
# Paths already set up by env_bootstrap.setup_paths() above
# If env_bootstrap not available, paths are set up in fallback above

# Import Fusior Forecast - canonical 14-day institutional forecaster
try:
    from modules.fusior_forecast import run as fusior_forecast_run
    logger.info("✅ Fusior Forecast (14-day institutional forecaster) loaded")
except ImportError as e:
    logger.error(f"❌ fusior_forecast not found: {e}")
    fusior_forecast_run = None

# Legacy modules removed - no longer importing in main.py:
# - fusion_forecast_integrated (available for deep_analysis_lab, but main.py uses fusior_forecast)
# - pattern_detection_integrated (removed)
# - deep_red_detector_integrated (removed)
# - quantum_forecaster (replaced by fusior_forecast)

# Import ai_recommender_integrated for AI recommendations
try:
    from modules.ai_recommender_integrated import run as ai_recommender_run
    logger.info("✅ AI Recommender integrated module loaded")
except ImportError:
    try:
        from modules.ai_recommender import run as ai_recommender_run
        logger.info("✅ AI Recommender module loaded")
    except ImportError:
        ai_recommender_run = None
        logger.warning("⚠️ AI Recommender not available")

# Import unified_executor for execute_all function
try:
    from modules.unified_executor import execute_all
    logger.info("✅ Unified Executor loaded")
except ImportError:
    execute_all = None
    logger.warning("⚠️ Unified Executor not available")

logger.info("✅ QuantumForecaster-14D active, legacy modules removed")

# Validate dependencies on startup (non-blocking, no auto-install)
try:
    from backend.startup.validate_dependencies import validate_dependencies_on_startup
    dep_results = validate_dependencies_on_startup(auto_install=False)
    if dep_results.get("status") == "OK":
        logger.debug("✅ All required dependencies available")
    elif dep_results.get("missing_after"):
        logger.warning(f"⚠️ Missing dependencies detected: {', '.join(dep_results.get('missing_after', []))}")
except ImportError:
    logger.debug("Dependency validator not available, skipping validation")
except Exception as e:
    logger.warning(f"⚠️ Dependency validation error (non-blocking): {e}")

logger.info("🚀 Quantum AI Cockpit Backend Initialized (v14.2)")

# Import other modules (risk, screener, etc.) - sentiment_engine removed
try:
    from modules import (
        risk_engine,
        screener_engine,
        portfolio_manager,
        watchlist_manager,
        market_overview_api
    )
except ImportError:
    from modules import (
        risk_engine,
        screener_engine,
        portfolio_manager,
        watchlist_manager,
        market_overview_api
    )

logger.info("✅ Legacy forecasting stack removed, Fusior Forecast active")

# Import risk_engine separately if available
try:
    try:
        from modules import risk_engine
    except ImportError:
        from modules import risk_engine
except ImportError:
    risk_engine = None
    print("⚠️ risk_engine not available")

# Check for shared context availability
try:
    from modules.shared_context import get_shared_context
    SHARED_CONTEXT_AVAILABLE = True
except ImportError:
    SHARED_CONTEXT_AVAILABLE = False
    logger.warning("Shared context not available")

# Import bindings API router
try:
    try:
        from modules.generate_bindings_api import router as bindings_router
    except ImportError:
        from modules.generate_bindings_api import router as bindings_router
    app.include_router(bindings_router)
    print("✅ Module bindings API router registered")
except ImportError as e:
    print(f"⚠️  Bindings API router not available: {e}")
except Exception as e:
    print(f"⚠️  Failed to register bindings API router: {e}")

# Include prediction router
try:
    app.include_router(predict_router, tags=['predict'])
    logger.info("✅ Prediction router registered")
except Exception as e:
    logger.warning(f"⚠️ Prediction router not available: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://yourdomain.com"  # Replace with production domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# -------------------------------------------------------
# ACTIVE CONNECTIONS
# -------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        living = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
                living.append(connection)
            except WebSocketDisconnect:
                continue
        self.active_connections = living

# Use existing ConnectionManager instance
ws_manager = ConnectionManager()
manager = ws_manager  # Keep for backward compatibility

# -------------------------------------------------------
# REST ENDPOINTS
# -------------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "✅ Operational",
        "system": "Quantum AI Cockpit",
        "mode": "Mission Control (Real-Time)",
        "version": "14.2",
        "env_sources": [k for k in os.environ.keys() if 'API' in k or 'KEY' in k],
        "available_endpoints": [
            "/api/patterns/{symbol}",
            "/api/forecast/{symbol}",
            "/api/sentiment/{symbol}",
            "/api/risk/{symbol}",
            "/api/deepred/{symbol}",
            "/api/screener",
            "/api/ai_recommendation/{symbol}",
            "/api/deep_analysis/{symbol}",
            "/api/recommendation/{symbol}",
            "/api/portfolio",
            "/api/watchlist",
            "/api/market_overview",
            "/api/top_gainers",
            "/api/market_ticker",
            "/api/must_buy",
            "/api/system/modules",
            "/api/system/modules/{module_name}",
            "/api/system/bindings",
            "/ws/alerts"
        ]
    }

def normalize_symbol(symbol: str) -> str:
    """Normalize and validate ticker symbol."""
    if not symbol:
        raise ValueError("Symbol cannot be empty")
    normalized = symbol.upper().strip()
    # Basic validation: 1-5 alphanumeric characters
    if not normalized or not all(c.isalnum() for c in normalized) or len(normalized) > 5:
        raise ValueError(f"Invalid symbol format: {symbol}")
    return normalized

@app.get("/api/patterns/{symbol}")
@app.get("/api/pattern_detection/{symbol}")
@timed_endpoint()
async def patterns(symbol: str):
    """Pattern detection endpoint - deprecated, returns error."""
    logger.warning(f"Pattern detection endpoint called for {symbol} - module removed")
    return {
        "error": "Pattern detection module has been removed. Use /api/forecast for forecasting.",
        "status": "error",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }, 410  # Gone

@app.get("/api/forecast/{symbol}")
@app.get("/api/fusion_forecast/{symbol}")
@timed_endpoint()
async def forecast(symbol: str):
    """Fusior Forecast endpoint - returns 14-day day-by-day forecast."""
    t0 = time.perf_counter()
    try:
        from modules.utils.json_serializer import serialize_for_plotly
        
        normalized_symbol = normalize_symbol(symbol)
        
        # Use Fusior Forecast (Quantum Swing Engine)
        if fusior_forecast_run is None:
            raise ImportError("fusior_forecast not available")
        
        # Call fusior forecast with timeout protection (60s max)
        try:
            result = await asyncio.wait_for(
                fusior_forecast_run(normalized_symbol, horizon_days=14),
                timeout=60.0
            )
        except asyncio.TimeoutError:
            latency = (time.perf_counter() - t0) * 1000.0
            logger.error(f"Fusior Forecast timed out after 60s for {normalized_symbol}")
            error_result = {
                "module": "fusior_forecast",
                "symbol": normalized_symbol,
                "status": "error",
                "error": "TIMEOUT: Module exceeded 60s timeout",
                "trend": "neutral",
                "confidence": 0.0,
                "forecast_days": [],
                "metrics": {},
                "visual_context": {
                    "traces": [],
                    "layout": {"title": f"{normalized_symbol} Forecast Error", "template": "plotly_dark"}
                },
                "diagnostics": {"latency_ms": round(latency, 2), "failed": True},
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            return serialize_for_plotly(qai_sanitize(error_result)), 500
        
        # Result is already JSON-safe from fusior_forecast
        if not isinstance(result, dict):
            latency = (time.perf_counter() - t0) * 1000.0
            logger.error(f"Fusior Forecast returned non-dict: {type(result)}")
            error_result = {
                "module": "fusior_forecast",
                "symbol": normalized_symbol,
                "status": "error",
                "error": "INVALID_RESPONSE_TYPE",
                "trend": "neutral",
                "confidence": 0.0,
                "forecast_days": [],
                "metrics": {},
                "visual_context": {
                    "traces": [],
                    "layout": {"title": f"{normalized_symbol} Forecast Error", "template": "plotly_dark"}
                },
                "diagnostics": {"latency_ms": round(latency, 2), "failed": True},
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            return serialize_for_plotly(qai_sanitize(error_result)), 500
        
        # Ensure all required fields are present (fusior_forecast should already provide these)
        if "trend" not in result:
            result["trend"] = "neutral"
        if "confidence" not in result:
            result["confidence"] = 0.0
        if "forecast_days" not in result:
            result["forecast_days"] = []
        if "metrics" not in result:
            result["metrics"] = {}
        if "visual_context" not in result:
            result["visual_context"] = {"traces": [], "layout": {"title": f"{normalized_symbol} Forecast", "template": "plotly_dark"}}
        
        # Add timestamp if missing
        if "timestamp" not in result:
            result["timestamp"] = datetime.now(timezone.utc).isoformat()
        if "symbol" not in result:
            result["symbol"] = normalized_symbol
        
        # Emit WebSocket event if forecast completed successfully (non-blocking)
        if result.get("status") != "error" and "error" not in result:
            try:
                if 'manager' in globals() and hasattr(manager, 'active_connections') and manager.active_connections:
                    await manager.broadcast({
                        "event": "analysis_complete",
                        "module": "fusior_forecast",
                        "symbol": normalized_symbol,
                        "status": "ok",
                        "timestamp": result.get("timestamp")
                    })
            except Exception as ws_err:
                logger.debug(f"WebSocket broadcast failed: {ws_err}")
        
        return serialize_for_plotly(qai_sanitize(result))
        
    except ValueError as e:
        latency = (time.perf_counter() - t0) * 1000.0
        error_result = {
            "module": "fusior_forecast",
            "symbol": symbol.upper() if symbol else "UNKNOWN",
            "status": "error",
            "error": f"INVALID_SYMBOL: {str(e)}",
            "trend": "neutral",
            "confidence": 0.0,
            "forecast_days": [],
            "metrics": {},
            "visual_context": {
                "traces": [],
                "layout": {"title": "Forecast Error", "template": "plotly_dark"}
            },
            "diagnostics": {"latency_ms": round(latency, 2), "failed": True},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        return serialize_for_plotly(qai_sanitize(error_result)), 400
    except Exception as e:
        latency = (time.perf_counter() - t0) * 1000.0
        logger.error(f"Fusior Forecast error: {e}", exc_info=True)
        error_result = {
            "module": "fusior_forecast",
            "symbol": symbol.upper() if symbol else "UNKNOWN",
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)}",
            "trend": "neutral",
            "confidence": 0.0,
            "forecast_days": [],
            "metrics": {},
            "visual_context": {
                "traces": [],
                "layout": {"title": "Forecast Error", "template": "plotly_dark"}
            },
            "diagnostics": {"latency_ms": round(latency, 2), "failed": True},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        return serialize_for_plotly(qai_sanitize(error_result)), 500

@app.get("/api/sentiment/{symbol}")
def sentiment(symbol: str):
    """Sentiment endpoint - deprecated, returns error."""
    logger.warning(f"Sentiment endpoint called for {symbol} - module removed")
    return {
        "error": "Sentiment engine module has been removed.",
        "status": "error",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }, 410  # Gone

@app.get("/api/deepred/{symbol}")
@app.get("/api/deep_red/{symbol}")
@timed_endpoint()
async def deepred(symbol: str):
    """Deep red detector endpoint - deprecated, returns error."""
    logger.warning(f"Deep red detector endpoint called for {symbol} - module removed")
    return {
        "error": "Deep red detector module has been removed. Use /api/forecast for forecasting.",
        "status": "error",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }, 410  # Gone

@app.get("/api/screener")
@timed_endpoint()
async def screener(source: str = Query("robinhood", description="Source: portfolio, watchlist, or robinhood")):
    """
    Enhanced screener endpoint - scans for opportunities.
    Uses stock scraper to analyze top 500 Robinhood stocks if source=robinhood.
    """
    try:
        from modules.screener_engine import get_screener_data
        from modules.stock_scraper_recommender import get_stock_recommendations_for_screener, find_stocks_worth_analysis
        
        if source == "robinhood":
            # Use stock scraper to get top stocks and analyze them
            stocks = await get_stock_recommendations_for_screener()
            
            # Find stocks worth analysis (buy-low-sell-high logic)
            recommendations = await find_stocks_worth_analysis(
                min_volume=1000000,
                max_price=500.0,
                min_price=1.0,
                look_for_dips=True  # Focus on stocks in the red that will gain
            )
            
            # Convert to screener format
            results = []
            for rec in recommendations:
                results.append({
                    "symbol": rec["symbol"],
                    "price": rec["current_price"],
                    "change_pct": rec["price_change_5d"],
                    "volume": rec["volume"],
                    "rsi": rec["rsi"],
                    "volatility": rec["volatility"],
                    "score": rec["score"],
                    "recommendation": rec["recommendation"],
                    "reasons": rec["reasons"],
                    "is_good_buy": rec["score"] >= 50,  # Green background if score >= 50
                    "timestamp": rec["timestamp"]
                })
            
            return {
                "status": "success",
                "source": "robinhood",
                "count": len(results),
                "results": results,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        else:
            # Use existing screener for portfolio/watchlist
            result = get_screener_data(source)
            return result
    except Exception as e:
        logger.error(f"Screener error: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "results": [],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@app.get("/api/ai_recommendation/{symbol}")
@app.get("/api/ai_recommender/{symbol}")
@timed_endpoint()
async def recommendation(symbol: str):
    """
    Get AI recommendation with insight data and trading signals.
    Returns JSON with emoji, summary, expected_move, horizon, rationale, and trading signals
    (entry_price, stop_loss, take_profit, position_size, risk_reward_ratio, entry_timing).
    """
    try:
        from modules.utils.json_serializer import serialize_for_plotly
        
        normalized_symbol = normalize_symbol(symbol)
        # Use integrated module run function (already imported)
        result = await ai_recommender_run(normalized_symbol)
        
        # Ensure JSON serializable
        if isinstance(result, dict):
            if "module" in result:
                # Already standardized, ensure serializable
                # Trading signals are already included in metrics.trading_signal
                return serialize_for_plotly(qai_sanitize(result))
            else:
                # Extract result and serialize
                extracted = result.get("result", result)
                return serialize_for_plotly(qai_sanitize(extracted if isinstance(extracted, dict) else {"data": extracted}))
        return serialize_for_plotly(qai_sanitize({"data": result}))
    except ValueError as e:
        return {"error": str(e), "status": "error"}, 400
    except Exception as e:
        import traceback
        logger.error(f"AI recommendation error: {e}\n{traceback.format_exc()}")
        return {"error": str(e), "status": "error"}, 500

@app.get("/api/ai_recommendation/{symbol}")
@app.get("/api/recommendation/{symbol}")  # Alias for backward compatibility
async def recommendation_async(symbol: str):
    """
    Async endpoint for AI recommendation.
    Returns FinalRecommendation envelope.
    """
    try:
        from modules.utils.json_serializer import serialize_for_plotly
        
        normalized_symbol = normalize_symbol(symbol)
        # Use integrated ai_recommender - returns FinalRecommendation format
        result = await ai_recommender_run(normalized_symbol)
        
        # Result should already be FinalRecommendation format from ai_recommender_integrated
        # Ensure it's JSON serializable
        return serialize_for_plotly(qai_sanitize(result if isinstance(result, dict) else {"data": result}))
    except ValueError as e:
        from modules.contracts import FinalRecommendation, BaseModuleEnvelope
        from modules.utils.json_serializer import serialize_for_plotly
        
        error_rec = FinalRecommendation(
            symbol=normalize_symbol(symbol),
            timestamp=BaseModuleEnvelope.now_iso(),
            status="error",
            action="HOLD",
            confidence=0.0,
            expected_move_5d=0.0,
            expected_move_20d=0.0,
            holding_horizon_days=10,
            rationale_bullets=[f"Error: {str(e)}"],
            contributing_signals={},
            risk_flags={},
            error=str(e),
            metrics={},
            visual_context={}
        )
        return serialize_for_plotly(qai_sanitize(error_rec.dict())), 400
    except Exception as e:
        import traceback
        logger.error(f"AI recommendation async error: {e}\n{traceback.format_exc()}")
        from modules.contracts import FinalRecommendation, BaseModuleEnvelope
        from modules.utils.json_serializer import serialize_for_plotly
        
        error_rec = FinalRecommendation(
            symbol=normalize_symbol(symbol),
            timestamp=BaseModuleEnvelope.now_iso(),
            status="error",
            action="HOLD",
            confidence=0.0,
            expected_move_5d=0.0,
            expected_move_20d=0.0,
            holding_horizon_days=10,
            rationale_bullets=[f"Error: {str(e)}"],
            contributing_signals={},
            risk_flags={},
            error=str(e),
            metrics={},
            visual_context={}
        )
        return serialize_for_plotly(qai_sanitize(error_rec.dict())), 500

@app.get("/api/risk/{symbol}")
def risk(symbol: str):
    """Get risk analysis for a symbol."""
    if risk_engine is None:
        return {"error": "Risk engine not available", "status": "error"}, 503
    try:
        normalized_symbol = normalize_symbol(symbol)
        return risk_engine.analyze_symbol(normalized_symbol)
    except ValueError as e:
        return {"error": str(e), "status": "error"}, 400
    except Exception as e:
        return {"error": str(e), "status": "error"}, 500

# Deep Analysis Lab endpoint
@app.get("/api/deep_analysis/{symbol}")
@timed_endpoint()
async def deep_analysis(symbol: str):
    """Deep Analysis Lab endpoint - orchestrates all modules."""
    try:
        from modules.utils.json_serializer import serialize_for_plotly
        
        normalized_symbol = normalize_symbol(symbol)
        from modules.deep_analysis_lab import run as deep_analysis_run
        result = await deep_analysis_run(normalized_symbol)
        
        # Ensure JSON serializable
        return serialize_for_plotly(qai_sanitize(result if isinstance(result, dict) else {"data": result}))
    except ValueError as e:
        return {"error": str(e), "status": "error"}, 400
    except Exception as e:
        logger.error(f"Deep analysis error: {e}", exc_info=True)
        return {"error": str(e), "status": "error"}, 500

# Import and register deep analysis router (if exists)
try:
    from backend.routers.deep_analysis_router import router as deep_analysis_router
    app.include_router(deep_analysis_router)
    logger.info("✅ Deep Analysis router registered")
except ImportError as e:
    logger.debug(f"Deep Analysis router not available: {e}")
except Exception as e:
    logger.debug(f"Failed to register Deep Analysis router: {e}")

# NOTE: Portfolio and Watchlist endpoints moved to watchdog-integrated versions below
# Using new WebSocket-integrated system with portfolio_watchdog
# Old endpoints commented out to avoid conflicts:
# @app.get("/api/portfolio")
# def portfolio():
#     return portfolio_manager.get_portfolio_summary()
#
# @app.get("/api/watchlist")
# def watchlist():
#     return watchlist_manager.get_watchlist()

@app.get("/api/market_overview")
@timed_endpoint()
async def overview():
    """Market overview endpoint - returns market snapshot with gainers and movers."""
    try:
        # Get top gainers from screener engine
        from modules.screener_engine import get_top_gainers
        gainers = await get_top_gainers(limit=10)
        
        # Get market sentiment summary
        overview_data = {
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "gainers": gainers[:10],
            "market_summary": {
                "total_gainers": len(gainers),
                "top_gainer": gainers[0] if gainers else None,
            },
            "session": "regular"  # Could be enhanced to detect pre/after hours
        }
        
        return overview_data
    except Exception as e:
        logger.error(f"Market overview error: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "gainers": [],
            "market_summary": {}
        }

# Top Gainers endpoint
@app.get("/api/top_gainers")
@timed_endpoint()
async def api_top_gainers(limit: int = Query(20, ge=1, le=100, description="Number of gainers to return"), 
                          session: str = Query("regular", description="Market session")):
    """
    Get top gainers for the current market session.
    
    Returns:
        Dictionary with status, session, count, data (list of gainers), and timestamp
    """
    try:
        from modules.screener_engine import get_top_gainers
        
        limit = max(1, min(limit, 100))  # Clamp between 1 and 100
        gainers = await get_top_gainers(limit=limit, session=session)
        
        return {
            "status": "success",
            "session": session,
            "count": len(gainers),
            "data": gainers,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"/api/top_gainers failed: {e}", exc_info=True)
        return {
            "status": "error",
            "session": session,
            "count": 0,
            "data": [],
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# MSNBC-style Market Ticker endpoint
@app.get("/api/market_ticker")
@timed_endpoint()
async def market_ticker(limit: int = Query(50, ge=10, le=200, description="Number of tickers to return")):
    """
    Get MSNBC-style scrolling market ticker data.
    
    Returns:
        List of ticker data with symbol, price, change, etc. for scrolling display
    """
    try:
        from modules.market_ticker import get_market_ticker_data
        
        limit = max(10, min(limit, 200))  # Clamp between 10 and 200
        ticker_data = await get_market_ticker_data(limit=limit)
        
        return {
            "status": "success",
            "count": len(ticker_data),
            "data": ticker_data,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"/api/market_ticker failed: {e}", exc_info=True)
        return {
            "status": "error",
            "count": 0,
            "data": [],
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# Must Buy Opportunities endpoint (news scraper)
@app.get("/api/must_buy")
@timed_endpoint()
async def must_buy_opportunities(limit: int = Query(20, ge=5, le=50, description="Number of opportunities to return")):
    """
    Scrape news to find must-buy trading opportunities.
    
    Returns:
        List of opportunities with symbol, news, sentiment, confidence, etc.
    """
    try:
        from modules.must_buy_scraper import scrape_news_for_opportunities
        
        limit = max(5, min(limit, 50))  # Clamp between 5 and 50
        opportunities = await scrape_news_for_opportunities(limit=limit)
        
        return {
            "status": "success",
            "count": len(opportunities),
            "data": opportunities,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"/api/must_buy failed: {e}", exc_info=True)
        return {
            "status": "error",
            "count": 0,
            "data": [],
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# Stock Scraper & Recommender endpoints
@app.get("/api/stock_scraper/robinhood_top_500")
@timed_endpoint()
async def robinhood_top_500():
    """Get top 500 stocks from Robinhood/popular sources."""
    try:
        from modules.stock_scraper_recommender import scrape_robinhood_top_500
        stocks = await scrape_robinhood_top_500()
        return {
            "status": "success",
            "count": len(stocks),
            "stocks": stocks,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get Robinhood top 500: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "stocks": [],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@app.get("/api/stock_scraper/recommendations")
@timed_endpoint()
async def stock_recommendations(
    min_volume: int = Query(1000000, description="Minimum daily volume"),
    max_price: float = Query(500.0, description="Maximum stock price"),
    min_price: float = Query(1.0, description="Minimum stock price"),
    look_for_dips: bool = Query(True, description="Look for stocks in the red")
):
    """Get stocks worth running Deep Analysis Lab on."""
    try:
        from modules.stock_scraper_recommender import find_stocks_worth_analysis
        recommendations = await find_stocks_worth_analysis(
            min_volume=min_volume,
            max_price=max_price,
            min_price=min_price,
            look_for_dips=look_for_dips
        )
        return {
            "status": "success",
            "count": len(recommendations),
            "recommendations": recommendations,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get stock recommendations: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "recommendations": [],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# AI Trading Bot endpoints
@app.get("/api/trading_bots")
@timed_endpoint()
async def get_trading_bots():
    """Get all AI trading bots."""
    try:
        from modules.ai_trading_bot import get_bot_system
        bot_system = get_bot_system()
        bots_status = bot_system.get_all_bots_status()
        return {
            "status": "success",
            "bots": bots_status,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get trading bots: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "bots": [],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@app.post("/api/trading_bots/create")
@timed_endpoint()
async def create_trading_bot(name: str = Query(..., description="Bot name")):
    """Create a new AI trading bot."""
    try:
        from modules.ai_trading_bot import get_bot_system
        bot_system = get_bot_system()
        bot = bot_system.create_bot(name)
        return {
            "status": "success",
            "bot": bot.to_dict(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to create trading bot: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@app.post("/api/trading_bots/{bot_id}/trade")
@timed_endpoint()
async def execute_bot_trade(
    bot_id: str,
    symbol: str = Query(..., description="Stock symbol"),
    action: str = Query(..., description="BUY or SELL"),
    shares: Optional[float] = Query(None, description="Number of shares")
):
    """Execute a trade for a bot."""
    try:
        from modules.ai_trading_bot import get_bot_system
        bot_system = get_bot_system()
        trade = await bot_system.execute_trade(bot_id, symbol, action, shares)
        if trade:
            return {
                "status": "success",
                "trade": asdict(trade),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        else:
            return {
                "status": "error",
                "error": "Trade execution failed",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    except Exception as e:
        logger.error(f"Failed to execute trade: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@app.post("/api/trading_bots/{bot_id}/analyze")
@timed_endpoint()
async def analyze_and_trade_bot(
    bot_id: str,
    symbols: str = Query(..., description="Comma-separated list of symbols")
):
    """Analyze symbols and execute trades for a bot."""
    try:
        from modules.ai_trading_bot import get_bot_system
        bot_system = get_bot_system()
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        await bot_system.analyze_and_trade(bot_id, symbol_list)
        return {
            "status": "success",
            "message": "Analysis and trading completed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to analyze and trade: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# News Scraper endpoints
@app.get("/api/news/{symbol}")
@timed_endpoint()
async def get_news_for_symbol(
    symbol: str,
    limit: int = Query(20, ge=5, le=50, description="Number of articles to return")
):
    """Get news articles for a specific symbol with sentiment analysis."""
    try:
        from modules.news_scraper import scrape_news_for_symbol
        news = await scrape_news_for_symbol(symbol.upper(), limit=limit)
        return {
            "status": "success",
            "symbol": symbol.upper(),
            "count": len(news),
            "articles": news,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get news for {symbol}: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "articles": [],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@app.get("/api/news/market/general")
@timed_endpoint()
async def get_general_market_news(
    limit: int = Query(50, ge=10, le=100, description="Number of articles to return")
):
    """Get general market news from all sources."""
    try:
        from modules.news_scraper import scrape_general_market_news
        news = await scrape_general_market_news(limit=limit)
        return {
            "status": "success",
            "count": len(news),
            "articles": news,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get general market news: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "articles": [],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@app.get("/api/news/alerts")
@timed_endpoint()
async def get_news_alerts(
    symbols: str = Query(..., description="Comma-separated list of symbols"),
    min_sentiment: float = Query(0.7, ge=0.0, le=1.0, description="Minimum sentiment strength")
):
    """Get news alerts for symbols with strong sentiment."""
    try:
        from modules.news_scraper import get_news_alerts
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        alerts = await get_news_alerts(symbol_list, min_sentiment_strength=min_sentiment)
        return {
            "status": "success",
            "count": len(alerts),
            "alerts": alerts,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get news alerts: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "alerts": [],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@app.get("/api/news/sentiment/{symbol}")
@timed_endpoint()
async def get_news_sentiment(symbol: str):
    """Get aggregated news sentiment for a symbol."""
    try:
        from modules.news_scraper import get_aggregated_news_sentiment
        sentiment = await get_aggregated_news_sentiment(symbol.upper())
        return {
            "status": "success",
            "data": sentiment,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get news sentiment for {symbol}: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# Advanced Training endpoints
@app.post("/api/training/train_all")
@timed_endpoint()
async def train_all_modules_endpoint(
    symbols: Optional[str] = Query(None, description="Comma-separated list of symbols"),
    months: int = Query(12, ge=6, le=24, description="Number of months of historical data (minimum 6)"),
    force_retrain: bool = Query(True, description="Retrain until minimum 75% accuracy is met")
):
    """Train all modules with advanced AI/ML techniques.
    
    Trains all modules with:
    - Minimum 6 months of historical data
    - Pattern recognition from moving close prices
    - Self-calibration and tuning
    - Retries until minimum 75% accuracy is achieved
    """
    try:
        from modules.advanced_module_trainer import train_all_modules_advanced
        
        symbol_list = None
        if symbols:
            symbol_list = [s.strip().upper() for s in symbols.split(",")]
        
        # Ensure minimum 6 months
        months = max(months, 6)
        
        # Run training
        results = await train_all_modules_advanced(symbol_list, months=months, force_retrain=force_retrain)
        
        return {
            "status": "success",
            "message": "Training completed",
            "results": results,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@app.get("/api/training/status")
@timed_endpoint()
async def get_training_status():
    """Get training status and results for all modules."""
    try:
        from pathlib import Path
        import json
        
        RESULTS_DIR = Path(__file__).resolve().parent / "output" / "training_results"
        
        # Find latest summary
        summary_files = list(RESULTS_DIR.glob("training_summary_*.json"))
        if summary_files:
            latest = max(summary_files, key=lambda p: p.stat().st_mtime)
            with open(latest, 'r') as f:
                summary = json.load(f)
            
            return {
                "status": "success",
                "summary": summary,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        else:
            return {
                "status": "no_training",
                "message": "No training has been run yet",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    except Exception as e:
        logger.error(f"Error getting training status: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@app.get("/api/training/module/{module_name}")
@timed_endpoint()
async def get_module_training_status(module_name: str):
    """Get training status for a specific module."""
    try:
        from pathlib import Path
        import json
        
        RESULTS_DIR = Path(__file__).resolve().parent / "output" / "training_results"
        results_file = RESULTS_DIR / f"{module_name}_training_results.json"
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            return {
                "status": "success",
                "module": module_name,
                "results": results,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        else:
            return {
                "status": "not_trained",
                "module": module_name,
                "message": "Module has not been trained yet",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    except Exception as e:
        logger.error(f"Error getting module training status: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# Trading Signals endpoint
@app.get("/api/trading_signals/{symbol}")
@timed_endpoint()
async def trading_signals(symbol: str):
    """
    Get trading signals for a symbol (entry/exit prices, stop-loss, take-profit, position sizing).
    
    Returns:
        Trading signal with entry_price, stop_loss, take_profit, position_size_pct,
        risk_reward_ratio, entry_timing, etc.
    """
    try:
        from modules.trading_signals import generate_trading_signal
        from modules.fusior_forecast import run as forecast_run
        from modules.ai_recommender_integrated import run as ai_recommender_run
        
        normalized_symbol = normalize_symbol(symbol)
        
        # Get forecast and AI recommendation
        forecast_result = await forecast_run(normalized_symbol)
        ai_result = await ai_recommender_run(normalized_symbol)
        
        # Get opportunity data if available
        opportunity = None
        if forecast_result and forecast_result.get("opportunities") and len(forecast_result["opportunities"]) > 0:
            opp = forecast_result["opportunities"][0]
            if isinstance(opp, dict):
                opportunity = opp
            elif hasattr(opp, 'to_dict'):
                opportunity = opp.to_dict()
        
        # Generate trading signal
        portfolio_size = float(os.getenv("DEFAULT_PORTFOLIO_SIZE", 100000))
        trading_signal = await generate_trading_signal(
            symbol=normalized_symbol,
            forecast_data=forecast_result or {},
            ai_recommendation=ai_result or {},
            portfolio_size=portfolio_size,
            opportunity=opportunity
        )
        
        if trading_signal:
            return {
                "status": "success",
                "data": trading_signal.to_dict(),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        else:
            return {
                "status": "error",
                "error": "Could not generate trading signal",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    except Exception as e:
        logger.error(f"/api/trading_signals/{symbol} failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# -------------------------------------------------------
# DASHBOARD & SYSTEM DIAGNOSTICS ENDPOINTS
# -------------------------------------------------------

@app.get("/api/dashboard")
@timed_endpoint()
async def dashboard(symbol: str = Query("AAPL", description="Stock symbol to analyze")):
    """
    Dashboard aggregator endpoint for Deep Analysis Lab.
    Combines fusior_forecast + ai_recommender_v2 + pattern insights.
    
    Returns:
        JSON with symbol, status, forecaster, deep_lab, patterns, deep_red
    """
    normalized_symbol = symbol.upper().strip() if symbol else "AAPL"
    logger.info(f"[Dashboard] Request started for symbol={normalized_symbol}")
    
    try:
        from modules.utils.json_serializer import serialize_for_plotly
        
        # 1) Get forecast from Fusior Forecast
        if fusior_forecast_run is None:
            raise ImportError("fusior_forecast not available")
        
        try:
            forecast = await asyncio.wait_for(fusior_forecast_run(normalized_symbol, horizon_days=14), timeout=60.0)
        except asyncio.TimeoutError:
            logger.error(f"[Dashboard] Fusior Forecast timed out after 60s for symbol={normalized_symbol}")
            return serialize_for_plotly(qai_sanitize({
                "symbol": normalized_symbol,
                "status": "error",
                "error": "Dashboard analysis timed out after 60 seconds",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })), 500
        
        if forecast.get("status") != "ok":
            return serialize_for_plotly(qai_sanitize({
                "symbol": normalized_symbol,
                "status": "error",
                "error": forecast.get("error", "forecast_failed"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })), 500
        
        # 2) Get AI recommendation from ai_recommender_v2
        try:
            from modules.ai_recommender_v2 import build_recommendation_from_forecast
            recommendation = build_recommendation_from_forecast(forecast)
        except ImportError:
            logger.warning("[Dashboard] ai_recommender_v2 not available, using fallback")
            recommendation = {
                "summary": f"{forecast.get('trend', 'neutral').capitalize()} trend with {int(forecast.get('confidence', 0) * 100)}% confidence",
                "risk_brief": "Moderate risk",
                "ai_recommendation": {
                    "action": "hold",
                    "risk_level": "medium",
                    "holding_horizon_days": 14,
                    "rationale_points": ["Forecast available", "Technical indicators analyzed"],
                },
            }
        except Exception as e:
            logger.warning(f"[Dashboard] ai_recommender_v2 failed: {e}")
            recommendation = {
                "summary": "Analysis incomplete",
                "risk_brief": "Unknown",
                "ai_recommendation": {
                    "action": "hold",
                    "risk_level": "unknown",
                    "holding_horizon_days": 14,
                    "rationale_points": [f"Error: {e}"],
                },
            }
        
        # 3) Extract signals and metrics
        signals = forecast.get("signals", {}) or {}
        metrics = forecast.get("metrics", {}) or {}
        
        # 4) Build patterns summary
        patterns = {
            "fib_cluster": bool(signals.get("fib_cluster", False)),
            "harmonic_pattern": signals.get("harmonic_pattern", "none"),
            "cycle_phase": signals.get("cycle_phase", "unknown"),
            "ema_ribbon_state": signals.get("ema_ribbon_state", "neutral"),
            "fib_levels": None,  # Can be added later from forecast metadata
        }
        
        # 5) Build deep_red summary
        deep_red = {
            "is_deep_red": bool(signals.get("deep_red", False)),
            "reason": None,  # Can be enhanced later
            "score": float(metrics.get("deep_red_score", 0.0)),
        }
        
        # 6) Construct final dashboard response
        dashboard_result = {
            "symbol": normalized_symbol,
            "status": "ok",
            "forecast": forecast,  # Include full forecast envelope with visual_context
            "forecaster": {
                "trend": forecast.get("trend", "neutral"),
                "confidence": float(forecast.get("confidence", 0.0)),
                "forecast_days": forecast.get("forecast_days", []),
                "metrics": metrics,
            },
            "deep_lab": {
                "summary": recommendation.get("summary", ""),
                "risk_brief": recommendation.get("risk_brief", ""),
                "ai_recommendation": recommendation.get("ai_recommendation", {}),
            },
            "patterns": patterns,
            "deep_red": deep_red,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        logger.info(f"[Dashboard] Request completed for symbol={normalized_symbol}")
        # Ensure JSON safety before returning
        from modules.utils.json_sanitize import ensure_json_safe
        sanitized = ensure_json_safe(dashboard_result)
        return serialize_for_plotly(sanitized)
        
    except Exception as e:
        logger.error(f"[Dashboard] Error for symbol={normalized_symbol}: {e}", exc_info=True)
        response = {
            "symbol": normalized_symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "error",
            "error": str(e),
            "trend": "neutral",
            "confidence": 0.0,
            "forecast": [0.0] * 14,
            "metrics": {"atr": 0.0, "rsi": 50.0, "volatility": 0.0, "momentum": 0.0},
            "visual_context": {"traces": [], "layout": {"title": "Error", "template": "plotly_dark"}},
        }
        return serialize_for_plotly(qai_sanitize(response))

@app.get("/api/system/modules")
async def api_modules(refresh: int = Query(0, description="Force refresh (1) or use cache (0)")):
    """Get list of all registered modules (allowlisted + health-gated)."""
    # Use the router from generate_bindings_api for consistency
    from modules.generate_bindings_api import api_system_modules
    return await api_system_modules()

@app.get("/api/system/health")
def api_health():
    """Get module health status for all modules (uses decorator-based registry)."""
    try:
        from modules.module_registry import build_registry
        reg = build_registry(force=False)
        by = {m["id"]: m.get("status", "unknown") for m in reg.get("modules", [])}
        overall = 50.0 if any(v == "unknown" for v in by.values()) else 100.0
        return {"overall": overall, "summary": "OK (synthetic)", "by_module": by}
    except Exception:
        # Fallback: still return a valid payload to keep frontend alive
        return {"overall": 50.0, "summary": "OK (fallback)", "by_module": {}}

@app.post("/api/run/{module_id}")
def api_run_one(module_id: str, symbol: str = Body(..., embed=True)):
    """Run a single module for a symbol."""
    try:
        res = run_module(module_id, symbol)
        vc = (res.get("result") or {}).get("visual_context")
        if vc:
            res["result"]["visual_context"] = coerce_visual_context(vc)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/deep_analysis/{symbol}")
def api_run_many(symbol: str, modules: List[str] = Body(default=[])):
    """Run multiple modules for a symbol (Deep Analysis)."""
    if not modules:
        modules = ["fusior_forecast", "risk_engine", "ai_recommender"]
    outs = []
    for m in modules:
        try:
            r = run_module(m, symbol)
            vc = (r.get("result") or {}).get("visual_context")
            if vc:
                r["result"]["visual_context"] = coerce_visual_context(vc)
            outs.append(r)
        except Exception as e:
            outs.append({"module": m, "symbol": symbol, "status": "error", "error": str(e)})
    return {"symbol": symbol, "modules": modules, "outputs": outs}

@app.get("/api/tickers/search")
def api_ticker_search(q: str = Query("", alias="query"), limit: int = Query(20, ge=1, le=100)):
    """Search for tickers by symbol or company name."""
    return {"results": search_tickers(q, limit)}

@app.get("/api/system/module_health")
def api_module_health():
    """Legacy alias for /api/system/health (back-compat)."""
    return api_health()

@app.get("/api/system/diagnostics")
async def system_diagnostics():
    """
    System diagnostics endpoint listing all modules and their import status.
    
    Returns:
        Dictionary with module registry, health status, and dependency graph
    """
    try:
        from modules import module_registry
        
        # Discover modules if not already done
        await module_registry.discover_modules()
        
        # Get registry and health
        registry = await module_registry.get_registry()
        health = await module_registry.get_health()
        
        # Get dependency graph
        dependency_graph = module_registry.resolve_dependencies()
        execution_order = module_registry.get_execution_order(dependency_graph)
        
        return {
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "registry": registry,
            "health": health,
            "dependency_graph": dependency_graph,
            "execution_order": [
                {"level": i+1, "modules": level} 
                for i, level in enumerate(execution_order)
            ],
            "summary": {
                "total_modules": len(registry),
                "healthy_modules": health["summary"]["healthy"],
                "degraded_modules": health["summary"]["degraded"],
                "failed_modules": health["summary"]["failed"],
                "execution_levels": len(execution_order)
            }
        }
    except Exception as e:
        logger.error(f"Error in system diagnostics: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@app.get("/api/insights")
async def get_insights(symbol: str = Query("AAPL", description="Stock symbol to analyze")):
    """
    Generate actionable AI insights for a symbol using unified executor.
    
    Returns:
        Dictionary with actionable insights, recommendations, and Buy the Dip opportunities
    """
    try:
        from modules.utils.json_serializer import serialize_for_plotly
        
        # Use unified executor
        if execute_all is None:
            raise ImportError("Unified executor not available")
        
        # Execute all modules
        normalized_symbol = symbol.upper().strip() if symbol else "AAPL"
        logger.info(f"🔍 Generating insights for {normalized_symbol}")
        module_results = await execute_all(normalized_symbol, visualize=False)
        
        # Generate insights using insights_generator if available
        try:
            from modules.insights_generator import InsightsGenerator

            # Prepare context for insights generator using unified envelope structure
            if SHARED_CONTEXT_AVAILABLE:
                from modules.shared_context import get_shared_context

                ctx = get_shared_context()
                context = await ctx.get_context(normalized_symbol)
                context["dependencies"] = module_results.get("modules", {})
                context["outputs"] = module_results.get("modules", {})
            else:
                context = {
                    "symbol": normalized_symbol,
                    "dependencies": module_results.get("modules", {}),
                    "outputs": module_results.get("modules", {}),
                    "data": None,
                    "meta": {},
                }

            generator = InsightsGenerator()
            insights_result = await generator.analyze(context)

            # Extract insight data
            insight_data = insights_result.get("data", {})

            return serialize_for_plotly(
                qai_sanitize(
                    {
                        "status": "ok",
                        "symbol": normalized_symbol,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "recommendation": insight_data.get("recommendation", "HOLD"),
                        "sentiment": insight_data.get("overall_sentiment", "neutral"),
                        "confidence": insight_data.get("confidence", 0.0),
                        "insight": insight_data.get("insight", ""),
                        "reasoning": insight_data.get("reasoning", []),
                        "buy_the_dip": insight_data.get("buy_the_dip", {}),
                        "signal_breakdown": insight_data.get("signal_breakdown", {}),
                        "current_price": insight_data.get("current_price"),
                        "modules_analyzed": list(module_results.get("modules", {}).keys()),
                    }
                )
            )
        except ImportError:
            # If insights_generator not available, return module results directly
            logger.warning("Insights generator not available, returning raw module results")
            return serialize_for_plotly(qai_sanitize(module_results))
    except Exception as e:
        logger.error(f"Error generating insights: {e}", exc_info=True)
        return serialize_for_plotly(qai_sanitize({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol.upper() if symbol else "UNKNOWN"
        }))

# -------------------------------------------------------
# REAL-TIME ALERT SYSTEM (WebSocket)
# -------------------------------------------------------
@app.websocket("/ws")
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """Simple WebSocket endpoint for connection testing and general streaming."""
    await manager.connect(websocket)
    try:
        await websocket.send_json({"type": "hello", "message": "Connected to Quantum AI Cockpit"})
        while True:
            # Keep connection alive with heartbeat
            await asyncio.sleep(30)
            await websocket.send_json({"type": "heartbeat", "timestamp": datetime.now(timezone.utc).isoformat()})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket stream error: {e}")
        manager.disconnect(websocket)

@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """Streams live pattern, forecast, sentiment, and screener alerts to the front-end"""
    from modules.utils.json_serializer import serialize_for_plotly
    
    await manager.connect(websocket)
    try:
        while True:
            # AI loop broadcasting real-time alerts from integrated modules
            live_alerts = []

            # Pull real alerts from screener
            try:
                screener_data = screener_engine.scan_opportunities()
                if screener_data and isinstance(screener_data, list):
                    for item in screener_data[:5]:
                        live_alerts.append({
                            "symbol": item.get("symbol"),
                            "score": float(item.get("ai_score", 0)) if isinstance(item.get("ai_score"), (int, float)) else 0,
                            "signal": str(item.get("signal", "")),
                            "timestamp": item.get("timestamp", datetime.now(timezone.utc).isoformat()),
                            "source": "Screener"
                        })
            except Exception as e:
                logger.warning(f"Screener data error: {e}")

            # Use Fusior Forecast for forecast alerts (legacy modules removed)
            try:
                if fusior_forecast_run is not None:
                    forecast_result = await fusior_forecast_run("AAPL", horizon_days=14)
                    if isinstance(forecast_result, dict) and "error" not in forecast_result:
                        live_alerts.append({
                            "symbol": "AAPL",
                            "signal": str(forecast_result.get("trend", "neutral")),
                            "confidence": float(forecast_result.get("confidence", 0)) if isinstance(forecast_result.get("confidence"), (int, float)) else 0,
                            "volatility": float(forecast_result.get("metrics", {}).get("volatility", 0)) if isinstance(forecast_result.get("metrics", {}).get("volatility"), (int, float)) else 0,
                            "source": "QuantumForecaster"
                        })
            except Exception as e:
                logger.warning(f"QuantumForecaster error: {e}")

            # Broadcast collected live alerts (serialized for JSON)
            if live_alerts:
                serialized_alerts = serialize_for_plotly(qai_sanitize({"alerts": live_alerts}))
                await manager.broadcast(serialized_alerts)

            await asyncio.sleep(10)  # adjust to tune broadcast frequency

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast({"message": "❌ Client disconnected from alert stream"})

# -------------------------------------------------------
# PORTFOLIO & WATCHLIST WEBSOCKET ENDPOINTS
# -------------------------------------------------------
try:
    from backend.scripts.portfolio_watchdog import (
        register_portfolio_connection,
        register_watchlist_connection,
        unregister_portfolio_connection,
        unregister_watchlist_connection,
        get_watchdog
    )
    PORTFOLIO_WATCHDOG_AVAILABLE = True
except ImportError as e:
    PORTFOLIO_WATCHDOG_AVAILABLE = False
    logger.warning(f"Portfolio watchdog not available: {e}")

@app.websocket("/ws/portfolio")
async def websocket_portfolio(websocket: WebSocket):
    """Real-time portfolio updates via WebSocket."""
    if not PORTFOLIO_WATCHDOG_AVAILABLE:
        await websocket.close(code=1003, reason="Portfolio watchdog not available")
        return
    
    await websocket.accept()
    register_portfolio_connection(websocket)
    
    try:
        # Send initial portfolio data
        watchdog = get_watchdog()
        if watchdog.last_portfolio_data:
            await websocket.send_json({
                "type": "portfolio_update",
                "data": watchdog.last_portfolio_data,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        # Keep connection alive
        while True:
            try:
                # Wait for messages from client (ping/pong)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # Echo or process message
                await websocket.send_json({
                    "type": "pong",
                    "message": data,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_json({
                    "type": "ping",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
    except WebSocketDisconnect:
        unregister_portfolio_connection(websocket)
    except Exception as e:
        logger.error(f"Error in portfolio WebSocket: {e}")
        unregister_portfolio_connection(websocket)

@app.websocket("/ws/watchlist")
async def websocket_watchlist(websocket: WebSocket):
    """Real-time watchlist updates via WebSocket."""
    if not PORTFOLIO_WATCHDOG_AVAILABLE:
        await websocket.close(code=1003, reason="Portfolio watchdog not available")
        return
    
    await websocket.accept()
    register_watchlist_connection(websocket)
    
    try:
        # Send initial watchlist data
        watchdog = get_watchdog()
        if watchdog.last_watchlist_data:
            await websocket.send_json({
                "type": "watchlist_update",
                "data": watchdog.last_watchlist_data,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        # Keep connection alive
        while True:
            try:
                # Wait for messages from client (ping/pong)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # Echo or process message
                await websocket.send_json({
                    "type": "pong",
                    "message": data,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_json({
                    "type": "ping",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
    except WebSocketDisconnect:
        unregister_watchlist_connection(websocket)
    except Exception as e:
        logger.error(f"Error in watchlist WebSocket: {e}")
        unregister_watchlist_connection(websocket)

# -------------------------------------------------------
# PORTFOLIO & WATCHLIST REST ENDPOINTS
# -------------------------------------------------------

# Pydantic models for request validation
class PortfolioHolding(BaseModel):
    symbol: str
    shares: float
    cost_basis: float
    sector: Optional[str] = "Unknown"
    purchase_date: Optional[str] = None

class WatchlistTicker(BaseModel):
    symbol: str
    notes: Optional[str] = ""

@app.get("/api/portfolio")
@timed_endpoint()
async def get_portfolio():
    """
    Get portfolio with per-symbol analysis for Plotly graphs.
    Uses integrated modules for comprehensive symbol analysis.
    """
    try:
        from modules.services.portfolio_service import get_portfolio_overview
        return await asyncio.wait_for(get_portfolio_overview(), timeout=30.0)
    except ImportError as e:
        logger.debug(f"Portfolio service not available: {e}")
        # Fallback to basic portfolio data
    except asyncio.TimeoutError:
        logger.warning("Portfolio service timed out")
        return {"error": "Portfolio service timeout", "status": "error"}
    
    if not PORTFOLIO_WATCHDOG_AVAILABLE:
        return {"error": "Portfolio watchdog not available", "status": "error"}
    try:
        watchdog = get_watchdog()
        portfolio_data = watchdog._load_portfolio()
        if portfolio_data:
            portfolio_data = await asyncio.wait_for(
                watchdog._update_portfolio_prices(portfolio_data),
                timeout=30.0
            )
        
        # Ensure proper structure for frontend
        if not portfolio_data:
            portfolio_data = {
                "positions": [],
                "holdings": [],
                "symbols": [],
                "total_value": 0,
                "total_equity": 0,
                "total_cost_basis": 0,
                "total_gain_loss": 0,
                "total_gain_loss_pct": 0,
                "status": "empty"
            }
        else:
            # Normalize structure - ensure positions/holdings/symbols arrays exist
            if "positions" not in portfolio_data:
                portfolio_data["positions"] = portfolio_data.get("holdings", portfolio_data.get("symbols", []))
            if "holdings" not in portfolio_data:
                portfolio_data["holdings"] = portfolio_data.get("positions", portfolio_data.get("symbols", []))
            if "symbols" not in portfolio_data:
                portfolio_data["symbols"] = portfolio_data.get("positions", portfolio_data.get("holdings", []))
        
        return portfolio_data
    except asyncio.TimeoutError:
        logger.warning("Portfolio update timed out")
        return {"error": "Portfolio update timeout", "status": "error"}
    except Exception as e:
        logger.error(f"Error getting portfolio overview: {e}", exc_info=True)
        return {"error": str(e), "status": "error"}

@app.post("/api/portfolio/add")
async def add_portfolio_holding(holding: PortfolioHolding):
    """Add a new holding to the portfolio."""
    if not PORTFOLIO_WATCHDOG_AVAILABLE:
        return {"error": "Portfolio watchdog not available"}
    
    try:
        watchdog = get_watchdog()
        portfolio_data = watchdog._load_portfolio()
        if not portfolio_data:
            return {"error": "Portfolio data not found"}
        
        # Check if holding already exists
        symbol = holding.symbol
        existing_holding = next((h for h in portfolio_data.get("holdings", []) if h.get("symbol") == symbol), None)
        
        if existing_holding:
            # Update existing holding (add shares)
            existing_shares = float(existing_holding.get("shares", 0))
            new_shares = float(holding.shares)
            existing_holding["shares"] = existing_shares + new_shares
            # Update cost basis (weighted average)
            total_cost = (existing_shares * float(existing_holding.get("cost_basis", 0)) +
                         new_shares * float(holding.cost_basis))
            existing_holding["cost_basis"] = total_cost / existing_holding["shares"]
        else:
            # Add new holding
            portfolio_data.setdefault("holdings", []).append({
                "symbol": symbol,
                "shares": float(holding.shares),
                "cost_basis": float(holding.cost_basis),
                "current_price": 0.0,
                "equity": 0.0,
                "gain_loss": 0.0,
                "gain_loss_pct": 0.0,
                "sector": holding.sector or "Unknown",
                "purchase_date": holding.purchase_date or datetime.now(timezone.utc).isoformat()
            })
        
        # Save updated portfolio
        with open(watchdog.portfolio_file, "w") as f:
            json.dump(portfolio_data, f, indent=2)
        
        # Update prices and broadcast
        portfolio_data = await watchdog._update_portfolio_prices(portfolio_data)
        await watchdog._broadcast_portfolio_update(portfolio_data)
        
        return {"status": "success", "portfolio": portfolio_data}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/portfolio/remove")
async def remove_portfolio_holding(data: dict = Body(...)):
    """Remove a holding from the portfolio."""
    if not PORTFOLIO_WATCHDOG_AVAILABLE:
        return {"error": "Portfolio watchdog not available"}
    
    try:
        symbol = data.get("symbol")
        if not symbol:
            return {"error": "Missing required field: symbol"}
        
        watchdog = get_watchdog()
        portfolio_data = watchdog._load_portfolio()
        if not portfolio_data:
            return {"error": "Portfolio data not found"}
        
        # Remove holding
        portfolio_data["holdings"] = [h for h in portfolio_data.get("holdings", []) if h.get("symbol") != symbol]
        
        # Save updated portfolio
        with open(watchdog.portfolio_file, "w") as f:
            json.dump(portfolio_data, f, indent=2)
        
        # Update prices and broadcast
        portfolio_data = await watchdog._update_portfolio_prices(portfolio_data)
        await watchdog._broadcast_portfolio_update(portfolio_data)
        
        return {"status": "success", "portfolio": portfolio_data}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/watchlist")
@timed_endpoint()
async def get_watchlist():
    """
    Get watchlist with per-symbol analysis for Plotly graphs.
    Uses integrated modules for comprehensive symbol analysis.
    """
    try:
        from modules.services.watchlist_service import get_watchlist_overview
        return await asyncio.wait_for(get_watchlist_overview(), timeout=30.0)
    except ImportError as e:
        logger.debug(f"Watchlist service not available: {e}")
        # Fallback to basic watchlist data
    except asyncio.TimeoutError:
        logger.warning("Watchlist service timed out")
        return {"error": "Watchlist service timeout", "status": "error"}
    
    if not PORTFOLIO_WATCHDOG_AVAILABLE:
        return {"error": "Watchlist watchdog not available", "status": "error"}
    try:
        watchdog = get_watchdog()
        watchlist_data = watchdog._load_watchlist()
        if watchlist_data:
            watchlist_data = await asyncio.wait_for(
                watchdog._update_watchlist_prices(watchlist_data),
                timeout=30.0
            )
        
        # Ensure proper structure for frontend
        if not watchlist_data:
            watchlist_data = {
                "tickers": [],
                "symbols": [],
                "status": "empty"
            }
        else:
            # Normalize structure - ensure tickers/symbols arrays exist
            if "tickers" not in watchlist_data:
                watchlist_data["tickers"] = watchlist_data.get("symbols", [])
            if "symbols" not in watchlist_data:
                watchlist_data["symbols"] = watchlist_data.get("tickers", [])
        
        return watchlist_data
    except asyncio.TimeoutError:
        logger.warning("Watchlist update timed out")
        return {"error": "Watchlist update timeout", "status": "error"}
    except Exception as e:
        logger.error(f"Error getting watchlist overview: {e}", exc_info=True)
        return {"error": str(e), "status": "error"}

@app.post("/api/watchlist/add")
async def add_watchlist_ticker(ticker: WatchlistTicker):
    """Add a ticker to the watchlist."""
    if not PORTFOLIO_WATCHDOG_AVAILABLE:
        return {"error": "Watchlist watchdog not available"}
    
    try:
        watchdog = get_watchdog()
        watchlist_data = watchdog._load_watchlist()
        if not watchlist_data:
            return {"error": "Watchlist data not found"}
        
        symbol = ticker.symbol
        
        # Check if ticker already exists
        existing_ticker = next((t for t in watchlist_data.get("tickers", []) if t.get("symbol") == symbol), None)
        
        if existing_ticker:
            return {"error": f"Ticker {symbol} already in watchlist"}
        
        # Add new ticker
        watchlist_data.setdefault("tickers", []).append({
            "symbol": symbol,
            "added_date": datetime.now(timezone.utc).isoformat(),
            "notes": ticker.notes or "",
            "current_price": 0.0,
            "price_change": 0.0,
            "price_change_pct": 0.0,
            "previous_price": 0.0
        })
        
        # Save updated watchlist
        with open(watchdog.watchlist_file, "w") as f:
            json.dump(watchlist_data, f, indent=2)
        
        # Update prices and broadcast
        watchlist_data = await watchdog._update_watchlist_prices(watchlist_data)
        await watchdog._broadcast_watchlist_update(watchlist_data)
        
        return {"status": "success", "watchlist": watchlist_data}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/watchlist/remove")
async def remove_watchlist_ticker(data: dict = Body(...)):
    """Remove a ticker from the watchlist."""
    if not PORTFOLIO_WATCHDOG_AVAILABLE:
        return {"error": "Watchlist watchdog not available"}
    
    try:
        symbol = data.get("symbol")
        if not symbol:
            return {"error": "Missing required field: symbol"}
        
        watchdog = get_watchdog()
        watchlist_data = watchdog._load_watchlist()
        if not watchlist_data:
            return {"error": "Watchlist data not found"}
        
        # Remove ticker
        watchlist_data["tickers"] = [t for t in watchlist_data.get("tickers", []) if t.get("symbol") != symbol]
        
        # Save updated watchlist
        with open(watchdog.watchlist_file, "w") as f:
            json.dump(watchlist_data, f, indent=2)
        
        # Update prices and broadcast
        watchlist_data = await watchdog._update_watchlist_prices(watchlist_data)
        await watchdog._broadcast_watchlist_update(watchlist_data)
        
        return {"status": "success", "watchlist": watchlist_data}
    except Exception as e:
        return {"error": str(e)}


# -------------------------------------------------------
# HEALTH CHECK ENDPOINT
# -------------------------------------------------------
@app.get("/health")
async def health():
    """Basic health check endpoint."""
    return {"status": "ok"}


# -------------------------------------------------------
# AI TRADER ENDPOINTS
# -------------------------------------------------------
@app.get("/api/ai_trader/status")
async def ai_trader_status():
    """Get AI trader status and performance."""
    try:
        from modules.ai_trader_engine import get_trader, TradingStyle
        trader = get_trader(style=TradingStyle.MODERATE)
        metrics = trader.get_performance_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Error getting AI trader status: {e}")
        return {"error": str(e)}


@app.post("/api/ai_trader/run")
async def ai_trader_run(symbols: List[str] = Body(...), style: str = "moderate"):
    """
    Run AI trader on a list of symbols.
    
    Args:
        symbols: List of ticker symbols
        style: Trading style (conservative, moderate, aggressive)
    """
    try:
        from modules.ai_trader_engine import run_trader, TradingStyle
        
        style_enum = TradingStyle.MODERATE
        if style.lower() == "conservative":
            style_enum = TradingStyle.CONSERVATIVE
        elif style.lower() == "aggressive":
            style_enum = TradingStyle.AGGRESSIVE
        
        results = await run_trader(symbols, style=style_enum)
        return results
    except Exception as e:
        logger.error(f"Error running AI trader: {e}")
        return {"error": str(e)}


@app.get("/api/ai_trader/logs")
async def ai_trader_logs(limit: int = 50):
    """Get recent trading logs."""
    try:
        from modules.ai_trader_engine import TRADING_LOG_FILE
        logs = []
        if TRADING_LOG_FILE.exists():
            with open(TRADING_LOG_FILE, 'r') as f:
                lines = f.readlines()
                for line in lines[-limit:]:
                    try:
                        logs.append(json.loads(line.strip()))
                    except:
                        pass
        return {"logs": logs, "count": len(logs)}
    except Exception as e:
        logger.error(f"Error getting trading logs: {e}")
        return {"error": str(e)}


@app.post("/api/forecast/train_tune")
async def train_tune_forecast(symbol: str = Body(...), months: int = 12):
    """
    Train, test, backtest, and tune forecast for a symbol.
    
    Args:
        symbol: Ticker symbol
        months: Months of history (6-12)
    """
    try:
        from modules.forecast_backtest_tuner import train_test_tune_symbol
        results = await train_test_tune_symbol(symbol, months=months)
        return results
    except Exception as e:
        logger.error(f"Error training/tuning forecast: {e}")
        return {"error": str(e)}


# -------------------------------------------------------
# WEBSOCKET ENDPOINT FOR TRADING SIGNALS
# -------------------------------------------------------
@app.websocket("/ws/trading-signals")
async def trading_signals(ws: WebSocket):
    """WebSocket endpoint for real-time trading signals."""
    await ws_manager.connect(ws)
    try:
        while True:
            # Send heartbeat every 5 seconds
            await ws.send_json({"type": "heartbeat", "timestamp": datetime.now(timezone.utc).isoformat()})
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(ws)


# -------------------------------------------------------
# START PORTFOLIO WATCHDOG BACKGROUND TASK
# -------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """
    Startup event handler - must be fast and non-blocking.
    Heavy initialization should be done in background tasks.
    """
    logger.info("🚀 Quantum AI Cockpit Backend starting up...")
    
    # Start portfolio watchdog as background task (non-blocking)
    if PORTFOLIO_WATCHDOG_AVAILABLE:
        try:
            watchdog = get_watchdog()
            # Start watchdog as background task (non-blocking)
            asyncio.create_task(watchdog.run())
            logger.info("✅ Portfolio watchdog started (background task)")
        except Exception as e:
            logger.error(f"Error starting portfolio watchdog: {e}")
    
    # Log startup completion
    logger.info("✅ Application startup complete — Ready to serve requests")
    logger.info("📊 Dashboard endpoint: GET /api/dashboard?symbol=AAPL")
    logger.info("🏥 Module health endpoint: GET /api/system/module_health")
    logger.info("🔮 Prediction endpoint: POST /predict")
    logger.info("🔌 WebSocket endpoint: /ws/trading-signals")
    logger.info("🎨 Cyberpunk Preview: http://127.0.0.1:8090/cyberpunk_preview.html")
    
    # Mount static files for cyberpunk preview
    try:
        from pathlib import Path
        backend_root = Path(__file__).resolve().parent
        static_dir = backend_root / "static"
        static_dir.mkdir(exist_ok=True)
        if static_dir.exists():
            app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
            logger.info(f"✅ Static files mounted at /static from {static_dir}")
    except Exception as e:
        logger.warning(f"⚠️ Could not mount static files: {e}")


@app.get("/cyberpunk_preview.html")
async def cyberpunk_preview_redirect():
    """Redirect to cyberpunk preview page."""
    from fastapi.responses import FileResponse
    from pathlib import Path
    backend_root = Path(__file__).resolve().parent
    preview_file = backend_root / "static" / "cyberpunk_preview.html"
    if preview_file.exists():
        return FileResponse(preview_file)
    else:
        return {"error": "Preview file not found", "path": str(preview_file)}

# -------------------------------------------------------
# ENTRYPOINT
# -------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    print("[QuantumCockpit] Starting FastAPI server on 127.0.0.1:8000")
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
