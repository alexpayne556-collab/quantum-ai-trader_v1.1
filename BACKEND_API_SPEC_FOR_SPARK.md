# 🎯 Backend API Specification for Spark Frontend

**Purpose**: Clean, production-ready REST API specification for building the Swing Trading Cockpit frontend  
**Backend Stack**: Flask + Python ML Models  
**Frontend**: Spark (React/Next.js) - to be built in parallel with training optimization  
**Date**: December 9, 2025

---

## 🏗️ System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    SPARK FRONTEND                           │
│              (React/Next.js Dashboard)                      │
│  • Real-time Charts • Signals Feed • Portfolio View        │
│  • Risk Metrics • Trade History • AI Chat                  │
└─────────────────────┬───────────────────────────────────────┘
                      │ REST API (JSON)
                      │ WebSocket (Real-time)
┌─────────────────────▼───────────────────────────────────────┐
│                  FLASK API SERVER                           │
│              (Port 5000, CORS Enabled)                      │
│  • Authentication • Rate Limiting • Caching                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┬─────────────┐
        │             │             │             │
┌───────▼──────┐ ┌───▼──────┐ ┌───▼──────┐ ┌───▼──────┐
│ ML ENSEMBLE  │ │  DATA    │ │ REGIME   │ │ PORTFOLIO│
│  3 Models    │ │ FETCHER  │ │CLASSIFIER│ │ MANAGER  │
│ XGB+RF+GB    │ │5 Sources │ │10 Regimes│ │Risk Mgmt │
└──────────────┘ └──────────┘ └──────────┘ └──────────┘
        │             │             │             │
        └─────────────┼─────────────┴─────────────┘
                      │
              ┌───────▼────────┐
              │  PostgreSQL    │
              │  + Redis Cache │
              └────────────────┘
```

---

## 📡 Core API Endpoints

### Base URL
```
http://localhost:5000/api/v1
```

### Authentication
All endpoints require API key in header:
```
X-API-Key: <user_api_key>
```

---

## 1️⃣ SIGNALS & PREDICTIONS

### `GET /signals/latest`
**Purpose**: Get latest trading signals from ML ensemble  
**Use Case**: Display on dashboard home, auto-refresh every 5 minutes

**Request**:
```bash
GET /api/v1/signals/latest?limit=20&confidence_min=0.7
```

**Query Parameters**:
- `limit` (optional, default=20): Number of signals to return
- `confidence_min` (optional, default=0.6): Minimum confidence threshold (0-1)
- `sector` (optional): Filter by sector (e.g., "Technology", "Healthcare")
- `regime` (optional): Filter by market regime (e.g., "BULL_LOW_VOL", "BEAR_HIGH_VOL")

**Response** (200 OK):
```json
{
  "timestamp": "2025-12-09T03:30:00Z",
  "market_regime": "CHOPPY_HIGH_VOL",
  "signals": [
    {
      "ticker": "AAPL",
      "signal": "BUY",
      "confidence": 0.85,
      "predicted_return": 0.034,
      "current_price": 277.89,
      "target_price": 287.34,
      "stop_loss": 272.32,
      "position_size": 0.08,
      "risk_reward_ratio": 1.52,
      "sector": "Technology",
      "market_cap": "2.8T",
      "avg_volume": "45.2M",
      "regime": "CHOPPY_HIGH_VOL",
      "model_votes": {
        "xgboost": "BUY",
        "random_forest": "BUY",
        "gradient_boosting": "HOLD"
      },
      "feature_importance": {
        "RSI_14": 0.18,
        "MACD_signal": 0.15,
        "volume_trend": 0.12
      },
      "generated_at": "2025-12-09T03:28:45Z",
      "expires_at": "2025-12-09T08:28:45Z"
    }
  ],
  "total_signals": 18,
  "filtered_count": 18,
  "regime_config": {
    "max_position_size": 0.10,
    "stop_loss_pct": 0.08,
    "take_profit_pct": 0.15
  }
}
```

---

### `GET /signals/ticker/{ticker}`
**Purpose**: Get signal history and prediction for specific ticker  
**Use Case**: Ticker detail page, chart annotations

**Request**:
```bash
GET /api/v1/signals/ticker/AAPL?days=30
```

**Query Parameters**:
- `days` (optional, default=30): Historical signal period

**Response** (200 OK):
```json
{
  "ticker": "AAPL",
  "current_signal": {
    "signal": "BUY",
    "confidence": 0.85,
    "predicted_return": 0.034,
    "target_price": 287.34,
    "stop_loss": 272.32,
    "generated_at": "2025-12-09T03:28:45Z"
  },
  "signal_history": [
    {
      "date": "2025-12-08",
      "signal": "HOLD",
      "confidence": 0.62,
      "actual_return": 0.012,
      "outcome": "correct"
    },
    {
      "date": "2025-12-07",
      "signal": "BUY",
      "confidence": 0.78,
      "actual_return": 0.045,
      "outcome": "correct"
    }
  ],
  "performance_stats": {
    "total_signals": 45,
    "correct_signals": 28,
    "accuracy": 0.622,
    "win_rate": 0.615,
    "avg_return": 0.023,
    "sharpe_ratio": 1.82
  }
}
```

---

### `POST /signals/generate`
**Purpose**: Generate new signals for custom ticker list  
**Use Case**: Manual refresh, custom watchlist analysis

**Request**:
```bash
POST /api/v1/signals/generate
Content-Type: application/json

{
  "tickers": ["AAPL", "TSLA", "NVDA"],
  "confidence_min": 0.7,
  "include_features": true
}
```

**Response** (200 OK):
```json
{
  "job_id": "sig_20251209_032845_abc123",
  "status": "completed",
  "execution_time_ms": 2340,
  "signals": [ /* same format as /signals/latest */ ]
}
```

---

## 2️⃣ MARKET DATA & CHARTS

### `GET /market/ohlcv/{ticker}`
**Purpose**: Get OHLCV data with technical indicators  
**Use Case**: Candlestick charts, technical analysis overlays

**Request**:
```bash
GET /api/v1/market/ohlcv/AAPL?interval=1h&period=5d&indicators=RSI,MACD,BB
```

**Query Parameters**:
- `interval`: `1m`, `5m`, `15m`, `1h`, `4h`, `1d`
- `period`: `1d`, `5d`, `1mo`, `3mo`, `1y`, `2y`
- `indicators` (optional): Comma-separated list (RSI, MACD, BB, EMA_20, etc.)

**Response** (200 OK):
```json
{
  "ticker": "AAPL",
  "interval": "1h",
  "data": [
    {
      "timestamp": "2025-12-09T03:00:00Z",
      "open": 277.50,
      "high": 278.90,
      "low": 276.80,
      "close": 277.89,
      "volume": 2345678,
      "indicators": {
        "RSI_14": 58.3,
        "MACD": 1.23,
        "MACD_signal": 1.10,
        "BB_upper": 285.40,
        "BB_middle": 278.20,
        "BB_lower": 271.00,
        "EMA_20": 276.50
      }
    }
  ],
  "total_bars": 32,
  "data_source": "twelve_data"
}
```

---

### `GET /market/regime`
**Purpose**: Get current market regime classification  
**Use Case**: Dashboard header, regime indicator widget

**Request**:
```bash
GET /api/v1/market/regime
```

**Response** (200 OK):
```json
{
  "current_regime": "CHOPPY_HIGH_VOL",
  "confidence": 0.88,
  "vix": 15.41,
  "spy_return_20d": 0.023,
  "yield_curve_10y2y": 0.45,
  "regime_since": "2025-12-05T14:30:00Z",
  "regime_duration_hours": 85,
  "regimes": [
    {
      "name": "BULL_LOW_VOL",
      "probability": 0.05,
      "description": "Strong uptrend, low volatility (VIX < 15, SPY > 2%)"
    },
    {
      "name": "CHOPPY_HIGH_VOL",
      "probability": 0.88,
      "description": "Sideways, high volatility (VIX > 20, SPY ±2%)"
    }
  ],
  "trading_config": {
    "recommended_position_size": 0.08,
    "stop_loss_pct": 0.08,
    "take_profit_pct": 0.12,
    "max_positions": 8
  },
  "updated_at": "2025-12-09T03:28:00Z"
}
```

---

## 3️⃣ PORTFOLIO & POSITIONS

### `GET /portfolio/summary`
**Purpose**: Get portfolio overview and performance  
**Use Case**: Portfolio dashboard, account header

**Request**:
```bash
GET /api/v1/portfolio/summary
```

**Response** (200 OK):
```json
{
  "account": {
    "id": "paper_123456",
    "type": "paper",
    "total_value": 105234.56,
    "cash": 42100.00,
    "equity": 63134.56,
    "buying_power": 84200.00,
    "day_return": 1234.56,
    "day_return_pct": 1.18,
    "total_return": 5234.56,
    "total_return_pct": 5.23
  },
  "positions": [
    {
      "ticker": "AAPL",
      "quantity": 100,
      "avg_entry_price": 270.50,
      "current_price": 277.89,
      "market_value": 27789.00,
      "unrealized_pl": 739.00,
      "unrealized_pl_pct": 2.73,
      "position_size_pct": 26.4,
      "signal_entry": "BUY",
      "entry_date": "2025-12-06T10:30:00Z",
      "target_price": 287.34,
      "stop_loss": 272.32
    }
  ],
  "risk_metrics": {
    "total_exposure": 0.60,
    "max_position_size": 0.26,
    "portfolio_beta": 1.12,
    "sharpe_ratio": 1.82,
    "max_drawdown": 0.08,
    "var_95": -2340.00
  },
  "updated_at": "2025-12-09T03:30:00Z"
}
```

---

### `POST /portfolio/order`
**Purpose**: Execute trade order (paper trading)  
**Use Case**: Click "Execute Trade" button, follow ML signal

**Request**:
```bash
POST /api/v1/portfolio/order
Content-Type: application/json

{
  "ticker": "AAPL",
  "side": "buy",
  "quantity": 100,
  "order_type": "limit",
  "limit_price": 277.50,
  "time_in_force": "gtc",
  "stop_loss": 272.32,
  "take_profit": 287.34,
  "signal_id": "sig_20251209_032845_abc123"
}
```

**Response** (201 Created):
```json
{
  "order_id": "ord_abc123xyz",
  "status": "filled",
  "ticker": "AAPL",
  "side": "buy",
  "quantity": 100,
  "filled_quantity": 100,
  "filled_price": 277.52,
  "total_cost": 27752.00,
  "commission": 0.00,
  "submitted_at": "2025-12-09T03:30:15Z",
  "filled_at": "2025-12-09T03:30:16Z"
}
```

---

### `GET /portfolio/history`
**Purpose**: Get trade history and closed positions  
**Use Case**: Performance analysis, trade journal

**Request**:
```bash
GET /api/v1/portfolio/history?days=30&status=closed
```

**Response** (200 OK):
```json
{
  "trades": [
    {
      "trade_id": "trd_xyz789",
      "ticker": "TSLA",
      "entry_date": "2025-12-01T10:30:00Z",
      "exit_date": "2025-12-05T14:20:00Z",
      "entry_price": 245.30,
      "exit_price": 256.80,
      "quantity": 50,
      "realized_pl": 575.00,
      "realized_pl_pct": 4.69,
      "hold_duration_hours": 99,
      "signal_entry": "BUY",
      "signal_confidence": 0.82,
      "exit_reason": "take_profit",
      "regime_entry": "BULL_LOW_VOL",
      "regime_exit": "CHOPPY_HIGH_VOL"
    }
  ],
  "summary": {
    "total_trades": 24,
    "winning_trades": 15,
    "losing_trades": 9,
    "win_rate": 0.625,
    "avg_win": 645.30,
    "avg_loss": -234.50,
    "profit_factor": 2.75,
    "total_realized_pl": 5234.56
  }
}
```

---

## 4️⃣ BACKTESTING & ANALYSIS

### `POST /backtest/run`
**Purpose**: Run backtest on trained models  
**Use Case**: Strategy validation, parameter optimization

**Request**:
```bash
POST /api/v1/backtest/run
Content-Type: application/json

{
  "tickers": ["AAPL", "TSLA", "NVDA"],
  "start_date": "2024-01-01",
  "end_date": "2024-12-31",
  "initial_capital": 100000,
  "confidence_threshold": 0.7,
  "max_position_size": 0.10,
  "stop_loss_pct": 0.08,
  "take_profit_pct": 0.15
}
```

**Response** (200 OK):
```json
{
  "backtest_id": "bt_20251209_033045",
  "status": "completed",
  "execution_time_ms": 8920,
  "results": {
    "total_trades": 156,
    "winning_trades": 92,
    "losing_trades": 64,
    "win_rate": 0.590,
    "total_return": 0.183,
    "sharpe_ratio": 1.76,
    "max_drawdown": 0.142,
    "profit_factor": 2.21,
    "avg_trade_duration_hours": 48,
    "final_equity": 118300.00
  },
  "equity_curve": [
    {"date": "2024-01-01", "equity": 100000.00},
    {"date": "2024-01-02", "equity": 100234.00}
  ],
  "monthly_returns": [
    {"month": "2024-01", "return": 0.023},
    {"month": "2024-02", "return": -0.012}
  ]
}
```

---

### `GET /backtest/results/{backtest_id}`
**Purpose**: Retrieve stored backtest results  
**Use Case**: View historical backtests, compare strategies

**Request**:
```bash
GET /api/v1/backtest/results/bt_20251209_033045
```

**Response**: Same as `/backtest/run` response

---

## 5️⃣ AI CHAT & INSIGHTS

### `POST /ai/chat`
**Purpose**: AI-powered market analysis and strategy advice  
**Use Case**: Chat widget, ask questions about signals

**Request**:
```bash
POST /api/v1/ai/chat
Content-Type: application/json

{
  "message": "Why did you recommend buying AAPL?",
  "context": {
    "ticker": "AAPL",
    "signal_id": "sig_20251209_032845_abc123"
  },
  "conversation_id": "conv_abc123" 
}
```

**Response** (200 OK):
```json
{
  "response": "I recommended buying AAPL because:\n1. **Strong Technical Setup**: RSI at 58.3 (neutral-bullish), MACD crossed above signal line\n2. **High Confidence**: Ensemble confidence 85% (XGBoost and Random Forest both voted BUY)\n3. **Favorable Regime**: Current CHOPPY_HIGH_VOL regime historically shows 61% win rate for tech stocks\n4. **Risk/Reward**: 1.52:1 ratio with stop loss at $272.32, target at $287.34\n5. **Volume Confirmation**: Above-average volume trend indicates accumulation\n\nRecommended position size: 8% of portfolio with tight stop loss.",
  "conversation_id": "conv_abc123",
  "model_used": "perplexity-sonar",
  "confidence": 0.92,
  "sources": [
    {"type": "technical_indicator", "data": {"RSI_14": 58.3, "MACD": 1.23}},
    {"type": "model_prediction", "confidence": 0.85}
  ]
}
```

---

### `POST /ai/explain-signal`
**Purpose**: Get detailed explanation of a signal  
**Use Case**: "Why this signal?" button on signal cards

**Request**:
```bash
POST /api/v1/ai/explain-signal
Content-Type: application/json

{
  "signal_id": "sig_20251209_032845_abc123"
}
```

**Response** (200 OK):
```json
{
  "signal_id": "sig_20251209_032845_abc123",
  "ticker": "AAPL",
  "explanation": {
    "summary": "Strong BUY signal based on bullish momentum and high model confidence",
    "technical_factors": [
      "RSI at 58.3 showing upward momentum without overbought conditions",
      "MACD crossover above signal line (bullish confirmation)",
      "Price above 20-day EMA indicating uptrend"
    ],
    "model_reasoning": {
      "xgboost": "High probability (0.87) due to volume surge and momentum indicators",
      "random_forest": "Confirmed BUY (0.82) - similar historical patterns led to 3.4% avg gain",
      "gradient_boosting": "Neutral HOLD (0.63) - cautious due to recent volatility"
    },
    "risk_factors": [
      "High VIX (15.4) suggests increased market volatility",
      "Position size limited to 8% due to choppy regime"
    ],
    "historical_performance": "Past 30 AAPL BUY signals: 68% win rate, +2.8% avg return"
  }
}
```

---

## 6️⃣ SYSTEM STATUS & HEALTH

### `GET /system/health`
**Purpose**: Check API and model health  
**Use Case**: System status page, monitoring dashboard

**Request**:
```bash
GET /api/v1/system/health
```

**Response** (200 OK):
```json
{
  "status": "healthy",
  "timestamp": "2025-12-09T03:30:00Z",
  "components": {
    "api_server": {"status": "up", "latency_ms": 12},
    "database": {"status": "up", "latency_ms": 8},
    "redis_cache": {"status": "up", "latency_ms": 2},
    "ml_models": {
      "status": "loaded",
      "models": ["xgboost", "random_forest", "gradient_boosting"],
      "last_trained": "2025-12-09T00:00:00Z",
      "version": "v1.1.0"
    },
    "data_sources": {
      "twelve_data": {"status": "up", "remaining_calls": 724},
      "finnhub": {"status": "up", "remaining_calls": 60},
      "fred": {"status": "up"},
      "alpaca": {"status": "up", "account_type": "paper"}
    }
  },
  "performance": {
    "avg_response_time_ms": 145,
    "requests_per_minute": 28,
    "cache_hit_rate": 0.82
  }
}
```

---

### `GET /system/metrics`
**Purpose**: Get model performance metrics  
**Use Case**: Model performance dashboard

**Request**:
```bash
GET /api/v1/system/metrics?period=7d
```

**Response** (200 OK):
```json
{
  "period": "7d",
  "model_performance": {
    "accuracy": 0.622,
    "precision": 0.681,
    "recall": 0.587,
    "f1_score": 0.630,
    "roc_auc": 0.712,
    "win_rate": 0.615,
    "avg_return_per_trade": 0.023,
    "sharpe_ratio": 1.82
  },
  "signal_stats": {
    "total_signals_generated": 342,
    "avg_confidence": 0.74,
    "signals_by_type": {
      "BUY": 128,
      "SELL": 94,
      "HOLD": 120
    }
  },
  "regime_distribution": {
    "BULL_LOW_VOL": 0.15,
    "CHOPPY_HIGH_VOL": 0.45,
    "BEAR_HIGH_VOL": 0.08
  }
}
```

---

## 7️⃣ WEBSOCKET (Real-Time Updates)

### Connection
```javascript
const ws = new WebSocket('ws://localhost:5000/ws');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'subscribe',
    channels: ['signals', 'prices', 'portfolio']
  }));
};
```

### Message Types

**Signal Update**:
```json
{
  "type": "signal",
  "data": {
    "ticker": "AAPL",
    "signal": "BUY",
    "confidence": 0.85,
    "timestamp": "2025-12-09T03:30:00Z"
  }
}
```

**Price Update**:
```json
{
  "type": "price",
  "ticker": "AAPL",
  "price": 277.89,
  "change": 1.23,
  "change_pct": 0.44,
  "timestamp": "2025-12-09T03:30:15Z"
}
```

**Portfolio Update**:
```json
{
  "type": "portfolio",
  "data": {
    "total_value": 105234.56,
    "day_return_pct": 1.18,
    "timestamp": "2025-12-09T03:30:00Z"
  }
}
```

---

## 🔒 Authentication & Security

### API Key Generation
```bash
POST /api/v1/auth/generate-key
Content-Type: application/json

{
  "user_id": "user123",
  "permissions": ["read", "write", "trade"]
}
```

**Response**:
```json
{
  "api_key": "qat_abc123xyz789...",
  "created_at": "2025-12-09T03:30:00Z",
  "expires_at": "2026-12-09T03:30:00Z"
}
```

### Rate Limits
- **Free Tier**: 60 requests/minute, 1000 requests/hour
- **Premium**: 300 requests/minute, 10000 requests/hour
- **Headers**: `X-RateLimit-Remaining`, `X-RateLimit-Reset`

---

## 📊 Data Models

### Signal Object
```typescript
interface Signal {
  ticker: string;
  signal: 'BUY' | 'SELL' | 'HOLD';
  confidence: number; // 0-1
  predicted_return: number;
  current_price: number;
  target_price: number;
  stop_loss: number;
  position_size: number; // 0-1 (% of portfolio)
  risk_reward_ratio: number;
  sector: string;
  market_cap: string;
  regime: string;
  model_votes: {
    xgboost: string;
    random_forest: string;
    gradient_boosting: string;
  };
  generated_at: string; // ISO 8601
  expires_at: string; // ISO 8601
}
```

### Position Object
```typescript
interface Position {
  ticker: string;
  quantity: number;
  avg_entry_price: number;
  current_price: number;
  market_value: number;
  unrealized_pl: number;
  unrealized_pl_pct: number;
  position_size_pct: number;
  entry_date: string; // ISO 8601
  target_price?: number;
  stop_loss?: number;
}
```

---

## 🚀 Quick Start for Frontend Devs

### 1. Start Backend (Local Dev)
```bash
cd /workspaces/quantum-ai-trader_v1.1
source .venv/bin/activate
python api_server.py
# Runs on http://localhost:5000
```

### 2. Test API (cURL)
```bash
# Get latest signals
curl -X GET "http://localhost:5000/api/v1/signals/latest?limit=10" \
  -H "X-API-Key: your_api_key_here"

# Get market regime
curl -X GET "http://localhost:5000/api/v1/market/regime" \
  -H "X-API-Key: your_api_key_here"
```

### 3. Frontend Integration (React Example)
```typescript
// services/api.ts
const API_BASE = 'http://localhost:5000/api/v1';
const API_KEY = process.env.NEXT_PUBLIC_API_KEY;

export async function getLatestSignals(limit = 20) {
  const response = await fetch(
    `${API_BASE}/signals/latest?limit=${limit}`,
    {
      headers: { 'X-API-Key': API_KEY }
    }
  );
  return response.json();
}

export async function getMarketRegime() {
  const response = await fetch(`${API_BASE}/market/regime`, {
    headers: { 'X-API-Key': API_KEY }
  });
  return response.json();
}

// components/SignalFeed.tsx
import { useEffect, useState } from 'react';
import { getLatestSignals } from '@/services/api';

export function SignalFeed() {
  const [signals, setSignals] = useState([]);

  useEffect(() => {
    const fetchSignals = async () => {
      const data = await getLatestSignals(20);
      setSignals(data.signals);
    };

    fetchSignals();
    const interval = setInterval(fetchSignals, 300000); // Refresh every 5 min
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="signal-feed">
      {signals.map(signal => (
        <SignalCard key={signal.ticker} signal={signal} />
      ))}
    </div>
  );
}
```

---

## 🎨 Recommended Frontend Components

### 1. **Dashboard Home**
- Real-time signal feed (auto-refresh)
- Market regime indicator
- Portfolio summary card
- Today's P&L chart

### 2. **Signals View**
- Filterable signal table (sector, confidence, regime)
- Signal cards with confidence meters
- "Execute Trade" buttons
- AI explanation modals

### 3. **Portfolio View**
- Position cards with P&L
- Allocation pie chart
- Risk metrics dashboard
- Trade history table

### 4. **Chart View**
- TradingView-style candlestick charts
- Technical indicators overlay
- Signal annotations (buy/sell arrows)
- Volume bars

### 5. **Backtest Lab**
- Parameter input form
- Results visualization (equity curve, drawdown chart)
- Trade list table
- Strategy comparison

### 6. **AI Chat**
- Chat interface with message history
- Quick question buttons ("Why this signal?", "What's the risk?")
- Signal context awareness
- Code block rendering for technical details

---

## 📝 Environment Variables (.env)

```bash
# API Server
API_PORT=5000
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
SECRET_KEY=your_secret_key_here

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/quantum_trader
REDIS_URL=redis://localhost:6379/0

# Market Data APIs (already configured)
TWELVE_DATA_API_KEY=d19ebe6706614dd897e66aa416900fd3
FINNHUB_API_KEY=d3qj8p9r01quv7kb49igd3qj8p9r01quv7kb49j0
ALPHA_VANTAGE_API_KEY=gL_pHRAJ6SQK0AK2MD0rSuP653GW733l
FRED_API_KEY=32829556722ddb7fd681d84ad9192026

# Trading (Paper)
ALPACA_API_KEY=PKRNFP4NMO4O2CDYRRBGLH2EFU
ALPACA_SECRET_KEY=7b85Wo48enKp36PkaB4fC1nZyHxscRSMNHX7ktkCuZjL
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# AI
PERPLEXITY_API_KEY=your_perplexity_api_key_hereSugdX6yxqiIorS526CYof8aqlcySXisRbIoNf84BBQ7szSOl
OPENAI_API_KEY=sk-proj-piQ_XzZL...
```

---

## 🔧 Backend Modules (Already Built)

### Core ML Stack
- ✅ `multi_model_ensemble.py` - 3-model voting system
- ✅ `feature_engine.py` - 49 technical indicators
- ✅ `regime_classifier.py` - 10 market regimes
- ✅ `data_source_manager.py` - API rotation system

### Supporting Modules
- ✅ `backtest_engine.py` - Strategy backtesting
- ✅ `portfolio_manager_optimal.py` - Position sizing & risk
- ✅ `risk_manager.py` - Stop loss & take profit logic
- ✅ `signal_service.py` - Signal generation service

### API Layer (TO BE BUILT)
- ⏳ `api_server.py` - Flask REST API (needs implementation)
- ⏳ `websocket_server.py` - Real-time WebSocket feed
- ⏳ `auth_middleware.py` - API key authentication
- ⏳ `rate_limiter.py` - Request throttling

---

## 📅 Parallel Development Timeline

### Week 1 (Dec 9-15) - TRAINING + API SKELETON
**Backend (You)**:
- Train baseline models on Colab Pro (2-4 hours)
- Hyperparameter optimization
- Target: 65-70% precision

**Frontend (Spark)**:
- Set up Next.js project
- Build static UI mockups (no API calls yet)
- Design component library

### Week 2 (Dec 16-22) - OPTIMIZATION + API INTEGRATION
**Backend (You)**:
- Feature engineering & label optimization
- Improve precision to 68-72%
- Build Flask API endpoints (signals, market, portfolio)

**Frontend (Spark)**:
- Integrate API calls for signals feed
- Build real-time chart components
- Implement portfolio dashboard

### Week 3-4 (Dec 23-Jan 5) - PAPER TRADING + FULL STACK
**Backend (You)**:
- Paper trading with Alpaca (validate live performance)
- WebSocket real-time feeds
- AI chat integration

**Frontend (Spark)**:
- Trade execution UI
- AI chat interface
- Backtest lab
- Final polish & testing

### Month 2+ - PRODUCTION READY
- IF models are profitable (win rate ≥58%) → Launch
- ELSE → Iterate optimization

---

## 🎯 Success Metrics

### Backend Goals
- ✅ Precision >60% (currently targeting 65-70%)
- ✅ Win rate ≥58% in paper trading
- ✅ Sharpe ratio ≥1.5
- ✅ Max drawdown <20%
- ✅ API response time <200ms (p95)

### Frontend Goals
- ✅ Signal feed auto-refresh (<5s latency)
- ✅ Chart rendering <1s for 1000 bars
- ✅ Mobile responsive design
- ✅ 90+ Lighthouse score
- ✅ Zero blocking operations on main thread

---

## 🆘 Support & Questions

**Backend Developer**: You (Python ML stack)  
**Frontend Developer**: Spark (React/Next.js)  
**This Document**: Single source of truth for API contract

**Questions?** Update this document as API evolves.

---

**Last Updated**: December 9, 2025  
**API Version**: v1.0 (draft spec)  
**Backend Status**: ML models ready, API implementation pending  
**Frontend Status**: Not started, ready to begin in parallel
