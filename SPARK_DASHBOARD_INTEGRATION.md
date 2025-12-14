# 🚀 Spark Dashboard Integration Guide

## Overview
The **NEW Underdog Trading System** plugs seamlessly into your Spark frontend through REST API endpoints. This provides Alpha 76 intraday predictions (5-hour horizon) for 76 small/mid-cap growth tickers.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│           SPARK FRONTEND (React/Vue/HTML)               │
│              Running on port 3000/5173/8080             │
└──────────────────┬──────────────────────────────────────┘
                   │ HTTP/AJAX
                   ↓
┌─────────────────────────────────────────────────────────┐
│          UNDERDOG API (Flask Backend)                   │
│              Running on port 5000                       │
│  Routes:                                                │
│    /api/underdog/predict          - Single prediction   │
│    /api/underdog/batch-predict    - Batch predictions   │
│    /api/underdog/top-signals      - Top 10 BUYs         │
│    /api/underdog/regime           - Market regime       │
│    /api/underdog/alpha76          - Watchlist           │
└──────────────────┬──────────────────────────────────────┘
                   │ Python imports
                   ↓
┌─────────────────────────────────────────────────────────┐
│       TRAINED MODELS (After Colab Pro Training)         │
│  • models/underdog_v1/xgboost.pkl                       │
│  • models/underdog_v1/random_forest.pkl                 │
│  • models/underdog_v1/gradient_boosting.pkl             │
│  • models/underdog_v1/scaler.pkl                        │
│  • models/underdog_v1/metadata.pkl                      │
└─────────────────────────────────────────────────────────┘
```

---

## Quick Start (3 Steps)

### Step 1: Start Underdog API (Backend)
```bash
cd /workspaces/quantum-ai-trader_v1.1
python underdog_api.py
```

**Output**:
```
🚀 UNDERDOG API - Flask Backend for Spark Dashboard
Alpha 76 Watchlist: 76 tickers
Starting server on http://localhost:5000
Ready for Spark dashboard integration! 🎯
```

### Step 2: Test API Endpoints
```bash
# Check health
curl http://localhost:5000/api/underdog/status

# Get Alpha 76 watchlist
curl http://localhost:5000/api/underdog/alpha76

# Get current regime
curl http://localhost:5000/api/underdog/regime

# Get prediction for RKLB
curl -X POST http://localhost:5000/api/underdog/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "RKLB"}'

# Get top 10 BUY signals
curl http://localhost:5000/api/underdog/top-signals
```

### Step 3: Connect from Spark Frontend
See JavaScript examples below.

---

## API Endpoints Reference

### 1. System Health Check
**GET** `/api/underdog/status`

**Response**:
```json
{
  "success": true,
  "status": "operational",
  "models": {
    "ensemble_trained": true,
    "feature_engine": true,
    "regime_classifier": true
  },
  "watchlist_size": 76,
  "timestamp": "2025-12-09T..."
}
```

**Use Case**: Display system status indicator in dashboard header

---

### 2. Get Alpha 76 Watchlist
**GET** `/api/underdog/alpha76`

**Response**:
```json
{
  "success": true,
  "watchlist": ["SYM", "IONQ", "RKLB", ...],
  "total_tickers": 76,
  "sectors": {
    "Autonomous": ["SYM", "IONQ", "RGTI", ...],
    "Space": ["RKLB", "ASTS", "LUNR", ...],
    "Biotech": ["VKTX", "NTLA", "BEAM", ...],
    ...
  },
  "timestamp": "2025-12-09T..."
}
```

**Use Case**: Populate ticker dropdown/selector in UI

---

### 3. Single Ticker Prediction
**POST** `/api/underdog/predict`

**Request**:
```json
{
  "ticker": "RKLB"
}
```

**Response** (Success):
```json
{
  "ticker": "RKLB",
  "sector": "Space",
  "signal": "BUY",
  "confidence": 0.892,
  "agreement": 1.0,
  "votes": {"xgboost": 2, "random_forest": 2, "gradient_boosting": 2},
  "vote_counts": {0: 0, 1: 0, 2: 3},
  "current_price": 24.56,
  "regime": "CHOPPY_HIGH_VOL",
  "regime_filter": "PASS",
  "position_size_multiplier": 0.3,
  "stop_loss_pct": 0.06,
  "timestamp": "2025-12-09T...",
  "success": true
}
```

**Response** (Error - Models Not Trained):
```json
{
  "ticker": "RKLB",
  "error": "Models not trained - train on Colab Pro first",
  "success": false
}
```

**Use Case**: Display prediction card when user clicks ticker

---

### 4. Batch Predictions
**POST** `/api/underdog/batch-predict`

**Request** (Specific tickers):
```json
{
  "tickers": ["RKLB", "IONQ", "SOFI", "APP", "VKTX"]
}
```

**Request** (All Alpha 76 - omit tickers):
```json
{}
```

**Response**:
```json
{
  "success": true,
  "predictions": [
    {
      "ticker": "RKLB",
      "signal": "BUY",
      "confidence": 0.892,
      ...
    },
    {
      "ticker": "IONQ",
      "signal": "HOLD",
      "confidence": 0.723,
      ...
    }
  ],
  "total": 20,
  "timestamp": "2025-12-09T..."
}
```

**Use Case**: Display signals table/grid for multiple tickers

---

### 5. Current Market Regime
**GET** `/api/underdog/regime`

**Response**:
```json
{
  "success": true,
  "regime": {
    "name": "CHOPPY_HIGH_VOL",
    "description": "Sideways market, high volatility - Wait mode",
    "position_size_multiplier": 0.3,
    "max_positions": 3,
    "stop_loss_pct": 0.06,
    "take_profit_pct": 0.05,
    "min_confidence": 0.85,
    "strategy_weights": {
      "momentum": 0.10,
      "mean_reversion": 0.35,
      "dark_pool": 0.25,
      "cross_asset": 0.20,
      "sentiment": 0.10
    }
  },
  "timestamp": "2025-12-09T..."
}
```

**Use Case**: Display regime banner at top of dashboard with color coding

---

### 6. Top BUY Signals
**GET** `/api/underdog/top-signals`

**Response**:
```json
{
  "success": true,
  "signals": [
    {
      "ticker": "APP",
      "signal": "BUY",
      "confidence": 0.914,
      "current_price": 234.12,
      "sector": "Software",
      "regime_filter": "PASS",
      ...
    },
    ...
  ],
  "regime": "CHOPPY_HIGH_VOL",
  "min_confidence": 0.85,
  "timestamp": "2025-12-09T..."
}
```

**Use Case**: Display "Hot Picks" section with top opportunities

---

### 7. Training/Backtest Summary
**GET** `/api/underdog/backtest-summary`

**Response** (After Colab training):
```json
{
  "success": true,
  "summary": {
    "training_date": "2025-12-09",
    "validation_accuracy": 0.63,
    "models": {
      "xgboost": {"accuracy": 0.65, "roc_auc": 0.71},
      "random_forest": {"accuracy": 0.62, "roc_auc": 0.68},
      "gradient_boosting": {"accuracy": 0.61, "roc_auc": 0.67}
    },
    "backtest": {
      "win_rate": 0.67,
      "avg_return": 0.023,
      "total_signals": 143
    }
  },
  "timestamp": "2025-12-09T..."
}
```

**Use Case**: Display model performance metrics in settings/about page

---

## Frontend Integration Examples

### Vanilla JavaScript (Fetch API)
```javascript
// Get top signals
async function getTopSignals() {
  const response = await fetch('http://localhost:5000/api/underdog/top-signals');
  const data = await response.json();
  
  if (data.success) {
    displaySignals(data.signals);
    displayRegime(data.regime);
  }
}

// Get prediction for specific ticker
async function getPrediction(ticker) {
  const response = await fetch('http://localhost:5000/api/underdog/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ ticker: ticker })
  });
  
  const data = await response.json();
  
  if (data.success) {
    displayPrediction(data);
  } else {
    console.error('Error:', data.error);
  }
}
```

---

### React (Axios)
```jsx
import axios from 'axios';
import { useState, useEffect } from 'react';

const API_BASE = 'http://localhost:5000/api/underdog';

function Dashboard() {
  const [signals, setSignals] = useState([]);
  const [regime, setRegime] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchTopSignals();
    fetchRegime();
  }, []);

  const fetchTopSignals = async () => {
    try {
      const response = await axios.get(`${API_BASE}/top-signals`);
      if (response.data.success) {
        setSignals(response.data.signals);
      }
    } catch (error) {
      console.error('Error fetching signals:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchRegime = async () => {
    try {
      const response = await axios.get(`${API_BASE}/regime`);
      if (response.data.success) {
        setRegime(response.data.regime);
      }
    } catch (error) {
      console.error('Error fetching regime:', error);
    }
  };

  const getPrediction = async (ticker) => {
    try {
      const response = await axios.post(`${API_BASE}/predict`, { ticker });
      if (response.data.success) {
        return response.data;
      }
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div>
      <RegimeBanner regime={regime} />
      <SignalsGrid signals={signals} />
    </div>
  );
}
```

---

### Vue.js
```vue
<template>
  <div class="dashboard">
    <div v-if="regime" class="regime-banner" :class="regime.name">
      Regime: {{ regime.name }}
    </div>
    
    <div class="signals-grid">
      <div v-for="signal in signals" :key="signal.ticker" class="signal-card">
        <h3>{{ signal.ticker }}</h3>
        <div class="signal-badge" :class="signal.signal">
          {{ signal.signal }}
        </div>
        <div>Confidence: {{ (signal.confidence * 100).toFixed(1) }}%</div>
        <div>Price: ${{ signal.current_price }}</div>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  data() {
    return {
      signals: [],
      regime: null,
      API_BASE: 'http://localhost:5000/api/underdog'
    };
  },
  
  mounted() {
    this.fetchData();
  },
  
  methods: {
    async fetchData() {
      try {
        const [signalsRes, regimeRes] = await Promise.all([
          axios.get(`${this.API_BASE}/top-signals`),
          axios.get(`${this.API_BASE}/regime`)
        ]);
        
        if (signalsRes.data.success) {
          this.signals = signalsRes.data.signals;
        }
        
        if (regimeRes.data.success) {
          this.regime = regimeRes.data.regime;
        }
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    },
    
    async getPrediction(ticker) {
      try {
        const response = await axios.post(`${this.API_BASE}/predict`, { ticker });
        return response.data;
      } catch (error) {
        console.error('Error:', error);
      }
    }
  }
};
</script>
```

---

## UI Components to Build

### 1. Regime Banner
```html
<div class="regime-banner BULL_LOW_VOL">
  <span class="regime-icon">🟢</span>
  <div>
    <strong>BULL LOW VOL</strong>
    <p>Full risk on • Position size: 100% • Min confidence: 65%</p>
  </div>
</div>
```

**Color Coding**:
- 🟢 BULL_LOW_VOL, BULL_MODERATE_VOL → Green
- 🟡 CHOPPY_LOW_VOL, BULL_HIGH_VOL → Yellow
- 🟠 BEAR_LOW_VOL, CHOPPY_HIGH_VOL → Orange
- 🔴 BEAR_HIGH_VOL, PANIC_EXTREME_VOL → Red

---

### 2. Signal Card
```html
<div class="signal-card">
  <div class="ticker-header">
    <h3>RKLB</h3>
    <span class="sector-badge">Space</span>
  </div>
  
  <div class="signal-badge BUY">BUY</div>
  
  <div class="confidence-bar">
    <div class="confidence-fill" style="width: 89.2%"></div>
    <span>89.2% Confidence</span>
  </div>
  
  <div class="price-info">
    <span>Price: $24.56</span>
    <span>Stop: $22.09 (-6%)</span>
  </div>
  
  <div class="model-votes">
    ✅ XGBoost: BUY
    ✅ Random Forest: BUY
    ✅ Gradient Boosting: BUY
  </div>
</div>
```

---

### 3. Signals Table
```html
<table class="signals-table">
  <thead>
    <tr>
      <th>Ticker</th>
      <th>Signal</th>
      <th>Confidence</th>
      <th>Price</th>
      <th>Sector</th>
      <th>Regime Filter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>APP</strong></td>
      <td><span class="badge BUY">BUY</span></td>
      <td>91.4%</td>
      <td>$234.12</td>
      <td>Software</td>
      <td>✅ PASS</td>
    </tr>
  </tbody>
</table>
```

---

## CORS Configuration

The API is configured to accept requests from:
- `http://localhost:3000` (React default)
- `http://localhost:5173` (Vite default)
- `http://localhost:8080` (Vue CLI default)

**Custom port?** Edit `underdog_api.py`:
```python
# Line ~35
origins = [
    "http://localhost:3000",  # React
    "http://localhost:5173",  # Vite
    "http://localhost:8080",  # Vue
    "http://localhost:4200"   # Add your custom port
]
```

---

## Next Steps

1. **Train models on Colab Pro** → Use `UNDERDOG_COLAB_TRAINER.ipynb`
2. **Download trained models** → Place in `models/underdog_v1/`
3. **Start API**: `python underdog_api.py`
4. **Test endpoints**: Use curl or Postman
5. **Connect Spark frontend**: Use JavaScript examples above
6. **Build UI components**: Regime banner, signal cards, tables

**Ready to integrate!** ��
