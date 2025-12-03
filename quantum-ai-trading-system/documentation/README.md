# 🚀 Quantum AI Trading System

**Production-grade AI-powered trading platform with institutional features**

## 🎯 What Makes This Different

Most trading platforms show charts. We provide **actionable intelligence**:

- 🤖 **AI-First Approach** - Clear buy/sell/hold recommendations with confidence scores
- 📊 **14-Day Forecasts** - Predictive price movements with confidence bands  
- 🔄 **Multi-Source Reliability** - 4 data sources with automatic failover
- ⚡ **Real-Time Streaming** - WebSocket-powered live market updates
- 🏛️ **Institutional Backtesting** - Monte Carlo analysis with professional metrics

## 🏗️ Architecture

```
Backend (FastAPI)     Frontend (React/Vite)
├── Elite Modules     ├── Real-time Dashboard
│   ├── Signal Gen    │   ├── AI Recommendations
│   ├── Backtest      │   ├── Forecast Charts  
│   ├── Risk Mgmt     │   └── Portfolio Health
│   └── AI Recomm     └── WebSocket Streaming
├── Data Orchestrator
└── Circuit Breakers
```

## 🚀 Quick Start

```bash
# Backend
cd backend
pip install -r requirements.txt
python main.py

# Frontend  
cd frontend
npm install
npm run dev
```

## 📊 Key Features

### 🤖 AI Recommendations
- **Signal Strength**: 0-100 confidence scores
- **Expected Moves**: 5-day and 20-day predictions
- **Risk Flags**: Volatility, correlation, liquidity warnings
- **Trading Signals**: Entry, exit, stop-loss levels

### 📈 Advanced Analytics
- **Multi-Factor Scoring**: 8 technical + fundamental factors
- **Regime Detection**: Bull/bear/sideways market states
- **Kelly Sizing**: Optimal position sizing based on edge
- **Monte Carlo**: 1000-simulation robustness testing

### 🔄 Reliability Features
- **Circuit Breakers**: Automatic API source switching
- **Rate Limiting**: Intelligent request throttling
- **Health Monitoring**: Real-time API status dashboard
- **Auto-Recovery**: Self-healing data pipelines

## 📋 API Endpoints

```python
# AI Analysis
GET /api/ai_recommendation/{symbol}
GET /api/forecast/{symbol}

# Market Data  
GET /api/screener
GET /api/top_gainers
GET /api/market_overview

# Backtesting
POST /api/backtest
GET /api/backtest/results/{id}
```

## 🎯 Performance Targets

- **Win Rate**: 70-80%
- **Sharpe Ratio**: >1.5  
- **Max Drawdown**: <15%
- **Latency**: <500ms for signals

## 🛠️ Tech Stack

**Backend:**
- FastAPI (Python 3.9+)
- Asyncio + uvloop
- Pandas + NumPy
- Plotly for charts

**Frontend:**
- React 19 + Vite
- Plotly.js for charts
- WebSocket streaming
- TailwindCSS

**Data Sources:**
- Polygon.io
- Financial Modeling Prep
- Twelve Data
- Alpha Vantage

## 📈 Backtest Results

```
Initial Capital: $10,000
Final Capital: $14,230 (42.3% return)
Win Rate: 78% (18 wins, 5 losses)
Sharpe Ratio: 2.1
Max Drawdown: 8.4%
```

## 🔧 Configuration

Copy `.env.example` to `.env` and add API keys:

```bash
POLYGON_API_KEY=your_key_here
FINANCIALMODELINGPREP_API_KEY=your_key_here
TWELVEDATA_API_KEY=your_key_here
```

## 🚀 Deployment

**Docker:**
```bash
docker-compose up -d
```

**Manual:**
```bash
python start_system.py  # Universal launcher
```

## 📖 Documentation

- [API Reference](docs/API.md)
- [Architecture Guide](docs/ARCHITECTURE.md)  
- [Deployment Guide](docs/DEPLOYMENT.md)

## 🏆 Why This Matters

Most trading platforms suffer from:
- ❌ Information overload (100+ indicators)
- ❌ Reactive analysis (only shows what happened)
- ❌ Single-source data (unreliable)
- ❌ No guidance on WHAT to do

**Our solution:**
- ✅ **One clear action** (STRONG_BUY/BUY/PASS)
- ✅ **Predictive insights** (14-day forecasts)
- ✅ **Enterprise reliability** (4 sources + failover)
- ✅ **AI reasoning** (shows WHY it recommends)

## 🤝 Contributing

1. Fork the repo
2. Create feature branch
3. Add tests for new features
4. Submit PR with description

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

**Built for traders who value speed, reliability, and actionable intelligence over pretty charts.**
