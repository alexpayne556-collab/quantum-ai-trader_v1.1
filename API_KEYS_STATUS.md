# 🔑 API Keys & Environment Configuration

## Current API Keys (Active)

### ✅ Market Data APIs
```
ALPHA_VANTAGE_API_KEY=9OS7LP4D495FW43S
  - Free tier: 25 requests/day, 5 requests/minute
  - Status: Active
  - Use: Historical data, fundamentals

POLYGON_API_KEY=iRXh2jGpwhcJxGWfW4ZRVn2C4s_v4ghr
  - Free tier: 5 API calls/minute, delayed data
  - Status: Active
  - Use: Intraday bars, real-time quotes

FINNHUB_API_KEY=d3qj8p9r01quv7kb49igd3qj8p9r01quv7kb49j0
  - Free tier: 60 calls/minute, real-time US stocks
  - Status: Active
  - Use: Primary real-time data source

FMP_API_KEY=15zYYtksuJnQsTBODSNs3MrfEedOSd3i
  - Free tier: 250 requests/day
  - Status: Active
  - Use: Fundamentals, earnings, ratios

EODHD_API_TOKEN=68f5419033db54.61168020
  - Free tier: 20 API requests/day
  - Status: Active
  - Use: End-of-day data, backup source
```

### ✅ AI APIs
```
PERPLEXITY_API_KEY=your_perplexity_api_key_hereSugdX6yxqiIorS526CYof8aqlcySXisRbIoNf84BBQ7szSOl
  - Status: Active
  - Use: AI research, market analysis, answering questions

OPENAI_API_KEY=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
  - Status: Active
  - Use: Alternative AI analysis, embeddings
```

### ⚠️ Missing/Not Configured
```
TWELVE_DATA_API_KEY=(empty)
  - Not critical - we have other data sources

ALPACA_API_KEY=(placeholder)
  - Needed for: Paper trading execution
  - Priority: Medium (Week 3-4 after models proven)

TWILIO/SENDGRID=(placeholders)
  - Needed for: SMS/Email alerts
  - Priority: Low (nice-to-have)

DISCORD_WEBHOOK=(placeholder)
  - Needed for: Discord notifications
  - Priority: Low
```

---

## What We Have vs What We Need

### ✅ HAVE - Ready to Train
- **Market data**: 4 active sources (Finnhub, Polygon, Alpha Vantage, FMP)
- **AI research**: Perplexity + OpenAI
- **Compute**: Colab Pro T4 GPU access
- **Code**: Complete training pipeline

### ⚠️ NEED - For Live Trading (Week 3+)
- **Alpaca keys**: For paper trading execution
- **Notification service**: SMS/Email/Discord (optional)

---

## Current System Capabilities

### Data Collection ✅
```python
# Primary: Finnhub (60 calls/min) - Real-time quotes
# Backup: Polygon (5 calls/min) - Intraday bars
# Backup: Alpha Vantage (5 calls/min) - Historical
# Backup: FMP (250/day) - Fundamentals
```

**Status**: Robust multi-source fallback system operational

### Training Pipeline ✅
```python
# Colab Pro T4 GPU available
# 76 tickers × 2 years × 1hr bars = ~1.3M rows
# Expected training time: 2-4 hours
```

**Status**: Ready to execute tonight

### Prediction System ✅
```python
# 3-model ensemble: XGBoost GPU + RF + GB
# 49 technical features
# 10 regime detection (VIX, SPY, yield curve)
```

**Status**: Code complete, untrained

### API Backend ✅
```python
# Flask API on port 5000
# 7 endpoints for predictions, regime, watchlist
# CORS configured for frontend ports
```

**Status**: Ready for post-training integration

---

## What Perplexity Should Help With

### Priority 1: Training Optimization (This Week)
**Questions for Perplexity**:
1. "What are optimal XGBoost hyperparameters for high-volatility small-cap stocks?"
2. "Should I use asymmetric thresholds (+3%/-2%) or symmetric for intraday trading?"
3. "What technical indicators work best for biotech/space sector momentum?"
4. "How do I handle class imbalance in 3-class classification (BUY/HOLD/SELL)?"

### Priority 2: Feature Engineering (Week 2)
**Questions for Perplexity**:
1. "What microstructure features predict intraday price spikes in small-caps?"
2. "Should I include sector-relative features or ticker-specific only?"
3. "How important is order flow data vs pure price/volume for 1hr bars?"

### Priority 3: Risk Management (Week 3)
**Questions for Perplexity**:
1. "What's a realistic max drawdown for small-cap intraday strategies?"
2. "Should position sizing be fixed or volatility-adjusted for Alpha 76?"
3. "How do regime shifts affect optimal stop-loss levels?"

---

## API Key Status Summary

| Service | Key Status | Rate Limit | Use Case | Priority |
|---------|-----------|------------|----------|----------|
| Finnhub | ✅ Active | 60/min | Real-time quotes | Critical |
| Polygon | ✅ Active | 5/min | Intraday bars | High |
| Alpha Vantage | ✅ Active | 25/day | Backup data | Medium |
| FMP | ✅ Active | 250/day | Fundamentals | Medium |
| EODHD | ✅ Active | 20/day | EOD data | Low |
| Perplexity | ✅ Active | Per plan | AI research | High |
| OpenAI | ✅ Active | Per plan | AI backup | Medium |
| Alpaca | ❌ Missing | N/A | Paper trading | Week 3 |
| Twelve Data | ❌ Empty | 800/day | Optional | Low |

---

## Next Steps

### Tonight (Dec 9)
1. ✅ API keys verified and working
2. 🔄 Upload training notebook to Colab Pro
3. 🔄 Train baseline models (2-4 hours)
4. 🔄 Download trained models

### Week 1 (Dec 10-16)
1. Hyperparameter optimization (use Perplexity for research)
2. Feature selection (49 → 30 best)
3. Label engineering (test different horizons)
4. Target: 65-70% accuracy, 60% win rate

### Week 2 (Dec 17-23)
1. Advanced ensemble optimization
2. Regime-adaptive training
3. Walk-forward validation
4. Target: 68-72% accuracy, 62-65% win rate

### Week 3-4 (Dec 24-Jan 6)
1. Get Alpaca paper trading keys
2. Deploy paper trading system
3. Track live signals (no real money)
4. Validate: 58%+ win rate, 1.5+ Sharpe

---

## Environment Variables File Location
```
/workspaces/quantum-ai-trader_v1.1/.env
```

**Status**: File exists with all API keys configured
**Backup**: `.env.example` template available
**Security**: .env is git-ignored (not pushed to GitHub)

---

## Summary for Perplexity

**What we have**:
- 4 active market data APIs (Finnhub, Polygon, Alpha Vantage, FMP)
- 2 active AI APIs (Perplexity, OpenAI)
- Complete training pipeline ready
- 76-ticker watchlist (Alpha 76) defined
- Colab Pro GPU access ready

**What we need help with**:
- Hyperparameter optimization strategies
- Feature engineering best practices
- Risk management for small-cap intraday trading
- Model validation techniques
- Real-world profitability benchmarks

**What we're NOT worried about yet**:
- Frontend (weeks away)
- Live trading keys (Week 3+)
- Notifications (nice-to-have)

**Goal**: Train a profitable model (60%+ precision) in 2 weeks before worrying about deployment infrastructure.
