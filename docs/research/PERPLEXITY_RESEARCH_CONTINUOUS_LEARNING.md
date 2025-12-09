# PERPLEXITY PRO RESEARCH REQUEST: PRODUCTION SELF-LEARNING TRADING SYSTEM

## CRITICAL CONTEXT
We're deploying a 24/7 pattern detection + AI forecasting system with REAL CAPITAL at risk. Need production-grade continuous learning, online model updates, and rigorous performance tracking.

---

## QUESTION 1: Online Learning & Model Retraining Strategy

```
I'm building a 24/7 algorithmic trading system (swing trading, 5-min updates, 20 tickers) 
with REAL CAPITAL on the line. Need a production-grade continuous learning pipeline.

Current architecture:
- Pattern detection: TA-Lib (60+ candlestick patterns) + custom EMA/VWAP detectors
- ML models: Calibrated HistGradientBoostingClassifier for BUY/SELL/HOLD prediction
- Forecast: 7-day price prediction with confidence intervals
- Update frequency: Every 5 minutes during market hours
- Deployment: Local machine (24/7 uptime)

REQUIREMENTS FOR ONLINE LEARNING:

1. **Incremental Model Updates**
   - How to update ML models WITHOUT full retraining every time?
   - Compare approaches:
     * Online learning algorithms (SGDClassifier, Passive-Aggressive, FTRL)
     * Mini-batch updates (retrain on last N samples)
     * Ensemble approach (keep old model + new model, weighted voting)
     * Meta-learning (learn which model to use based on market regime)
   
   - For HistGradientBoostingClassifier (current model):
     * Can it be updated incrementally? Or must switch to online-capable model?
     * If switching: Which online learner has best accuracy for financial time series?
     * How to preserve calibration during incremental updates?

2. **Performance Tracking & Trigger Conditions**
   - What metrics trigger retraining?
     * Win rate drops below X%? (what threshold for 60% target win rate?)
     * Brier score increases? (calibration degradation)
     * Prediction confidence vs actual outcome divergence?
     * Market regime change detection (volatility spike, trend reversal)?
   
   - How often to evaluate: Every trade? Daily? Weekly?
   - How to avoid overfitting to recent noise vs adapting to regime change?

3. **A/B Testing & Model Validation**
   - How to A/B test new model vs old model in LIVE trading?
     * Paper trading new model while old model trades real capital?
     * Split capital allocation (80% old, 20% new) with performance comparison?
     * Shadow mode: new model generates signals but doesn't execute?
   
   - Statistical significance testing:
     * How many trades needed to validate new model is better? (50? 100? 200?)
     * Chi-squared test for win rate improvement?
     * Paired t-test for profit per trade?

4. **Online Feature Engineering**
   - As new patterns emerge (e.g., new TA-Lib pattern becomes predictive), how to:
     * Auto-detect feature importance changes?
     * Add new features without retraining entire pipeline?
     * Prune features that become irrelevant (feature decay)?
   
   - Feature drift detection: How to monitor when indicators (RSI, MACD) lose predictive power?

5. **Model Rollback & Safety**
   - If new model performs WORSE than old model, how to rollback safely?
     * Automatic rollback triggers? (3 consecutive losing trades? 5% drawdown?)
     * How to preserve model checkpoints (every N hours? every N trades?)
     * Version control for models: naming convention, metadata storage?

6. **Production Implementation**
   - Python libraries for online learning in production:
     * River (online ML library) vs scikit-learn vs custom implementation?
     * How to serialize/deserialize incremental models (joblib? pickle? ONNX?)
   
   - Architecture:
     * Separate training process vs prediction process? (async updates)
     * Queue-based system for new data → model update → deployment?
     * Docker containers for isolated model versions?

PROVIDE:
- Code examples for incremental HistGradientBoostingClassifier updates (if possible)
- Alternative online learning models with comparable accuracy (>60% direction prediction)
- Performance metric tracking dashboard design (Grafana? Custom Dash?)
- Retraining trigger logic (Python pseudocode)
- A/B testing framework for model comparison (statistical tests + code)
- Production deployment checklist for continuous learning systems
```

---

## QUESTION 2: Rigorous Logging & Performance Analytics for Live Trading

```
For the same 24/7 trading system with REAL CAPITAL, I need institutional-grade logging 
and performance analytics to track every decision, optimize strategies, and debug failures.

LOGGING REQUIREMENTS:

1. **Trade Execution Logging**
   What to log for EVERY signal generated:
   - Ticker, timestamp (microsecond precision), signal type (BUY/SELL/HOLD)
   - Pattern(s) detected (with confidence scores)
   - ML model prediction + probability
   - Entry price, stop loss, take profit levels
   - Position size (shares, dollar amount, % of portfolio)
   - Confluence score (how many signals agreed?)
   - Market conditions: volatility (ATR), trend (ADX), volume ratio
   
   - Log format: JSON? CSV? SQL database? Time-series DB (InfluxDB)?
   - Storage: How much disk space for 24/7 logging (estimate for 20 tickers, 5-min updates)?
   - Retention policy: Keep raw logs for how long? (30 days? 1 year? Forever?)

2. **Model Performance Metrics (Real-Time)**
   Track these metrics continuously:
   - Win rate (rolling 20 trades, 50 trades, 100 trades)
   - Profit factor (gross profit / gross loss)
   - Sharpe ratio (rolling 30-day window)
   - Max drawdown (current, historical max)
   - Average hold time per trade
   - Brier score (calibration accuracy)
   - Confusion matrix (predicted vs actual outcomes)
   
   - How to calculate rolling metrics efficiently? (circular buffer? SQL window functions?)
   - Alert thresholds: When to notify admin? (win rate < 55%? drawdown > 10%?)

3. **Pattern Detection Analytics**
   For each pattern type (Hammer, Engulfing, EMA Ribbon, etc.):
   - Historical success rate (% of times pattern led to profitable trade)
   - Average profit per pattern occurrence
   - Pattern frequency (how often detected per ticker)
   - False positive rate (pattern detected but signal failed)
   
   - Use this to dynamically weight patterns (e.g., if Hammer success rate drops, 
     reduce its weight in confluence scoring)
   
   - How to auto-detect pattern degradation? (30-day rolling win rate per pattern?)

4. **Error & Exception Logging**
   Critical for debugging 24/7 system:
   - API failures (yfinance timeout, rate limit hit)
   - Data quality issues (missing candles, bad ticks, price spikes)
   - Model prediction errors (NaN outputs, negative probabilities)
   - Execution failures (order rejected, insufficient capital)
   
   - Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
   - Rotation: Daily log files? Size-based rotation (max 100MB per file)?
   - Alerting: Email/Discord on CRITICAL errors? (with rate limiting)

5. **Dashboard Visualization**
   Real-time monitoring dashboard showing:
   - Equity curve (live P&L chart)
   - Current open positions (ticker, entry price, current P&L, time held)
   - Pattern detection heatmap (which patterns firing most?)
   - Model confidence distribution (are we taking high-confidence trades?)
   - System health (API response times, memory usage, CPU %)
   
   - Technology stack: Plotly Dash? Grafana + InfluxDB? Streamlit?
   - Update frequency: Real-time (WebSocket) or polling (every 5 sec)?

6. **Backtesting vs Live Performance Comparison**
   Critical question: Is live performance matching backtest expectations?
   - Log backtest results: expected win rate, profit factor, max drawdown
   - Compare live results to backtest after 30 days, 90 days, 6 months
   - Slippage analysis: backtest assumed $X slippage, what's actual slippage?
   - Latency analysis: backtest assumed instant execution, what's actual delay?
   
   - How to quantify "backtest decay"? (live win rate 5% lower than backtest = normal?)

7. **Regulatory & Audit Trail**
   If scaling to manage external capital (future):
   - Immutable audit log (blockchain? append-only database?)
   - Reproducibility: Can I replay any historical trade decision from logs?
   - Compliance: What records required for SEC/FINRA? (trade blotter, order history)

PROVIDE:
- Python logging configuration (structured JSON logs with rotation)
- SQL schema for trade execution log table
- Performance metrics calculation (rolling Sharpe, profit factor, Brier score)
- Real-time dashboard mockup (Dash code example)
- Alert trigger logic (when to notify on degraded performance)
- Storage estimation formula (logs per day × retention period)
- Backtest vs live performance comparison framework
```

---

## QUESTION 3: Market Regime Detection & Adaptive Strategy Switching

```
My trading system needs to ADAPT to different market conditions (trending vs ranging, 
high vol vs low vol, bull vs bear) by detecting regime changes and switching strategies.

BACKGROUND:
- Currently using fixed strategy: pattern detection + ML forecast for all market conditions
- Problem: Strategies that work in trending markets FAIL in ranging markets
- Need: Automatic regime detection + strategy selection

REGIME DETECTION:

1. **Market Regimes to Detect**
   - Trending vs Ranging (how to quantify? ADX > 25 = trending?)
   - High volatility vs Low volatility (VIX? ATR percentile?)
   - Bull market vs Bear market vs Sideways
   - Mean-reverting vs Momentum-driven
   - High liquidity vs Low liquidity (bid-ask spread, volume)
   
   - How many regimes total? (4? 6? 12?)
   - Can regimes overlap? (e.g., "Trending + High Vol + Bullish")

2. **Regime Detection Algorithms**
   Compare approaches:
   - Rule-based (if ADX > 25 AND ATR > X → trending regime)
   - Hidden Markov Models (HMM) for regime inference
   - Clustering (k-means on indicator vectors, identify distinct states)
   - Machine learning (train classifier to predict regime from indicators)
   
   - Which is most reliable for live trading? (low latency, high accuracy)
   - How to handle regime transitions? (hysteresis to avoid flapping between states?)

3. **Strategy Selection Per Regime**
   Example strategy matrix:
   - Trending + High Vol → EMA ribbon + MACD momentum strategy
   - Ranging + Low Vol → Mean reversion (VWAP pullbacks, Bollinger Band bounces)
   - Bull market → Aggressive long bias (70% long, 30% short)
   - Bear market → Defensive (50/50 or cash-heavy)
   
   - How to backtest each strategy IN EACH REGIME separately?
   - How to blend strategies during transitions? (gradual shift vs hard switch?)

4. **Online Regime Learning**
   - As new regimes emerge (e.g., "Crypto winter 2026"), how to detect them?
   - Anomaly detection: when current market doesn't fit known regimes?
   - Auto-create new regime cluster? Or alert human to define new strategy?

5. **Performance by Regime**
   Critical analytics:
   - Win rate per strategy per regime (e.g., "EMA strategy wins 75% in trending markets")
   - Regime duration (how long do regimes last? hours? days? weeks?)
   - Regime prediction accuracy (did we correctly identify regime in hindsight?)
   
   - Use this to improve regime detection model over time

6. **Implementation**
   - Real-time regime detection (update every 5 minutes?)
   - Strategy switching latency (how fast to switch models/logic?)
   - Model storage: One model per regime? Or meta-model that selects sub-models?

PROVIDE:
- Regime detection algorithm comparison (HMM vs rule-based vs clustering)
- Python code for ADX/ATR-based regime classifier
- Strategy switching logic (pseudocode or flowchart)
- Backtest framework that evaluates strategies PER REGIME
- Performance tracking by regime (SQL queries + visualization)
- Transition handling (hysteresis, gradual blending)
```

---

## QUESTION 4: Automated Feature Selection & Pattern Weighting

```
With 60+ TA-Lib patterns + custom patterns + 20+ technical indicators, I need a system 
to automatically identify which features are ACTUALLY PREDICTIVE and weight them accordingly.

CURRENT PROBLEM:
- Not all patterns are equally useful (some are noise)
- Pattern effectiveness changes over time (regime-dependent)
- Manual feature selection is slow and subjective

NEED:
- Automatic feature importance ranking
- Dynamic weighting (patterns that work get higher weight)
- Pruning (remove useless features to reduce overfitting)

QUESTIONS:

1. **Feature Importance Calculation**
   For each pattern/indicator, calculate:
   - Predictive power (mutual information with target variable?)
   - Consistency (does it work across all tickers or just some?)
   - Stability (does importance change drastically week-to-week?)
   
   - Methods:
     * Permutation importance (scikit-learn)
     * SHAP values (for tree models)
     * Recursive feature elimination (RFE)
     * Custom: backtest each feature individually, rank by win rate
   
   - Which method is fastest for 80+ features × 20 tickers?

2. **Dynamic Pattern Weighting**
   - Start with equal weights: all patterns = 1.0
   - After 30 days: patterns with >60% success rate get weight increased to 1.2
   - After 60 days: patterns with <50% success rate get weight reduced to 0.5 or removed
   
   - Mathematical framework:
     * Bayesian updating? (posterior weight based on observed success rate)
     * Exponential moving average of success rate?
     * Thompson sampling? (explore-exploit tradeoff)
   
   - How to avoid overfitting to recent lucky patterns?

3. **Feature Pruning**
   - If a pattern has <45% success rate over 100 occurrences → PRUNE IT
   - How to detect multicollinearity? (MACD and RSI might be correlated → keep only one?)
   - Variance Inflation Factor (VIF) to detect redundant features?
   
   - Pruning schedule: Weekly? Monthly? After every 100 trades?

4. **Ensemble Feature Selection**
   - Train multiple models with different feature subsets
   - Use voting: if 3 out of 5 models agree on a feature, keep it
   - Or: meta-learner combines models (stacking)

5. **Implementation**
   - Store feature importance scores in database (per ticker, per week)
   - Visualization: heatmap showing which features are hot/cold
   - Alerting: "Feature X was important but now useless - investigate market change"

PROVIDE:
- Feature importance calculation code (SHAP, permutation, or custom)
- Dynamic weighting algorithm (Bayesian or EMA-based)
- Feature pruning logic (VIF, correlation matrix, success rate threshold)
- SQL schema for storing feature performance history
- Visualization: feature importance heatmap over time
```

---

## USAGE INSTRUCTIONS

1. **Copy Question 1** → Paste into Perplexity Pro → Get online learning strategy
2. **Copy Question 2** → Get logging & analytics architecture  
3. **Copy Question 3** → Get market regime detection framework
4. **Copy Question 4** → Get automated feature selection system

5. **Bring back answers** → I'll implement:
   - Continuous learning pipeline with incremental model updates
   - Production logging system (JSON + SQL + dashboards)
   - Regime detection + strategy switching
   - Auto feature importance + dynamic weighting

---

## SUCCESS CRITERIA

After implementation, system should:
✅ Learn continuously from every trade (win or loss)
✅ Log every decision with full context (reproducible audit trail)
✅ Adapt to market regime changes automatically
✅ Prune bad patterns, boost good patterns dynamically
✅ Alert on performance degradation BEFORE significant capital loss
✅ Provide real-time dashboard showing: equity curve, pattern heatmap, model confidence
✅ Run 24/7 without human intervention (auto-restart on crashes)

This is PRODUCTION-GRADE. Real money on the line. No shortcuts.
