# 🏆 PERPLEXITY UNDERDOG CHAMPION BRIEF
## Making Retail ML Trading Competitive with Institutions

**Date:** December 10, 2025  
**System:** Quantum AI Trader v1.1  
**Goal:** Optimize 12+ ML modules for small-cap 1hr swing trading  
**Current Status:** Baseline untrained → Target 68-72% precision  

---

## 📊 SYSTEM OVERVIEW

### **Trading Universe:**
- **76 Tickers** across 6 sectors (biotech, space, AI/quantum, energy, crypto-related, retail/consumer)
- **Timeframe:** 1-hour bars, 2 years historical data
- **Expected Dataset:** ~250,000 rows (70 successful downloads × 3,500 bars per ticker)
- **Volatility Profile:** 3-8% daily moves (high-volatility small-caps)
- **Strategy:** Swing trading (hold 1-21 days), not day trading

### **Current Performance (Untrained Baseline):**
- ❌ Precision: 50-55% (random chance)
- ❌ Sharpe Ratio: 0.5-0.8
- ❌ Max Drawdown: 15-20%
- ❌ Win Rate: 50%

### **Target Performance (After Full Optimization):**
- ✅ Precision: **68-72%** ← Competitive edge
- ✅ Sharpe Ratio: **1.8-2.2** ← Institutional-grade
- ✅ Max Drawdown: **8-10%** ← Capital preservation
- ✅ Win Rate: **65-70%** ← Consistent profitability

### **Training Environment:**
- **Hardware:** Google Colab Pro, T4 GPU (15GB VRAM)
- **Time Budget:** 7-14 hours GPU training + 1-2 weeks optimization
- **Capital:** $10,000 initial (virtual paper trading on Alpaca)

---

## 🧠 MODULE #1: MULTI-MODEL ENSEMBLE

### **Architecture:**
3-model voting system: XGBoost (GPU) + RandomForest (CPU) + GradientBoosting (CPU)

### **Current Hyperparameters:**
```python
XGBoost:
  max_depth: 6
  learning_rate: 0.1
  n_estimators: 200
  subsample: 0.8
  colsample_bytree: 0.8
  tree_method: 'gpu_hist'
  
RandomForest:
  n_estimators: 200
  max_depth: 10
  min_samples_split: 20
  
GradientBoosting:
  n_estimators: 200
  max_depth: 5
  learning_rate: 0.1
```

### **Current Issues:**
- Labels: BUY (>2%), SELL (<-2%) may be too tight for 3-8% daily vol stocks
- Voting confidence: (agreement × 0.7) + (probability × 0.3) — weights are arbitrary
- max_depth=6 may be too shallow for complex small-cap patterns
- n_estimators=200 may be too low for 250k rows

### **Questions for Perplexity:**

**Q1: XGBoost GPU Optimization for Small-Cap 1hr Bars**
> We have 250,000 rows of small-cap 1-hour bar data with 3-8% daily volatility. Current XGBoost hyperparameters: max_depth=6, learning_rate=0.1, n_estimators=200, tree_method='gpu_hist' on T4 GPU (15GB VRAM).
> 
> Should we:
> - Increase max_depth to 8-10 to capture complex small-cap patterns?
> - Reduce learning_rate to 0.01-0.05 for better generalization?
> - How many n_estimators are optimal for 250k rows without overfitting?
> - What subsample/colsample_bytree ratios work best for high-volatility stocks?

**Q2: Ensemble Confidence Weighting Strategy**
> Our 3-model ensemble calculates confidence = (agreement × 0.7) + (probability × 0.3). 
> 
> Should these weights:
> - Vary by market regime (e.g., 0.9/0.1 in PANIC for safety, 0.5/0.5 in BULL for growth)?
> - Adapt to ticker-specific volatility (more weight on agreement for 8% vol biotechs)?
> - Be learned from data instead of hardcoded?

**Q3: Label Threshold Optimization for High-Volatility Stocks**
> We use binary labels: BUY (future return >2%), SELL (<-2%), HOLD (between). For tickers with 3-8% daily moves:
> 
> Should we:
> - Use wider fixed thresholds (±3% or ±5%)?
> - Use ATR-based dynamic thresholds: ±(1.5 × ATR% per bar)?
> - Use sector-specific thresholds (biotech ±4%, utilities ±1.5%)?
> - Which approach gives best precision on small-cap swing trading?

---

## 📈 MODULE #2: FEATURE ENGINE

### **Current Features (30+):**
- **Momentum:** RSI(14), Stochastic(14,3,3), MACD(12,26,9), ROC(5,10,20), Williams %R(14)
- **Trend:** EMAs(8,21,50,200), Bollinger Bands(20,2std), ADX(14), ATR(14)
- **Volume:** OBV, VWAP, Volume MA(20), Volume trend EMA(10)
- **Microstructure:** Spread proxy, Amihud illiquidity, bar position, price pressure, roll spread
- **Price Patterns:** Body size, shadows, consecutive bars, higher highs/lower lows

### **Current Issues:**
- All periods are standard (RSI=14, MACD=12/26/9) — not optimized for high-vol small-caps
- No sector-specific features (biotech FDA calendars, space launch schedules)
- Microstructure features designed for liquid stocks — validity for $1M-$50M daily volume?

### **Questions for Perplexity:**

**Q4: Indicator Period Optimization for High-Volatility Small-Caps**
> Standard technical indicator periods: RSI=14, MACD=12/26/9, EMAs=8/21/50/200, ADX=14, ATR=14. 
> 
> For high-volatility small-caps on 1-hour bars:
> - **Biotech stocks** (8% daily volatility): Should we shorten periods (RSI=7, MACD=6/13/5) for faster response?
> - **Space stocks** (5% daily volatility): Keep standard periods or moderate shortening?
> - **General rule:** How do we scale indicator periods based on ATR or realized volatility?
> - Is there research on optimal TA periods for small-cap swing trading?

**Q5: Sector-Specific Feature Engineering**
> We trade 76 tickers across 6 sectors. Should we add:
> - **Biotech:** FDA approval calendars, clinical trial phase transitions, patent expiry dates
> - **Space:** SpaceX/Blue Origin launch schedules, satellite deployment announcements
> - **AI/Quantum:** Earnings call sentiment, product release cycles
> - **Energy:** Oil/gas price correlations, geopolitical event indicators
> 
> Which sectors benefit most from custom features? How to avoid overfitting to calendar events?

**Q6: Microstructure Features for Low-Liquidity Stocks**
> We calculate microstructure features (Amihud illiquidity, roll spread, price pressure) assuming liquid markets.
> 
> For small-caps with $1M-$50M daily volume:
> - Are these features still valid or do they introduce noise?
> - Should we replace with simpler features (bid-ask spread proxy, volume spikes)?
> - What liquidity threshold makes microstructure analysis unreliable?

---

## 🌍 MODULE #3: REGIME CLASSIFIER

### **Current Architecture:**
10 market regimes based on VIX + SPY returns + yield curve

### **Regimes:**
- BULL × {LOW_VOL (<15 VIX), MODERATE_VOL (15-20), HIGH_VOL (>20)}
- BEAR × {LOW_VOL, MODERATE_VOL, HIGH_VOL}
- CHOPPY × {LOW_VOL, HIGH_VOL}
- PANIC (VIX>30, SPY<-5%)
- EUPHORIA (VIX<12, SPY>10%)

### **Current Thresholds:**
```python
VIX Levels:
  LOW: < 15
  MODERATE: 15-20
  HIGH: > 20
  PANIC: > 30
  EUPHORIA: < 12

SPY Returns (20-day):
  BULL: > +3%
  BEAR: < -3%
  CHOPPY: -3% to +3%
```

### **Current Issues:**
- Thresholds calibrated for large-cap SPY, not small-caps
- Single regime per day — no intraday regime transitions
- No ticker-specific regime detection (biotech may be in PANIC while SPY is BULL)

### **Questions for Perplexity:**

**Q7: Small-Cap Regime Threshold Calibration**
> Our regime classifier uses VIX levels (15/20/30) and SPY returns (±3%) to classify market state.
> 
> For a small-cap universe:
> - Should we use different VIX thresholds? (Small-caps often move independently of VIX)
> - Should we calculate sector-specific VIX (biotech volatility index, space sector ATR)?
> - Or use ticker-specific volatility percentiles (80th percentile = HIGH_VOL for that stock)?
> - What's the correlation between SPY regimes and small-cap performance?

**Q8: Regime-Specific Models vs Single Model with Regime Feature**
> We have 10 regimes and 250k total rows.
> 
> Should we:
> - **Option A:** Train 10 separate ensemble models (one per regime) = 25k rows each
> - **Option B:** Train single model with regime as categorical feature
> - **Option C:** Train 3 models (BULL/BEAR/CHOPPY) and ignore volatility levels
> 
> Trade-off: Regime specialization vs data sufficiency. Which approach gives best risk-adjusted returns?

---

## 📊 MODULE #4: QUANTILE FORECASTER

### **Current Architecture:**
Multi-quantile regression (10%, 25%, 50%, 75%, 90% percentiles) with separate models per quantile

### **Current Hyperparameters:**
```python
Model: GradientBoostingRegressor (CPU-only)
Quantiles: [0.10, 0.25, 0.50, 0.75, 0.90]
Max Iterations: 200
Learning Rate: 0.05
Horizons: 1-bar, 3-bar, 5-bar, 10-bar, 21-bar
Total Models: 5 quantiles × 5 horizons = 25 models
```

### **Current Issues:**
- GradientBoostingRegressor is CPU-only (slow on 250k rows)
- Training 25 models sequentially takes 75-100 minutes
- Uncertainty bounds may be too wide for actionable trading

### **Questions for Perplexity:**

**Q9: GPU-Accelerated Quantile Regression**
> We train 25 quantile models (5 quantiles × 5 horizons) using GradientBoostingRegressor on CPU.
> 
> For T4 GPU training:
> - Should we switch to HistGradientBoostingRegressor (GPU-compatible)?
> - Or use LightGBM/CatBoost with quantile loss (which has better GPU support)?
> - Can we parallelize quantile training (train 5 quantiles simultaneously on GPU)?
> - What's the expected speedup: 75-100 min CPU → ? min GPU?

**Q10: Optimal Forecast Horizons for Swing Trading**
> We forecast 1/3/5/10/21 bars (1 day to 1 month).
> 
> For small-cap swing trading:
> - Which horizon is most predictive? (Hypothesis: 3-5 day sweet spot)
> - Should we focus training budget on 2-3 horizons vs all 5?
> - Does longer horizon (21-day) provide useful uncertainty bounds or just noise?
> - What Sharpe ratio improvement does multi-horizon forecasting give vs single-horizon?

---

## 🤖 MODULE #5: AI RECOMMENDER

### **Current Architecture:**
Per-ticker Logistic Regression classifier predicting 7-day direction

### **Current Hyperparameters:**
```python
Model: LogisticRegression(class_weight='balanced')
Horizon: 7 days
Threshold: Fixed ±2% OR Adaptive ATR% × 1.5
Feature Selection: SelectKBest (n_features=12)
Training: 76 models (one per ticker)
Expected Time: 5-10 min per ticker = 6-13 hours total
```

### **Current Issues:**
- Per-ticker approach = 76 models × 10 min = 13 hours CPU time
- LogisticRegression is simple — may miss nonlinear patterns
- No feature sharing across tickers (each model learns independently)

### **Questions for Perplexity:**

**Q11: Per-Ticker vs Global Model Architecture**
> We train 76 separate Logistic Regression models (one per ticker, 5-10 min each = 6-13 hours).
> 
> Should we:
> - **Per-Ticker:** Keep 76 models for ticker-specific patterns
> - **Global:** Train single model with ticker as one-hot encoded feature
> - **Hybrid:** Train sector-level models (6 models for 6 sectors)
> - **Transfer Learning:** Pre-train on all tickers, fine-tune per ticker
> 
> Which gives best precision for small-cap swing trading? What's the training time trade-off?

**Q12: Adaptive Label Threshold Optimization**
> We use adaptive labels: threshold = ATR% × 1.5 (clipped 0.5%-8%).
> 
> For sector-specific optimization:
> - **Biotech** (high vol): Should multiplier be 2.0 or higher?
> - **Utilities** (low vol): Should multiplier be 1.0 or lower?
> - Can we learn optimal multiplier per ticker via cross-validation?
> - Or use machine learning (meta-learning) to predict optimal threshold from ticker features?

---

## 🛡️ MODULE #6: RISK MANAGER

### **Current Architecture:**
Regime-based position sizing with 10% max drawdown circuit breaker

### **Current Parameters:**
```python
Initial Capital: $10,000
Max Drawdown: 10% (halt trading)
Risk Per Trade: 2% of capital
Position Size Formula: shares = risk_amount / price_risk
  where price_risk = |entry - stop_loss|

Regime Limits:
  PANIC: 0% position (no trading)
  BEAR: 2% position (1 trade max)
  CORRECTION: 3% position (2 trades max)
  BULL: 5% position (5 trades max)
```

### **Current Issues:**
- 2% risk per trade on $10k = $200 per trade — aggressive for small-caps
- 10% max drawdown = $1,000 loss before halt — may be too tight for swing trading
- No volatility-adjusted position sizing (treat all tickers same)

### **Questions for Perplexity:**

**Q13: Optimal Risk Per Trade for Small-Cap Swing Trading**
> We risk 2% per trade on $10,000 account ($200 risk per trade).
> 
> For small-caps with 3-8% daily volatility:
> - Should we reduce to 1% for capital preservation?
> - Or increase to 3% for faster growth?
> - Should risk% vary by regime (1% in BEAR, 3% in BULL)?
> - What's the optimal risk% for maximizing long-term Sharpe ratio vs CAGR?

---

## 📂 MODULE #7: PORTFOLIO MANAGER

### **Current Architecture:**
Hierarchical Risk Parity (HRP) optimization with event-sourced position tracking

### **Current Parameters:**
```python
Optimization: HRP (correlation clustering + equal risk weighting)
Linkage Method: 'ward' (minimize variance)
Rebalance Frequency: Not specified
Risk Metrics Weights:
  Sharpe: 30%
  Sortino: 25%
  Max Drawdown: -20%
  Calmar: 15%
  VaR 95%: -10%
```

### **Current Issues:**
- No rebalancing schedule (daily/weekly/monthly?)
- Transaction costs not modeled
- HRP assumes Gaussian returns (small-caps have fat tails)

### **Questions for Perplexity:**

**Q14: Optimal Rebalancing Frequency for Swing Trading**
> We use Hierarchical Risk Parity for portfolio optimization.
> 
> For swing trading (1-21 day holds):
> - **Daily rebalance:** Best risk control, but high transaction costs
> - **Weekly rebalance:** Balance costs vs drift from optimal
> - **Monthly rebalance:** Lowest costs, but large drift during volatility spikes
> - **Event-driven:** Rebalance when drawdown >X% or correlation structure changes
> 
> What frequency maximizes Sharpe ratio after transaction costs (assume 0.1% per trade)?

---

## 🔮 MODULE #8: FORECAST ENGINE

### **Current Architecture:**
ATR-based 24-day price path generator

### **Current Formula:**
```python
daily_return = (direction × confidence × ATR × 0.5 × decay_factor) + random_shock
forecast_price = last_close + daily_return

Where:
  direction: +1 (BULLISH), -1 (BEARISH), 0 (NEUTRAL)
  confidence: Model probability (0-1)
  ATR: 14-period Average True Range
  0.5: Scaling factor ← NEEDS OPTIMIZATION
  decay_factor: 1.0 (days 1-10) → 0.0 (day 24)
  random_shock: Normal(0, ATR × 0.2)
```

### **Current Issues:**
- 0.5 scaling factor is arbitrary (no calibration)
- Same scaling for all volatility levels (should vary by ATR%)
- Decay schedule (linear) not validated

### **Questions for Perplexity:**

**Q15: ATR Scaling Factor Calibration for 1hr Bars**
> Our forecast uses: base_move = direction × confidence × ATR × 0.5
> 
> For 1-hour bars:
> - Is 0.5 optimal or should it be calibrated per ticker?
> - High-vol stocks (biotech 8% daily): Use 0.3 for dampening?
> - Low-vol stocks (utilities 2% daily): Use 0.7 for amplification?
> - How to determine optimal scaling via walk-forward validation?

---

## 📡 MODULE #9: DATA SOURCE MANAGER

### **Current Architecture:**
7 data sources with automatic rotation on rate limits

### **Current Priority:**
1. Twelve Data (800/day, 8/min) - Primary
2. Finnhub (60/min) - Backup 1
3. Polygon (5/min) - Backup 2
4. yfinance (unlimited) - Fallback
5-7. Alpha Vantage, FMP, EODHD

### **Current Issues:**
- No data quality validation (missing bars, bad ticks)
- Priority based on rate limits, not data accuracy
- No latency tracking for real-time trading

### **Questions for Perplexity:**

**Q16: Data Quality vs Latency Trade-off**
> We use 7 data sources prioritized by rate limits: Twelve Data → Finnhub → Polygon → yfinance.
> 
> For 1-hour bars on small-caps:
> - Should we prioritize **data quality** (fewest missing bars) over rate limits?
> - How to measure real-time data quality (missing bars, outliers, timestamp accuracy)?
> - Which source has best small-cap coverage for tickers under $1B market cap?
> - Should we use multiple sources and cross-validate (majority vote on OHLCV)?

---

## 🏆 UNDERDOG CHAMPION QUESTIONS

### **Q17: Retail ML Edge - What Are Institutions Missing?**
> We've built a 10+ module ML system optimized for small-cap 1hr swing trading:
> - Multi-model ensemble (XGBoost GPU + RF + GB)
> - 30+ technical features + microstructure
> - 10 market regimes with position sizing
> - Quantile forecasting for uncertainty
> - Per-ticker AI recommenders
> - HRP portfolio optimization
> - Multi-source data validation
> 
> **CRITICAL QUESTIONS:**
> 1. What blind spots do ML trading systems typically have? (Overfitting, regime changes, black swans?)
> 2. What optimizations haven't we considered? (Meta-learning, ensemble of ensembles, Bayesian optimization?)
> 3. Where can retail traders with ML beat institutions? (Speed? Niche markets? Alternative data?)
> 4. What's the biggest mistake retail ML traders make? (Over-optimization? Too many features? Wrong metrics?)
> 5. How do we avoid curve-fitting while still achieving 68-72% precision?
> 
> **UNDERDOG ADVANTAGE:** 
> We're not constrained by:
> - Regulatory restrictions on position sizes
> - Need to deploy billions (liquidity constraints)
> - Quarterly performance pressure
> - Legacy infrastructure
> 
> How do we exploit these advantages for small-cap swing trading?

---

## 📈 EXPECTED PERFORMANCE TRAJECTORY

### **Week 1: GPU Training (Dec 10-16)**
**Tasks:**
- Train Multi-Model Ensemble (20-30 min)
- Train Quantile Forecaster (75-100 min)
- Train AI Recommender (6-13 hours)

**Expected Metrics:**
- Precision: 50-55% → **55-60%** (+5-10%)
- Sharpe: 0.5-0.8 → **0.8-1.2** (+0.3-0.4)
- Max DD: 15-20% → **12-15%** (-3-5%)

### **Week 2: Hyperparameter Optimization (Dec 17-23)**
**Tasks:**
- Optimize XGBoost depth/learning rate/estimators
- Tune feature engine periods (RSI, MACD, EMA)
- Calibrate regime thresholds for small-caps
- Adjust risk per trade and position sizing

**Expected Metrics:**
- Precision: 55-60% → **60-65%** (+5%)
- Sharpe: 0.8-1.2 → **1.2-1.5** (+0.3)
- Max DD: 12-15% → **10-12%** (-2-3%)

### **Week 3: Advanced Features (Dec 24-30)**
**Tasks:**
- Add sector-specific features (FDA calendars, launches)
- Implement regime-specific models (3-10 models)
- Optimize data source validation
- Add ensemble confidence weighting

**Expected Metrics:**
- Precision: 60-65% → **65-70%** (+5%)
- Sharpe: 1.2-1.5 → **1.5-2.0** (+0.3-0.5)
- Max DD: 10-12% → **8-10%** (-2%)

### **Week 4: Production Validation (Dec 31 - Jan 6, 2026)**
**Tasks:**
- Walk-forward backtesting (12 months out-of-sample)
- Paper trading on Alpaca ($100k virtual)
- Stress testing (2020 COVID crash, 2022 bear market)
- Final hyperparameter tuning

**Target Metrics:**
- ✅ Precision: **68-72%** 
- ✅ Sharpe Ratio: **1.8-2.2**
- ✅ Max Drawdown: **8-10%**
- ✅ Win Rate: **65-70%**
- ✅ Profit Factor: **1.8-2.2**
- ✅ CAGR: **45-65%** (assuming 68% precision, 2:1 reward:risk)

---

## 🎯 SUCCESS CRITERIA

### **Minimum Viable Performance (MVP):**
- Precision ≥60% (10% above random)
- Sharpe ≥1.2 (better than buy-and-hold)
- Max DD ≤12% (capital preservation)
- Win Rate ≥55% (consistent profitability)

### **Competitive Performance (Target):**
- Precision ≥68% (institutional-grade)
- Sharpe ≥1.8 (top decile retail traders)
- Max DD ≤10% (professional risk management)
- Win Rate ≥65% (sustainable edge)

### **World-Class Performance (Stretch):**
- Precision ≥75% (rare in retail)
- Sharpe ≥2.5 (hedge fund level)
- Max DD ≤8% (elite risk control)
- Win Rate ≥70% (consistent alpha)

---

## 📝 PERPLEXITY INSTRUCTIONS

### **How to Use This Brief:**

1. **Copy entire document** to Perplexity Pro chat
2. **Ask questions sequentially** (Q1-Q17) OR all at once if context window allows
3. **Request specific citations** from academic papers, quant finance research, institutional whitepapers
4. **Focus on actionable recommendations**: 
   - Specific hyperparameter values
   - Algorithm choices (LightGBM vs XGBoost vs CatBoost)
   - Threshold calibration methods
   - Best practices from successful retail quant traders

5. **Document answers** in PERPLEXITY_ANSWERS_DEC10.md with format:
```markdown
## Q1: XGBoost GPU Optimization

**Answer:** [Perplexity response]

**Actionable Changes:**
- max_depth: 6 → 8
- learning_rate: 0.1 → 0.03
- n_estimators: 200 → 500

**Citations:**
- [Paper/Source 1]
- [Paper/Source 2]

**Expected Improvement:** +3-5% precision
```

---

## 🚀 NEXT ACTIONS AFTER PERPLEXITY RESEARCH

1. **Implement all 17 recommendations** (prioritize by expected improvement)
2. **Retrain all modules** with optimized hyperparameters
3. **Run walk-forward validation** (12 months out-of-sample)
4. **Compare metrics**: Baseline vs Optimized
5. **Paper trade** on Alpaca for 2 weeks validation
6. **Go live** with $1,000 real capital (10% of $10k target)
7. **Scale up** to $10,000 after 1 month profitable paper trading

---

**Intelligence edge, not speed edge.** 🚂

**Document Generated:** December 10, 2025  
**Purpose:** Comprehensive research brief for Perplexity Pro AI optimization  
**Status:** READY FOR PERPLEXITY RESEARCH ✅  
**Expected Research Time:** 2-4 hours (depending on response depth)
