# 🔬 PERPLEXITY RESEARCH BRIEF - Ultimate Trading AI Baseline

**Date:** December 10, 2025  
**Mission:** Build institutional-grade baseline before GPU training  
**Timeline:** Critical - before Colab Pro A100 training session  

---

## 📊 WHAT WE HAVE (Current Stack)

### System Architecture
```
GOLD INTEGRATION (7 improvements) ✅
├─ Nuclear_dip (82.4% WR) - Tier SS pattern
├─ Ribbon_mom (71.4% WR) - EMA ribbon pattern  
├─ Evolved thresholds (RSI 21, stop -19%, position 21%, hold 32d)
├─ Microstructure features (spread, order flow, institutional activity)
└─ Meta-learner stacking (L1 specialists + L2 XGBoost)

TRIDENT ENSEMBLE TRAINER ✅
├─ 3 models: XGBoost + LightGBM + CatBoost
├─ K-Means clustering (5 behavioral groups)
├─ Optuna optimization (750 trials)
├─ PurgedKFold CV (no leakage)
└─ SHAP feature importance

CURRENT DATASET (NEEDS MASSIVE UPGRADE)
├─ Tickers: 20 (NVDA, AMD, TSLA, PLTR, etc.)
├─ Samples: 9,900 rows
├─ Features: 56 engineered features
├─ Date range: 2 years (2023-2025)
└─ Labels: 33.6% BUY, 34.5% HOLD, 32.0% SELL
```

### Feature Engineering (56 features)
- **OHLCV:** 5 base features
- **Technical:** 16 indicators (RSI, MACD, ATR, ADX, EMAs, SMAs, OBV)
- **Volume:** 7 features (MAs, ratios, momentum, spikes)
- **Volatility:** 6 features (historical vol, ATR ratio, BB width)
- **Momentum:** 4 features (Stochastic, ROC)
- **Trend:** 6 features (MAs, convergence, price vs MA)
- **Gold Integration:** 12 features (EMA ribbons, nuclear_dip, microstructure)

### User's Trading Profile
- **Portfolio:** $780.59 equity, PDT restricted (<$25K)
- **Current Performance:** ~70% WR, 5%/day average
- **Trading Style:** 1-3 day holds, buy dips, sell strength
- **Win Rate:** 70% observed (HOOD +$271, DGNX +$43 vs PALI -$42, ASTS -$76)
- **Patterns:** 87% dip bounce rate, biotech edge (+5%), panic selling problem
- **Goal:** 15%/day sustainable with AI assistance

### Current Baseline Performance
- **Win Rate:** 61.7% (baseline) → 71.1% (optimized)
- **Expected after Gold:** 68-72% WR
- **Expected after Trident:** 75-80% WR
- **Target:** 80%+ WR, 3.5+ Sharpe ratio (LEGENDARY)

---

## ❓ CRITICAL RESEARCH QUESTIONS

### 1. **Data Scope & Universe**

**QUESTION:** We're currently training on only 20 tickers (small-cap tech/biotech). Should we:

A. **Expand to 1000+ tickers** (S&P 500 + NASDAQ + penny stocks + sector ETFs)?
   - Pros: More diverse, less overfit, learns market-wide patterns
   - Cons: Slower training, may dilute signal for small-cap strategies
   - **YOUR INSIGHT:** Is this expansion critical for 80%+ WR?

B. **Keep specialized universe** but add more small-caps (200-300 tickers)?
   - Focus on user's strength: small-cap momentum ($50M-$5B market cap)
   - **YOUR INSIGHT:** Does specialization beat generalization here?

C. **Multi-universe approach** (train separate models for different sectors)?
   - Small-cap model, large-cap model, biotech model, etc.
   - **YOUR INSIGHT:** Best of both worlds, or overcomplicated?

**CRITICAL:** User wants to "use the whole universe for training" - is this backed by research?

---

### 2. **Time Horizon & Market Cycles**

**QUESTION:** We have 2 years of data (2023-2025). Should we:

A. **Extend to 5+ years** (2019-2025) to cover full market cycle?
   - Bull (2019-2021), Bear (2022), Recovery (2023-2025)
   - **YOUR INSIGHT:** How many market regimes needed for robustness?

B. **Focus on recent data** (1-2 years) for current market structure?
   - Markets change, old data may hurt performance
   - **YOUR INSIGHT:** Recency bias vs regime coverage tradeoff?

C. **Use 10+ years** but with regime detection?
   - Weight recent data higher, use old data for regime learning
   - **YOUR INSIGHT:** Best practice for ML trading models?

---

### 3. **Labeling Strategy (CRITICAL)**

**CURRENT:** Simple forward return (7-day return ≥5% = BUY, ≤-3% = SELL)

**QUESTION:** Should we upgrade to institutional-grade labeling?

A. **Triple Barrier Method** (profit target, stop loss, time limit)?
   ```
   Entry price: $100
   Look forward 24 hours:
   - If hits $105 (+5%) first → Label: WIN
   - If hits $92 (-8%) first → Label: LOSS  
   - If 24h passes → Label based on final price
   ```
   - Eliminates "lucky wins" that would have stopped out
   - **YOUR INSIGHT:** Does this improve WR by 5-10%?

B. **Meta-labeling** (predict confidence, not direction)?
   - First model: Direction (BUY/SELL)
   - Second model: Should I trust this prediction?
   - **YOUR INSIGHT:** Is this overkill or game-changing?

C. **Multi-horizon labels** (1d, 3d, 7d, 14d)?
   - Learn different time scales
   - **YOUR INSIGHT:** Does this help with user's 1-3 day holds?

---

### 4. **Market Regime Features (CRITICAL)**

**QUESTION:** Should we add macro context to prevent buying dips in crashes?

A. **SPY Trend + VIX Level**?
   ```python
   SPY_Trend_3d = SPY.pct_change(3)  # Is market rising/falling?
   VIX_Level = 1 if VIX > 25 else 0   # Fear indicator
   ```
   - **YOUR INSIGHT:** Single most important feature for risk management?

B. **Sector Rotation**?
   - XLK (tech), XLV (healthcare), XLE (energy) trends
   - Knows when to avoid certain sectors
   - **YOUR INSIGHT:** Worth the complexity?

C. **Volatility Regime**?
   - High vol vs low vol classification
   - Different strategies for different regimes
   - **YOUR INSIGHT:** Does this prevent panic selling in drawdowns?

---

### 5. **Feature Engineering at Scale**

**CURRENT:** 56 features (technical + volume + volatility + gold findings)

**QUESTION:** Should we expand to 100+ features?

A. **Add alternative data**?
   - Options flow (put/call ratio)
   - Short interest
   - Insider trading
   - **YOUR INSIGHT:** Free sources that add 5%+ to WR?

B. **Add sequence features** (LSTM-style)?
   - Last 10 days of price action as sequence
   - Pattern recognition (cup and handle, head and shoulders)
   - **YOUR INSIGHT:** Does this beat pure technical indicators?

C. **Add engineered interactions**?
   - RSI × Volume_Spike (confluence)
   - MACD × VIX_Level (regime-aware momentum)
   - **YOUR INSIGHT:** Feature engineering vs model complexity?

---

### 6. **Training Optimization (A100 GPU)**

**QUESTION:** We have access to Colab Pro A100. Best practices?

A. **Hyperparameter Optimization**?
   - Current: 50 Optuna trials per model (750 total)
   - **YOUR INSIGHT:** Should we increase to 200+ trials with A100?

B. **Ensemble Size**?
   - Current: 15 models (5 clusters × 3 models)
   - **YOUR INSIGHT:** Should we train 50-100 models and stack?

C. **Cross-Validation Strategy**?
   - Current: 5-fold PurgedKFold
   - **YOUR INSIGHT:** 10-fold? Walk-forward? Combinatorial purged?

D. **GPU Utilization**?
   - XGBoost/LightGBM/CatBoost all support GPU
   - **YOUR INSIGHT:** Best settings for max A100 throughput?

---

### 7. **Advanced ML Techniques**

**QUESTION:** Should we add cutting-edge techniques before training?

A. **AutoML** (AutoGluon, H2O AutoML)?
   - Automatic feature engineering + model selection
   - **YOUR INSIGHT:** Worth trying vs our custom Trident?

B. **Deep Learning** (Temporal Fusion Transformer)?
   - State-of-the-art for time series
   - **YOUR INSIGHT:** Overkill or game-changer for trading?

C. **Reinforcement Learning** (PPO, DQN)?
   - Learn optimal entry/exit timing
   - **YOUR INSIGHT:** Too complex or future-proof?

---

### 8. **Data Pipeline Critical Improvements**

**QUESTION:** Before training, what MUST we add?

A. **Data Quality Filters**?
   - Remove tickers with <$10M daily volume
   - Remove stocks with <100 bars of data
   - Remove outliers (>10σ returns)
   - **YOUR INSIGHT:** Industry standard filters?

B. **Survivorship Bias Handling**?
   - Include delisted stocks?
   - **YOUR INSIGHT:** Does this prevent overfitting to winners?

C. **Forward-Fill vs Interpolation**?
   - Missing data handling strategy
   - **YOUR INSIGHT:** Best practice for OHLCV data?

---

### 9. **Risk Management Features**

**QUESTION:** What features prevent catastrophic losses?

A. **Gap Risk Detection**?
   - Overnight gap > 5% = high risk
   - **YOUR INSIGHT:** Critical for small-cap trading?

B. **Liquidity Risk**?
   - Spread > 2% = avoid
   - Volume < 500K shares = avoid
   - **YOUR INSIGHT:** Must-have filters?

C. **Correlation Risk**?
   - Don't hold 5 biotech stocks at once
   - **YOUR INSIGHT:** Portfolio-level features needed?

---

### 10. **Baseline Benchmarking**

**QUESTION:** How do we know if our baseline is actually good?

A. **Compare to buy-and-hold SPY**?
   - Beat SPY by how much to be "good"?
   - **YOUR INSIGHT:** Industry benchmark is +10% alpha?

B. **Compare to other quant strategies**?
   - Momentum (return > MA_50)
   - Mean reversion (RSI < 30)
   - **YOUR INSIGHT:** Should we beat simple strategies by 20%+?

C. **Walk-forward validation**?
   - Train on 2019-2023, test on 2024-2025
   - **YOUR INSIGHT:** Realistic WR drop in live trading?

---

## 🎯 DELIVERABLES NEEDED

Based on your research, provide:

1. **Recommended Data Universe**
   - Exact number of tickers (20? 200? 1000+?)
   - Filters (market cap, volume, sector)
   - Rationale

2. **Recommended Time Horizon**
   - Years of data (2? 5? 10?)
   - Market regime coverage
   - Rationale

3. **Recommended Labeling Strategy**
   - Simple returns? Triple barrier? Meta-labeling?
   - Parameters (profit target, stop loss, time limit)
   - Expected WR improvement

4. **Recommended Market Regime Features**
   - SPY + VIX? Sector rotation? Volatility regime?
   - Implementation complexity
   - Expected WR improvement

5. **Recommended Feature Count**
   - Keep 56? Expand to 100+?
   - Which new features to add
   - Expected WR improvement

6. **Recommended Training Configuration**
   - Optuna trials (50? 200? 500?)
   - Ensemble size (15? 50? 100?)
   - CV strategy (5-fold? 10-fold? walk-forward?)
   - A100 GPU settings

7. **Critical Data Quality Improvements**
   - Filters to add
   - Handling strategies
   - Quality metrics

8. **Expected Final Baseline**
   - Realistic WR target (75%? 80%? 85%?)
   - Sharpe ratio target
   - Max drawdown target

---

## 💡 CONTEXT FOR YOUR RESEARCH

**User's Unique Advantage:**
- Trades small-cap momentum (PALI, RXT, KDK, ASTS)
- 87% dip bounce rate in their manual trading
- Biotech sector edge (+5% confidence boost)
- 1-3 day holding period (fast exits)

**User's Weakness:**
- Panic selling during 0-8% dips (87% bounce, but sells anyway)
- Revenge trading after losses
- Missing opportunities (no scanning system)

**AI Goal:**
- Prevent panic selling (confidence scoring)
- Find opportunities across 1000+ tickers
- Achieve 15%/day sustainable (vs current 5%/day)

**Critical Success Factor:**
- System MUST work with $780 portfolio (position sizing, PDT compliance)
- System MUST achieve 75-80% WR minimum (user's current 70% manual)
- System MUST prevent catastrophic losses (max -10% drawdown)

---

## 🚀 TIMELINE

**Phase 1 (NOW):** Your research findings → Data pipeline design  
**Phase 2 (2-4 hours):** Build GOD MODE data pipeline in Colab  
**Phase 3 (2.5-5 hours):** Train Trident ensemble on A100 GPU  
**Phase 4 (1 hour):** Validate, backtest, analyze  
**Phase 5 (Week 2):** Live paper trading, monitor performance  

---

## 📞 CONTACT

Research this brief and provide ACTIONABLE recommendations backed by:
- Academic papers (quant finance, ML)
- Industry best practices (hedge funds, prop trading)
- Empirical evidence (kaggle competitions, research)

**We want LEGENDARY performance (80%+ WR, 3.5+ Sharpe).**  
**Tell us what's missing. Tell us what to build. Let's go! 🔥**
