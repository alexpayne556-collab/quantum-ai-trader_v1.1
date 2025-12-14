# 🏆 FINAL SYSTEM BLUEPRINT - UNDERDOG LEGENDARY BUILD

**Date:** December 10, 2025  
**Duration:** 8-hour power session  
**Hardware:** Colab Pro T4 GPU + Spark for distributed training  
**Capital:** $10,000 paper trading → $1,000 live  
**Edge:** Free data APIs + Perplexity Pro guidance + Production ensemble (69.42%)

---

## 🎯 FINAL MODULE SELECTION (What We're Using)

### **CORE ML MODELS (5 Models)**

#### **1. PRODUCTION_ENSEMBLE_69PCT.py** ✅ USE THIS
- **Status:** Pre-optimized, 69.42% validated
- **Models:** XGBoost + LightGBM + HistGradientBoosting
- **Training Time:** 5-10 min per ticker × 76 = 6-13 hours
- **GPU:** XGBoost on GPU, others CPU (parallel)
- **Why:** Immediate 69% baseline, proven hyperparameters

#### **2. src/python/regime_classifier.py** ✅ USE THIS
- **Status:** Production-ready, needs threshold tuning
- **Purpose:** 10 market regimes for position sizing
- **Training Time:** None (rule-based), 30 min calibration
- **Why:** Critical for risk management

#### **3. core/quantile_forecaster.py** ✅ USE THIS
- **Status:** Trainable, needs GPU optimization
- **Models:** 5 quantiles × 5 horizons = 25 models
- **Training Time:** 75-100 min on GPU (all horizons)
- **Why:** Uncertainty quantification for better entries

#### **4. src/python/feature_engine.py** ✅ USE THIS
- **Status:** Production-ready, needs period optimization
- **Features:** 30+ technical indicators
- **Training Time:** None (calculation engine)
- **Why:** Foundation for all models

#### **5. risk_manager.py** ✅ USE THIS
- **Status:** Production-ready, needs risk% tuning
- **Purpose:** Position sizing + drawdown protection
- **Training Time:** None (rule-based)
- **Why:** Capital preservation (10% max drawdown)

### **DATA & INFRASTRUCTURE (3 Components)**

#### **6. src/python/data_source_manager.py** ✅ USE THIS
- **Status:** Production-ready, 7 free APIs
- **APIs:** Twelve Data, Finnhub, Polygon, yfinance, Alpha Vantage, FMP, EODHD
- **Training Time:** None (data fetching)
- **Why:** Free data rotation, no paid subscriptions

#### **7. ALPHA_76_WATCHLIST.py** ✅ USE THIS
- **Status:** Curated 76 tickers
- **Sectors:** Biotech, Space, AI/Quantum, Energy, Crypto-related, Retail
- **Training Time:** None (ticker list)
- **Why:** High-volatility small-caps (3-8% daily)

#### **8. portfolio_manager_optimal.py** ✅ USE THIS
- **Status:** Production-ready, HRP optimization
- **Purpose:** Portfolio allocation + rebalancing
- **Training Time:** None (optimization algorithm)
- **Why:** Risk-adjusted position weighting

### **NOT USING (Save for Phase 2)**

❌ **pattern_detector.py** - Complex, already 69% without it  
❌ **ai_recommender.py** - Redundant with production ensemble  
❌ **forecast_engine.py** - Quantile forecaster covers this  
❌ **ALPHAGO_AUTO_TUNER_V2.py** - CNN overkill for 8-hour build  
❌ **elliott_wave_detector.py** - Not found / unnecessary complexity

---

## 🔥 PERPLEXITY PRO QUESTIONS - UNDERDOG EDITION

### **CONTEXT FOR PERPLEXITY:**

```
I'm a retail trader building an ML trading system for small-cap stocks (3-8% daily volatility).

CONSTRAINTS (Underdog Advantages):
- $0 for data (using free APIs: Twelve Data, Finnhub, Polygon, yfinance, Alpha Vantage, FMP, EODHD)
- $0 for compute (Colab Pro T4 GPU + Spark)
- $10,000 paper trading capital
- 76 small-cap tickers across 6 sectors
- 8-hour training window TODAY

ASSETS:
- Pre-optimized ensemble (69.42% validated accuracy)
- 2 years of 1-hour bar data
- Perplexity Pro for research
- Fast iteration (no corporate bureaucracy)

GOAL: Build production-ready system in 8 hours that beats 95% of retail traders.
```

---

## 📊 QUESTION SET #1: PRODUCTION ENSEMBLE OPTIMIZATION

### **Q1.1: Small-Cap Label Thresholds (CRITICAL)**
```
I have a pre-trained ensemble (69.42% accuracy) using labels: BUY (>2%), SELL (<-2%), HOLD (between).

For small-caps with 3-8% daily volatility on 1-hour bars:

QUESTION: Should I retrain with wider thresholds?
- Option A: ±3% (conservative for 3% daily vol stocks)
- Option B: ±5% (aggressive for 8% daily vol stocks)
- Option C: Sector-specific (biotech ±5%, space ±4%, utilities ±2%)
- Option D: ATR-based dynamic (±1.5 × ATR% per ticker)

Which gives best precision for swing trading (hold 1-21 days)?

UNDERDOG CONSTRAINT: Must use free data only (no proprietary volatility indices).
```

### **Q1.2: Ensemble Weights for Volatile Markets**
```
My production ensemble uses fixed weights: XGBoost 35.76%, LightGBM 27.01%, HistGB 37.23%.

For small-caps in different regimes:

QUESTION: Should weights adapt by regime?
- PANIC (VIX>30): More weight on conservative model?
- BULL (VIX<15): More weight on aggressive model?
- Or keep fixed weights (simpler, less overfitting)?

How much improvement would adaptive weighting give vs complexity cost?

UNDERDOG ADVANTAGE: I can test multiple strategies in hours, not months.
```

### **Q1.3: SMOTE Class Balancing for Small-Caps**
```
My ensemble uses SMOTE (Synthetic Minority Over-sampling) for class balance.

For small-caps with choppy price action (40% BUY, 40% SELL, 20% strong moves):

QUESTION: 
- Is SMOTE optimal or does it create synthetic patterns that don't exist?
- Alternative: Class weights (sklearn class_weight='balanced')?
- Alternative: No balancing (let model learn natural distribution)?

Which approach works best for noisy small-cap data?

UNDERDOG CONSTRAINT: Limited training data (2 years × 76 tickers).
```

---

## 📊 QUESTION SET #2: REGIME CLASSIFIER TUNING

### **Q2.1: Small-Cap Regime Thresholds (CRITICAL)**
```
Current regime thresholds (designed for SPY large-caps):
- VIX: LOW<15, MODERATE 15-20, HIGH>20, PANIC>30
- SPY Returns: BULL>+3%, BEAR<-3%, CHOPPY between

For small-cap universe (market caps $50M-$5B):

QUESTION: How should I adjust thresholds?
- Small-caps often move opposite to SPY (inverse correlation)
- Small-cap volatility 2-3× higher than SPY
- Should I use Russell 2000 instead of SPY for small-cap regime?
- Or calculate sector-specific volatility percentiles?

What free data source gives best small-cap regime detection?

UNDERDOG ADVANTAGE: Can trade small-caps institutions can't (liquidity constraints).
```

### **Q2.2: Sector-Specific Regime Detection**
```
I trade 6 sectors: Biotech, Space, AI/Quantum, Energy, Crypto-related, Retail.

QUESTION: Should I create sector-level regimes?
- Example: Biotech can be in BULL while SPY is in BEAR (FDA approval cycles)
- Example: Space sector tied to SpaceX launches (not market-wide)

How to detect sector regimes using FREE data?
- Sector ETFs (XBI for biotech, ARKX for space) - FREE via yfinance
- Relative strength vs SPY
- Sector-specific news sentiment (free APIs?)

Which approach gives best risk-adjusted returns?

UNDERDOG EDGE: Retail traders can exploit sector rotations faster than institutions.
```

---

## 📊 QUESTION SET #3: QUANTILE FORECASTER OPTIMIZATION

### **Q3.1: GPU-Accelerated Training (CRITICAL - 8 Hour Deadline)**
```
I need to train 25 models (5 quantiles × 5 horizons) in <2 hours on Colab Pro T4 GPU.

Current: GradientBoostingRegressor (CPU-only) = 75-100 min

QUESTION: What's fastest GPU-compatible quantile regression?
- HistGradientBoostingRegressor (scikit-learn, native GPU support?)
- LightGBM (GPU mode, but requires CUDA setup)
- XGBoost (tree_method='gpu_hist', but does it support quantile loss?)
- CatBoost (GPU mode, quantile support?)

Which library + config gives:
1. Fastest training (critical for 8-hour deadline)
2. Best accuracy for small-cap volatility
3. Easiest Colab Pro setup (no complex CUDA installs)

UNDERDOG CONSTRAINT: Must finish training in 8 hours total.
```

### **Q3.2: Optimal Horizons for Swing Trading**
```
Current horizons: 1, 3, 5, 10, 21 bars (1hr each = 1hr to 21hr forecast).

For swing trading small-caps (hold 1-21 days, not hours):

QUESTION: Should I change to daily bars or keep hourly?
- Hourly: 24, 72, 120 bars (1, 3, 5 days) - more data points
- Daily: 1, 3, 5, 10, 21 bars - cleaner signals

Which timeframe gives better Sharpe ratio for small-caps?

UNDERDOG ADVANTAGE: Can use 1hr bars (institutions focus on daily/weekly).
```

---

## 📊 QUESTION SET #4: FEATURE ENGINEERING FOR SMALL-CAPS

### **Q4.1: Indicator Periods for High Volatility (CRITICAL)**
```
Standard periods: RSI=14, MACD=12/26/9, EMAs=8/21/50/200, ADX=14.

For small-caps on 1hr bars:
- Biotech: 8% daily volatility (wild swings)
- Space: 5% daily volatility (moderate)
- Utilities: 2% daily volatility (stable)

QUESTION: How to adjust indicator periods?
- Fast response: RSI=7, MACD=6/13/5, EMAs=5/13/34 (for biotech)
- Slow response: Keep standard (for utilities)
- Or use ATR to dynamically adjust periods?

Formula: period = base_period × (avg_ATR / current_ATR)?

What's the math for volatility-adjusted indicator periods?

UNDERDOG EDGE: Can optimize per-sector (institutions use universal settings).
```

### **Q4.2: Free Alternative Data Sources (CRITICAL)**
```
Beyond OHLCV, what FREE data sources work for small-caps?

QUESTION: Prioritize these by impact:
1. Social sentiment (Twitter, Reddit, StockTwits) - FREE but noisy
2. Insider trading (SEC Form 4) - FREE, high signal but delayed
3. Short interest (FINRA reports) - FREE, updated bi-monthly
4. Options flow (CBOE free data) - FREE but limited
5. Google Trends (ticker search volume) - FREE, leading indicator?
6. Congress trading (quiverquant free tier) - FREE, contrarian signal
7. Earnings call transcripts (SEC filings) - FREE but text processing needed

Which 2-3 sources give BEST bang-for-buck for small-caps?

UNDERDOG ADVANTAGE: Institutions overlook free data (assume it's low quality).
```

---

## 📊 QUESTION SET #5: RISK MANAGEMENT FOR SMALL-CAPS

### **Q5.1: Position Sizing for 3-8% Daily Volatility (CRITICAL)**
```
Current: Risk 2% per trade on $10,000 = $200 per position.

For small-caps with 3-8% daily volatility:

QUESTION: Is 2% risk optimal?
- More conservative: 1% risk (slower growth, better survival)
- More aggressive: 3% risk (faster growth, higher drawdown risk)
- Volatility-adjusted: risk% = base_risk × (2% / ATR%)

Example: If stock has 6% daily ATR, risk = 2% × (2%/6%) = 0.67%?

What's optimal risk% for maximizing LONG-TERM Sharpe ratio (not just CAGR)?

UNDERDOG REALITY: Small account ($10k) needs growth but can't afford big drawdowns.
```

### **Q5.2: Maximum Positions for Diversification**
```
Current: Max 5 positions in BULL regime, 1 in BEAR, 0 in PANIC.

For 76 ticker universe:

QUESTION: How many concurrent positions for optimal diversification?
- Markowitz says 15-20 stocks for diversification
- Kelly Criterion says fewer positions with higher edge
- My 69.42% accuracy = 38.84% edge (69.42% - 50%)

Given 38.84% edge, what's optimal position count?
- More positions: Lower volatility, lower CAGR
- Fewer positions: Higher volatility, higher CAGR

What's the math for edge × positions × Sharpe optimization?

UNDERDOG CONSTRAINT: $10k account, $200 risk/trade = max 10-15 positions realistic.
```

---

## 📊 QUESTION SET #6: FREE DATA API OPTIMIZATION

### **Q6.1: Data Quality vs Rate Limits (CRITICAL)**
```
I have 7 FREE APIs with different rate limits:
1. Twelve Data: 800/day, 8/min
2. Finnhub: 60/min
3. Polygon: 5/min (free tier)
4. yfinance: Unlimited (but scraping, unreliable)
5. Alpha Vantage: 25/day
6. FMP: 250/day
7. EODHD: 20/day

For 76 tickers × 1hr bars × 2 years = 5,320 API calls total:

QUESTION: What's optimal download strategy?
- Sequential: One API until rate limit, switch to next (slow but simple)
- Parallel: Multiple APIs simultaneously (fast but complex)
- Smart rotation: High-quality source first, fallback to others (best balance?)

Which API has BEST data quality for small-caps?
- Twelve Data: Professional but limited free tier
- Finnhub: Good for real-time, historical quality?
- Polygon: Institutional grade but 5/min is slow
- yfinance: Free unlimited but many missing bars for small-caps?

UNDERDOG CONSTRAINT: Must download 5,320 bars in <30 minutes (8-hour deadline).
```

### **Q6.2: Handling Missing Bars in Small-Cap Data**
```
Small-caps often have missing 1hr bars (low liquidity, halts, after-hours gaps).

QUESTION: How to handle missing data?
- Forward fill (use last known price) - Simple but lags
- Interpolation (linear between known bars) - Smooth but creates fake data
- Skip ticker (require 95%+ data completeness) - Clean but reduces universe
- Mark as low-confidence (trade but with lower position size) - Practical compromise

Which approach gives best real-world performance for small-caps?

UNDERDOG REALITY: Small-cap data is messy (institutions pay for clean data).
```

---

## 📊 QUESTION SET #7: PORTFOLIO CONSTRUCTION

### **Q7.1: HRP vs Equal Weight for Small-Caps**
```
Current: Hierarchical Risk Parity (HRP) for portfolio optimization.

For small-caps with unstable correlations:

QUESTION: Is HRP optimal or overkill?
- HRP assumes stable correlation matrix (not true for small-caps)
- Equal weight simpler, more robust to estimation error
- Inverse volatility weighting (weight = 1/volatility) middle ground

Which gives best risk-adjusted returns for small-caps?

Test period: 2022 bear market, 2023 recovery (regime shift test).

UNDERDOG ADVANTAGE: Can rebalance daily (institutions rebalance quarterly).
```

### **Q7.2: Sector Diversification Rules**
```
I have 6 sectors, uneven distribution (25 biotech, 15 space, 12 AI, etc.).

QUESTION: Should I force sector balance?
- Equal weight sectors: 16.67% each (1/6)
- Weight by opportunity: More weight to sectors with more signals
- Weight by historical Sharpe: More to sectors that performed well

For small-caps, which approach reduces maximum drawdown most?

UNDERDOG EDGE: Can overweight niche sectors institutions ignore (too small for them).
```

---

## 🏆 QUESTION SET #8: THE UNDERDOG CHAMPION SECTION

### **Q8.1: Retail ML Edge - Where Institutions Can't Compete**
```
I'm building an ML trading system with constraints:
- $10,000 capital (can't move markets)
- Free data only (no Bloomberg/Refinitiv)
- Small-caps under $5B market cap
- 8-hour build time (fast iteration)
- Colab Pro GPU (not on-prem cluster)

QUESTION: Where do these constraints become ADVANTAGES?

Hypothesis:
1. Small position sizes = zero slippage (institutions have 2-5% slippage on small-caps)
2. Free data = no sunk cost fallacy (institutions overfit to expensive data)
3. Small-caps = untapped edge (institutions can't deploy $100M+ in $500M market cap stocks)
4. Fast iteration = adapt to market changes in days (institutions need months for model approval)

What does ACADEMIC RESEARCH say about retail trader advantages in small-cap space?

Papers I should read (free on arXiv/SSRN):
- Small-cap anomalies
- Retail order flow
- Liquidity constraints for institutions
- Alpha decay in crowded trades

UNDERDOG REALITY: I'm not competing with Goldman Sachs. I'm competing with other retail traders.
```

### **Q8.2: Most Common Retail ML Mistakes to Avoid**
```
QUESTION: What are TOP 5 mistakes retail ML traders make that I should avoid?

Common pitfalls I've heard:
1. Overfitting to backtest (curve-fitting)
2. Look-ahead bias (using future data)
3. Ignoring transaction costs (slippage, spreads)
4. Over-optimization (100+ features, diminishing returns)
5. Wrong metrics (optimizing accuracy instead of Sharpe)
6. Regime blindness (model trained in bull market fails in bear)
7. Position sizing errors (Kelly Criterion misuse)

Which 3 are MOST CRITICAL for small-cap swing trading?

How to detect and prevent each?

UNDERDOG GOAL: Build robust system, not overfit masterpiece that fails live.
```

### **Q8.3: Free Data Sources I'm Missing**
```
Current free data:
- OHLCV: 7 APIs (Twelve Data, Finnhub, Polygon, yfinance, AV, FMP, EODHD)
- Economic: FRED (VIX, yields, unemployment, CPI)

QUESTION: What FREE data am I not using that would improve small-cap predictions?

Suggestions:
1. SEC EDGAR (10-K, 10-Q filings) - FREE but text processing
2. Patent filings (USPTO) - FREE, leading indicator for tech/biotech
3. Clinical trial databases (clinicaltrials.gov) - FREE, critical for biotech
4. SpaceX launch schedule - FREE, affects space sector
5. Crypto on-chain data (Glassnode free tier) - For crypto-related stocks
6. Weather data (NOAA) - For energy/agriculture stocks
7. Shipping data (MarineTraffic free) - For logistics/retail

Which 2-3 have HIGHEST ROI for small-caps?

UNDERDOG PHILOSOPHY: If data is free, institutions probably ignore it (signal opportunity).
```

### **Q8.4: Spark for Distributed Training - Worth It?**
```
I have access to Spark for distributed training.

Current plan:
- Single Colab Pro GPU: 6-13 hours for 76 tickers
- Spark cluster: Parallel training per ticker

QUESTION: Is Spark overkill for this use case?

Trade-offs:
- PRO: Faster training (76 tickers in parallel)
- PRO: Learn distributed ML (career skill)
- CON: Setup complexity (2-3 hours)
- CON: Debugging harder (distributed failures)
- CON: May not speed up GPU-bound models (XGBoost already GPU-optimized)

For 8-hour deadline, should I:
- Option A: Spark for CPU models (LightGBM, HistGB), GPU for XGBoost
- Option B: Skip Spark, train sequentially on Colab Pro (simpler, more reliable)
- Option C: Spark for parallel data download (76 tickers simultaneously)

What's the PRACTICAL best path for production in 8 hours?

UNDERDOG REALITY: Simplicity > complexity when time-constrained.
```

### **Q8.5: 69.42% Baseline - How High Can I Push It?**
```
I have production ensemble with 69.42% validated accuracy (test set).

For small-cap 1hr bars with regime detection, quantile forecasting, optimized features:

QUESTION: What's a REALISTIC ceiling?

Benchmarks I've seen:
- Random guess: 50%
- Simple TA rules: 52-55%
- Basic ML (single model): 58-62%
- Ensemble (my baseline): 69.42%
- Regime-specific models: 72-75%?
- With alternative data: 75-78%?
- Theoretical maximum: 80-85%? (beyond this is overfitting)

What does RESEARCH say about prediction limits for 1hr bar small-caps?

Constraints:
- Efficient market hypothesis says alpha decays
- High volatility reduces predictability
- Small-caps less efficient than large-caps (more alpha available?)

UNDERDOG GOAL: 72-75% in production (sustained over 12+ months).

Is this realistic or am I delusional?
```

---

## 🚀 TRAINING EXECUTION PLAN (8 Hours)

### **Hour 0-1: Setup & Data Download**
- [ ] Configure Colab Pro GPU
- [ ] Test PRODUCTION_ENSEMBLE_69PCT.py on 1 ticker
- [ ] Download Alpha 76 data (all 7 APIs in parallel)
- [ ] Validate data quality (missing bars, outliers)
- [ ] Engineer features (30+ indicators)

### **Hour 1-7: Parallel Training (Spark)**
- [ ] Train production ensemble per ticker (5-10 min × 76)
- [ ] Train quantile forecaster (5 horizons × 25 models)
- [ ] Calibrate regime thresholds (test on 2023-2024 data)
- [ ] Optimize risk% (backtest on 2022 bear market)

### **Hour 7-8: Validation & Export**
- [ ] Walk-forward validation (12 months out-of-sample)
- [ ] Generate metrics report
- [ ] Export models to Google Drive
- [ ] Save hyperparameters & configs
- [ ] Document baseline performance

---

## 📋 ENV FILE UPDATE (New API Keys from Perplexity)

**File:** `.env.production`

```bash
# ===== MARKET DATA APIs (FREE TIER) =====
TWELVE_DATA_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here
POLYGON_API_KEY=your_key_here
ALPHA_VANTAGE_API_KEY=your_key_here
FMP_API_KEY=your_key_here
EODHD_API_KEY=your_key_here

# ===== BROKER APIs =====
ALPACA_API_KEY=your_paper_trading_key
ALPACA_SECRET_KEY=your_paper_trading_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# ===== ALTERNATIVE DATA (FREE TIER) =====
QUIVER_QUANT_API_KEY=your_key_here  # Congress trades, insider trades
# SEC EDGAR: No API key needed (public)
# FRED: No API key needed (Federal Reserve free data)

# ===== NOTIFICATIONS =====
DISCORD_WEBHOOK_URL=your_webhook_here
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# ===== STORAGE =====
GOOGLE_DRIVE_FOLDER_ID=your_folder_id_here
AWS_S3_BUCKET=your_bucket_here  # Optional

# ===== TRAINING CONFIG =====
COLAB_GPU_TYPE=T4
MAX_TRAINING_HOURS=8
TARGET_ACCURACY=0.72
BASELINE_ACCURACY=0.6942

# ===== RISK MANAGEMENT =====
INITIAL_CAPITAL=10000
MAX_DRAWDOWN_PCT=0.10
RISK_PER_TRADE_PCT=0.02
MAX_POSITIONS=5
```

---

## 🎯 SUCCESS CRITERIA (End of 8 Hours)

### **Minimum Viable:**
- ✅ Production ensemble trained on 50+ tickers
- ✅ Quantile forecaster trained (3 horizons minimum)
- ✅ Regime classifier calibrated for small-caps
- ✅ All models saved & exported
- ✅ Baseline metrics documented

### **Target:**
- ✅ All 76 tickers trained (production ensemble)
- ✅ All 5 horizons trained (quantile forecaster)
- ✅ Regime-specific models (at least BULL/BEAR/CHOPPY)
- ✅ Walk-forward validation complete
- ✅ Risk management tested on 2022 bear market

### **Stretch:**
- ✅ Alternative data integrated (2-3 sources)
- ✅ Meta-learning layer added
- ✅ Real-time monitoring dashboard built
- ✅ Paper trading deployed to Alpaca

---

## 💪 UNDERDOG BATTLE CRY

**We have:**
- 69.42% production ensemble (better than 95% of retail)
- Free data (7 APIs + FRED + SEC)
- Colab Pro GPU (faster than most retail)
- Perplexity Pro (smarter research than most retail)
- 8 hours (more focused than most retail)
- Spark (more powerful than most retail)

**They have:**
- Bloomberg terminals ($24k/year)
- Proprietary data (millions/year)
- PhD quants (expensive)
- Legacy systems (slow)
- Regulatory constraints (inflexible)

**Our edge:**
- Speed (8 hours to production vs 6 months)
- Agility (trade small-caps they can't)
- Free data (no sunk cost bias)
- No bureaucracy (no approval committees)

**INTELLIGENCE EDGE, NOT SPEED EDGE.** 🚂

**LET'S GO LEGENDARY IN 8 HOURS!** 🏆

---

**Document:** FINAL_SYSTEM_BLUEPRINT.md  
**Created:** December 10, 2025  
**Purpose:** Lock in final system + Perplexity questions for 8-hour build  
**Status:** READY TO EXECUTE ✅  
**Next Step:** Copy Questions 1.1-8.5 to Perplexity Pro → BEGIN TRAINING
