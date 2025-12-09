# 🎯 COMPLETE PERPLEXITY RESEARCH AGENDA
## 37 Questions to Answer Before Building (December 9, 2025)

**Purpose**: Eliminate all architectural unknowns before implementation  
**Timeline**: 3-4 hours tomorrow morning  
**Output**: Complete answers documented in `PERPLEXITY_FORECASTER_ARCHITECTURE.md`

---

## 📋 BATCH 1: Meta-Learner Architecture (Core - 6 questions)

### Q1: Optimal Ensemble Method for Trading Forecaster
```
Given a trading forecaster combining:
- Pattern detection signals (Elliott Wave, candlesticks, S/R)
- Regime classification (12-regime system: VIX × breadth × ATR)
- Research features (40-60 engineered: dark pool, sentiment, microstructure)
- Pre-catalyst signals (options OI, insider trades, news)
- Cross-asset correlations (BTC, yields, VIX leads)

Which ensemble architecture minimizes overfitting on small samples (~100 trades/regime) while maximizing predictive accuracy:
A) Weighted average with regime-specific weights
B) Gradient boosting (XGBoost/LightGBM)
C) Bayesian model averaging
D) Small neural network (3 layers, 32-16-8 units)
E) Hierarchical ensemble (patterns→regime→research→final)

Provide: Implementation recommendation, expected accuracy improvement, computational cost, pros/cons for each.
```

### Q2: Regime-Aware Weight Matrices
```
For 12 market regimes (BULL/BEAR/NEUTRAL × LOW/NORMAL/ELEVATED/EXTREME_VOL × STABLE/MODERATE/VOLATILE_ATR):

How should signal weights change dynamically:
- BULL_LOW_VOL_STABLE: favor patterns (40%?), regime (30%?), research (20%?)?
- BEAR_EXTREME_VOL_VOLATILE: favor catalysts (40%?), research (30%?), cross-asset (20%?)?
- NEUTRAL_NORMAL_VOL_MODERATE: favor microstructure (35%?), regime (25%?)?

Provide: Complete 12×5 weight matrix (12 regimes × 5 signal types) optimized for Sharpe ratio. Cite research if available (Hamilton 2024 regime-switching, others).
```

### Q3: Confidence Calibration Method
```
Problem: Raw model confidence 70% → actual win-rate only 55% (overconfident).

Which calibration method works best for small samples per regime (~100 trades):
A) Platt scaling (logistic regression: P_cal = 1/(1+exp(A*P_raw + B)))
B) Isotonic regression (non-parametric, monotonic)
C) Beta calibration (extension of Platt for both under/over-confidence)
D) Temperature scaling (neural network calibration)
E) Histogram binning with Laplace smoothing

Provide: Method recommendation, expected calibration error (ECE target <0.05?), sample code, when to recalibrate (every 20 trades?).
```

### Q4: Signal Agreement Boosting
```
When all 5 signals agree (patterns + regime + research + catalyst + cross-asset all say BUY):
- Should confidence be boosted? By how much? (1.2× multiplier?)
- Should position size increase? (2× normal size?)
- Historical evidence: does signal agreement predict higher win-rate?

When only 1-2 signals fire:
- Should confidence be reduced? (0.7× multiplier?)
- Should position size decrease or skip trade entirely?
- Threshold: Minimum how many signals must agree to trade?

Provide: Signal agreement formula, confidence adjustment rules, position sizing rules.
```

### Q5: Handling Missing Signals
```
In production, some signals may be unavailable:
- Dark pool data missing (API down)
- Sentiment data stale (EODHD rate limit hit)
- Insider data incomplete (Finnhub lag)

Fallback strategies:
A) Skip ticker if critical signal missing
B) Use last known value (with staleness penalty)
C) Impute from correlated tickers
D) Reduce signal weight proportionally
E) Use regime-specific default values

Provide: Recommended fallback per signal type, staleness thresholds (max age before ignore), imputation methods.
```

### Q6: Meta-Learner Training Strategy
```
Should weights be:
A) Fixed (hand-tuned based on research, never updated)
B) Learned once (train on 2021-2024, freeze)
C) Periodically retrained (every month on rolling 18-month window)
D) Adaptive (online learning, update after each trade)

Pros/cons of each? Risk of overfitting? Computational cost? Expected accuracy improvement?

Provide: Recommended strategy, retraining schedule, validation protocol (how to ensure new weights > old weights).
```

---

## 📋 BATCH 2: Feature Engineering & Selection (Critical - 6 questions)

### Q7: Feature Selection from 60 Candidates
```
Starting with 60 engineered features (microstructure, sentiment, momentum, volatility, cross-asset):

How to reduce to top 15-20 without losing predictive power:
A) SHAP importance ranking (keep top 15)
B) Correlation filtering (drop features with r>0.8)
C) Recursive feature elimination (RFE)
D) L1 regularization (LASSO) automatic selection
E) Forward selection (add one feature at a time, keep if Sharpe improves)

Provide: Recommended method, expected accuracy loss if any, computational cost, sector-specific feature importance (AI vs Quantum vs Robotaxi).
```

### Q8: Microstructure Proxies from Free Data
```
Without paid Level 2 data, how to proxy institutional activity:

1. Dark pool ratio: Use yfinance minute volume clustering? FINRA ATS weekly data (2-week lag)? Other?
2. Spread compression: Use (High-Low)/Close ratio? Intraday VWAP deviation? Other?
3. Order flow imbalance: Volume-weighted price momentum? Tick direction from minute bars? Other?
4. Smart money accumulation: After-hours volume? Block trades (volume spikes >3σ)? Other?

Provide: Best free proxy for each, validation method (correlation with paid data if available), expected signal strength.
```

### Q9: Sentiment Integration
```
EODHD sentiment score (-1 to +1) aggregated daily:

How to use effectively:
A) Raw sentiment as single feature
B) Sentiment trend (5-day change)
C) Sentiment divergence (price up, sentiment down → bearish)
D) Sentiment-weighted volume (high volume + positive sentiment → bullish)
E) News tier classification (break news vs earnings vs supply chain vs macro)

Research shows sentiment alone has R²≈0.01 (useless). Combined with price action: R²≈0.35 (meaningful).

Provide: Optimal sentiment feature engineering, combination rules, when to ignore sentiment (low article count?).
```

### Q10: Cross-Asset Lead-Lag Relationships
```
Research discovered:
- BTC returns lead tech stocks by 6-24 hours (correlation 0.72 in bull regimes)
- 10Y yield changes lead sector rotation by 3-5 days
- VIX term structure predicts volatility regime by 1-3 days
- Sector breadth divergences predict corrections by 2-7 days

How to validate these leads without look-ahead bias:
- Use BTC data from T-1 day (available at market close)?
- Use yield data from T (FRED updates daily)?
- What correlation threshold to trust the signal (r>0.5?)?

Provide: Timestamp alignment checklist, lag validation method, correlation thresholds per asset class.
```

### Q11: Regime-Specific Feature Importance
```
Hypothesis: Different features matter in different regimes:
- BULL: Patterns (EMA crosses, breakouts) matter most
- BEAR: Catalysts (insider trades, analyst downgrades) matter most
- CHOP: Microstructure (spread, volume) matters most

How to validate:
A) Train separate models per regime, compare feature importance
B) Use interaction terms (feature × regime indicator)
C) SHAP values conditional on regime

Provide: Recommended approach, expected accuracy improvement from regime-specific features, computational cost.
```

### Q12: Feature Staleness & Drift Detection
```
Features calculated on 2023-2025 data may drift by 2027:
- Market structure changes (more ETF trading)
- Algo evolution (HFTs adapting to patterns)
- Regime shifts (new regime not seen before)

How to detect drift:
A) Population Stability Index (PSI) per feature (threshold PSI>0.2 → retrain)
B) Kolmogorov-Smirnov test (distribution shift)
C) Rolling correlation with returns (if drops >20% → feature weakening)
D) Performance monitoring (if Sharpe drops >0.1 → investigate features)

Provide: Recommended drift detection method, alert thresholds, retraining trigger logic.
```

---

## 📋 BATCH 3: Training & Validation (Critical - 6 questions)

### Q13: Regime-Aware Cross-Validation
```
Standard k-fold CV causes look-ahead bias and regime mixing.

For trading, need regime-aware time-series CV:
A) Walk-forward (train 18mo, test 3mo, roll forward)
B) Blocked time-series CV (5 folds, preserve time order)
C) Regime-stratified CV (ensure train/test have same regime distribution)
D) Purged k-fold (drop samples near fold boundaries to prevent leakage)

Which method prevents overfitting while giving reliable out-of-sample estimates?

Provide: Recommended CV strategy, implementation code, expected train/test performance gap (5%? 10%?).
```

### Q14: Class Imbalance Handling
```
Label distribution: CHOP 40%, UP 35%, DOWN 25% (imbalanced).

Standard model predicts CHOP for everything (high accuracy, no profit).

Solutions:
A) Class weighting (penalize CHOP misclassification less)
B) Threshold tuning (predict UP if confidence >60% not 50%)
C) SMOTE (synthetic minority oversampling)
D) Cost-sensitive learning (custom loss function)
E) Separate models (one for UP, one for DOWN, ensemble)

Provide: Recommended approach for trading (prioritize precision over recall?), expected win-rate improvement.
```

### Q15: Small Sample Overfitting Prevention
```
Each regime has ~100 trades (small sample). Risk of overfitting.

Regularization strategies:
A) L2 regularization (ridge regression on weights)
B) Dropout (neural networks only)
C) Early stopping (monitor validation loss)
D) Ensemble (bagging: train on bootstrap samples)
E) Shrinkage (pull estimates toward global mean)

Which combination works best for small-sample trading systems?

Provide: Recommended regularization stack, expected generalization improvement, hyperparameter tuning guidance.
```

### Q16: Sample Size Requirements
```
How many trades needed for 95% confidence in win-rate estimate?

Current: ~100 trades per regime per year.

Statistical power calculation:
- Assume true win-rate = 60%, want ±5% precision
- Required sample size = ?
- If insufficient, how to aggregate regimes (combine BULL_LOW + BULL_NORMAL)?

Provide: Sample size formula, aggregation strategy if needed, confidence intervals for current sample size.
```

### Q17: Train/Test Split Ratios
```
For 5 years data (2020-2025), what split ratio:
A) 60/20/20 (train/val/test)
B) 70/15/15
C) 80/10/10
D) Walk-forward (no fixed split, rolling windows)

Trade-off: More training data → better model, but less test data → less reliable validation.

Provide: Recommended split for 5 years data, rationale, expected performance variance.
```

### Q18: Holdout Test Set Strategy
```
Final test set (never touched until end):
- Should it be most recent data (2024-2025)?
- Should it span all regimes (stratified sample)?
- Should it include stress events (2020 crash)?

Provide: Recommended holdout strategy, size (10%? 20%?), composition.
```

---

## 📋 BATCH 4: Calibration & Position Sizing (Critical - 6 questions)

### Q19: Calibration Curve Fitting
```
Per-regime calibration: 12 regimes × ~100 trades each = small sample per curve.

Risk of overfitting calibration itself.

Solutions:
A) Pool similar regimes (BULL_LOW + BULL_NORMAL → single curve)
B) Hierarchical model (global curve + regime-specific adjustments)
C) Regularized calibration (penalize deviation from global curve)
D) Only calibrate regimes with >100 samples, use global for others

Provide: Recommended approach, expected calibration error (ECE), recalibration frequency.
```

### Q20: Kelly Criterion for Multi-Outcome Trading
```
Standard Kelly: f* = (p*b - q) / b
- p = win probability
- b = win amount / loss amount

For trading with stops/targets:
- Win outcome: +7.5% (take-profit)
- Loss outcome: -3.5% (stop-loss)
- Chop outcome: ±1% (time decay)

Modified Kelly formula?

Provide: Multi-outcome Kelly formula, implementation, fractional Kelly recommendation (25%? 50%?), expected Sharpe improvement.
```

### Q21: Volatility Targeting
```
Target portfolio volatility = 15% annualized.

Position sizing formula:
position_size = (target_vol / ticker_vol) × base_size

But tickers have different vol regimes:
- AI: 30% vol (size down by 0.5×)
- Utilities: 15% vol (size up by 1.0×)
- Quantum: 50% vol (size down by 0.3×)

How to combine Kelly + vol-targeting + regime adjustment?

Provide: Complete position sizing formula, example calculations, expected portfolio vol achieved.
```

### Q22: Stop-Loss Optimization
```
Sector-specific stops:
- AI: -3% (low vol, tight stops)
- Quantum: -8% (high vol, wide stops)
- Robotaxi: -4%

But regime matters too:
- EXTREME_VOL: widen stops by 1.5×?
- LOW_VOL: tighten stops by 0.8×?

How to combine sector + regime + ticker vol into optimal stop?

Provide: Stop-loss formula, expected hit rate (target 20-30%?), validation method.
```

### Q23: Take-Profit vs Trailing Stop
```
Fixed take-profit (e.g., +7.5%) vs trailing stop (e.g., trail by 5%):

Pros/cons:
- Fixed TP: Locks profit, but misses big moves
- Trailing stop: Captures trends, but gives back profit

Which works better for 5-15d holding period?

Provide: Recommended approach per regime (trailing in strong trends, fixed in chop?), parameter tuning guidance.
```

### Q24: Portfolio-Level Risk Limits
```
Constraints:
- Max 5% per position
- Max 30% per sector
- Max total allocation 100%

But should these change by regime?
- BULL: Allow up to 40% in hot sector?
- BEAR: Cap at 20% per sector (diversify)?

Provide: Regime-specific constraint matrix, rationale, expected risk reduction.
```

---

## 📋 BATCH 5: Implementation & Production (Medium Priority - 7 questions)

### Q25: Model Retraining Schedule
```
Options:
A) Daily (retrain on rolling 18-month window)
B) Weekly (every Monday)
C) Monthly (first trading day)
D) On-demand (only when drift detected)

Trade-off: More frequent → adapts faster, but risks overfitting to noise.

Provide: Recommended schedule, computational cost, validation that new model > old model.
```

### Q26: Execution Slippage Modeling
```
Using only free data, what is defensible slippage model:
A) Fixed bps (5 bps for liquid tickers, 15 bps for illiquid)
B) Spread-based (use (High-Low)/Close as spread proxy)
C) Volume-based (higher slippage for low-volume days)
D) Historical execution analysis (backtest what actual fills would be)

Provide: Recommended model, parameter values per liquidity tier, validation method.
```

### Q27: Timestamp Alignment Guardrails
```
To prevent look-ahead bias:
- Market close: 4:00 PM ET
- yfinance data available: ~5:00 PM ET (1hr lag)
- EODHD sentiment: 6:00 PM ET (2hr lag)
- FRED macro: next day 8:30 AM

How to align timestamps safely:
- Use data only from T-1 day?
- Use intraday data with strict cutoff?
- Validate no future data in features?

Provide: Timestamp alignment checklist, code pattern, validation tests.
```

### Q28: Backtesting Realism Checks
```
Common backtest pitfalls:
- Survivorship bias (only tickers that survived)
- Look-ahead bias (using future data)
- Cherry-picking (testing on same period as tuning)
- Ignoring costs (slippage, commissions, market impact)

Validation checklist for clean backtest?

Provide: Realism checklist, validation tests, expected performance haircut from ideal backtest.
```

### Q29: Drift Monitoring Metrics
```
Production monitoring:
A) PSI (Population Stability Index) per feature
B) KS (Kolmogorov-Smirnov) test per feature
C) Rolling win-rate (last 20 trades)
D) Rolling Sharpe (last 50 trades)
E) Confidence calibration error (predicted vs actual)

Which metrics alert earliest to model degradation?

Provide: Recommended monitoring stack, alert thresholds, escalation rules (when to pause trading).
```

### Q30: Ablation Test Matrix
```
To measure incremental value:
1. Patterns only
2. Patterns + regimes
3. Patterns + regimes + research features
4. Full system (+ calibration + position sizing)

Expected Sharpe improvement at each step:
1 → 2: +0.1 Sharpe?
2 → 3: +0.15 Sharpe?
3 → 4: +0.05 Sharpe?

Provide: Ablation experiment design, stopping rules (if component doesn't add value, remove it), validation protocol.
```

### Q31: Scenario Replay Validation
```
Stress test on known events:
- 2020 COVID crash (Feb-Apr)
- 2022 rates shock (Jan-Jun)
- 2023 AI melt-up (full year)

Expected performance:
- 2020: Survive with DD <25%? Sharpe negative OK?
- 2022: Adapt to bear regime? Sharpe >0.3?
- 2023: Capture upside? Sharpe >0.7?

Provide: Scenario testing protocol, pass/fail criteria, how to improve if fails.
```

---

## 📋 BATCH 6: Research Gaps & Advanced Topics (Lower Priority - 6 questions)

### Q32: Sequence Models vs Tabular Ensembles
```
For time-series with 40-60 features:
A) Tabular (XGBoost, Random Forest): Fast, interpretable, handles small samples
B) Sequence (LSTM, GRU, TCN): Captures temporal dependencies, needs more data
C) Transformer: Best for long-range dependencies, very data-hungry

Given ~100 samples per regime, which architecture?

Provide: Architecture recommendation, expected accuracy improvement, computational cost, when to upgrade to sequence model.
```

### Q33: Transfer Learning from Related Assets
```
Pre-train model on SPY (1000s of data points), fine-tune on low-sample tickers (quantum stocks with 100 data points):

Does transfer learning help?
- SPY patterns transfer to NVDA? (both tech)
- SPY patterns transfer to IonQ? (less clear)

Provide: Transfer learning protocol, expected accuracy improvement, when it helps vs hurts.
```

### Q34: Options Data Integration (If Available)
```
If free options data becomes available (e.g., via Schwab API):

Which options signals matter most:
A) Put/call ratio (directional sentiment)
B) Implied volatility skew (risk aversion)
C) Open interest changes (institutional positioning)
D) Options gamma exposure (market maker hedging)

Provide: Top 3 options signals, feature engineering, expected accuracy improvement.
```

### Q35: Multi-Ticker Correlations
```
Should forecaster consider correlations between tickers:
- If NVDA forecast UP, should MSFT forecast also be UP (tech correlation)?
- Portfolio construction: Avoid 5 correlated AI longs (concentration risk)

Correlation adjustment strategies:
A) Penalize forecasts for correlated tickers
B) Adjust position sizes (reduce if correlated)
C) Ignore (treat each ticker independently)

Provide: Recommended approach, expected risk reduction, computational cost.
```

### Q36: Regime Transition Detection
```
Detecting regime changes in real-time (not just classifying current regime):

Leading indicators:
- VIX spike (volatility regime shift)
- Breadth breakdown (bull → bear transition)
- Yield curve inversion (recession risk)

How to detect transitions 1-3 days early:

Provide: Transition detection algorithm, validation on historical regime changes, expected lead time.
```

### Q37: Continuous Learning / Online Updates
```
Should model update after each trade:
- Win → boost signals that contributed
- Loss → reduce signals that contributed

Pros: Adapts quickly to changing market
Cons: Overfits to recent noise

Provide: Online learning protocol, when it helps (regime shifts) vs hurts (noise), safeguards (only update if N trades since last update).
```

---

## ✅ EXECUTION PLAN FOR TOMORROW (December 9)

### Morning Session (3-4 hours):
1. Open Perplexity Pro (unlimited queries)
2. Copy-paste questions in batches (Batch 1-4 critical)
3. Document answers in `PERPLEXITY_FORECASTER_ARCHITECTURE.md`
4. Extract implementation decisions (architecture choices, formulas, thresholds)

### Priority Order:
- **Batch 1** (Meta-Learner): Q1-Q6 (MUST ANSWER - core architecture)
- **Batch 2** (Features): Q7-Q12 (MUST ANSWER - feature selection)
- **Batch 3** (Training): Q13-Q18 (MUST ANSWER - avoid overfitting)
- **Batch 4** (Calibration): Q19-Q24 (MUST ANSWER - position sizing)
- **Batch 5** (Implementation): Q25-Q31 (nice to have)
- **Batch 6** (Advanced): Q32-Q37 (future optimization)

### Output Format:
```markdown
# Q1: Optimal Ensemble Method

**Answer:**
[Perplexity's full response]

**Implementation Decision:**
- Architecture: XGBoost (fastest, handles small samples)
- Hyperparameters: max_depth=3, n_estimators=100
- Expected improvement: +0.15 Sharpe vs weighted average
- Computational cost: <100ms per forecast

**Code Snippet:**
[Copy relevant code from Perplexity if provided]
```

**By end of Day 1, you'll have all architectural unknowns resolved. Then you build with confidence.**
