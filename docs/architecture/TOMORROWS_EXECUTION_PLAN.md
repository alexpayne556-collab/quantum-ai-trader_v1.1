# ⚡ TOMORROW'S EXECUTION PLAN: Perplexity Research + System Integration

**Timeline**: December 9, 2025 (Tomorrow)  
**Goal**: Gather Perplexity answers to all 25 questions, prepare for Week 1 implementation  
**Status**: All prep complete - ready to execute

---

## 🎯 MORNING SESSION (Perplexity Research) - 2-3 hours

### Step 1: Open Perplexity & Copy Questions
```
Go to: perplexity.ai
→ New chat
→ Copy-paste Batch 1 questions (Q1.1, Q1.2, Q1.3, Q3.1, Q3.2)
→ Wait for responses (~10 minutes)
→ Copy responses into PERPLEXITY_FORECASTER_ARCHITECTURE.md
```

### Step 2: Process Batch 1 (Forecaster Architecture)
**Questions to ask Perplexity:**

Q1.1 (5 min):
```
Given:
- Pattern detection system (Elliott Wave patterns, candlestick formations, support/resistance)
- Regime-aware recommender (ADX-based trending vs choppy states)
- 40-60 engineered features from free APIs (dark pool proxy, options skew, VWAP, breadth, BTC/yields)
- Pre-catalyst detection signals (24-48 hours ahead)

What is the optimal ensemble architecture (weighted average, Bayesian averaging, gradient boosting, 
or neural network meta-learner) that:
1. Combines 5-10 signals into a single confidence score (0-100%)
2. Learns optimal weights per regime (bull/bear/chop) and per sector (AI/quantum/robotaxi)
3. Calibrates confidence to actual win-rate (e.g., 70% confidence = 70% actual win-rate)
4. Handles regime switches without retraining
5. Prevents overfitting through regime-aware cross-validation

Expected output: Confidence score + position size recommendation
```

Q1.2 (5 min):
```
How should signal weights change dynamically based on:
- Current regime (BULL_LOW_VOL: favor patterns; EXTREME_VOL: favor catalysts; CHOP: favor microstructure)
- Sector (AI_INFRA: favor patterns; QUANTUM: favor catalysts; ROBOTAXI: favor microstructure)
- Time to event (far from catalyst: use patterns; near catalyst: use pre-catalyst signals)
- Signal agreement (all 5 agree: boost weight; only 1-2 agree: reduce weight)

Provide specific weight matrices (9×5×3) that maximize Sharpe ratio per sector.
```

Q1.3 (5 min):
```
How to build a confidence calibration curve that:
1. Learns mapping between predicted confidence and actual accuracy per regime
2. Adjusts confidence DOWN in low-accuracy regimes (HIGH_VOL)
3. Adjusts confidence UP in high-accuracy regimes (BULL_LOW_VOL)
4. Handles sector-specific differences (AI vs QUANTUM vs ROBOTAXI)
5. Updates dynamically without overfitting (only ~100 recent trades per regime)

Provide: Mathematical formulation + implementation approach (Platt scaling, isotonic regression, neural net)
```

Q3.1 (5 min):
```
For predicting next 5-15 day return DIRECTION (UP/CHOP/DOWN) + confidence score:
- Input: 40-50 features at close of day
- Output: Direction probability + Confidence (0-100%) + Optimal position size (0-5%) + Hold period (3-15d)

Which model architecture minimizes overfitting while maximizing predictive power (small sample: ~100 trades/regime)?
- Logistic Regression
- Random Forest / Gradient Boosting
- Neural Network (LSTM or transformer)
- Bayesian approach
- Ensemble

Provide implementation recommendations for each with pros/cons.
```

Q3.2 (5 min):
```
For trading forecasters with time-series data:
1. What is the correct time-series cross-validation methodology?
   - Walk-forward validation
   - Regime-based stratified CV
   - Other?

2. How to measure performance fairly across regimes?
   - Separate Sharpe per regime
   - Weighted Sharpe
   - Other metrics?

3. How many trades needed for 95% confidence in win-rate estimate?
   - Current: ~100 trades per regime per year
   - Sample size calculation?

Provide: Recommended CV strategy + implementation code
```

**After Batch 1**: You'll have answers on:
- Meta-learner architecture (weighted average vs Bayesian vs ensemble)
- Weight matrices by regime/sector
- Calibration approach (Platt scaling vs isotonic vs neural)
- Model selection (random forest vs gradient boosting)
- CV methodology (walk-forward + regime-aware)

---

### Step 3: Process Batch 2 (Integration & Features)
**Questions to ask Perplexity:**

Q2.1 (5 min):
```
Current system: 5-10 custom indicators (ADX, RSI, MACD, EMA crosses, etc)
Research provides: 40-60 additional features

How to integrate without:
1. Feature explosion / dimensionality curse
2. Multicollinearity
3. Overfitting to recent data
4. Breaking existing backtests
5. Adding computational complexity (need <1s per ticker)

Which 10-15 research features have HIGHEST predictive power AND lowest correlation with existing indicators?
Feature selection approach: PCA, correlation filtering, SHAP importance ranking?

Provide: Feature selection methodology + final 10-15 features recommended
```

Q2.2 (5 min):
```
Current: ADX-based regime (trending vs choppy, 2 states)
Research: 12-regime system (VIX × breadth × ATR = 4×3×3 combinations)

Should we REPLACE ADX or AUGMENT it?
If augment: How to weight ADX vs 12-regime in final forecaster?
What's the expected improvement from ensemble vs single regime?

Provide: 12-regime parameter table (barriers, EMAs, hold times, optimal signal weights, expected win-rates)
```

Q2.3 (5 min):
```
Research discovered:
- BTC returns lead tech stocks by 6-24 hours
- 10Y yield changes lead sector rotation by 3-5 days
- VIX structure (term spread) predicts volatility regime by 1-3 days
- Breadth divergences predict corrections by 2-7 days

How to:
1. Incorporate these as INPUT features to forecaster?
2. Adjust signal weight based on current cross-asset setup?
3. Create sector-specific cross-asset models:
   - AI_INFRA: 70% BTC correlation, 30% yield correlation
   - QUANTUM: 50% BTC correlation, 40% yield correlation, 10% cybersecurity
   - ROBOTAXI: 30% BTC, 70% yield correlation
4. Validate cross-asset leads without look-ahead bias?

Provide: Cross-asset feature matrix + sector-specific weights
```

Q4.1 (5 min):
```
Free data validation for trading signals from 9 APIs:
- yfinance, SEC EDGAR, FRED, CoinGecko, Reddit, Google Trends, Twitter, Finnhub

For each data source:
1. Known data quality issues and how to detect them?
2. Validation checks before using in forecaster?
3. How to handle missing data (forward fill? interpolate? skip?)?
4. How to validate historical free data matches production data?

Provide: Data validation checklist + quality score methodology
```

Q4.2 (5 min):
```
Features calculated on 2023-2025 data will they work on 2026-2027 data?

How to:
1. Detect when feature distributions shift significantly?
2. Trigger retraining when drift detected?
3. Maintain backward compatibility with old features?
4. Test features on out-of-sample future data?

Provide: Feature drift detection methodology + retraining schedule
```

Q4.3 (5 min):
```
Different sectors respond differently to same features:
- AI_INFRA: 0.82 Sharpe (patterns matter most)
- QUANTUM: 0.68 Sharpe (catalysts matter most)
- ROBOTAXI: 0.61 Sharpe (microstructure matters)

How to:
1. Calculate SHAP importance per sector?
2. Build sector-specific forecasters (separate model per sector) vs global?
3. Which approach has higher Sharpe?
4. Handle tickers spanning multiple sectors (MSFT is both AI and cloud)?

Provide: Sector-specific SHAP importance rankings + model structure recommendation
```

**After Batch 2**: You'll have answers on:
- Which 10-15 research features to use (avoid feature explosion)
- How to extend ADX to 12-regime (replace vs augment)
- Cross-asset feature matrix (BTC/yields/VIX weights by sector)
- Data quality validation checklist
- Feature drift detection methodology
- Sector-specific importance and model architecture

---

### Step 4: Process Batch 3 (Implementation & Validation)
**Questions to ask Perplexity:**

Q5.1 (5 min):
```
Forecaster training schedule options:
1. Daily: Update with new day's data, retrain on last 1-2 years
2. Weekly: Train every Monday on last 1-2 years
3. Monthly: Train at month-end
4. On-demand: Trigger only when regime shift detected

For each:
- Expected accuracy over 1-month horizon?
- How does it handle regime shifts?
- Computational cost?
- How to validate new model > old model?

Recommend: Optimal retraining schedule
```

Q5.2 (5 min):
```
Forecaster outputs: confidence (72%) + expected return (+5.2%) + optimal horizon (8 days)

Translate to:
1. Position size: Is 2.5% right? Formula: position_size = f(confidence, sharpe, regime, vol)?
2. Stop loss & take profit:
   - Pre-breakout signals → wide stops (±5-7%)?
   - Pre-catalyst signals → tight stops (±2-3%)?
   - Sector-specific (AI ±3%, quantum ±8%, robotaxi ±4%)?
3. Position scaling (pyramid in/out)
4. Risk per trade: Fixed 0.5% risk? Fractional Kelly? Full Kelly?

Provide: Position sizing formula + risk management framework
```

Q5.3 (5 min):
```
Backtesting protocol before deployment:

1. Historical backtesting (2021-2025):
   - Walk-forward validation (train 2021-2023, test 2024-2025)
   - Separate by regime + sector
   - Expected: Sharpe >0.5, win-rate 55%+, max drawdown <15%

2. Forward testing (paper trading, 2-4 weeks):
   - Compare predicted confidence vs actual win-rate
   - Adjust calibration if needed
   - Expected: confidence = actual win-rate ±5%

3. Live deployment:
   - Start 1% capital for 1 month
   - Monitor regime changes, drift, anomalies
   - Scale to 5% after month 1
   - Scale to 10%+ after quarter

Metrics to monitor? When to pause/reduce sizes?

Provide: Complete backtest + forward-test + live deployment protocol
```

Q6.1 (5 min):
```
Elliott Wave integration:
- Your pattern detector identifies impulse waves (5 waves) and corrective waves (ABC)

How to integrate with research forecaster:
1. When Elliott identifies 5-wave impulse: weight patterns MORE?
2. When Elliott identifies A-B-C correction: weight catalysts/microstructure MORE?
3. Combine Elliott Wave probability with research confidence?
4. Update adaptive horizon?
   - Wave 3 of impulse: extend horizon (riding trend)
   - Wave 5: shorten horizon (expect reversal)

Provide: Integration rules (weight adjustment matrices, horizon adjustment rules)
```

Q6.2 (5 min):
```
ADX regime combination:
Current: ADX-based (trending vs choppy, 2 states)
Research: 12-regime system (VIX × breadth × ATR = 4×3×3)

Should we:
1. REPLACE ADX with 12-regime?
2. AUGMENT (ensemble both)?

If augment:
- How to weight ADX vs 12-regime?
- What's improvement from ensemble vs single?
- Expected accuracy gain?

Provide: Combination approach + accuracy improvement estimate
```

Q6.3 (5 min):
```
Meta-learner evolution:
Current: weights(pattern_signal) + weights(ADX_regime) + other signals
Need to add: research_forecaster (new 40-60 features)

How should meta-learner:
1. Weight research forecaster relative to existing signals?
2. Be structured: fixed weights vs adaptive vs hierarchical?
3. Learn optimal weights from historical data?

Provide: Updated meta-learner architecture + weight assignment methodology
```

Q7.1 (5 min):
```
Multiday predictions (5-15 day horizon):

Approaches:
1. Rolling window: "Given today's features, predict next 5 days" (expensive)
2. Composite return: "Predict 5-day return %" (efficient)
3. Regime-adjusted: Different models per regime?

Which is most effective for trading?

Provide: Recommended approach + validation methodology
```

Q7.2 (5 min):
```
Class imbalance in training data:
Most days are CHOP (40%), fewer UP (35%) or DOWN (25%)

Standard model predicts CHOP for everything, high accuracy but doesn't make money.

How to handle:
1. Class weighting (give more weight to UP/DOWN)
2. Threshold adjustment (predict UP if confidence > 65% not 50%)
3. Separate model per class
4. Oversampling (SMOTE, generate synthetic samples)
5. Custom loss function

Which works best for trading?

Provide: Recommended class imbalance strategy + implementation
```

Q8.1 (5 min):
```
Realistic accuracy targets by horizon:
- 1-day: 52-55% baseline
- 5-day: 48-52% (compounding errors)
- 15-day: 45-50% (even harder)

With regime awareness + catalysts:
- Trending + clear catalyst: 60-70%?
- Choppy + no catalyst: 45-50%?

For each sector:
- AI: 55-60%?
- Quantum: 50-55%?
- Robotaxi: 48-53%?

Minimum accuracy to be profitable? (position 2.5%, Sharpe >0.5 needs 54%+ win-rate)

How to improve 50% → 55%?

Provide: Realistic accuracy targets + breakdown by factor contribution
```

Q8.2 (5 min):
```
Sharpe ratio estimation:
- 55% win-rate, position 2.5%, stops -2.5%, targets +7.5%
- Average win: +7.5%, average loss: -2.5%
- Expected Sharpe: 0.6?

How to:
1. Calculate Sharpe per sector from historical data?
2. Estimate from win-rate + risk/reward?
3. How does regime affect Sharpe?
4. What Sharpe needed to justify effort?
   - 1% time: Sharpe >0.3
   - 5% time: Sharpe >0.5
   - 10% time: Sharpe >0.7

Target Sharpe per sector?

Provide: Sharpe calculation methodology + sector-specific targets
```

Q9.1 (5 min):
```
Market structure changes since 2023:
- More passive investing (ETF flows)
- More algos (retail bots, hedge fund algos)
- Regulatory changes (SEC dark pool rules)
- Venue changes (SPACs, direct listings, crypto)

How have these affected:
1. Dark pool activity (volume clustering still valid)?
2. Options market (skew patterns same)?
3. Breadth patterns (still lead sector rotation)?
4. News diffusion (Reddit/Twitter still lead professional media)?

Assessment: Signal strength change, new signals, deteriorated signals?

Provide: Market structure assessment + signal validity update
```

Q9.2 (5 min):
```
Emerging free data sources not covered:

1. Blockchain data (whale movements, exchange flows)
2. Alternative data (satellite imagery, job postings, patent filings, supply chain)
3. Sentiment APIs (newspapers, earnings transcripts, LinkedIn)
4. Regulatory data (USPTO, FDA, FCC)

For each: predictive power, lead time, cost, reliability

Top 3 emerging sources ranked by predictive power?

Provide: Emerging data sources assessment
```

Q9.3 (5 min):
```
Trading ensemble architecture:

Need to combine 5 subsystems:
1. Pattern detection (Elliott, candlesticks)
2. Regime detection (ADX, 12-regime)
3. Research forecaster (40-60 features)
4. Cross-asset sentiment (BTC, yields, VIX)
5. News/catalysts (options, insiders)

Ensemble approaches:
1. Simple weighted average
2. Stacked meta-learner
3. Bayesian ensemble
4. Hierarchical ensemble
5. Boosting/bagging

For trading specifically, which has best Sharpe ratio per subsystem?

Provide: Recommendation + accuracy vs complexity trade-offs
```

**After Batch 3**: You'll have answers on:
- Retraining schedule (daily/weekly/monthly/on-demand)
- Position sizing formula (confidence + Sharpe + regime)
- Backtest/forward-test/live protocol
- Elliott Wave integration rules
- ADX + 12-regime combination approach
- Meta-learner evolution strategy
- Multiday prediction approach
- Class imbalance handling
- Realistic accuracy targets (55-60% for integrated)
- Sharpe ratio estimation (>0.5 target)
- Confidence calibration evaluation
- Market structure changes assessment
- Emerging data sources
- Ensemble architecture recommendation

---

## 📝 AFTERNOON SESSION (Documentation) - 1 hour

After Perplexity research completes:

### Step 1: Create PERPLEXITY_FORECASTER_ARCHITECTURE.md
```
Save all Perplexity answers into new markdown file:

PERPLEXITY_FORECASTER_ARCHITECTURE.md
├── Batch 1 Answers (Meta-learner design)
├── Batch 2 Answers (Feature selection + integration)
├── Batch 3 Answers (Implementation + validation)
└── Key Takeaways for Week 1 Implementation

This becomes the specification document for week 1 development.
```

### Step 2: Map Answers to Implementation Tasks
```
For each answer, create 1-2 action items:

Example - Q1.1 (Meta-learner architecture):
Answer: "Gradient boosting with 3 learners: pattern model, regime model, research model"
Action: Week 1 task: "Implement GradientBoostingClassifier with 3-learner ensemble"

Example - Q2.1 (Feature selection):
Answer: "Use SHAP importance ranking, keep top 10 features, drop >0.7 correlation"
Action: Week 1 task: "Run SHAP analysis on 40-60 features, select top 10-15 non-correlated"
```

### Step 3: Create WEEK1_IMPLEMENTATION_TASKS.md
```
Based on Perplexity answers, break into actionable tasks:

Week 1 Implementation Checklist:
- [ ] Build research_features.py (implement 10-15 selected features)
- [ ] Test on 100-ticker universe (validate <1s per ticker)
- [ ] Build integrated_meta_learner.py (gradient boosting + pattern + regime + research)
- [ ] Train on 2021-2023 data (walk-forward CV)
- [ ] Test on 2024-2025 data (out-of-sample)
- [ ] Build confidence_calibrator.py (Platt scaling per regime)
- [ ] Backtest: compare forecaster output vs market returns
- [ ] Validate: 55%+ accuracy, Sharpe >0.5

Each task has dependencies (e.g., can't build meta-learner before research_features).
Order tasks by dependency.
```

---

## 🎯 END OF DAY CHECKLIST

By end of tomorrow (EOD):

- [ ] Asked all 25 questions to Perplexity
- [ ] Got comprehensive answers on:
  - [ ] Meta-learner architecture (1.1, 1.2, 1.3)
  - [ ] Feature selection (2.1, 4.3)
  - [ ] Regime combination (2.2, 6.2)
  - [ ] Cross-asset integration (2.3)
  - [ ] Data validation (4.1, 4.2)
  - [ ] Model training (3.1, 3.2)
  - [ ] Retraining schedule (5.1)
  - [ ] Position sizing (5.2)
  - [ ] Backtesting (5.3)
  - [ ] Integration rules (6.1, 6.3)
  - [ ] Multiday predictions (7.1)
  - [ ] Class imbalance (7.2)
  - [ ] Accuracy targets (8.1, 8.2)
  - [ ] Market assessment (9.1, 9.2, 9.3)
- [ ] Created PERPLEXITY_FORECASTER_ARCHITECTURE.md (answers)
- [ ] Created WEEK1_IMPLEMENTATION_TASKS.md (action items)
- [ ] Ready to start Week 1 development

---

## 📚 Reference Documents Available

When asking Perplexity, you can reference:

1. **INTEGRATION_PLAN_RESEARCH_TO_FORECASTER.md** (Cells 1-6 Colab code)
2. **DISCOVERY_LAYERS_2_THROUGH_4_COMPLETE.md** (12-regime system, 40-60 features)
3. **COMPLETE_100_TICKER_UNIVERSE.md** (100+ tickers, liquidity filters)
4. **DISCOVERY_LAYERS_5_THROUGH_9_COMPLETE.md** (pre-catalyst, news, deployment)
5. **SYSTEM_ARCHITECTURE_MAPPING.md** (current system + what to build)

If Perplexity asks for examples, cite these documents. Perplexity can see your workspace files.

---

## 🚀 Week 1 Development (After Perplexity Research)

Once you have Perplexity answers:

**Monday (Day 1-2)**:
- Build research_features.py (10-15 features)
- Test on 100-ticker universe
- Validate calculations

**Tuesday (Day 3)**:
- Build integrated_meta_learner.py (based on Perplexity recommendation)
- Implement weight matrices (by regime, by sector)
- Train on 2021-2023 data

**Wednesday (Day 4)**:
- Build confidence_calibrator.py (Platt scaling or Perplexity recommendation)
- Create calibration curves per regime
- Validate confidence = actual win-rate

**Thursday (Day 5)**:
- Backtest integrated forecaster (2021-2025 walk-forward)
- Calculate Sharpe ratio per sector
- Compare vs baseline (pattern + regime only)
- Target: 55-65% accuracy, Sharpe >0.5

**Friday (Day 6-7)**:
- Paper trade on live data
- Track predicted confidence vs actual accuracy
- Adjust calibration if needed
- Prepare for live deployment

---

## ✅ Success Criteria for Tomorrow

- [ ] All 25 Perplexity questions answered thoroughly
- [ ] Answers provide specific implementation guidance (not just theory)
- [ ] Answers address your specific constraints (free APIs, 100+ tickers, existing system)
- [ ] Clear technical recommendations (which model, which features, which approach)
- [ ] Ready to hand off to Week 1 development team (or yourself)

---

## 📞 If You Get Stuck

**Common issues tomorrow:**

1. **Perplexity gives vague answer**: Ask follow-up with specific example
   ```
   Example: "How should meta-learner combine 5 signals"
   If vague: "Specifically, if pattern says BUY (60% confidence), regime says TREND_UP (80%), 
   research says BREAKOUT_SCORE 4/5, how to combine into single confidence score?"
   ```

2. **Perplexity doesn't know your system**: Provide context
   ```
   "I have existing pattern detector (Elliott Wave) and ADX-based regime detector. 
   I want to add research features (40-60). How to integrate without replacing existing?"
   ```

3. **Answer too theoretical**: Ask for implementation details
   ```
   If Perplexity says "use stacked meta-learner":
   "Can you show Python pseudocode for training this on trading data with regime-aware CV?"
   ```

4. **Conflicting answers**: Ask Perplexity to reconcile
   ```
   "Earlier you said use random forest, now you say gradient boosting. 
   For trading with ~100 trades/regime, which is better and why?"
   ```

---

**You're ready for tomorrow! Let's go build this integrated forecaster. 🚀**

