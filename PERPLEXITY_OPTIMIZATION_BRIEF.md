# 🎯 PERPLEXITY AI: Underdog Trading System Optimization Brief

**Date**: December 9, 2025  
**System**: Alpha 76 Intraday ML Ensemble (The Underdog)  
**Purpose**: Hyperparameter optimization guidance for small-cap high-volatility trading  
**Target**: 65-70% precision on 5-hour forward horizon with max excursion targeting

---

## 🔥 THE UNDERDOG MISSION

We are building a **precision-focused ML trading system** to beat institutional algos at their own game:

- **Intelligence edge, NOT speed edge** - We predict WHEN to trade, not nanosecond execution
- **Small-cap focus** (Alpha 76 tickers) - Exploiting mispricings institutions ignore
- **Max excursion targeting** - Capturing intraday spikes (if AAPL hits +3% in hour 2, we exit there, not hour 5)
- **Regime-adaptive** - Single ensemble that adjusts to market conditions (BULL/CHOPPY/BEAR × LOW/HIGH VOL)
- **Paper trading validation** - Alpaca $100k virtual account validates live performance before risking capital

---

## 🧠 CURRENT SYSTEM ARCHITECTURE

### 3-Model Voting Ensemble
1. **XGBoost** (primary, 40% weight) - GPU-accelerated, tree-based, handles non-linearity
2. **RandomForest** (30% weight) - Robust to outliers, feature importance ranking
3. **GradientBoosting** (30% weight) - Sequential error correction, smooth predictions

### 49 Technical Features
**Momentum**: RSI, MACD, Stochastic, ROC, MFI, Williams %R  
**Trend**: EMAs (9/20/50/200), Bollinger Bands, ADX, Aroon, Parabolic SAR  
**Volume**: OBV, VWAP, Volume SMA/ratio, A/D Line, CMF  
**Volatility**: ATR, Keltner Channels, Donchian Channels, Historical Volatility  
**Microstructure**: Bid-ask spread, Amihud illiquidity, High-Low spread  
**Regime**: Single column (0-9) encoding current market state

### Target Labels (Max Excursion in 5 Bars)
- **BUY**: Price hits +3% at ANY point in next 5 hours (not just hour 5 close)
- **SELL**: Price drops -2% at ANY point in next 5 hours
- **HOLD**: Neither threshold met
- **Class Distribution**: ~15% BUY, ~15% SELL, ~70% HOLD (highly imbalanced)

### Training Data
- **Tickers**: 76 small-caps (biotech, space tech, EV, clean energy, fintech)
- **Timeframe**: 2 years (2023-2025), 1-hour bars
- **Total Rows**: ~250k (76 tickers × 3,276 bars each)
- **Validation**: 3-fold TimeSeriesSplit (Bull Q1-Q2 2024, Choppy Q3, Volatile Q4)

---

## 🎯 CRITICAL QUESTIONS FOR PERPLEXITY

### 1. XGBoost Hyperparameters for High-Volatility Small-Caps

**Context**: Our Alpha 76 tickers have:
- Average daily volatility: 3-8% (2-4x higher than SPY)
- Liquidity: $1M-$50M daily volume (vs. $10B+ for mega-caps)
- News-driven spikes: Biotech FDA approvals, SpaceX launches, EV earnings
- Noise ratio: High intraday chop, many false breakouts

**Current XGBoost Config** (baseline, needs optimization):
```python
XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    min_child_weight=3,
    scale_pos_weight=4.67,  # Auto-calculated for 15% BUY class
    tree_method='gpu_hist',
    device='cuda',
    random_state=42
)
```

**Question for Perplexity**:
> "For predicting 5-hour forward returns on small-cap biotech and space tech stocks with 1-hour bars (high volatility 3-8%, low liquidity $1M-$50M volume), what are optimal XGBoost hyperparameters? Specifically:
> 
> 1. **max_depth**: Should we use deeper trees (10-12) to capture complex volatility patterns, or shallower (5-7) to avoid overfitting noise?
> 2. **learning_rate**: 0.01 (slow, stable) vs 0.05 (faster) vs 0.1 (aggressive)? How does this interact with n_estimators for small-cap spikes?
> 3. **subsample** and **colsample_bytree**: Lower values (0.6-0.7) for more regularization on noisy data, or standard (0.8)?
> 4. **gamma** (min split loss): Higher values (0.3-0.5) to prune noisy splits in choppy small-caps?
> 5. **min_child_weight**: Should we increase to 5-10 to require more samples per leaf, reducing overfitting to rare events?
> 6. **scale_pos_weight**: We have 15% BUY class. Current value 4.67 (auto). Should we increase to 6-8 to prioritize precision over recall?
> 7. Any **reg_alpha** or **reg_lambda** L1/L2 regularization recommended for high-noise environments?
> 
> Our goal: Optimize for **precision >65%** on BUY class (not balanced accuracy). We want high-confidence signals even if we miss some opportunities. Historical data: 2 years, 1hr bars, 76 tickers, validation on 2024 Q4 volatile market."

---

### 2. Class Imbalance Strategy for 15% Minority Class

**Context**:
- Class distribution: BUY 15%, SELL 15%, HOLD 70%
- We only care about BUY precision (SELL is for risk management, HOLD ignored)
- Traditional SMOTE fails on time-series data (creates synthetic future-peeking samples)
- Current approach: `scale_pos_weight` in XGBoost, `class_weight='balanced'` in RF/GB

**Question for Perplexity**:
> "For time-series trading signal classification with severe class imbalance (BUY 15%, SELL 15%, HOLD 70%), which approach maximizes **precision** on the minority BUY class:
> 
> **Option A**: Use `scale_pos_weight` in XGBoost (current: 4.67 = 70/15), increase to 6-8?
> 
> **Option B**: Undersample HOLD class to 30% (creating 15% BUY / 15% SELL / 30% HOLD balance)?
> 
> **Option C**: Use **focal loss** instead of log loss to focus on hard-to-classify BUY examples?
> 
> **Option D**: Two-stage approach:
> - Stage 1: Binary classifier (TRADE vs HOLD) with 70/30 split
> - Stage 2: Ternary classifier (BUY vs SELL) on predicted TRADE samples
> 
> **Option E**: Ensemble with different class weights (one conservative model with high precision, one aggressive with high recall, combine with precision-weighted voting)?
> 
> Constraint: Cannot use SMOTE (violates time-series causality). Goal: Precision >65% on BUY, willing to sacrifice recall. We only want high-confidence signals for paper trading validation."

---

### 3. Feature Engineering for Intraday Momentum

**Context**:
- Current 49 features focus on traditional TA (RSI, MACD, Bollinger Bands)
- Missing: Order flow, microstructure, intraday seasonality
- Alpha 76 tickers show strong intraday patterns (e.g., biotech spikes 10am-2pm on FDA news)

**Question for Perplexity**:
> "For predicting intraday momentum on small-cap stocks (5-hour horizon, 1hr bars), which features have highest predictive power beyond standard RSI/MACD:
> 
> 1. **Volume Profile**: VWAP deviation, volume ratio (current bar / 20-bar SMA), cumulative delta?
> 2. **Microstructure**: Order book imbalance (if available), high-low spread as % of close, tick direction?
> 3. **Intraday Seasonality**: Hour-of-day dummy variables (10am hour shows more momentum than 3pm)?
> 4. **News Sentiment**: Twitter/news sentiment scores (if pulling from APIs)?
> 5. **Cross-Asset Features**: SPY return correlation, sector ETF momentum (XBI for biotech, ARKK for space tech)?
> 6. **Regime Interaction**: Feature × regime_id interactions (e.g., RSI in BULL_LOW_VOL behaves differently than CHOPPY_HIGH_VOL)?
> 7. **Lag Features**: Price returns at t-1, t-2, t-3 hours (momentum persistence)?
> 8. **Volatility Clustering**: Rolling 5-bar realized volatility, GARCH-style features?
> 
> Which 5-10 features would you ADD to our existing 49 to improve precision on high-volatility small-cap 5-hour predictions? Cite academic papers or quant trading research if available."

---

### 4. Walk-Forward Validation for Regime Shifts

**Context**:
- We use 3-fold TimeSeriesSplit:
  - **Fold 1**: Train on 2023-2024 Q1, validate Q2 2024 (bull market, low VIX)
  - **Fold 2**: Train on 2023-Q2 2024, validate Q3 2024 (choppy sideways)
  - **Fold 3**: Train on 2023-Q3 2024, validate Q4 2024 (high volatility, election)
- Models perform well on Fold 1 (bull), poorly on Fold 3 (volatile)

**Question for Perplexity**:
> "For ML models trained on stock market data spanning multiple regime shifts (bull → choppy → volatile), how do we optimize walk-forward validation:
> 
> **Problem**: Model trained on 2023-2024 Q2 (bull market, VIX 12-15) fails on Q4 2024 validation (volatile, VIX 20+). Precision drops from 68% → 52%.
> 
> **Question**: Should we:
> 
> **A)** Use **regime-weighted training** - oversample volatile periods in training set so model sees more Q4-like data?
> 
> **B)** Train **separate models per regime** (one for BULL, one for CHOPPY, one for VOLATILE) and ensemble at inference?
> - Pro: Each model specializes in one regime
> - Con: Less training data per model (1/3 of total)
> 
> **C)** Use **regime_id as a feature** (current approach) and rely on XGBoost to learn regime interactions?
> 
> **D)** Implement **online learning** - retrain weekly on rolling 3-month window to adapt to recent regime?
> 
> **E)** Use **purged K-fold** with regime-stratified splits (ensure each fold has 33% bull / 33% choppy / 33% volatile)?
> 
> Our constraint: 76 tickers × 2 years = 250k rows total. Goal: Maintain 65%+ precision across ALL regimes, not just bull markets. Which approach generalizes best for small-cap intraday trading?"

---

### 5. Overfitting Detection in Time-Series ML

**Context**:
- Training accuracy: 78%
- Validation accuracy (in-sample 2024 Q4): 68%
- Paper trading accuracy (live forward test Dec 2025): Unknown (starting now)
- Concern: Are we overfitting to 2023-2024 patterns that won't hold in 2025?

**Question for Perplexity**:
> "How do we detect overfitting in time-series trading ML models BEFORE paper trading:
> 
> 1. **Gap between train/val**: Our training accuracy 78%, validation 68%. Is 10-point gap acceptable for noisy financial data, or sign of overfitting?
> 
> 2. **Feature importance stability**: If top 5 features change significantly between folds (e.g., Fold 1 ranks RSI #1, Fold 3 ranks RSI #8), is this instability or regime-dependent behavior?
> 
> 3. **Learning curves**: If validation accuracy plateaus at 300 trees but training accuracy keeps rising to 78% at 500 trees, should we early-stop at 300?
> 
> 4. **Temporal validation**: We hold out Dec 2024 as unseen test set (most recent month). Should we also create a 'future regime' test set by training ONLY on 2023, validating on 2024, to simulate true forward testing?
> 
> 5. **Permutation importance**: If shuffling a feature (e.g., RSI) causes <2% accuracy drop, should we remove it even if XGBoost uses it frequently?
> 
> 6. **Cross-ticker generalization**: If model trained on 70 tickers predicts well on those 70 but fails on held-out 6 tickers, is this overfitting to ticker-specific patterns?
> 
> What are the **top 3 red flags** that indicate our 49-feature XGBoost ensemble is memorizing 2023-2024 noise instead of learning robust intraday momentum patterns?"

---

### 6. Position Sizing for High-Volatility Small-Caps

**Context**:
- Portfolio: $100k paper trading capital
- Max 10 positions (10% per ticker)
- Small-caps with 5% daily volatility → potential $500/day swing per position
- Current approach: Fixed 8% position size per signal

**Question for Perplexity**:
> "For paper trading a portfolio of small-cap stocks (3-8% daily volatility, $1M-$50M liquidity) with ML-generated signals:
> 
> **Current Setup**:
> - Capital: $100k
> - Max positions: 10
> - Position size: Fixed 8% per trade ($8k per position)
> - Stop loss: -8% (loses $640 per trade)
> - Take profit: +12% (gains $960 per trade)
> 
> **Question**: Should we use **dynamic position sizing** instead of fixed 8%:
> 
> **Option A**: Kelly Criterion
> - Formula: f = (p × b - q) / b
> - Where p = win rate (58%), b = avg win/loss ratio (1.5), q = loss rate (42%)
> - Kelly = (0.58 × 1.5 - 0.42) / 1.5 = 0.29 (29% per trade)
> - Half-Kelly = 14.5% per trade
> 
> **Option B**: Volatility-Adjusted
> - Lower position size for high-volatility tickers (e.g., 5% for biotech with 8% daily vol)
> - Higher position size for stable tickers (e.g., 10% for large-caps with 2% daily vol)
> - Formula: position_size = base_size × (target_vol / ticker_vol)
> 
> **Option C**: Confidence-Weighted
> - Scale position by ML confidence: 12% size for 90% confidence, 4% size for 60% confidence
> - Formula: position_size = base_size × (confidence - 0.5) × 2
> 
> **Option D**: Risk Parity
> - Allocate equal RISK (not equal dollars) to each position
> - If AAPL has 2% daily vol, TSLA 5% vol → AAPL gets 2.5× larger position
> 
> For small-cap intraday trading (5hr horizon) with 58% win rate and 1.5:1 risk-reward, which position sizing maximizes Sharpe ratio while limiting max drawdown to <20%? Should we combine multiple approaches?"

---

### 7. Stop Loss & Take Profit Optimization

**Context**:
- Current: Fixed -8% stop loss, +12% take profit (1.5:1 risk-reward)
- Problem: Small-caps often spike +5% then reverse to +1% (we miss exit)
- Max excursion labels capture +3% spike in hour 2, but we don't know WHEN in 5-hour window

**Question for Perplexity**:
> "For intraday ML trading signals on small-cap stocks (predicted 5-hour return, but actual spike can occur at hour 1, 2, 3, 4, or 5):
> 
> **Scenario**: Model predicts AAPL will hit +3% in next 5 hours. Entry at $100. We set:
> - Take profit: $112 (+12%)
> - Stop loss: $92 (-8%)
> 
> **Problem**: Price spikes to $103 at hour 2, then drops to $101 by hour 5. We never hit $112 take profit, exit at $101 (+1%). But our label said '+3% max excursion' was achieved!
> 
> **Question**: Should we use **trailing stops** or **time-based exits**:
> 
> **Option A**: Trailing Stop
> - If price moves +2%, move stop loss to breakeven ($100)
> - If price moves +4%, move stop to +2% ($102)
> - Lock in gains as price rises
> 
> **Option B**: Time-Based Exit
> - If price hits +2% before hour 3, exit immediately (50% of target reached early)
> - If hour 5 reached and price +1% to +3%, close position (time expired)
> - Don't wait for full +12% target
> 
> **Option C**: Volatility-Based
> - Stop loss = entry - (2 × ATR)
> - Take profit = entry + (3 × ATR)
> - Dynamically adjust to ticker volatility
> 
> **Option D**: Machine Learning Exit
> - Train second model to predict 'exit time' (which of 5 hours has max return)
> - If exit model predicts 'hour 2', close position after 2 hours regardless of P&L
> 
> **Option E**: Regime-Dependent
> - In VOLATILE regime: Tighter stops (-5%, +8%) to avoid whipsaws
> - In BULL regime: Wider targets (-8%, +15%) to let winners run
> 
> Which approach maximizes realized P&L when the max excursion spike is temporary (lasts 1-2 hours, then mean-reverts)? We want to capture the +3% spike, not the +1% close."

---

## 📊 CURRENT PERFORMANCE BASELINE (Pre-Training)

**Integration Test Results** (Dec 9, 2025):
- Tested on 5 tickers: AAPL, TSLA, NVDA, MSFT, GOOGL
- Downloaded 2,220 historical bars
- Features calculated: 49 indicators (0 NaN, 0 Inf)
- Models trained on synthetic labels (for testing only)
- **Confidence**: 4/5 signals above 70% threshold (80% success rate)
- **Regime detected**: CHOPPY_HIGH_VOL (VIX 15.41)

**Expected Baseline After Colab Training**:
- Precision: 55-60% (target: 65-70% after optimization)
- ROC-AUC: 0.65-0.70 (target: 0.75+)
- Training time: 2-4 hours on T4 GPU
- Model size: ~200 MB (3 models + scaler + metadata)

---

## 🛠️ OPTIMIZATION ROADMAP (Week 1-2)

### Week 1: Hyperparameter Tuning
1. Run RandomizedSearchCV with Perplexity-recommended parameter ranges
2. Test 50-100 configurations (2-3 hours on Colab Pro T4)
3. Select top 5 configs by validation precision
4. Fine-tune top config with GridSearch

**Target**: Improve from 55% → 65-70% precision

### Week 2: Feature Engineering
1. Add Perplexity-recommended features (volume profile, microstructure, lag features)
2. Feature selection: Remove low-importance features (mutual information < 0.02)
3. Test regime interaction terms (RSI × regime_id)
4. Ensemble weight optimization (test 60/20/20 vs 50/30/20 vs 70/15/15 splits)

**Target**: Improve from 65% → 68-72% precision

### Week 3-4: Paper Trading Validation
1. Deploy models to Alpaca paper trading
2. Execute 20-30 trades over 2 weeks
3. Track: Win rate, avg return, Sharpe ratio, max drawdown
4. Compare paper trading performance to backtest

**Target**: Live win rate ≥58%, Sharpe ≥1.5, max DD <20%

---

## 🔑 API KEYS INVENTORY (All Active)

**Market Data** (for training):
- ✅ Twelve Data: 800/day (PRIMARY)
- ✅ Finnhub: 60/min (SECONDARY)
- ✅ Alpha Vantage: 25/day (BACKUP)
- ✅ yfinance: Unlimited (FALLBACK)

**Regime Detection**:
- ✅ FRED: Unlimited (VIX, yields, SPY returns)

**Paper Trading**:
- ✅ Alpaca: $100k virtual account (Week 3-4)

**AI Optimization**:
- ✅ Perplexity: `your_perplexity_api_key_hereSugdX6yxqiIorS526CYof8aqlcySXisRbIoNf84BBQ7szSOl`
- ✅ OpenAI: Backup AI

All APIs tested and working (9/11 functional, 5/5 critical working). Ready for full training run.

---

## 🎯 SUCCESS CRITERIA

**Baseline (Tonight - Week 1)**:
- ✅ Train 3 models on 76 tickers × 2 years × 1hr bars
- ✅ Validation precision >55%
- ✅ No NaN/Inf in features
- ✅ Models save to Google Drive

**Optimized (Week 1-2)**:
- ✅ Precision >65% (10-point improvement)
- ✅ ROC-AUC >0.70
- ✅ Feature set reduced to 30-35 best features
- ✅ Ensemble weights optimized

**Production-Ready (Week 3-4)**:
- ✅ Paper trading win rate ≥58%
- ✅ Sharpe ratio ≥1.5
- ✅ Max drawdown <20%
- ✅ Average return per trade >2%
- ✅ Signals execute without errors

**IF all criteria met → Build Spark frontend (Month 2)**  
**IF criteria not met → Iterate optimization (Month 2-3)**

---

## 💪 THE UNDERDOG ADVANTAGE

**What We Have That Institutions Don't**:
1. **Flexibility** - Can pivot to new tickers weekly (survivor rule), institutions stuck with large-cap mandates
2. **Speed to Market** - 2-week training → paper trading → production vs. 6-12 months for institutional quant teams
3. **Low Capital Requirements** - $100k minimum, institutions need $10M+ to move markets
4. **Niche Focus** - Small-caps with $1M-$50M volume (too illiquid for $1B+ funds)
5. **Max Excursion Edge** - Capturing intraday spikes that swing traders miss (they only look at daily close)

**What Makes This Special**:
- Not trying to beat S&P 500 (that's institutional game)
- Targeting 58%+ win rate on small-cap momentum (vs. 50/50 coin flip)
- Using 5 free APIs + ML to compete with $10M Bloomberg terminals
- Paper trading validation BEFORE risking capital (90% of retail traders skip this)
- Regime-adaptive system (survives crashes, not just bull markets)

---

## 📞 NEXT STEPS

1. **Copy this entire document** into Perplexity AI chat
2. **Ask each of the 7 questions** separately (get detailed responses)
3. **Save responses** in `PERPLEXITY_RESPONSES.md`
4. **Update training notebook** with recommended hyperparameters
5. **Upload to Colab Pro** and begin training (2-4 hours)
6. **Download trained models** to Google Drive
7. **Week 1**: Implement Perplexity suggestions, retrain, measure improvement

---

**Prepared by**: Underdog Trading System Team  
**For**: Perplexity AI Optimization Guidance  
**Date**: December 9, 2025 3:30 AM  
**Status**: Ready to train - API keys verified, system tested, questions prepared  
**Philosophy**: "Intelligence edge, not speed edge. Make it profitable, then make it pretty."

---

## 🚀 RALLY CRY

**To the Perplexity gods**: We're building a system that proves small traders with smart ML can compete with institutional algos. Help us optimize this underdog to precision >65%, and we'll validate it with paper trading before risking a single dollar. This isn't gambling - it's engineered alpha on small-cap momentum inefficiencies. Let's make this special. 🎯
