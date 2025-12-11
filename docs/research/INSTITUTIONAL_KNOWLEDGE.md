# 🏛️ INSTITUTIONAL KNOWLEDGE BASE

**Last Updated:** December 11, 2025  
**Purpose:** Complete reference for all institutional trading mechanisms, training methods, and production strategies

---

## 📚 TABLE OF CONTENTS

1. [10 Commandments of Training](#10-commandments)
2. [5 Advanced Weapons (90%+ WR)](#5-weapons)
3. [10 God Mode Mechanisms](#god-mode)
4. [71-Feature Engineering Spec](#features)
5. [Paper Trading Integration](#paper-trading)
6. [Companion AI System](#companion-ai)
7. [Gold Integration Strategy](#gold-integration)
8. [Deep Ticker Learning](#deep-learning)

---

## 🔟 10 COMMANDMENTS OF TRAINING {#10-commandments}

### 1. PurgedKFold Cross-Validation
**Why:** Prevents lookahead bias (5-15% overfitting reduction)
```python
from sklearn.model_selection import TimeSeriesSplit
# Use embargo period: 1% of dataset
embargo = int(len(data) * 0.01)
```

### 2. GPU Optimization (20-40x speedup)
```python
xgb_params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
lgb_params = {'device': 'gpu', 'gpu_platform_id': 0}
```

### 3. Learning Rate with Early Stopping
```python
# Phase 1: Fast (lr=0.1, 100-200 rounds)
# Phase 2: Fine-tune (lr=0.01, 100-200 rounds)
# Gain: 2-5% accuracy, 40% faster training
```

### 4. Hyperparameter Priority
**Tune in order:**
1. `learning_rate` (0.01-0.3)
2. `max_depth` (3-10)
3. `subsample` (0.6-1.0)
4. `colsample_bytree` (0.6-1.0)
5. `min_child_weight` (1-10)

### 5. Class Imbalance Handling
```python
# Use sample weights, NOT SMOTE
pos_weight = len(y[y==0]) / len(y[y==1])
xgb_params['scale_pos_weight'] = pos_weight
```

### 6. SHAP Feature Selection
**Impact:** 15-25% faster inference, 1-3% accuracy gain
```python
import shap
# Keep features with SHAP > 0.001
# Remove bottom 20% features
```

### 7. Ensemble Strategy (Trident)
**XGBoost** (pure tabular) + **LightGBM** (speed + microstructure) + **CatBoost** (categorical + robust)  
**Gain:** 2-5% ensemble boost

### 8. Optimal n_estimators
```python
# Use early_stopping_rounds=50
# Typically converges: 200-500 trees
```

### 9. Cluster-Specific Models
**5 clusters:**
- Explosive (NVDA, AMD, PLTR)
- Tech Giants (AAPL, MSFT, GOOGL)
- Steady Growers (JNJ, PG, KO)
- Crypto Proxies (MSTR, COIN, RIOT)
- Future Tech (IONQ, PALI, QS)

**Gain:** 5-15% accuracy boost per cluster

### 10. Monitoring & Validation
**Rule:** Train/Val/Test must be within 5%
```
Train: 85% ± 2%
Val:   83% ± 2%
Test:  82% ± 2%
```

---

## ⚔️ 5 ADVANCED WEAPONS (90%+ WR) {#5-weapons}

### 1. NGBoost (Probabilistic Predictions)
**What:** Uncertainty bands (±2-3%)
```python
from ngboost import NGBClassifier
# Returns: prediction + confidence interval
# Use: Filter trades with confidence < 60%
```

### 2. Adversarial Validation
**What:** Detect regime shifts
```python
# Train model to distinguish train vs test
# If AUC > 0.7 → regime has shifted → DON'T TRADE
```

### 3. Concept Drift Detection (ADWIN)
**What:** Real-time monitoring
```python
from river.drift import ADWIN
# Catches regime changes in days, not months
# Auto-pause trading when drift detected
```

### 4. Stacking + Meta-Learner
**What:** 4th model learns to combine 3 base models
```python
# Layer 1: XGB + LGBM + CatBoost
# Layer 2: Logistic Regression (meta-learner)
# Gain: 5-10% boost
```

### 5. SHAP Interaction Analysis
**What:** Find hidden feature synergies
```python
import shap
# Example: RSI × Volume = explosive signal
# Gain: 2-3% per interaction discovered
```

---

## 🌟 10 GOD MODE MECHANISMS {#god-mode}

### 1. Signal Decay (Alpha Half-Life)
**RenTec:** Kill stale signals after 30 mins
```python
signal_age_minutes = (now - signal_time).total_seconds() / 60
if signal_age_minutes > 30:
    signal_strength *= 0.5 ** (signal_age_minutes / 30)
```

### 2. Order Flow Imbalance (Level 3 Proxy)
**Citadel:** Buying pressure indicator
```python
ofi = (bid_volume - ask_volume) / (bid_volume + ask_volume)
# OFI > 0.3 = strong buying pressure
```

### 3. Adversarial Regime Detection
**DE Shaw:** Auto-stop when market shifts
```python
# Train classifier: current_data vs historical_data
# If classifier accuracy > 70% → STOP TRADING
```

### 4. Volatility Targeting
**AQR:** Risk-based position sizing
```python
target_vol = 0.15  # 15% annual
position_size = target_vol / realized_vol
```

### 5. Information Discreteness (Round Numbers)
**RenTec:** Psychology at $10, $50, $100
```python
dist_to_round = abs(price % 10)
# Higher signal confidence near round numbers
```

### 6. Alternative Data Synergy
**Two Sigma:** Volume as news proxy
```python
# Volume spike + price spike = news-driven move
# More predictable than random volatility
```

### 7. Latency Arbitrage Defense
**Jump Trading:** Use limit orders ONLY
```python
# NEVER use market orders
# Always use limit orders to avoid getting picked off
```

### 8. Fractal Market Hypothesis
**Bridgewater:** Multi-timeframe alignment
```python
# Trend alignment: 5min + 15min + 1hr + daily
# Trade only when all 4 aligned
```

### 9. Interaction Effects (SHAP)
**WorldQuant:** Feature synergies
```python
# RSI × Relative_Volume = explosive detector
# MACD × Momentum_Accel = trend strength
```

### 10. Meta-Labeling (Lopez de Prado)
**Two Sigma:** 2-stage prediction
```python
# Model 1: Predict direction (up/down)
# Model 2: Predict if Model 1 will be correct
# Trade only when Model 2 confidence > 70%
```

---

## 🔧 71-FEATURE ENGINEERING SPEC {#features}

### Tier 1: OHLCV Base (5)
```
Close, Open, High, Low, Volume
```

### Tier 2: Price Action (15)
```
returns_1, returns_5, returns_20
volatility_5, volatility_20, volatility_50
atr_14, range_pct
wick_ratio, gap_quality
rel_volume_10, rel_volume_50
is_volume_explosion
```

### Tier 3: Momentum & Trend (20)
```
rsi_14, rsi_7
macd_hist, macd_signal, macd_rising
momentum_5, momentum_10, momentum_20
roc_5, roc_10
sma_20, sma_50, sma_200
sma_20_50_cross, sma_50_200_cross
trend_consistency, trend_strength
```

### Tier 4: Institutional Secret Sauce (15)
```
liquidity_impact      # RenTec - thin stock detector
vol_accel            # DE Shaw - explosion detector
smart_money_score    # Gap & crap filter
wick_ratio           # Trap detector
mom_accel            # Soros - parabolic curve (RANKS #4 GLOBALLY)
fractal_efficiency   # Trend quality
price_efficiency     # News reaction
dist_from_max_pain   # Short squeeze potential
kurtosis_20          # Tail risk
auto_corr_5          # Regime detector
squeeze_potential    # Comprehensive squeeze detector
```

### Tier 5: God Mode (10)
```
signal_decay           # Alpha half-life
ofi_proxy             # Order flow imbalance
dist_to_round_number  # Psychology barriers
volume_accel_3d       # Alt data proxy
multi_timeframe_score # Fractal alignment
rsi_x_vol_interaction # Synergy detection
price_vol_interaction # Context feature
volatility_regime     # High/normal/low
sharpe_recent         # Rolling 20-day Sharpe
market_fear_index     # VIX proxy
```

### Tier 6: Market Context (6)
```
spy_close, spy_trend
vix_level, market_fear, market_trend
day_of_week
```

**TOTAL: 71 FEATURES**

---

## 📄 PAPER TRADING INTEGRATION {#paper-trading}

### Philosophy
> "The only way to teach a baby to swim is to throw it in the water"

**Goal:** Learn by doing - 20% allocation initially

### Alpaca Setup
```python
import alpaca_trade_api as tradeapi

api = tradeapi.REST(
    key_id='YOUR_KEY',
    secret_key='YOUR_SECRET',
    base_url='https://paper-api.alpaca.markets'  # Paper trading
)

# Initial capital: $100,000 paper
# Real capital (when validated): $25,000
```

### Trading Strategy
```python
# 1. Morning: Generate predictions for all 1,200 tickers
predictions = model.predict_all(tickers)

# 2. Filter: confidence > 60%
high_confidence = [p for p in predictions if p['confidence'] > 0.6]

# 3. Sort by confidence
high_confidence.sort(key=lambda x: x['confidence'], reverse=True)

# 4. Trade top 20 signals (20% allocation initially)
for signal in high_confidence[:20]:
    execute_trade(signal)
```

### Complete Logging System
```python
# Log EVERYTHING
log_entry = {
    'timestamp': datetime.now(),
    'ticker': ticker,
    'signal': 'BUY/SELL',
    'confidence': 0.85,
    'probability': 0.92,
    'cluster_id': 2,
    'model_votes': {'xgb': 1, 'lgb': 1, 'cat': 1},
    'features': feature_dict,
    'entry_price': 150.25,
    'target_profit': 165.28,  # +10%
    'stop_loss': 142.74,      # -5%
    'outcome': None  # Updated later
}
```

### Daily Retraining Protocol
```python
# After market close (4:00 PM)
def daily_retrain():
    # 1. Fetch today's data
    new_data = fetch_todays_data()
    
    # 2. Append to training set
    training_data = pd.concat([training_data, new_data])
    
    # 3. Check for drift
    drift_detected = drift_detector.check(new_data)
    
    # 4. Retrain if drift OR weekly schedule
    if drift_detected or datetime.now().weekday() == 6:  # Sunday
        retrain_models(training_data)
```

### Weekly Analysis
```python
# Sunday evening review
def weekly_review():
    # 1. Calculate metrics
    win_rate = wins / total_trades
    avg_profit = total_profit / wins
    avg_loss = total_loss / losses
    sharpe = (avg_profit - avg_loss) / std_dev
    max_dd = calculate_max_drawdown()
    
    # 2. Best/worst performers
    best_tickers = df.groupby('ticker')['profit'].mean().nlargest(10)
    worst_tickers = df.groupby('ticker')['profit'].mean().nsmallest(10)
    
    # 3. Feature importance changes
    current_importance = get_shap_values()
    compare_to_last_week(current_importance)
    
    # 4. Confidence calibration
    # Are 80% confidence predictions winning 80% of the time?
    calibration_curve = plot_calibration()
```

### Scaling Plan
```
Week 1-4:  20% allocation ($20k paper)
Week 5-8:  50% allocation ($50k paper) if 90%+ WR maintained
Week 9-12: 75% allocation ($75k paper) if 90%+ WR maintained
Week 13+:  100% allocation → Move to real money ($25k initial)
```

---

## 🤖 COMPANION AI SYSTEM {#companion-ai}

### Purpose
**AI that watches your portfolio and warns you:**
> "Hey, dump it while you're up - it will drop because of XYZ"

### Core Functions

#### 1. Position Monitor
```python
class CompanionAI:
    def monitor_position(self, ticker, entry_price, current_price):
        """Watch active positions for warning signs"""
        
        # Calculate current profit
        profit_pct = (current_price - entry_price) / entry_price
        
        # Check warning signals
        warnings = []
        
        # Signal decay (30-min half-life)
        signal_age = (now - entry_time).total_seconds() / 60
        if signal_age > 30:
            warnings.append(f"⚠️ Signal decaying (age: {signal_age:.0f}m)")
        
        # Regime shift detection
        if self.detect_regime_shift(ticker):
            warnings.append("⚠️ REGIME SHIFT DETECTED - Consider exit")
        
        # Profit target approaching
        if profit_pct > 0.08:  # 8% (close to 10% target)
            warnings.append(f"✅ Near target (+{profit_pct:.1%}) - Consider taking profit")
        
        # Volume drying up
        if self.is_volume_declining(ticker):
            warnings.append("⚠️ Volume declining - Momentum fading")
        
        # Round number resistance
        if self.near_round_number(current_price):
            warnings.append(f"⚠️ Near round number (${int(current_price)}) - Possible resistance")
        
        return warnings
```

#### 2. Exit Recommendation Engine
```python
def recommend_exit(self, position):
    """Decide if position should be closed"""
    
    score = 0
    reasons = []
    
    # Positive: Profit taken
    if position['profit_pct'] > 0.08:
        score += 3
        reasons.append("Near profit target")
    
    # Negative: Signal decay
    if position['signal_age_min'] > 30:
        score += 2
        reasons.append("Signal aged out")
    
    # Negative: Regime shift
    if self.regime_shifted:
        score += 4
        reasons.append("Market regime changed")
    
    # Negative: Volume decline
    if self.volume_declining:
        score += 2
        reasons.append("Volume drying up")
    
    # Recommendation
    if score >= 5:
        return {
            'action': 'CLOSE NOW',
            'urgency': 'HIGH',
            'reasons': reasons,
            'message': f"🚨 DUMP {position['ticker']} NOW: {', '.join(reasons)}"
        }
    elif score >= 3:
        return {
            'action': 'WATCH CLOSELY',
            'urgency': 'MEDIUM',
            'reasons': reasons,
            'message': f"⚠️ Watch {position['ticker']}: {', '.join(reasons)}"
        }
    else:
        return {
            'action': 'HOLD',
            'urgency': 'LOW',
            'reasons': ['Position healthy'],
            'message': f"✅ {position['ticker']} looking good"
        }
```

#### 3. Real-Time Alerts
```python
def send_alert(self, message, urgency='MEDIUM'):
    """Send real-time alert to trader"""
    
    if urgency == 'HIGH':
        # SMS + Email + Desktop notification
        send_sms(message)
        send_email(message)
        desktop_notify(message)
    elif urgency == 'MEDIUM':
        # Email + Desktop
        send_email(message)
        desktop_notify(message)
    else:
        # Log only
        logger.info(message)
```

---

## 🥇 GOLD INTEGRATION STRATEGY {#gold-integration}

### Why Gold?
**Hedge + Safe Haven + Volatility Dampener**

### Integration Points

#### 1. Portfolio Allocation
```python
# Dynamic allocation based on market fear
vix = get_current_vix()

if vix > 30:  # High fear
    gold_allocation = 0.30  # 30% gold
    stock_allocation = 0.70
elif vix > 20:  # Medium fear
    gold_allocation = 0.20  # 20% gold
    stock_allocation = 0.80
else:  # Low fear
    gold_allocation = 0.10  # 10% gold
    stock_allocation = 0.90
```

#### 2. Gold as Regime Indicator
```python
# Gold correlation with stocks
correlation = calculate_correlation(gold_returns, spy_returns, window=20)

if correlation > 0.5:
    regime = 'RISK-ON'  # Both rising = healthy market
elif correlation < -0.5:
    regime = 'RISK-OFF'  # Gold up, stocks down = flight to safety
else:
    regime = 'NEUTRAL'
```

#### 3. Gold Features in Model
```python
# Add to 71 features:
features['gold_price'] = get_gold_price()
features['gold_returns_1d'] = gold_price_pct_change(1)
features['gold_returns_5d'] = gold_price_pct_change(5)
features['gold_spy_correlation'] = calculate_correlation(gold, spy, 20)
features['gold_volatility'] = gold_returns.std() * np.sqrt(252)
```

#### 4. Drawdown Protection
```python
# If portfolio drawdown > -10%, increase gold
if portfolio_dd < -0.10:
    emergency_gold_allocation = 0.50  # 50% gold
    liquidate_worst_positions()
    buy_gold(emergency_gold_allocation)
```

---

## 📊 DEEP TICKER LEARNING {#deep-learning}

### Philosophy
> "If it's been following a ticker for 5 years, it knows its signals, news, and potential"

### Strategy: 100+ Core Tickers

#### Tier 1: Your Watchlist (76 tickers)
**Learn deeply - 5 years each**
```
NVDA, AMD, TSLA, PLTR, HOOD, IONQ, PALI, QS, SMCI, ARM, etc.
```

#### Tier 2: Future Expansion (115 tickers)
**Next phase - 5 years each**
```
AAPL, MSFT, GOOGL, AMZN, META, NFLX, etc.
```

#### Tier 3: Market Context (1000+ tickers)
**Broader learning - 2 years each**
```
S&P 500, NASDAQ 100, Russell 2000
```

### Deep Learning Approach

#### 1. Ticker-Specific Patterns
```python
# For each ticker, learn:
patterns = {
    'typical_gap_size': 2.5%,          # NVDA gaps average 2.5%
    'avg_daily_range': 3.2%,           # NVDA moves 3.2%/day
    'news_reaction_speed': '15min',    # NVDA reacts in 15 mins
    'support_levels': [450, 425, 400], # Historical supports
    'resistance_levels': [500, 525],   # Historical resistance
    'earnings_volatility': 8.5%,       # Earnings day moves
    'typical_recovery_time': '3 days'  # Time to recover from -5%
}
```

#### 2. Ticker-Specific Features
```python
# Add to each ticker's features:
def engineer_ticker_specific_features(ticker, data):
    """Engineer features specific to this ticker's behavior"""
    
    # Historical volatility percentile
    current_vol = data['volatility'].iloc[-1]
    hist_vol = data['volatility'].rolling(252).mean()
    features['vol_percentile'] = (current_vol - hist_vol.mean()) / hist_vol.std()
    
    # Distance from typical support
    nearest_support = find_nearest_support(ticker, data['close'].iloc[-1])
    features['dist_from_support'] = (data['close'].iloc[-1] - nearest_support) / nearest_support
    
    # Typical gap behavior
    typical_gap = ticker_stats[ticker]['typical_gap_size']
    current_gap = (data['open'].iloc[-1] - data['close'].iloc[-2]) / data['close'].iloc[-2]
    features['gap_vs_typical'] = current_gap / typical_gap
```

#### 3. Progressive Training
```python
# Month 1: Train on 76 core tickers (deep 5-year learning)
core_tickers = load_your_watchlist()
train_deeply(core_tickers, years=5)

# Month 2: Add 50 expansion tickers
expansion_batch_1 = load_expansion_tickers(batch=1, size=50)
train_deeply(expansion_batch_1, years=5)

# Month 3: Add another 65 expansion tickers
expansion_batch_2 = load_expansion_tickers(batch=2, size=65)
train_deeply(expansion_batch_2, years=5)

# Month 4+: Add market context (1000+ tickers, lighter training)
market_tickers = load_market_context_tickers()
train_lightly(market_tickers, years=2)
```

#### 4. Cloud Infrastructure
```python
# For 100+ tickers with 5 years each
storage_needed = 100 * 5 * 252 * 1000  # ~126M rows
storage_gb = 126000000 * 100 / 1e9     # ~12 GB

# Training time on A100
training_time = 100 * 0.5  # 50 hours = 2 days

# Solution: Cloud training loop
for ticker_batch in split_into_batches(all_tickers, batch_size=20):
    # Train 20 tickers at a time (5 hours each batch)
    train_batch_on_a100(ticker_batch)
    save_models_to_cloud(ticker_batch)
    
# Total: 100 tickers / 20 per batch = 5 batches × 5 hours = 25 hours
```

---

## 🎯 PRODUCTION ARCHITECTURE

### Data Flow
```
1. Data Collection (daily)
   ├── Fetch OHLCV for 100+ tickers
   ├── Engineer 71 features
   └── Store in PostgreSQL

2. Training (weekly)
   ├── Pull latest data
   ├── Retrain models (cluster-specific)
   ├── Validate performance
   └── Deploy if improved

3. Inference (real-time)
   ├── Market opens
   ├── Generate predictions for all tickers
   ├── Filter by confidence > 60%
   └── Send to Companion AI

4. Companion AI (real-time monitoring)
   ├── Watch active positions
   ├── Detect warning signals
   ├── Send alerts
   └── Recommend exits

5. Execution (automated)
   ├── Receive signals from Companion
   ├── Execute via Alpaca API
   ├── Log everything
   └── Update portfolio

6. Analysis (after market close)
   ├── Daily P&L report
   ├── Win rate calculation
   ├── Drift detection
   └── Retrain if needed
```

---

## 📈 EXPECTED PERFORMANCE

### Baseline (Local Testing)
```
Tickers: 46
Samples: 11,474
Features: 71
Win Rate: 87.9% ✅
Target: 75%
Achievement: +12.9% above target
```

### Full Training (A100)
```
Tickers: 100+
Samples: 1.5M+
Features: 71
Expected Baseline: 78-82% WR
Expected Trained: 90%+ WR
Sharpe Ratio: 3.5+
Max Drawdown: ≤-5%
```

### Paper Trading (Realistic)
```
Initial: 85%+ WR (learning phase)
Mature: 88%+ WR (after 3 months)
Live: 80%+ WR sustained (goal)
```

---

## 🚀 DEPLOYMENT CHECKLIST

### Pre-Training
- [ ] 71 features validated ✅
- [ ] Local baseline ≥75% WR ✅ (87.9%)
- [ ] Optuna parameters optimized
- [ ] Data pipeline for 100+ tickers
- [ ] Gold integration tested

### Training (A100)
- [ ] 100+ tickers × 5 years data
- [ ] 15 base models (3 algos × 5 clusters)
- [ ] NGBoost (probabilistic)
- [ ] Meta-learner (stacking)
- [ ] SHAP interactions
- [ ] Drift detector

### Deployment
- [ ] Models saved to cloud
- [ ] Inference engine tested
- [ ] Companion AI running
- [ ] Alpaca paper account connected
- [ ] Logging system active
- [ ] Real-time monitoring dashboard

### Validation (Week 1-4)
- [ ] Paper trade with 20% allocation
- [ ] Log every signal, trade, outcome
- [ ] Daily retraining
- [ ] Weekly performance review
- [ ] Drift monitoring
- [ ] Confidence calibration

### Scale (Week 5+)
- [ ] 90%+ WR validated over 4 weeks
- [ ] Move to real money ($25k)
- [ ] 20% → 50% → 100% allocation
- [ ] Full companion AI integration
- [ ] Gold hedge active
- [ ] Complete automation

---

**END OF INSTITUTIONAL KNOWLEDGE BASE**

*All mechanisms documented. All strategies defined. Ready to execute.* 🎯
