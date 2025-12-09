# 🚀 COMPREHENSIVE BUILD PLAN - QUANTUM AI TRADER v1.1
## 4-Week Implementation Strategy with Colab Pro T4 GPU Training

**Date**: December 8, 2025 23:55 ET  
**Target Completion**: January 5, 2026  
**Resources**: Local dev (12hr/day) + Colab Pro (T4 GPU for training)

---

## 📊 CURRENT STATE AUDIT

### ✅ **PRODUCTION-READY (70% Complete)**
1. **Pattern Detection** (advanced_pattern_detector.py) - 85% complete, 751 LOC
2. **AI Recommender** (ai_recommender_adv.py) - 90% complete, 238 LOC  
3. **Market Regime** (market_regime_manager.py) - 95% complete, 300 LOC
4. **Backtest Engine** (backtest_engine.py) - 80% complete, 380 LOC
5. **Core Infrastructure** - 95-100% complete

### ⚠️ **NEEDS COMPLETION (30% Remaining)**
1. **Research Features** - 35% → 100% (need dark pool, sentiment, cross-asset)
2. **Meta-Learner** - 45% → 100% (need weight optimization from Perplexity Q1-Q6)
3. **Confidence Calibrator** - 70% → 100% (need method selection from Q3)
4. **Position Sizer** - 75% → 100% (need Kelly/vol-targeting from Q20-Q21)
5. **Test Suite** - 60% → 100% (need ablation, scenario tests)

### 🆕 **NEW MODULES FROM PERPLEXITY + DARK POOL RESEARCH**
1. **Dark Pool Signals** (IFI, A/D, OBV, VROC, SMI) - 0% → 100%
2. **Drift Monitor** (PSI, KS test, rolling metrics) - 0% → 100%
3. **Feature Store** (SQLite cache for historical data) - 0% → 100%
4. **Colab Training Pipeline** (T4 GPU hyperparameter tuning) - 0% → 100%

---

## 🎯 MODULE BUILD ORDER (Priority-Sorted)

### **WEEK 1: FOUNDATION + DARK POOL SIGNALS** (Dec 9-15)

#### **Module 1: Dark Pool & Microstructure Signals** 🔥 **START HERE**
**Priority**: CRITICAL (enables institutional flow detection)  
**Location**: `src/features/dark_pool_signals.py`  
**Dependencies**: yfinance (already installed)  
**Estimated**: 12 hours implementation + 4 hours testing

**What We Build**:
```python
class DarkPoolSignals:
    def institutional_flow_index(ticker, days=20) -> float  # IFI formula
    def accumulation_distribution(ticker, lookback=20) -> float  # A/D line
    def obv_institutional(ticker, lookback=20) -> float  # Enhanced OBV
    def volume_acceleration_index(ticker, lookback=20) -> float  # VROC
    def smart_money_index(ticker, lookback=20) -> dict  # SMI composite (0-100)
```

**Training Strategy**: No ML needed (pure formula-based), validate on 2023-2024 data  
**Success Metric**: SMI edge >30bps after costs, correlation with next-day returns >0.15

---

#### **Module 2: Research Features Completion** 🔥
**Priority**: CRITICAL (core signal generation)  
**Location**: `src/features/research_features.py` (already 35% done)  
**Dependencies**: Dark Pool Signals (Module 1), EODHD API, FRED API  
**Estimated**: 16 hours implementation + 4 hours testing

**What We Complete**:
- ✅ Layer 1: Dark pool ratio (use Module 1), after-hours volume, supply chain leads
- ✅ Layer 2: Breadth rotation, cross-asset correlations (BTC, yields, VIX)
- ✅ Layer 3: Sentiment integration (EODHD 5-feature engineering from Q9)
- ✅ Layer 4-9: Microstructure proxies, regime-specific features

**Training Strategy**: Feature selection via SHAP (Q7) - reduce 60 → 15-20 features  
**Colab Use**: Run SHAP on T4 GPU (15min vs 2hr local)  
**Success Metric**: Top 15 features identified, correlation matrix r<0.8

---

#### **Module 3: Feature Store (Data Cache)** 
**Priority**: HIGH (speeds up training 10x)  
**Location**: `src/data/feature_store.py`  
**Dependencies**: SQLite3 (built-in)  
**Estimated**: 8 hours implementation + 2 hours testing

**What We Build**:
```python
class FeatureStore:
    def save_features(ticker, date, features: dict) -> None
    def load_features(ticker, start_date, end_date) -> pd.DataFrame
    def check_staleness(ticker, max_age_hours=24) -> bool
    def bulk_insert(dataframe) -> None  # Fast batch insert
```

**Training Strategy**: Cache all 60 features for 100+ tickers, 5 years = 125K rows  
**Success Metric**: Feature retrieval <100ms (vs 30s API calls), 95% hit rate

---

### **WEEK 2: META-LEARNER + CALIBRATION** (Dec 16-22)

#### **Module 4: Integrated Meta-Learner Finalization** 🔥
**Priority**: CRITICAL (core forecasting engine)  
**Location**: `src/models/integrated_meta_learner.py` (already 45% done)  
**Dependencies**: Research Features (Module 2), Perplexity Q1-Q6 answers  
**Estimated**: 20 hours implementation + 8 hours training + 4 hours testing

**What We Complete**:
1. **Architecture Decision** (from Q1): Hierarchical ensemble vs XGBoost
2. **Weight Matrices** (from Q2): Complete 12×5 regime×signal matrix
3. **Signal Agreement** (from Q4): Boost/reduce confidence based on alignment
4. **Missing Signal Handling** (from Q5): Fallback strategies per signal type
5. **Training Strategy** (from Q6): Monthly retraining on 18mo window

**Colab Training Pipeline**:
```python
# Train on Colab Pro T4 GPU (4hr session)
def train_meta_learner_colab():
    # 1. Download features from Feature Store (5min)
    # 2. Grid search hyperparameters (XGBoost: 100 combos × 5 CV folds = 3hr on T4)
    # 3. Train final model on best params (20min)
    # 4. Upload model artifact to Drive (1min)
    # 5. Download to local, deploy (5min)
```

**Success Metric**: Validation Sharpe >0.6 (vs 0.4 baseline), win-rate 58-65%

---

#### **Module 5: Confidence Calibrator Finalization** 🔥
**Priority**: CRITICAL (accurate position sizing depends on this)  
**Location**: `src/calibration/confidence_calibrator.py` (already 70% done)  
**Dependencies**: Meta-Learner (Module 4), Perplexity Q3, Q19 answers  
**Estimated**: 12 hours implementation + 4 hours validation

**What We Complete**:
1. **Method Selection** (from Q3): Beta calibration (best for small samples)
2. **Per-Regime Calibration** (from Q19): Hierarchical global + regime adjustments
3. **Recalibration Logic**: Every 50 trades or ECE >0.10
4. **Validation**: Expected Calibration Error (ECE) <0.05

**Colab Use**: Train 12 calibration curves in parallel (5min on T4 vs 30min local)  
**Success Metric**: ECE <0.05, Brier score <0.15, log-loss <0.5

---

#### **Module 6: Position Sizer Optimization** 🔥
**Priority**: CRITICAL (converts predictions to $ positions)  
**Location**: `src/position_sizing/position_sizer.py` (already 75% done)  
**Dependencies**: Calibrator (Module 5), Perplexity Q20-Q24 answers  
**Estimated**: 12 hours implementation + 4 hours validation

**What We Complete**:
1. **Kelly Formula** (from Q20): Multi-outcome Kelly for 3 outcomes (win/loss/chop)
2. **Volatility Targeting** (from Q21): Combine Kelly + vol-target + regime
3. **Stop-Loss Optimization** (from Q22): Sector × regime × ATR formula
4. **Take-Profit Logic** (from Q23): Regime-dependent (trailing in trends, fixed in chop)
5. **Portfolio Constraints** (from Q24): Regime-specific limits (40% sector in bull, 20% in bear)

**Success Metric**: Backtest portfolio vol = 15% ± 2%, max DD <15%, Kelly prevents overbet

---

### **WEEK 3: TESTING + OPTIMIZATION** (Dec 23-29)

#### **Module 7: Comprehensive Test Suite**
**Priority**: HIGH (prevents production bugs)  
**Location**: `tests/unit/`, `tests/integration/`, `tests/backtests/`  
**Dependencies**: All modules 1-6  
**Estimated**: 16 hours implementation + 8 hours execution

**What We Build**:
1. **Unit Tests** (85% coverage target):
   - Timestamp validation (no look-ahead bias)
   - Feature determinism (same inputs → same outputs)
   - Constraint validation (Kelly 0-0.5, confidence 0-1)
   - API mocking (test without hitting rate limits)

2. **Integration Tests**:
   - End-to-end forecast (patterns → regime → research → meta → calibrator → sizer)
   - Data pipeline (API → Feature Store → Model)
   - Error handling (API down, missing data, stale cache)

3. **Ablation Study** (from Q30):
   - Patterns only → Sharpe baseline
   - + Regimes → +0.1 Sharpe?
   - + Research features → +0.15 Sharpe?
   - + Dark pool signals → +0.08 Sharpe?
   - + Calibration → +0.05 Sharpe?
   - + Position sizing → +0.05 Sharpe?

4. **Scenario Replay** (from Q31):
   - 2020 COVID crash: Survive with DD <25%?
   - 2022 rate shock: Sharpe >0.3?
   - 2023 AI melt-up: Sharpe >0.7?

**Colab Use**: Run 100 backtest variations in parallel (2hr on T4)  
**Success Metric**: All tests pass, ablation shows incremental value, scenarios pass

---

#### **Module 8: Drift Monitor & Retraining Pipeline**
**Priority**: HIGH (prevents model degradation)  
**Location**: `src/monitoring/drift_monitor.py`  
**Dependencies**: Feature Store, trained models  
**Estimated**: 10 hours implementation + 2 hours testing

**What We Build**:
```python
class DriftMonitor:
    def calculate_psi(train_dist, prod_dist) -> float  # Population Stability Index
    def ks_test(train_data, prod_data) -> float  # Kolmogorov-Smirnov
    def rolling_sharpe(trades, window=50) -> float  # Performance monitoring
    def calibration_error(predicted, actual) -> float  # ECE drift
    def trigger_retrain() -> bool  # Decision logic: PSI>0.2 OR Sharpe drop >0.15
```

**Success Metric**: Detect drift within 1 week, trigger retrain automatically

---

### **WEEK 4: PRODUCTION DEPLOYMENT** (Dec 30 - Jan 5)

#### **Module 9: Colab Training Orchestrator**
**Priority**: MEDIUM (automates GPU training)  
**Location**: `notebooks/colab_training_pipeline.ipynb`  
**Dependencies**: All modules 1-8  
**Estimated**: 8 hours implementation + 4 hours testing

**What We Build**:
1. **Data Download**: Sync Feature Store to Colab (Google Drive mount)
2. **Hyperparameter Tuning**: Optuna + XGBoost on T4 GPU (4hr session)
3. **Model Training**: Final model on best params
4. **Model Upload**: Save to Drive, download to local
5. **Deployment**: Automated via GitHub Actions

**Colab Pro Benefits**:
- T4 GPU: 10-15x faster than local CPU for XGBoost
- High RAM: 25GB (vs 8GB local) - process 5 years × 100 tickers in memory
- Background execution: Train while you sleep (12hr sessions)

**Success Metric**: Full retraining (100 tickers, 5yr data, 1000 hyperparam combos) in <4hr

---

#### **Module 10: Production API Wrapper**
**Priority**: MEDIUM (enables Streamlit frontend)  
**Location**: `src/api/forecaster_api.py`  
**Dependencies**: All modules 1-9  
**Estimated**: 8 hours implementation + 2 hours testing

**What We Build**:
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/forecast/{ticker}")
async def get_forecast(ticker: str) -> dict:
    # Returns: signal (BUY/SELL/HOLD), confidence (0-100), 
    # position_size (% portfolio), reasoning (top 5 signals)

@app.get("/portfolio/status")
async def get_portfolio() -> dict:
    # Returns: positions, P&L, risk metrics, regime

@app.post("/retrain")
async def trigger_retrain() -> dict:
    # Kicks off Colab training pipeline
```

**Success Metric**: <200ms response time, 99.9% uptime, deployed to cloud (Render/Railway)

---

#### **Module 11: Monitoring Dashboard**
**Priority**: LOW (nice to have, frontend for Spark UI)  
**Location**: `dashboards/streamlit_monitor.py`  
**Dependencies**: API (Module 10)  
**Estimated**: 12 hours implementation (you build this with Spark)

**Your Frontend** (Spark-based):
- Real-time forecasts table (ticker, signal, confidence, reasoning)
- Portfolio heatmap (positions, P&L, sector allocation)
- Performance charts (equity curve, rolling Sharpe, drawdown)
- Drift alerts (PSI per feature, calibration error)
- Regime tracker (current regime, transition probability)

---

## 📋 DETAILED MODULE 1 IMPLEMENTATION (START NOW)

### **Module 1: Dark Pool & Microstructure Signals** 

**File**: `src/features/dark_pool_signals.py`

**Implementation Steps**:

1. **Create Base Class** (1 hour):
```python
import yfinance as yf
import numpy as np
import pandas as pd
from typing import Dict, Tuple

class DarkPoolSignals:
    """
    Institutional flow detection using free data sources.
    Based on Perplexity research: IFI, A/D, OBV, VROC, SMI formulas.
    """
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.data_cache = {}  # In-memory cache
```

2. **Implement IFI - Institutional Flow Index** (2 hours):
```python
def institutional_flow_index(self, days: int = 20) -> Dict[str, float]:
    """
    IFI = (Large Buy Volume - Large Sell Volume) / Total Volume
    
    Returns:
        dict with keys: IFI, IFI_score (0-100), interpretation
    """
    # Get minute data (1m bars for last 30 days)
    data = yf.download(self.ticker, interval="1m", period="30d", progress=False)
    recent = data.tail(days * 390)  # 390 minutes per trading day
    
    # Define "large" as volume > 90th percentile
    vol_threshold = recent['Volume'].quantile(0.9)
    
    # Calculate directional volume
    recent['price_direction'] = np.sign(recent['Close'].diff())
    recent['large_vol'] = np.where(recent['Volume'] > vol_threshold, recent['Volume'], 0)
    recent['buy_vol'] = np.where(recent['price_direction'] > 0, recent['large_vol'], 0)
    recent['sell_vol'] = np.where(recent['price_direction'] < 0, recent['large_vol'], 0)
    
    # Calculate IFI
    buy_total = recent['buy_vol'].sum()
    sell_total = recent['sell_vol'].sum()
    total_vol = recent['Volume'].sum()
    
    ifi = (buy_total - sell_total) / total_vol if total_vol > 0 else 0
    
    # Convert to 0-100 scale (assume IFI ranges -0.3 to +0.3)
    ifi_score = max(0, min(100, (ifi / 0.3 + 1) * 50))
    
    interpretation = 'BULLISH' if ifi > 0.15 else 'BEARISH' if ifi < -0.15 else 'NEUTRAL'
    
    return {
        'IFI': ifi,
        'IFI_score': ifi_score,
        'buy_volume': buy_total,
        'sell_volume': sell_total,
        'interpretation': interpretation
    }
```

3. **Implement A/D Line** (1.5 hours)
4. **Implement OBV Enhanced** (1.5 hours)
5. **Implement VROC** (1.5 hours)
6. **Implement SMI Composite** (2 hours)
7. **Add Caching & Error Handling** (1 hour)
8. **Write Unit Tests** (2 hours)

**Validation**:
```python
# Test on known institutional accumulation (NVDA Q1 2023)
signals = DarkPoolSignals("NVDA")
smi_2023_q1 = signals.smart_money_index(lookback=20)
assert smi_2023_q1['SMI'] > 70, "Should detect NVDA accumulation in Q1 2023"

# Test on known distribution (META Q4 2022)
signals_meta = DarkPoolSignals("META")
smi_2022_q4 = signals_meta.smart_money_index(lookback=20)
assert smi_2022_q4['SMI'] < 40, "Should detect META distribution in Q4 2022"
```

---

## 🔄 DAILY WORKFLOW (12-Hour Focused Sessions)

### **Morning Block (Hours 0-4): Research + Planning**
1. **Hour 0-1**: Run Perplexity questions (if Day 1), review previous day's work
2. **Hour 1-3**: Implement current module (focus block, no distractions)
3. **Hour 3-4**: Write tests, validate on historical data

### **Midday Block (Hours 4-8): Implementation**
4. **Hour 4-7**: Continue module implementation (3-hour deep work)
5. **Hour 7-8**: Integration testing, debug issues

### **Evening Block (Hours 8-12): Training + Documentation**
6. **Hour 8-10**: Colab training (if needed), hyperparameter tuning
7. **Hour 10-11**: Document code, update README, commit to GitHub
8. **Hour 11-12**: Plan next day, update progress tracker

---

## 📊 SUCCESS METRICS PER WEEK

### **Week 1 Success Criteria**:
- ✅ Dark Pool Signals module complete, tested, SMI edge >30bps
- ✅ Research Features 60 → 15 features via SHAP, correlation r<0.8
- ✅ Feature Store operational, 95% cache hit rate
- ✅ All Perplexity Q1-Q24 answered, decisions extracted

### **Week 2 Success Criteria**:
- ✅ Meta-Learner trained, validation Sharpe >0.6 (vs 0.4 baseline)
- ✅ Confidence Calibrator ECE <0.05, Brier <0.15
- ✅ Position Sizer portfolio vol = 15% ± 2%
- ✅ Full backtest 2021-2025: Sharpe >0.7, win-rate 58-65%, DD <15%

### **Week 3 Success Criteria**:
- ✅ Test suite 85% coverage, all tests pass
- ✅ Ablation study shows incremental value: patterns (0.4) → +regimes (0.5) → +research (0.65) → +dark pool (0.73) → +calibration (0.78) → +sizing (0.83)
- ✅ Scenario tests pass: 2020 (survive), 2022 (adapt), 2023 (capture)
- ✅ Drift monitor operational, detect shifts within 1 week

### **Week 4 Success Criteria**:
- ✅ Colab training pipeline: 100 tickers × 5yr in <4hr
- ✅ Production API deployed to cloud, <200ms response
- ✅ Monitoring dashboard live (your Spark frontend)
- ✅ Paper trading started (1 week validation before real money)

---

## 🎯 GO/NO-GO DECISION (January 5, 2026)

**GO Criteria** (proceed to live trading with 1% capital):
1. Paper trading win-rate ≥55% (N≥20 trades)
2. Sharpe ratio >0.4 (on paper trades)
3. Calibration error <10% (predicted vs actual confidence)
4. Max drawdown <20% (during paper trading week)
5. All drift monitors operational (no false positives)

**NO-GO Criteria** (back to optimization):
1. Win-rate <50% → investigate feature drift
2. Sharpe <0.3 → retrain meta-learner
3. Calibration error >15% → retrain calibrator
4. Max DD >25% → tighten stop-losses
5. Drift detected but not flagged → improve monitoring

---

## 🔧 TOOLS & INFRASTRUCTURE

### **Local Development**:
- Python 3.10 (.venv already configured)
- VS Code with Copilot (already active)
- Git version control (commit after each module)

### **Colab Pro Setup**:
```python
# Colab notebook template
!pip install xgboost lightgbm optuna shap yfinance pandas numpy scikit-learn

from google.colab import drive
drive.mount('/content/drive')

# Sync Feature Store
!cp /content/drive/MyDrive/quantum-trader/feature_store.db .

# Run training (use T4 GPU)
import xgboost as xgb
model = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0)
```

### **API Keys Needed** (sign up this week):
- ✅ yfinance (no key needed)
- ✅ Polygon (already active)
- ⚠️ Finnhub (renew: https://finnhub.io/register)
- ⚠️ FRED (sign up: https://fred.stlouisfed.org/docs/api/api_key.html)
- ⚠️ FINRA (no key, manual download: https://www.finra.org/finra-data)

### **Deployment**:
- Cloud: Render or Railway (free tier, FastAPI deployment)
- Frontend: Your Spark UI (separate repo/build)
- Monitoring: Hosted dashboard + Discord/email alerts

---

## 📝 PROGRESS TRACKER

**Day 1 (Dec 9)**: Perplexity research (4hr) + Start Module 1 (8hr)  
**Day 2 (Dec 10)**: Complete Module 1 (4hr) + Start Module 2 (8hr)  
**Day 3 (Dec 11)**: Complete Module 2 (10hr) + Start Module 3 (2hr)  
**Day 4 (Dec 12)**: Complete Module 3 (8hr) + Integration testing (4hr)  
**Day 5 (Dec 13)**: Buffer day (complete any Week 1 TODOs)  

**Day 6-10 (Dec 16-20)**: Modules 4-6 (Meta-Learner, Calibrator, Position Sizer)  
**Day 11-15 (Dec 23-27)**: Modules 7-8 (Testing, Drift Monitor)  
**Day 16-20 (Dec 30-Jan 3)**: Modules 9-10 (Colab Pipeline, API)  
**Day 21 (Jan 4)**: Paper trading setup, monitoring  
**Day 22 (Jan 5)**: Go/No-Go decision  

---

## 🚀 START COMMAND

**Right now, we build Module 1: Dark Pool Signals**

Next message will contain the complete production-grade implementation of `dark_pool_signals.py` with all 5 formulas (IFI, A/D, OBV, VROC, SMI), unit tests, validation, and integration hooks.

**Your grandfather is watching. Let's build something that works.**
