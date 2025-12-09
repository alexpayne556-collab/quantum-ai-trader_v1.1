# 🔗 SYSTEM ARCHITECTURE MAPPING: From Existing → Complete Integrated Forecaster

**Purpose**: Show EXACT current state of your system + what needs to be added + integration points  
**Date**: December 8, 2025  
**Status**: Ready for Implementation After Perplexity Research

---

## 📊 PART 1: CURRENT SYSTEM INVENTORY

### 1.1 What You Already Have (Pattern Detection)

**File**: `pattern_detector.py` (from workspace)

```python
# EXISTING CAPABILITIES:
- Elliott Wave pattern recognition (impulse 5-wave, corrective ABC)
- Candlestick patterns (hammer, doji, engulfing, etc.)
- Support/Resistance levels (pivot points, Fibonacci retracements)
- Trend lines (higher highs/lows, lower highs/lows)
- Volume analysis (accumulation/distribution)

# OUTPUTS:
- pattern_signal: 'STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL'
- pattern_confidence: 0-100% (how sure we are about this pattern)
- pattern_type: 'ELLIOTT_5_WAVE', 'ABC_CORRECTION', 'HAMMER', etc.
- time_to_reversal: estimated days until pattern completes
```

### 1.2 What You Already Have (Regime Aware Recommender)

**File**: `ai_recommender_adv.py` or similar (from workspace)

```python
# EXISTING CAPABILITIES:
- ADX (Average Directional Index) based regime classification
- RSI divergences (pattern/overbought detection)
- MACD trending filters
- EMA configuration per regime (fast/slow for trending, medium for chop)

# OUTPUTS:
- regime_state: 'STRONG_TREND_UP', 'WEAK_TREND_UP', 'CHOPPY', 'STRONG_TREND_DOWN'
- adc_value: 0-100 (directional strength)
- optimal_ema: tuple of (fast, slow, signal) periods
- confidence: 0-100% (how strong the regime signal is)
- position_recommendation: 'LONG', 'NEUTRAL', 'SHORT'
```

### 1.3 What You Already Have (Meta-Learner)

**File**: `COMPREHENSIVE_SYSTEM_TEST_AND_TRAINER.py` or similar

```python
# EXISTING CAPABILITIES:
- Combines pattern_signal + regime_state into single recommendation
- Applies confidence filter (only trade when both signals align)
- Backtesting engine
- Performance tracking (win-rate, Sharpe ratio, drawdown)

# CURRENT ARCHITECTURE:
pattern_signal (40%) + regime_state (60%) → confidence filter → TRADE/HOLD/SKIP

# ACCURACY BASELINE:
- Current win-rate: ~34% on 5-15d horizon
- Current Sharpe: ~0.25-0.35
- Main weakness: Missing cross-asset context + catalyst awareness
```

### 1.4 What You Already Have (Universe/Backtester)

**File**: `backtest_engine.py`, `test_backtest_engine.py`

```python
# EXISTING CAPABILITIES:
- 50-ticker watchlist (AI, robotaxi, some quantum)
- Daily OHLCV data (yfinance)
- Position sizing logic
- Stop-loss & take-profit execution
- P&L tracking per trade
- Metrics: win-rate, Sharpe, max drawdown, return

# GAPS:
- Only 50 tickers (will expand to 100+)
- No pre-catalyst detection (missing 24-48hr edge)
- No cross-asset features (BTC, yields, VIX)
- No 12-regime system (only ADX)
- No dark pool/microstructure signals
```

---

## 📈 PART 2: WHAT RESEARCH ADDS (9 Layers)

### 2.1 Layer 1: Hidden Universe Construction
**From**: PERPLEXITY_RESEARCH_ANSWERS_FREE_TIER.md

```python
# NEW SIGNALS:
- dark_pool_ratio: volume clustering pattern (free from yfinance minute data)
- after_hours_volume: institutional activity proxy
- supply_chain_leads: ASML→NVDA (21-35 days ahead, SEC EDGAR)
- breadth_rotation: sector breadth divergences (5-10 days ahead)
- cross_asset_correlations: BTC→tech, yields→rotation (6-48 hours ahead)

# FEEDS INTO:
forecaster as Tier 1 features (highest SHAP importance 0.08-0.12)
```

### 2.2 Layers 2-4: Microstructure, Regimes, Features
**From**: DISCOVERY_LAYERS_2_THROUGH_4_COMPLETE.md

```python
# NEW SYSTEM:
- 12-regime classification (VIX × breadth × ATR, not just ADX)
- pre_breakout_fingerprint: 5-feature combo (spread, dark pool, skew, VWAP, RSI)
- 40-60 engineered features (ranked by SHAP importance)

# PERFORMANCE:
- Pre-breakout detection: 95% precision, 2-4 days early
- 12-regime system: 340 bps/year better than 2-regime (Hamilton 2024)
- Features cover: microstructure, sentiment, catalysts, cross-asset

# FEEDS INTO:
forecaster as primary signal sources (replaces some existing indicators)
```

### 2.3 Layers 5-9: Catalysts, Training, Deployment
**From**: DISCOVERY_LAYERS_5_THROUGH_9_COMPLETE.md

```python
# NEW CAPABILITIES:
- Pre-catalyst detection: 24-48hrs ahead (options OI, Form 4, analyst clusters)
- News tier classification: 5 tiers (instant to 2-4 week supply chain leads)
- Regime-aware cross-validation: prevent overfitting
- Sector-specific Sharpe ratios: AI 0.82, Quantum 0.68, Robotaxi 0.61
- Confidence calibration: meta-confidence models per regime
- Streamlit dashboard: real-time regime + candidates + risk alerts
- Complete free API reference: 9 sources, $0/month

# FEEDS INTO:
forecaster as secondary signals + validation framework + deployment tools
```

---

## 🎯 PART 3: THE MISSING PIECE - THE FORECASTER

### 3.1 What Forecaster Must Do

```python
# INPUT (at market close each day):
{
    'ticker': 'NVDA',
    'date': '2025-12-08',
    
    # From pattern detector (existing):
    'pattern_signal': 'BUY',
    'pattern_confidence': 0.68,
    'pattern_type': 'ELLIOTT_5_WAVE',
    
    # From regime detector (existing):
    'regime_state': 'STRONG_TREND_UP',
    'adc_value': 42,
    'optimal_ema': (8, 21, 55),
    
    # From research Layer 1 (NEW):
    'dark_pool_ratio': 0.42,
    'ah_volume_pct': 0.18,
    'supply_chain_signal': 1.0,  # 1=ASML leading
    'breadth_rotation': 0.73,  # 73% of AI above SMA
    
    # From research Layers 2-4 (NEW):
    'pre_breakout_score': 4,  # 4/5 features firing
    'regime_12': 'BULL_LOW_VOL_STABLE',
    'spread_compression': 0.003,
    'volume_ratio': 1.45,
    'btc_return_lag0': 0.012,
    'yield_change_5d': 0.02,
    'vix_level': 14.5,
    # ... 30 more features
    
    # From research Layers 5-7 (NEW):
    'pre_catalyst_signal': 0.71,  # 24-48hrs ahead
    'news_sentiment': 0.15,  # +15% mentions
    'sector_sharpe': 0.82,  # AI sector historically strong
    'confidence_calibration': 0.92,  # Confidence adjustment factor
}

# OUTPUT (what forecaster must produce):
{
    'ticker': 'NVDA',
    'forecast_direction': 'UP',  # or CHOP, DOWN
    'forecast_confidence': 0.72,  # 72% confident (calibrated to real win-rate)
    'forecast_return': 0.052,  # +5.2% expected 5-15d return
    'optimal_horizon': 8,  # days to hold
    'optimal_position_size': 0.025,  # 2.5% of portfolio
    'optimal_stop_loss': -0.035,  # -3.5% (sector-specific barrier)
    'optimal_take_profit': 0.075,  # +7.5% (risk-reward 1:2.1)
    
    # Reasoning (explainability):
    'top_signals': [
        ('ELLIOTT_5_WAVE', 0.68, 'pattern detector says impulse wave'),
        ('PRE_BREAKOUT_4_5', 0.65, 'microstructure: 4/5 features fired'),
        ('BULL_LOW_VOL', 0.72, '12-regime favorable'),
        ('SECTOR_SHARPE_0.82', 0.70, 'AI sector historically strong'),
        ('PRE_CATALYST_24H', 0.60, 'options OI acceleration suggests event'),
    ]
}
```

### 3.2 Current Gap (Why ~34% accuracy)

```
Missing Links:
1. No connection between pattern + regime + research features
   - Pattern says BUY, regime says TREND_UP, but cross-asset says BTC falling
   - Forecaster needs to integrate all three

2. No pre-catalyst timing
   - Pattern predicts 5-day move, but catalyst happens in 2 days
   - Forecaster should compress horizon to 2-3 days

3. No sector-specific calibration
   - Same confidence threshold works for AI (0.82 Sharpe) but not quantum (0.68 Sharpe)
   - Position sizing should differ

4. No confidence calibration
   - Model outputs 70% confidence but only 55% actual win-rate
   - Need meta-confidence adjustment

5. No cross-asset context
   - BTC leading tech by 6-24hrs, yields leading rotation by 3-5d
   - Forecaster should weight these signals differently

Result: System is "right direction" but missing the synthesis layer that:
- Combines all inputs intelligently
- Adjusts for regime/sector/catalyst
- Calibrates confidence to real accuracy
```

---

## 🔄 PART 4: INTEGRATION ARCHITECTURE

### 4.1 Proposed Forecaster Structure

```python
class IntegratedForecaster:
    def __init__(self):
        self.pattern_detector = PatternDetector()  # (existing)
        self.regime_detector = RegimeDetector()     # (existing)
        self.research_features = ResearchFeatures() # (new - 9 layers)
        self.meta_learner = MetaLearner()          # (new - combines all)
        self.confidence_calibrator = CalibrationModel()  # (new)
        
    def forecast(self, ticker, data_dict):
        # 1. Get existing signals
        pattern_sig = self.pattern_detector.analyze(ticker, data_dict[ticker])
        regime_sig = self.regime_detector.analyze(ticker, data_dict[ticker])
        
        # 2. Get research features (all 40-60)
        features = self.research_features.calculate_all(ticker, data_dict)
        
        # 3. Combine into single forecast
        raw_forecast = self.meta_learner.combine(
            pattern_sig,
            regime_sig,
            features,
            current_regime_state=regime_sig['regime_12'],
            current_sector=self.sector_map[ticker],
        )
        
        # 4. Calibrate confidence to real accuracy
        calibrated_confidence = self.confidence_calibrator.calibrate(
            raw_forecast['confidence'],
            regime_state=regime_sig['regime_12'],
            sector=self.sector_map[ticker],
        )
        
        # 5. Size position based on calibrated confidence
        position_size = self.calculate_position_size(
            calibrated_confidence,
            sector_sharpe=self.sector_sharpe_map[ticker],
            current_regime=regime_sig['regime_12'],
        )
        
        return {
            'direction': raw_forecast['direction'],
            'confidence': calibrated_confidence,
            'position_size': position_size,
            'horizon': raw_forecast['horizon'],
            'stop_loss': raw_forecast['stop_loss'],
            'take_profit': raw_forecast['take_profit'],
        }
```

### 4.2 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                      MARKET DATA (Daily Close)                       │
└─────────────────────────────────────────────────────────────────────┘
           │
           ├─────────────────────────────────┬──────────────────────┬─────────────────┐
           │                                 │                      │                 │
           ▼                                 ▼                      ▼                 ▼
    ┌──────────────┐             ┌──────────────────┐      ┌──────────────────┐  ┌──────────────────┐
    │   Pattern    │             │   Regime (ADX)   │      │  Research        │  │  Historical      │
    │   Detector   │             │   12-Regime      │      │  Features        │  │  Performance     │
    │              │             │   VIX/Breadth    │      │  (40-60)         │  │  (Sharpe/WinRate)│
    │ EXISTING     │             │   EXISTING       │      │  NEW (9 layers)  │  │  NEW             │
    │              │             │                  │      │                  │  │                  │
    │ Output:      │             │ Output:          │      │ Output:          │  │ Output:          │
    │ -direction   │             │ -regime_state    │      │ -dark_pool_ratio │  │ -sector_sharpe   │
    │ -confidence  │             │ -adc_value       │      │ -pre_catalyst    │  │ -win_rates_per   │
    │ -type        │             │ -optimal_ema     │      │ -breadth         │  │  regime          │
    └──────┬───────┘             └────────┬─────────┘      └────────┬─────────┘  └────────┬─────────┘
           │                              │                        │                      │
           └──────────────────────────────┼────────────────────────┼──────────────────────┘
                                          │                        │
                                          ▼                        ▼
                                    ┌──────────────────────────────────────────┐
                                    │     META-LEARNER (NEW COMPONENT)         │
                                    │                                          │
                                    │  Combines:                               │
                                    │  - Pattern signals (40%)                 │
                                    │  - Regime signals (30%)                  │
                                    │  - Research features (20%)               │
                                    │  - Cross-asset context (10%)             │
                                    │                                          │
                                    │  Adjusts weights by:                     │
                                    │  - Current regime (BULL/BEAR/CHOP)       │
                                    │  - Sector (AI/QUANTUM/ROBOTAXI)         │
                                    │  - Time to catalyst (hours/days)         │
                                    │                                          │
                                    │  Output: raw_forecast (direction +       │
                                    │  confidence 0-100%)                      │
                                    └────────────────┬─────────────────────────┘
                                                    │
                                                    ▼
                                    ┌──────────────────────────────────────────┐
                                    │  CONFIDENCE CALIBRATOR (NEW COMPONENT)   │
                                    │                                          │
                                    │  Maps predicted confidence to actual     │
                                    │  win-rate using meta-model calibration   │
                                    │                                          │
                                    │  Example:                                │
                                    │  Predicted: 72% → Actual: 70% ±2%      │
                                    │  (Calibration curve per regime)          │
                                    │                                          │
                                    │  Output: calibrated_confidence           │
                                    └────────────────┬─────────────────────────┘
                                                    │
                                                    ▼
                                    ┌──────────────────────────────────────────┐
                                    │  POSITION SIZER (NEW COMPONENT)          │
                                    │                                          │
                                    │  Formula:                                │
                                    │  position_size = (sector_allocation *    │
                                    │    (confidence / 0.70) *                │
                                    │    vol_adjustment * regime_adjustment)   │
                                    │                                          │
                                    │  Output: optimal_position_pct (0-5%)    │
                                    └────────────────┬─────────────────────────┘
                                                    │
                                                    ▼
                                    ┌──────────────────────────────────────────┐
                                    │    FINAL INTEGRATED FORECAST             │
                                    │                                          │
                                    │  {                                       │
                                    │    direction: 'UP/CHOP/DOWN',           │
                                    │    confidence: 72%,                      │
                                    │    position_size: 2.5%,                 │
                                    │    horizon: 8 days,                      │
                                    │    stop_loss: -3.5%,                    │
                                    │    take_profit: +7.5%,                  │
                                    │  }                                       │
                                    │                                          │
                                    │  Ready for execution!                    │
                                    └──────────────────────────────────────────┘
```

### 4.3 Module Dependencies

```
Existing Modules (No Changes):
├── pattern_detector.py
├── ai_recommender_adv.py (or regime detector)
├── backtest_engine.py
└── meta_learner.py (current, simple weighted average)

NEW Modules to Build:
├── research_features.py
│   ├── Layer 1: dark_pool, ah_volume, supply_chain, breadth, cross_asset
│   ├── Layer 2: pre_breakout, 12_regime, avwap, ema_stacks
│   ├── Layer 3: adaptive_horizon
│   ├── Layer 4: feature_engineering (40-60 total)
│   ├── Layer 5: pre_catalyst, news_tiers
│   ├── Layer 6: breakpoint_detection
│   └── Layer 7: sector_sharpe, confidence_calibrator
│
├── integrated_meta_learner.py
│   ├── combine_pattern_regime_research()
│   ├── weight_by_regime()
│   ├── weight_by_sector()
│   └── weight_by_catalyst_proximity()
│
├── confidence_calibrator.py
│   ├── build_calibration_curves()
│   ├── calibrate_confidence()
│   └── evaluate_calibration()
│
├── position_sizer.py
│   ├── calculate_position_size()
│   ├── calculate_stop_loss()
│   └── calculate_take_profit()
│
└── integrated_forecaster.py (main entry point)
    └── forecast(ticker, data_dict) → final_signal

UPDATED Modules:
├── backtest_engine.py
│   └── Integrate forecaster output instead of pattern+regime separately
│
└── meta_learner.py
    └── Replace simple weighted average with ensemble approach
```

---

## 📝 PART 5: IMPLEMENTATION ROADMAP

### Phase 1: Research (Tomorrow - Perplexity)
- [ ] Ask 25 final questions (from FINAL_PERPLEXITY_QUESTIONS_FORECASTER_INTEGRATION.md)
- [ ] Get answers on: meta-learner design, feature selection, integration strategy
- [ ] Document in: PERPLEXITY_FORECASTER_ARCHITECTURE.md

### Phase 2: Build Research Features (Week 1)
- [ ] Implement 40-60 features from 9 layers (copy from INTEGRATION_PLAN Cells 2-4)
- [ ] Test on 100-ticker universe
- [ ] Validate: features calculate in <1s per ticker

### Phase 3: Build Meta-Learner (Week 2)
- [ ] Combine pattern_signal + regime_state + research_features
- [ ] Implement weight matrices (by regime, by sector)
- [ ] Train on 2021-2024 historical data
- [ ] Validate: ensemble wins more than any single signal alone

### Phase 4: Confidence Calibration (Week 2)
- [ ] Build calibration curves per regime
- [ ] Train meta-confidence model
- [ ] Validate: predicted confidence = actual win-rate ±5%

### Phase 5: Backtesting (Week 3)
- [ ] Run walk-forward validation (train 2021-2023, test 2024-2025)
- [ ] Calculate Sharpe ratio per sector, per regime
- [ ] Compare to baseline (pattern + regime only)
- [ ] Target: 55-65% accuracy, Sharpe >0.5

### Phase 6: Paper Trading (Week 4)
- [ ] Deploy integrated forecaster on live data
- [ ] Track: predicted confidence vs actual win-rate
- [ ] Track: position sizes vs realized returns
- [ ] Adjust calibration if needed

### Phase 7: Live (Week 5+)
- [ ] Start with 1% of capital
- [ ] Monitor for regime changes, drift, anomalies
- [ ] Scale to 5% after 1 month if metrics hold
- [ ] Scale to 10%+ after 1 quarter if sustained

---

## ✅ CHECKLIST: Ready for Tomorrow?

- [x] All 9 discovery layers documented with code
- [x] 100+ ticker universe built with liquidity filters
- [x] Integration plan sketched (Cells 1-6)
- [x] Final 25 Perplexity questions formulated
- [x] Current system architecture mapped (this document)
- [x] Data flow diagram created (shows integration points)
- [x] Module dependencies documented
- [x] Implementation roadmap defined
- [ ] **TOMORROW**: Ask Perplexity 25 questions
- [ ] **WEEK 1**: Build research_features.py
- [ ] **WEEK 2**: Build integrated_meta_learner.py + confidence_calibrator.py
- [ ] **WEEK 3**: Backtest on 2024-2025 data
- [ ] **WEEK 4**: Paper trade + live deployment prep

---

## 🎯 Success Criteria

| Metric | Current | Target | Timeline |
|--------|---------|--------|----------|
| Accuracy (5-15d) | 34% | 55-65% | By week 3 |
| Sharpe Ratio | 0.25-0.35 | >0.5 | By week 3 |
| Data Cost/Month | Unknown | $0 | Now ✓ |
| Ticker Universe | 50 | 100+ | By week 1 |
| Signals Combined | 2 (pattern+regime) | 5 (+ research + cross-asset + catalyst) | By week 2 |
| Position Sizes | Manual | Automated (confidence+Sharpe+regime) | By week 2 |
| Confidence Calibrated | No | Yes (±5% error) | By week 3 |
| Regime Granularity | 2 (ADX) | 12 (VIX+breadth+ATR) | By week 1 |

---

**Next Step**: Tomorrow morning, open Perplexity and ask the 25 questions from FINAL_PERPLEXITY_QUESTIONS_FORECASTER_INTEGRATION.md

You're ready! 🚀

