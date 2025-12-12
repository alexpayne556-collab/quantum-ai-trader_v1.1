# 🏆 MASTER RESEARCH FINDINGS - VALIDATED PARAMETERS
## Complete Summary for Training Module Development
**Generated:** December 12, 2025  
**Source:** 48 research documents, 42 JSON files, production codebase analysis

---

## 📊 SECTION 1: PATTERN WIN RATES & OPTIMAL CONDITIONS

### 🥇 TIER SS - LEGENDARY (82%+ WR)

| Pattern | Win Rate | Total PnL | Trades | Optimal Conditions |
|---------|----------|-----------|--------|-------------------|
| **AI:nuclear_dip** | **82.35%** | +$31,667 | 1,700 | RSI < 21, MACD rising, 21d return < -5% |

**Integration:**
```python
# Nuclear Dip Detection
conditions = {
    'rsi_below': 21,          # Much lower than human's 35
    'ret_21d_below': -5,      # 5%+ drawdown in past 21 days
    'macd_rising': True,      # MACD histogram turning positive
    'bounce_min_pct': 8       # Wait for 8% confirmation (vs human's 5%)
}
# Weight: 2.0 (base), 2.5 in bear market, 0.5 in bull
```

---

### 🥈 TIER S - EXCELLENT (71%+ WR)

| Pattern | Win Rate | Total PnL | Trades | Optimal Conditions |
|---------|----------|-----------|--------|-------------------|
| **H:ribbon_mom** | **71.43%** | +$14,630 | 1,400 | EMA 8>13>21, MACD+, momentum>5% |
| **H:dip_buy** | **71.43%** | +$12,326 | 700 | RSI<35, 21d return<-8%, vol>0.8x |

**Integration:**
```python
# Ribbon Momentum Detection
ribbon_mom_conditions = {
    'ema_8_above_13': True,
    'ema_13_above_21': True,
    'macd_hist_above': 0,
    'momentum_5d_min': 5.0  # 5%+ momentum
}
# Weight: 1.8 (base), 2.0 in bull, 1.0 in bear

# Dip Buy Detection
dip_buy_conditions = {
    'rsi_below': 35,
    'returns_21d_below': -8,  # 8%+ pullback
    'volume_ratio_above': 0.8
}
exit_conditions = {
    'rsi_above': 60,
    'profit_target': 8,
    'stop_loss': -5
}
```

---

### 🥉 TIER A - GOOD (65-71% WR)

| Pattern | Win Rate | Total PnL | Trades | Optimal Conditions |
|---------|----------|-----------|--------|-------------------|
| **H:bounce** | **66.10%** | +$65,225 | 5,900 | RSI<40, MACD+, BB_pct<0.3 |
| **AI:quantum_mom** | **65.63%** | +$36,657 | 3,200 | Momentum aligned, volume surge |
| **AI:trend_cont** | **62.96%** | +$20,838 | 2,700 | Trend continuation signals |

---

### 📋 WINNING PATTERNS FROM `winning_patterns.json` (100% WR - Small Sample)

| Pattern | Trades | Avg PnL | Avg Hold Days | Entry Conditions |
|---------|--------|---------|---------------|------------------|
| **RSI_BOUNCE** | 7 | +14.4% | 4.3 | RSI<40, MACD+, BB_pct<0.3 |
| **DIP_BUY** | 170 | +11.7% | 4.4 | RSI<35, 21d ret<-8%, vol>0.8x |
| **MEAN_REVERSION** | 15 | +11.2% | 6.6 | RSI<45, EMA8>21, ret_21d∈[-5,0] |
| **VOLUME_BREAKOUT** | 96 | +10.7% | 3.7 | Vol>1.5x, MACD+, EMA8>21 |
| **MOMENTUM** | 165 | +10.4% | 5.6 | EMA8>21, MACD+, RSI∈[50,70], trend>0.3 |
| **BB_BOUNCE** | 9 | +9.3% | 9.1 | BB_pct<0.2, RSI<45 |

---

## 📈 SECTION 2: OPTIMAL TECHNICAL INDICATOR THRESHOLDS

### RSI Thresholds (Evolved via Genetic Algorithm - 30 generations)

| Threshold | Human Baseline | Evolved Optimal | Improvement |
|-----------|---------------|-----------------|-------------|
| **Oversold (BUY)** | 35 | **21** | Buy DEEPER dips |
| **Overbought (SELL)** | 70 | **76** | Ride trends longer |

### RSI Period Variations

| Period | Best Use Case | When to Use |
|--------|--------------|-------------|
| RSI(7) | High volatility stocks | 3-8% daily vol (biotech) |
| RSI(14) | Standard stocks | 1-3% daily vol |
| RSI(21) | Low volatility | <1% daily vol |
| RSI(50) | Long-term trend | Weekly analysis |

### MACD Settings
```python
# Standard (works well)
macd_fast = 12
macd_slow = 26
macd_signal = 9

# Key Signals:
# - MACD histogram > 0 = bullish momentum
# - MACD histogram rising = acceleration
# - MACD line cross above signal = buy trigger
```

### EMA Periods

| EMA | Purpose | Signal |
|-----|---------|--------|
| **EMA 8** | Short-term momentum | Price above = bullish |
| **EMA 13** | Intermediate trend | Ribbon component |
| **EMA 21** | Primary trend | Key pullback support |
| **EMA 34** | Fibonacci alignment | Confluence zone |
| **EMA 55** | Longer trend | Strong support |
| **EMA 200** | Major trend | Bull/bear boundary |

### Optimal Ribbon Signals
```python
# BULLISH RIBBON (70%+ WR when confirmed):
ema_8 > ema_13 > ema_21 > ema_34 > ema_55

# COMPRESSION (Breakout Setup):
ribbon_width < 2%  # All EMAs within 2% of each other

# Golden Cross Strength:
golden_cross = (ema_50 - ema_200) / ema_200  # Positive = bullish
```

### Bollinger Bands

| BB Position | Signal | Win Rate |
|-------------|--------|----------|
| **BB_pct < 0.2** | Oversold bounce setup | 64% |
| **BB_pct < 0.3** | Combined with RSI<40 | 68% |
| **BB_pct > 0.8** | Overbought, take profits | N/A |

### Volume Thresholds

| Ratio | Signal | Action |
|-------|--------|--------|
| **Vol > 2.5x avg** | Breakout confirmation | Enter |
| **Vol > 1.5x avg** | Volume surge | Confirm pattern |
| **Vol < 0.8x avg** | Exit signal (volume dying) | Close position |

---

## 💰 SECTION 3: RISK MANAGEMENT PARAMETERS

### Position Sizing (Evolved Config - 71.1% WR)

| Parameter | Human Baseline | Evolved Optimal | Rationale |
|-----------|---------------|-----------------|-----------|
| **Position Size** | 15% | **21%** | Larger positions with edge |
| **Max Positions** | 10 | **11** | More diversification |
| **Max Single Position** | 20% | **25%** | Concentration when confident |
| **Max Sector Exposure** | 30% | **40%** | Allow sector momentum |
| **Cash Reserve** | 30% | **20%** | Deploy more capital |

### Stop Loss & Take Profit (Evolved)

| Parameter | Human Baseline | Evolved Optimal | Insight |
|-----------|---------------|-----------------|---------|
| **Stop Loss** | -12% | **-19%** | Wider stops let winners run |
| **Profit Target 1** | 8% | **14%** | Higher first target |
| **Profit Target 2** | 15% | **25%** | Much higher second target |
| **Trailing Stop** | 8% | **11%** | Tighter trail to lock profits |
| **Max Hold Days** | 60 | **32** | Faster turnover |

### Regime-Specific Exit Parameters (Tested on 2,649 trades)

| Signal | Regime | Take Profit | Stop Loss | Win Rate |
|--------|--------|-------------|-----------|----------|
| **dip_buy** | bear | 10% | -10% | **84%** |
| **momentum** | bear | 5% | -10% | **88%** |
| **bounce** | bear | 8% | -10% | **75%** |
| **trend** | bull | 15% | -8% | 68% |
| **dip_buy** | bull | 12% | -8% | 70% |
| **bounce** | sideways | 15% | -10% | 64% |

### Kelly Criterion Adjustments
```python
# Base Kelly calculation
edge = win_rate - 0.50  # Example: 0.69 - 0.50 = 0.19
kelly_pct = edge / 0.19  # ~100% (too aggressive)

# Fractional Kelly (recommended)
half_kelly = kelly_pct * 0.5  # 50%
quarter_kelly = kelly_pct * 0.25  # 25%

# Production recommendation: 2% risk per trade for $10k account
# This is conservative but survivable
```

### Volatility-Adjusted Position Sizing
```python
# Target volatility approach (AQR-style)
target_vol = 0.15  # 15% annualized
realized_vol = current_ticker_volatility  # e.g., 0.30 for high-vol stock

position_size = base_size * (target_vol / realized_vol)
# High-vol stock: position_size = 0.21 * (0.15 / 0.30) = 10.5%
# Low-vol stock: position_size = 0.21 * (0.15 / 0.10) = 31.5% (cap at max)
```

---

## 🎯 SECTION 4: FEATURE IMPORTANCE RANKINGS

### Top 50 Features (from SHAP Analysis)

**Tier 1 - Critical (Weight: 1.5+)**
1. `Dist_to_Fib_0_786` - Distance to 78.6% Fibonacci
2. `Dist_to_Fib_0_236` - Distance to 23.6% Fibonacci
3. `Dist_to_FibExt_1_272` - Fibonacci extension
4. `RSI_7` - Short-term RSI
5. `Range` - Daily range
6. `EMA_8_Slope` - Short EMA momentum
7. `Price_vs_EMA_8` - Price position vs fast EMA
8. `MACD_Hist` - Histogram value

**Tier 2 - Important (Weight: 1.2-1.5)**
9. `Near_Fib_0_382` - Proximity to golden pocket
10. `RSI_14` - Standard RSI
11. `Beta_SPY` - Market correlation
12. `BB_Width` - Volatility measure
13. `Stoch_D` - Stochastic
14. `Stoch_K` - Stochastic
15. `Correlation_SPY` - Market correlation

**Tier 3 - Useful (Weight: 1.0-1.2)**
16. `BB_Upper` - Upper band level
17. `RSI_50` - Long-term RSI
18. `Plus_DI` - Directional movement
19. `VIX_Level` - Market fear
20. `Near_Fib_0_5` - Midpoint proximity
21. `RSI_14_Percentile_90d` - RSI context
22. `EMA_Ribbon_Compression` - Breakout setup
23. `OBV` - On-Balance Volume
24. `Volume_MA_20` - Volume trend
25. `ATR` - Volatility

### Interaction Features (SHAP Discovered)
```python
# High-value interactions (RenTec-style)
'rsi_x_volume' = RSI * relative_volume  # Explosive detector
'trend_x_vol' = trend_strength * volatility  # Context feature
'price_vol_interaction' = price_change * volume_change

# Implementation:
features['RSI_x_Volume'] = features['RSI_14'] * features['Volume_Ratio']
features['Trend_x_Vol'] = features['Trend_Alignment'] * features['Volatility']
```

### Feature Weights by Category
```python
FEATURE_CATEGORY_WEIGHTS = {
    'percentile_features': {
        'window': 90,
        'weight': 1.2  # Rolling percentiles add context
    },
    'second_order_features': {
        'rsi_momentum_weight': 1.1,
        'volume_acceleration_weight': 1.15
    },
    'cross_asset_features': {
        'spy_correlation_weight': 1.25,
        'vix_level_weight': 1.3
    },
    'interaction_terms': {
        'rsi_x_volume_weight': 1.1,
        'trend_x_volatility_weight': 1.15
    }
}
```

---

## 🌡️ SECTION 5: MARKET REGIME DETECTION

### Regime Classification Thresholds

| Regime | VIX Range | SPY 20d Return | Characteristics |
|--------|-----------|----------------|-----------------|
| **BULL_LOW_VOL** | < 15 | > +3% | Steady uptrend |
| **BULL_MOD_VOL** | 15-20 | > +3% | Volatile uptrend |
| **BULL_HIGH_VOL** | 20-30 | > +3% | Choppy bull |
| **BEAR_LOW_VOL** | < 15 | < -3% | Quiet decline |
| **BEAR_MOD_VOL** | 15-20 | < -3% | Standard selloff |
| **BEAR_HIGH_VOL** | 20-30 | < -3% | Fear selling |
| **CHOPPY_LOW_VOL** | < 15 | -3% to +3% | Range-bound |
| **CHOPPY_MOD_VOL** | 15-20 | -3% to +3% | Directionless |
| **PANIC** | > 30 | Any | Extreme fear |
| **EUPHORIA** | < 12 | > +10% | Extreme greed |

### Regime Detection Implementation
```python
def classify_regime(vix, spy_ret_20d, ribbon_bullish):
    """Classify current market regime"""
    
    # Special regimes first
    if vix > 30:
        return 'PANIC'
    if spy_ret_20d > 10 and vix < 12:
        return 'EUPHORIA'
    
    # Standard regime classification
    if spy_ret_20d > 3:
        trend = 'BULL'
    elif spy_ret_20d < -3:
        trend = 'BEAR'
    else:
        trend = 'CHOPPY'
    
    # Volatility level
    if vix < 15:
        vol = 'LOW_VOL'
    elif vix < 20:
        vol = 'MOD_VOL'
    else:
        vol = 'HIGH_VOL'
    
    return f"{trend}_{vol}"
```

### Simple Regime Classification (for signal switching)
```python
def classify_regime_simple(ret_21d, ribbon_bullish):
    """Simple 3-regime classifier for signal weights"""
    if ret_21d > 5 and ribbon_bullish:
        return 'bull'
    elif ret_21d < -5 and not ribbon_bullish:
        return 'bear'
    else:
        return 'sideways'
```

### Regime-Specific Position Limits

| Regime | Max Positions | Position Size | Cash Reserve |
|--------|---------------|---------------|--------------|
| BULL | 5-7 | 21% | 20% |
| BEAR | 1-2 | 10% | 50% |
| SIDEWAYS | 3-4 | 15% | 30% |
| PANIC | 0 | 0% | 100% |
| EUPHORIA | 3 | 15% | 40% (expect reversal) |

---

## 📊 SECTION 6: MULTI-TIMEFRAME ANALYSIS

### Timeframe Weights
```python
TIMEFRAME_WEIGHTS = {
    '1d': 0.40,   # Daily - primary signal
    '4h': 0.30,   # 4-hour - confirmation
    '1h': 0.20,   # Hourly - entry timing
    '15m': 0.10   # 15-min - precision entry (optional)
}
```

### Confluence Requirements
```python
CONFLUENCE_PARAMETERS = {
    'min_timeframe_agreement': 2,  # At least 2 timeframes must agree
    'confidence_cap': 0.85,        # Never exceed 85% confidence
    
    'rsi_boost': {
        'oversold_threshold': 30,
        'overbought_threshold': 70,
        'boost_multiplier': 1.15
    },
    'volume_boost': {
        'high_volume_threshold': 1.5,  # 1.5x average
        'boost_multiplier': 1.1
    },
    'regime_boost': {
        'strong_trend_threshold': 0.7,
        'boost_multiplier': 1.2
    }
}
```

### Fractal Market Alignment (Bridgewater-style)
```python
# Trade only when multiple timeframes align
def check_timeframe_alignment(ticker):
    alignments = []
    for tf in ['5min', '15min', '1hr', 'daily']:
        trend = get_trend(ticker, tf)
        alignments.append(trend)
    
    # All must agree for high-confidence trade
    if all(t == 'UP' for t in alignments):
        return 'STRONG_BUY'
    elif all(t == 'DOWN' for t in alignments):
        return 'STRONG_SELL'
    else:
        return 'WAIT'
```

---

## 🏦 SECTION 7: DARK POOL / INSTITUTIONAL FLOW SIGNALS

### Smart Money Index (SMI) - Composite Score
```python
# Weights for SMI calculation
SMI_WEIGHTS = {
    'IFI': 0.30,   # Institutional Flow Index (30%)
    'A_D': 0.25,   # Accumulation/Distribution (25%)
    'OBV': 0.25,   # On-Balance Volume (25%)
    'VROC': 0.20   # Volume Rate of Change (20%)
}

# Signal thresholds
SMI_THRESHOLDS = {
    'STRONG_BUY': 70,
    'BUY': 60,
    'NEUTRAL_HIGH': 60,
    'NEUTRAL_LOW': 40,
    'SELL': 40,
    'STRONG_SELL': 30
}
```

### Institutional Flow Index (IFI)
```python
def calculate_ifi(df, lookback=7):
    """
    Detect institutional block trades from OHLCV data
    Uses 90th percentile volume as proxy for large trades
    """
    large_vol_threshold = df['Volume'].quantile(0.90)
    large_trades = df[df['Volume'] > large_vol_threshold]
    
    # Buy volume: close > open
    buy_vol = large_trades[large_trades['Close'] > large_trades['Open']]['Volume'].sum()
    # Sell volume: close < open
    sell_vol = large_trades[large_trades['Close'] < large_trades['Open']]['Volume'].sum()
    
    total = buy_vol + sell_vol
    ifi = (buy_vol - sell_vol) / total if total > 0 else 0
    
    # Score: IFI of 0.3 = 100, -0.3 = 0
    score = (ifi / 0.3 + 1) * 50
    return min(100, max(0, score))
```

### Order Flow Imbalance (OFI) - Proxy
```python
def calculate_ofi_proxy(df):
    """
    Close Location Value: proxy for order flow without L2 data
    +1 = close at high (buying pressure)
    -1 = close at low (selling pressure)
    """
    ofi = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    return ofi.fillna(0)
```

### Microstructure Features (FREE - from OHLCV only)
```python
MICROSTRUCTURE_FEATURES = {
    'spread_proxy': "(High - Low) / Close",
    # Wider spreads = institutional block trades
    
    'order_flow_clv': "((Close-Low) - (High-Close)) / (High-Low)",
    # Close Location Value: +1=buying, -1=selling
    
    'institutional_activity': "Volume / abs(Close - Open)",
    # High volume + small candle = dark pool accumulation
}
```

---

## 🧬 SECTION 8: EDGE DISCOVERIES

### 1. Buy Deeper Dips Than Humans
- **Human threshold:** RSI < 35
- **AI optimal:** RSI < 21
- **Improvement:** +10.2% win rate

### 2. Wider Stops Let Winners Run
- **Human stop:** -12%
- **AI optimal:** -19%
- **Result:** Fewer stopped out, higher avg profit

### 3. Faster Turnover Is Better
- **Human max hold:** 60 days
- **AI optimal:** 32 days
- **Result:** Faster capital rotation, more opportunities

### 4. Wait for Bigger Bounces
- **Human bounce threshold:** 5%
- **AI optimal:** 8%
- **Result:** Better confirmation, fewer false signals

### 5. Signal Decay (RenTec-style)
```python
# Kill stale signals after 30 minutes
signal_age_minutes = (now - signal_time).total_seconds() / 60
if signal_age_minutes > 30:
    signal_strength *= 0.5 ** (signal_age_minutes / 30)
```

### 6. Round Number Psychology
```python
# Price behavior near $10, $50, $100, $500 levels
dist_to_round = abs(price % 10)
if dist_to_round < 0.5:  # Within 50 cents of round number
    resistance_probability = 'HIGH'
```

### 7. Meta-Labeling (Two-Stage Prediction)
```python
# Model 1: Predict direction (up/down)
# Model 2: Predict if Model 1 will be correct
# Trade only when Model 2 confidence > 70%
```

### 8. Adversarial Regime Detection
```python
# Train classifier: current_data vs historical_data
# If classifier accuracy > 70% → REGIME SHIFT → STOP TRADING
# This detects distribution shifts automatically
```

---

## 🏗️ SECTION 9: MODEL ARCHITECTURE RECOMMENDATIONS

### Production Ensemble (69.42% Validated)
```python
ENSEMBLE_CONFIG = {
    'models': {
        'XGBoost': {
            'weight': 0.3576,
            'params': {
                'max_depth': 9,
                'learning_rate': 0.23,
                'n_estimators': 308,
                'subsample': 0.68,
                'colsample_bytree': 0.98,
                'min_child_weight': 5,
                'gamma': 0.17,
                'reg_alpha': 2.63,
                'reg_lambda': 5.60
            }
        },
        'LightGBM': {
            'weight': 0.2701,
            'params': {
                'num_leaves': 187,
                'max_depth': 12,
                'learning_rate': 0.14,
                'n_estimators': 300,
                'subsample': 0.74,
                'colsample_bytree': 0.89,
                'min_child_samples': 21,
                'reg_alpha': 1.36,
                'reg_lambda': 0.004
            }
        },
        'HistGradientBoosting': {
            'weight': 0.3723,
            'params': {
                'max_iter': 492,
                'max_depth': 9,
                'learning_rate': 0.27,
                'min_samples_leaf': 13,
                'l2_regularization': 2.01
            }
        }
    }
}
```

### Meta-Learner Architecture (Stacking)
```python
# Level 1: Specialized Models
L1_MODELS = {
    'pattern_model': 'LogisticRegression',  # Linear, interpretable
    'research_model': 'XGBoost',            # 60 features, non-linear
    'darkpool_model': 'XGBoost'             # Institutional flow
}

# Level 2: Meta-Learner
META_LEARNER = {
    'model': 'XGBoost',
    'max_depth': 2,  # Constrained to prevent overfitting
    'inputs': ['L1_probabilities', 'regime_indicators']
}
```

---

## 📋 SECTION 10: TRAINING COMMANDMENTS

### The 10 Commandments of Training

1. **PurgedKFold Cross-Validation** - Prevents lookahead bias (5-15% overfitting reduction)
2. **GPU Optimization** - `tree_method='gpu_hist'` for 20-40x speedup
3. **Two-Phase Learning Rate** - Fast (0.1) then fine-tune (0.01)
4. **Hyperparameter Priority** - lr → max_depth → subsample → colsample → min_child
5. **Class Weights NOT SMOTE** - Use sample weights, SMOTE can break time series
6. **SHAP Feature Selection** - Keep features with SHAP > 0.001, remove bottom 20%
7. **Trident Ensemble** - XGBoost + LightGBM + CatBoost (2-5% boost)
8. **Early Stopping** - n_estimators via early_stopping_rounds=50
9. **Cluster-Specific Models** - 5-15% accuracy boost per cluster
10. **Train/Val/Test within 5%** - If gap > 5%, you're overfitting

### 5 Advanced Weapons (90%+ WR Potential)

1. **NGBoost** - Probabilistic predictions with uncertainty bands
2. **Adversarial Validation** - Detect regime shifts before trading
3. **Concept Drift (ADWIN)** - Real-time monitoring
4. **Stacking + Meta-Learner** - 5-10% Sharpe boost
5. **SHAP Interactions** - Find hidden feature synergies

---

## 🎯 QUICK REFERENCE: ALL PARAMETERS IN ONE PLACE

### Entry Parameters
```python
ENTRY_PARAMS = {
    'rsi_oversold': 21,
    'rsi_overbought': 76,
    'momentum_min_pct': 4,
    'bounce_min_pct': 8,
    'drawdown_trigger_pct': -6,
    'volume_surge_threshold': 1.5,
    'volume_breakout_threshold': 2.5,
    'ema_ribbon_compression': 0.02,  # 2% width = breakout setup
}
```

### Exit Parameters
```python
EXIT_PARAMS = {
    'profit_target_1': 14,
    'profit_target_2': 25,
    'stop_loss': -19,
    'trailing_stop': 11,
    'max_hold_days': 32,
    'signal_decay_halflife_mins': 30,
}
```

### Position Sizing
```python
POSITION_PARAMS = {
    'base_position_size': 0.21,  # 21% of portfolio
    'max_positions': 11,
    'max_single_position': 0.25,  # 25%
    'max_sector_exposure': 0.40,  # 40%
    'cash_reserve_min': 0.20,     # 20%
    'target_volatility': 0.15,    # 15% annualized
}
```

### Signal Weights by Regime
```python
SIGNAL_WEIGHTS = {
    'bull': {
        'nuclear_dip': 0.5, 'trend': 2.0, 'ribbon_mom': 2.0,
        'rsi_divergence': 0.8, 'dip_buy': 1.2, 'bounce': 1.0, 'momentum': 0.7
    },
    'bear': {
        'nuclear_dip': 2.5, 'trend': 1.5, 'ribbon_mom': 1.0,
        'rsi_divergence': 0.7, 'dip_buy': 1.8, 'bounce': 1.8, 'momentum': 0.5
    },
    'sideways': {
        'nuclear_dip': 1.5, 'trend': 0.0, 'ribbon_mom': 0.8,
        'rsi_divergence': 1.2, 'dip_buy': 2.0, 'bounce': 1.8, 'momentum': 0.3
    }
}
```

---

## ✅ VALIDATION CHECKLIST FOR TRAINING MODULES

- [ ] Use evolved thresholds (RSI 21, not 35)
- [ ] Implement regime-aware signal weights
- [ ] Include all 50+ SHAP-ranked features
- [ ] Add microstructure features (spread_proxy, OFI, institutional_activity)
- [ ] Configure meta-learner stacking
- [ ] Set proper stop loss (-19%) and take profit (14%/25%)
- [ ] Implement signal decay (30-min half-life)
- [ ] Use PurgedKFold for cross-validation
- [ ] Apply timeframe weights (1d: 40%, 4h: 30%, 1h: 20%, 15m: 10%)
- [ ] Configure regime-specific position limits

---

**END OF MASTER RESEARCH FINDINGS**

*This document contains all validated parameters extracted from 48+ research documents in the quantum-ai-trader repository.*
