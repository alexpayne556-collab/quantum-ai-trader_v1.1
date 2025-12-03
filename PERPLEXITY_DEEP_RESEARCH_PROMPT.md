# 🔬 Perplexity Deep Research Prompt: Advanced ML/AI Stock Prediction

Copy and paste this entire prompt into Perplexity Pro (or Claude/GPT-4 with web search) for cutting-edge research on improving swing trading prediction accuracy.

---

## Research Query

**I'm building an ML-based swing trading system targeting 60-70%+ accuracy on 5-day forward BUY/SELL/HOLD predictions. Current accuracy is ~43% with XGBoost/LightGBM ensembles using 150+ technical features including EMA ribbons, Fibonacci retracements, candlestick patterns, RSI, MACD, volume analysis, and cross-asset correlations (SPY, VIX).**

**Please research and provide specific, actionable guidance from recent academic papers (2022-2025) and quantitative finance research on:**

---

### 1. FEATURE ENGINEERING BEYOND TECHNICAL INDICATORS

- What **alternative data sources** (sentiment, order flow, options flow, dark pool activity) have shown predictive power in peer-reviewed research?
- Which **market microstructure features** (bid-ask spread dynamics, trade imbalance, volume-weighted price levels) improve short-term predictions?
- How do **regime-aware features** (volatility regimes, trend strength states, market correlation regimes) enhance model performance?
- What **cross-sectional features** (sector momentum, industry rotation signals, factor exposures) add predictive value?
- Are there **time-series transformations** (wavelet decomposition, Fourier features, Hilbert transform) that capture patterns missed by traditional indicators?

---

### 2. MODEL ARCHITECTURES FOR FINANCIAL TIME SERIES

- What are the **state-of-the-art architectures** for stock prediction as of 2024-2025?
  - Temporal Fusion Transformers (TFT)
  - Informer / Autoformer / PatchTST
  - N-BEATS / N-HiTS
  - TabNet for tabular financial data
  - Temporal Convolutional Networks (TCN)
- How do **attention mechanisms** specifically help with financial data?
- What **hybrid approaches** (CNN-LSTM, Transformer-GBM ensembles) have shown success?
- How should **multi-task learning** (predicting direction + magnitude + volatility jointly) be implemented?
- What **ensemble strategies** beyond simple averaging work best? (stacking, blending, snapshot ensembles)

---

### 3. TRAINING TECHNIQUES FOR NOISY FINANCIAL DATA

- What **loss functions** are optimal for imbalanced BUY/SELL/HOLD classification?
  - Focal loss, asymmetric loss, profit-weighted loss
- How should **label smoothing** and **confidence calibration** be applied?
- What **data augmentation** techniques work for financial time series?
  - Time warping, magnitude warping, jittering, mixup
- How do you implement **walk-forward optimization** and **purged cross-validation** correctly?
- What **regularization techniques** specifically prevent overfitting on financial data?
  - Temporal dropout, feature dropout schedules, early stopping strategies

---

### 4. HANDLING NON-STATIONARITY AND REGIME CHANGES

- How do papers address **concept drift** in financial markets?
- What **online learning** or **continual learning** approaches maintain accuracy over time?
- How should models **detect and adapt to regime changes** (bull/bear/sideways)?
- What **meta-learning** approaches allow quick adaptation to new market conditions?
- How do you implement **dynamic feature selection** that adapts to changing market conditions?

---

### 5. SPECIFIC RESEARCH PAPERS TO FIND

Please search for and summarize key findings from:

1. **"Deep Learning for Financial Time Series Forecasting: A Survey"** (2023-2024 versions)
2. **Papers on Temporal Fusion Transformers for stock prediction**
3. **Research on combining technical analysis with deep learning**
4. **Studies comparing XGBoost/LightGBM vs neural networks for trading**
5. **Papers on market regime detection and regime-dependent models**
6. **Research on feature importance and selection for financial ML**
7. **Studies on realistic backtesting methodologies (avoiding look-ahead bias)**
8. **Papers on transaction cost-aware model training**

---

### 6. PRACTICAL IMPLEMENTATION QUESTIONS

- What **minimum data requirements** (samples, history length) are needed for reliable training?
- How do you properly **evaluate financial ML models** beyond accuracy?
  - Sharpe ratio of predictions, maximum drawdown, profit factor
- What **feature importance methods** (SHAP, permutation importance) are most reliable for financial features?
- How do you **avoid common pitfalls** (survivorship bias, look-ahead bias, overfitting to specific market regimes)?
- What **inference latency considerations** matter for real-time trading?

---

### 7. SPECIFIC TECHNIQUES TO RESEARCH

Please provide implementation details for:

1. **Quantile regression** for predicting price ranges instead of point estimates
2. **Conformal prediction** for uncertainty quantification in trading signals
3. **Multi-horizon forecasting** (predicting 1-day, 3-day, 5-day, 10-day simultaneously)
4. **Attention-based feature selection** that learns which indicators matter when
5. **Adversarial training** to make models robust to market manipulation patterns
6. **Reinforcement learning** integration for position sizing and exit optimization

---

### 8. BENCHMARK EXPECTATIONS

- What **realistic accuracy targets** should I expect for 5-day swing trading?
- What do **top quantitative funds** report as achievable edge?
- How much improvement can **alternative data** provide over price/volume alone?
- What **Sharpe ratios** are achievable with ML-based systematic strategies?

---

## Format Requested

Please provide:

1. **Summary of key findings** with specific accuracy improvements cited
2. **Ranked list of techniques** by expected impact and implementation difficulty
3. **Code snippets or pseudocode** for top 3 recommended techniques
4. **Links to papers** with arxiv IDs where possible
5. **Warnings about common failures** in financial ML

---

## Current System Context

```
- Models: XGBoost + LightGBM ensemble
- Features: 150+ (RSI, MACD, EMA ribbons, Fibonacci, candlesticks, volume, VIX, SPY correlation)
- Labels: BUY (+2% in 5 days), SELL (-2% in 5 days), HOLD (between)
- Training: 5 years daily data, 10 liquid tickers (SPY, QQQ, AAPL, MSFT, etc.)
- Current accuracy: ~43% (target: 60%+)
- Compute: Google Colab Pro GPU (T4/A100)
```

---

**Please be specific and cite sources. I need actionable techniques I can implement, not general advice.**
