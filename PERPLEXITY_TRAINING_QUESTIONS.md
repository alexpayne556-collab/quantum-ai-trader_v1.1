# Deep Research Questions for Perplexity Pro
**Goal:** Discover the best training strategies for each module in our Golden Architecture to achieve uncanny, production-grade predictions.

---

## 1. Visual Pattern Engine (GASF-CNN)

### Q1.1 — Optimal CNN Architecture for GASF Images
> We convert stock OHLCV data into Gramian Angular Summation Field (GASF) images and feed them to a CNN for pattern recognition. What are the best CNN architectures (ResNet, EfficientNet, Vision Transformer, custom shallow nets) for classifying 64×64 or 128×128 GASF images? Include recommended layer depths, kernel sizes, and regularization techniques. Provide PyTorch or TensorFlow code snippets for the top 2 architectures.

### Q1.2 — Data Augmentation for Financial Images
> What data augmentation strategies (rotation, scaling, noise injection, mixup, cutout) are effective for GASF images in stock prediction? Are there domain-specific augmentations that preserve temporal structure while increasing training diversity?

### Q1.3 — Transfer Learning from ImageNet vs Training from Scratch
> Should we use ImageNet pretrained weights for financial GASF image classification, or train from scratch? What are the trade-offs and how much data do we need for each approach to converge?

### Q1.4 — Multi-Timeframe Visual Stacking
> How can we stack GASF images from multiple timeframes (daily, 4H, 1H) into multi-channel inputs for the CNN? What's the best channel arrangement and fusion strategy?

---

## 2. Regime Detection (Hidden Markov Model)

### Q2.1 — HMM State Selection
> How do we determine the optimal number of hidden states (2, 3, 4, or more) for an HMM detecting market regimes (bull, bear, sideways, high-volatility)? What metrics (BIC, AIC, log-likelihood, out-of-sample accuracy) should guide this choice?

### Q2.2 — Feature Engineering for Regime Detection
> What input features (returns, volatility, volume ratios, RSI, ATR, VIX correlation) produce the most stable and predictive HMM regimes? Provide a ranked list with empirical evidence.

### Q2.3 — Online vs Batch HMM Training
> Should we retrain the HMM periodically in batch mode or use online/incremental updates as new data arrives? What's the best retraining cadence for daily equity data?

### Q2.4 — Regime-Conditional Model Switching
> Once regimes are detected, how do we best condition downstream models (ensemble classifiers, RL agents) on the current regime? Should we train separate models per regime or use regime as a feature?

---

## 3. Triple-Barrier Labeling

### Q3.1 — Barrier Parameter Tuning
> For triple-barrier labeling with profit-take (pt), stop-loss (sl), and time horizon (max_holding), what's the best methodology to tune these parameters? Should they be volatility-adjusted (ATR-based), fixed percentages, or learned via optimization?

### Q3.2 — Class Imbalance Handling
> Triple-barrier labeling often produces imbalanced classes (many HOLDs). What are the best strategies: SMOTE, class weights, under-sampling, focal loss, or label smoothing? Provide code examples.

### Q3.3 — Meta-Labeling Integration
> How do we implement meta-labeling (Marcos López de Prado's approach) on top of triple-barrier labels to filter low-confidence signals? Step-by-step guide with Python code.

---

## 4. Advanced Feature Engineering

### Q4.1 — Feature Importance and Selection
> We generate 100+ technical indicators. What's the best approach to select the top 20-30 features: mutual information, SHAP values, recursive feature elimination, or L1 regularization? Provide a workflow.

### Q4.2 — Feature Stationarity and Differencing
> How do we ensure features are stationary for ML models? When should we use fractional differencing vs simple returns vs z-scoring? Include code for fractional differencing.

### Q4.3 — Cross-Asset Features
> How do we incorporate cross-asset signals (e.g., SPY momentum, VIX level, sector ETF relative strength) as features for individual stock prediction without data leakage?

---

## 5. Ensemble ML Training (XGBoost, LightGBM, RandomForest)

### Q5.1 — Hyperparameter Tuning Strategy
> What's the most efficient hyperparameter search strategy for XGBoost/LightGBM on financial data: grid search, random search, Bayesian optimization (Optuna), or Hyperband? Provide Optuna code for tuning learning_rate, max_depth, n_estimators, colsample_bytree.

### Q5.2 — Preventing Overfitting on Financial Data
> Financial data is noisy and non-stationary. What regularization, early stopping, and cross-validation strategies best prevent overfitting? Include purged K-fold setup.

### Q5.3 — Stacking and Meta-Learning
> We stack XGBoost, LightGBM, and RandomForest with a LogisticRegression meta-learner. What's the optimal way to generate out-of-fold predictions for the meta-learner? Should the meta-learner use probabilities or hard predictions?

### Q5.4 — Probability Calibration
> How do we calibrate predicted probabilities from gradient boosting models for better decision-making? Compare Platt scaling vs isotonic regression with code.

---

## 6. Symbolic Regression / Genetic Programming (Logic Engine)

### Q6.1 — PySR Configuration for Financial Formulas
> What are the best PySR settings (population_size, generations, operators, complexity penalties) for discovering trading rules from OHLCV data? Provide a working config.

### Q6.2 — Interpretable Alpha Discovery
> How can we use genetic programming to discover interpretable alpha formulas (e.g., `(close - sma_20) / atr_14 > threshold`)? What fitness functions (Sharpe ratio, accuracy, profit factor) work best?

### Q6.3 — Avoiding Overfitting in Symbolic Regression
> Symbolic regression can easily overfit. What strategies (parsimony pressure, validation split, ensemble of formulas) prevent this?

---

## 7. Reinforcement Learning Execution (SAC Agent)

### Q7.1 — Reward Function Design
> What reward functions work best for RL-based trade execution: raw PnL, risk-adjusted returns (Sharpe), differential Sharpe, or shaped rewards with drawdown penalties? Provide implementations.

### Q7.2 — State Space Design
> What should the RL agent's state include: current position, unrealized PnL, technical indicators, regime state, time features? How do we normalize these for stable training?

### Q7.3 — Action Space: Discrete vs Continuous
> Should our trading agent use discrete actions (BUY/SELL/HOLD) or continuous position sizing? What are the trade-offs and which SAC variant handles each better?

### Q7.4 — Sample Efficiency and Replay Buffer
> RL is sample-hungry. What techniques (prioritized experience replay, HER, offline RL with historical data) improve sample efficiency for trading agents?

### Q7.5 — Curriculum Learning for Trading
> Should we train the RL agent on progressively harder market conditions (trending → sideways → volatile)? How to implement curriculum learning?

---

## 8. Validation: Combinatorial Purged Cross-Validation (CPCV)

### Q8.1 — CPCV Implementation
> Provide a complete Python implementation of Combinatorial Purged Cross-Validation with embargo periods for financial time series. Explain the purging and embargo logic.

### Q8.2 — Walk-Forward vs CPCV
> When should we use traditional walk-forward validation vs CPCV? Can we combine them for more robust validation?

### Q8.3 — Statistical Significance Testing
> After CPCV, how do we assess if our model's performance is statistically significant and not due to chance? Include permutation tests and t-tests.

---

## 9. GPU Training Optimization

### Q9.1 — Mixed Precision Training
> How do we enable mixed precision (FP16) training for PyTorch CNNs on T4/A100 GPUs to speed up GASF image training? Provide code.

### Q9.2 — Batch Size and Learning Rate Scaling
> What batch sizes work best for T4 (16GB) vs A100 (40GB) for our CNN? How should we scale learning rate with batch size?

### Q9.3 — Multi-GPU Training
> If using multiple GPUs on Colab Pro, what's the simplest way to parallelize training: DataParallel, DistributedDataParallel, or DeepSpeed?

### Q9.4 — XGBoost/LightGBM GPU Mode
> How do we enable GPU training for XGBoost and LightGBM? What parameters need to change and what speedup can we expect?

---

## 10. End-to-End Training Pipeline

### Q10.1 — Training Order and Dependencies
> In what order should we train our modules: (1) Feature Engineering → (2) Regime Detection → (3) Labeling → (4) Visual CNN → (5) Ensemble ML → (6) Meta-Labeler → (7) RL Agent? Are there circular dependencies?

### Q10.2 — Hyperparameter Orchestration
> How do we orchestrate hyperparameter tuning across multiple modules without combinatorial explosion? Nested optimization, sequential tuning, or joint optimization?

### Q10.3 — Model Versioning and Experiment Tracking
> What's the best lightweight experiment tracking for Colab: MLflow, Weights & Biases, or simple JSON logs? How do we version models with Google Drive?

---

## 11. AI Recommender System

### Q11.1 — Recommendation Model Architecture
> Our AI Recommender uses prediction outputs to rank tickers. What model architecture (learning-to-rank, multi-armed bandit, transformer-based ranker) produces the best daily picks?

### Q11.2 — Incorporating Confidence Scores
> How should the recommender weight ensemble confidence, regime state, and visual pattern strength when ranking tickers? Linear combination, learned weights, or attention mechanism?

### Q11.3 — Portfolio-Level Optimization
> Once we have ranked tickers, how do we construct a portfolio: equal weight top-N, risk-parity, mean-variance optimization with predictions as expected returns?

---

## 12. Self-Discovering AI (AlphaZero-Style Pattern Discovery)

### Q12.1 — AlphaZero for Trading: Is It Possible?
> AlphaGo/AlphaZero learned Go without human knowledge, discovering strategies humans never conceived. Can we apply the same principle to stock trading — letting an AI discover profitable patterns through self-play and reinforcement learning without feeding it human-designed indicators? What architectures (Monte Carlo Tree Search + neural nets, pure RL, evolutionary strategies) work best for financial pattern discovery?

### Q12.2 — Self-Supervised Representation Learning for Markets
> How do we train a model to learn meaningful representations of market states WITHOUT labels? Techniques like contrastive learning (SimCLR), masked autoencoders, or variational autoencoders applied to OHLCV sequences — which discovers the most predictive latent features? Provide PyTorch code for a contrastive learning approach on price windows.

### Q12.3 — Evolutionary Alpha Discovery (Rubik's Cube Approach)
> Can we use genetic algorithms or evolutionary strategies to evolve trading rules from scratch — starting with random operations on OHLCV data and letting survival-of-the-fittest discover profitable formulas humans never imagined? How do we design the fitness function to avoid overfitting while rewarding novel discoveries?

### Q12.4 — Neural Architecture Search for Trading Models
> Instead of hand-designing CNN/LSTM architectures, can we use Neural Architecture Search (NAS) to let the AI discover its own optimal network topology for price prediction? What NAS methods (DARTS, ENAS, random search) are feasible on Colab Pro?

### Q12.5 — Multi-Agent Market Simulation (Self-Play)
> AlphaZero improved by playing against itself. Can we create a market simulation where multiple AI agents trade against each other, forcing them to discover robust strategies through competition? How do we prevent mode collapse where all agents converge to the same strategy?

### Q12.6 — Curiosity-Driven Exploration for New Patterns
> In RL, curiosity-driven exploration (ICM, RND) rewards agents for discovering novel states. Can we apply this to trading — rewarding the AI for finding unusual market conditions that lead to profitable trades? This could discover "black swan" patterns before they happen.

### Q12.7 — Transformer World Models for Market Dynamics
> Can we train a transformer-based world model (like in Dreamer/MuZero) that learns the underlying dynamics of the market, then use it to imagine future scenarios and plan optimal trades? How much historical data is needed to learn a useful market world model?

---

## 13. Achieving "Uncanny" Prediction Accuracy

### Q13.1 — Realistic Accuracy Targets
> What prediction accuracy (or precision/recall for BUY signals) is realistically achievable for daily stock direction with state-of-the-art methods? Cite recent research.

### Q13.2 — Signal Quality vs Quantity Trade-off
> Should we aim for many weak signals or few high-confidence signals? How do we tune the confidence threshold for production?

### Q13.3 — Detecting Model Degradation
> How do we detect when our trained models start degrading due to regime shifts? Online monitoring metrics and automated retraining triggers.

---

## Copy-Paste Ready Prompts for Perplexity Pro

Below are the top 10 questions to paste directly into Perplexity Pro for deep research:

---

**Prompt 1:**
```
We convert stock OHLCV data into 64x64 Gramian Angular Summation Field (GASF) images and classify them with CNNs to predict next-day direction. What are the best CNN architectures (ResNet-18, EfficientNet-B0, custom 4-layer CNN, Vision Transformer) for this task? Provide PyTorch code for the top architecture with recommended hyperparameters for training on a T4 GPU.
```

**Prompt 2:**
```
For Hidden Markov Model (HMM) based market regime detection, how do we determine the optimal number of states (2, 3, or 4)? What features (returns, volatility, volume, RSI) work best as observations? Provide Python code using hmmlearn with model selection via BIC.
```

**Prompt 3:**
```
Explain triple-barrier labeling for ML-based trading strategies. How do we tune the profit-take, stop-loss, and time horizon parameters using ATR-based volatility scaling? Provide Python code for generating labels and handling class imbalance with SMOTE.
```

**Prompt 4:**
```
For XGBoost and LightGBM on financial classification tasks, what's the best hyperparameter tuning strategy using Optuna? Include GPU training mode, early stopping, and purged K-fold cross-validation. Provide complete Python code.
```

**Prompt 5:**
```
How do we implement Combinatorial Purged Cross-Validation (CPCV) with embargo periods for time series ML models? Provide a complete Python implementation following Marcos López de Prado's methodology.
```

**Prompt 6:**
```
For a Soft Actor-Critic (SAC) reinforcement learning agent trading stocks, what reward function design works best: raw PnL, Sharpe ratio, or differential Sharpe? What state features should include? Provide a PyTorch + stable-baselines3 implementation.
```

**Prompt 7:**
```
We use PySR (symbolic regression) to discover interpretable trading rules from price data. What configuration (operators, population size, complexity penalties) discovers robust formulas without overfitting? Provide a working PySR config and example output interpretation.
```

**Prompt 8:**
```
How do we calibrate prediction probabilities from XGBoost for trading decisions? Compare Platt scaling vs isotonic regression with sklearn. When is calibration essential and how do we validate calibration quality?
```

**Prompt 9:**
```
For multi-ticker stock prediction, how do we incorporate cross-asset features (SPY momentum, VIX level, sector ETF strength) without data leakage? Provide a feature engineering pipeline with proper temporal alignment.
```

**Prompt 10:**
```
What's a realistic accuracy target for next-day stock direction prediction using SOTA methods (gradient boosting + CNN + HMM regime)? Cite recent quantitative finance research on achievable precision/recall and how to measure statistical significance of results.
```

**Prompt 11 (AlphaZero for Trading):**
```
AlphaGo/AlphaZero mastered Go through self-play without human knowledge, discovering strategies humans never conceived. Can we apply the same principle to stock trading — using Monte Carlo Tree Search + neural networks or pure reinforcement learning to discover profitable patterns WITHOUT human-designed indicators? What architecture would work, how would we define the "game state" and "winning", and what are the key challenges compared to board games?
```

**Prompt 12 (Self-Supervised Market Representations):**
```
How do we apply self-supervised learning (contrastive learning like SimCLR, masked autoencoders, or VAEs) to stock OHLCV sequences to learn meaningful representations WITHOUT labels? The goal is to discover latent patterns the model finds predictive on its own. Provide PyTorch code for contrastive learning on 20-day price windows.
```

**Prompt 13 (Evolutionary Alpha Discovery):**
```
Can we use genetic algorithms or evolutionary strategies to evolve trading formulas from scratch — starting with random mathematical operations on OHLCV data and letting survival-of-the-fittest discover profitable rules humans never imagined (like solving a Rubik's cube)? How do we design the fitness function (Sharpe ratio? profit factor?) to avoid overfitting while rewarding novel discoveries? Provide a working DEAP or PySR config.
```

**Prompt 14 (Curiosity-Driven Trading):**
```
In reinforcement learning, curiosity-driven exploration (Intrinsic Curiosity Module, Random Network Distillation) rewards agents for discovering novel states. Can we apply this to trading — rewarding an AI for finding unusual market microstructure patterns that correlate with profitable trades? This could theoretically discover "black swan" setups before they happen. Provide implementation guidance.
```

---

## Quick Reference Checklist

| Module | Key Training Question | Priority |
|--------|----------------------|----------|
| Visual CNN | Architecture selection for GASF | HIGH |
| Regime HMM | Number of states + features | HIGH |
| Labeling | Triple-barrier parameter tuning | HIGH |
| Ensemble | Optuna hyperparameter search | HIGH |
| Validation | CPCV implementation | HIGH |
| RL Execution | Reward function design | MEDIUM |
| Logic Engine | PySR config for alphas | MEDIUM |
| Recommender | Ranking model choice | MEDIUM |
| GPU Training | Mixed precision setup | HIGH |
| Meta-Labeling | Confidence filtering | MEDIUM |
| **Self-Discovery AI** | **AlphaZero-style pattern learning** | **HIGH** |
| Contrastive Learning | Self-supervised representations | HIGH |
| Evolutionary Search | Genetic alpha discovery | HIGH |
| Curiosity-Driven RL | Novel pattern exploration | MEDIUM |

---

*Generated for quantum-ai-trader_v1.1 — December 2025*
