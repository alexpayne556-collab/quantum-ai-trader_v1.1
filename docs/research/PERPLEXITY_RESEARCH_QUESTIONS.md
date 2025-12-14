# Perplexity Pro Research Questions
## Technical Deep Dive for Production-Ready Trading System

---

## 🎯 Context
We're building a comprehensive algorithmic trading system with:
- Elliott Wave pattern detection
- **EMA+ATR based 24-day price forecasting** (current issue: shape broadcasting errors)
- Risk management with position sizing
- Watchlist scanner (current issue: API dependency)
- Trade executor validation (current issue: pandas Series ambiguity)

**Goal**: Find proven, open-source patterns and "secret sauce" approaches that solve these specific errors while maintaining production-grade reliability.

---

## 📊 Question 1: Robust Time Series Forecasting with EMA+ATR
**Query for Perplexity:**
```
What are the most robust open-source implementations for generating deterministic 
price forecasts using EMA crossovers and ATR volatility bands? Specifically looking 
for Python libraries or patterns that:
- Generate N-day forward price paths (e.g., 24 days) from historical OHLCV
- Avoid numpy broadcasting errors when aligning forecast arrays with actual prices
- Use EMA(12)/EMA(26) trend + ATR(14) volatility to model drift and oscillation
- Have been validated in production trading systems

Include GitHub repos, papers, or QuantConnect/Alpaca strategies that demonstrate 
this pattern with clean numpy/pandas array handling.
```

**Why this matters:** Our current forecast implementation hits "operands could not be broadcast together with shapes (24,0) (23,)" suggesting array length mismatch between generated forecast and test data alignment.

---

## 🛡️ Question 2: Pandas Series Truth-Value Ambiguity - Production Patterns
**Query for Perplexity:**
```
What are the established patterns for avoiding "truth value of a Series is ambiguous" 
errors in production trading risk management systems? Looking for:
- How to safely convert pandas Series to scalars when computing Sharpe ratios, drawdowns
- Best practices for position sizing calculations that mix DataFrame columns with scalar math
- Open-source risk management libraries (e.g., empyrical, quantstats, pyfolio) that 
  handle Series-to-float conversions cleanly
- Patterns used in Zipline, Backtrader, or VectorBT for vectorized backtest simulations

Include specific code snippets showing .iloc[], .item(), or .values usage that prevents 
ambiguity errors when building portfolio value arrays.
```

**Why this matters:** Our Risk Manager hits conversion errors like "Could not convert [array([0.00037694]) array([4.18..." and Trade Executor validation fails with Series ambiguity when checking order conditions.

---

## 🔍 Question 3: Local-Only Stock Screening Without External APIs
**Query for Perplexity:**
```
How do production trading systems implement watchlist scanners and momentum/volume 
screens using ONLY local OHLCV data, without requiring external API keys? Looking for:
- Open-source implementations of technical scanners (RSI, volume breakouts, momentum)
- Patterns for computing relative strength, volume percentiles, and breakout detection 
  from pandas DataFrames alone
- How Zipline/Quantopian/QuantConnect handle universe selection with local data
- Libraries like TA-Lib, pandas-ta, or vectorbt patterns for scan-based filtering

Focus on approaches that don't depend on Finnhub, Alpha Vantage, or similar external 
APIs\u2014everything computed from yfinance downloaded historical data.
```

**Why this matters:** Our Watchlist Scanner currently fails with "module 'config' has no attribute 'FINNHUB_API_KEY'" because it's trying to call external services instead of computing signals locally.

---

## 📈 Question 4: Elliott Wave Target Validation in Live Markets
**Query for Perplexity:**
```
What are the proven methods for validating Elliott Wave price targets in algorithmic 
trading systems? Specifically:
- How to measure "hit rate" of Fibonacci extension targets (1.0, 1.618, 2.618) 
  in forward test periods
- Open-source Elliott Wave libraries (e.g., elwave, pyti) and their validation approaches
- Academic papers or trading blogs showing statistical significance tests for wave patterns
- How to combine wave-based targets with ATR stops for position exit rules

Include any GitHub repos or QuantConnect strategies that demonstrate Elliott Wave 
target tracking with real trade outcomes.
```

**Why this matters:** Our Elliott Wave trainer is running but finding 0% hit rates\u2014we need to validate if our target checking logic is correct or if we need alternative pattern validation approaches.

---

## 💰 Question 5: Position Sizing with Kelly Criterion + ATR Stops
**Query for Perplexity:**
```
What are the production-tested implementations of position sizing that combine:
- ATR-based stop loss placement (e.g., 1.5x ATR(14) below entry)
- Kelly Criterion or fixed fractional risk per trade (e.g., 2% of equity at risk)
- Dynamic capital allocation based on recent win rate and R-multiple

Looking for:
- Python libraries (e.g., riskfolio-lib, quantstats, ffn) with working examples
- How Interactive Brokers API, Alpaca, or QuantConnect handle position size calculation
- Patterns for converting risk-per-trade percentage into share quantities given stop distance
- Backtesting frameworks that validate this approach (Backtrader, VectorBT, Zipline)

Include any research papers or blogs from professional quant traders on optimal position sizing.
```

**Why this matters:** Our Risk Manager is simulating position sizing but needs validation that our ATR-stop + risk-per-trade math is correct and aligns with industry practice.

---

## 🔧 Question 6: Trade Executor Slippage + Fill Estimation Models
**Query for Perplexity:**
```
What are the state-of-the-art models for estimating slippage and fill quality in 
algorithmic trading systems? Looking for:
- How to model market impact as a function of order size vs average daily volume
- Bid-ask spread estimation from OHLC data when tick data isn't available
- Libraries like quantlib, zipline.finance, or backtrader that include slippage models
- Research papers on cost models (e.g., Almgren-Chriss, implementation shortfall)

Specifically need patterns for converting volatility (ATR) and volume metrics into:
- Commission (bps)
- Market impact (price movement from order)
- Adverse selection (spread cost)
- Total expected slippage per trade

Include any GitHub implementations or QuantConnect modules that demonstrate this.
```

**Why this matters:** Our Trade Executor returns slippage as pandas Series instead of clean floats, and we need validated slippage models to ensure realistic backtest expectations.

---

## 🧪 Question 7: Comprehensive Trading System Testing Patterns
**Query for Perplexity:**
```
What are the best practices for testing multi-module algorithmic trading systems 
where different components (wave detection, forecasting, risk, execution) need to 
work together? Looking for:
- How professional quant shops structure integration tests for trading systems
- Patterns for mocking market data, signals, and portfolio state in unit tests
- Open-source examples from Zipline, Backtrader, or QuantConnect showing module integration
- How to validate that risk manager, forecaster, and executor produce consistent outputs
- Testing patterns that catch pandas/numpy shape mismatches before production

Include any pytest patterns, testing frameworks (e.g., pytest-mock, hypothesis), or 
CI/CD approaches used in trading system development.
```

**Why this matters:** We're hitting multiple shape/type errors across modules\u2014need a systematic testing approach to catch these issues early and ensure module compatibility.

---

## 🚀 Question 8: Production Trading System Architecture - The "Secret Sauce"
**Query for Perplexity:**
```
What architectural patterns distinguish successful production algorithmic trading 
systems from academic backtests? Looking for the "secret sauce" around:
- How top quant funds (Renaissance, Two Sigma, Citadel) structure their data pipelines
- Event-driven vs batch processing for real-time signal generation
- State management patterns (how to track positions, pending orders, risk limits)
- How to decouple forecasting, risk management, and execution for independent scaling
- Open-source examples of production-grade trading architectures (e.g., Jesse, Freqtrade, Hummingbot)

Include any blog posts, conference talks, or open-source repos that reveal non-obvious 
design decisions that separate toy systems from production systems.
```

**Why this matters:** We want to ensure our comprehensive trainer is architected correctly so each module (forecast, risk, scanner, executor) can evolve independently and scale to real-time operation.

---

## 📝 Question 9: Continuous Learning Systems for Trading Models
**Query for Perplexity:**
```
How do production trading systems implement continuous learning and model retraining? 
Looking for:
- Online learning patterns (incremental updates vs full retraining schedules)
- How to detect model drift and trigger retraining in live markets
- Libraries like river (online ML), vowpal wabbit, or scikit-multiflow for streaming updates
- Logging and monitoring patterns that track prediction accuracy over time
- How hedge funds handle model versioning and A/B testing of strategies

Include any open-source MLOps patterns specific to financial time series, or examples 
from trading platforms like QuantConnect showing automated retraining pipelines.
```

**Why this matters:** Our comprehensive trainer runs once\u2014but production systems need continuous validation and adaptation as market regimes change.

---

## 🎓 Bonus Question 10: Regulatory & Risk Controls for Algo Trading
**Query for Perplexity:**
```
What regulatory compliance and risk control patterns are required for production 
algorithmic trading systems in US markets? Looking for:
- SEC Rule 15c3-5 (Market Access Rule) requirements for pre-trade risk checks
- Position limits, order throttling, and kill switch implementations
- Open-source examples of risk controls in broker APIs (IBKR, Alpaca, TD Ameritrade)
- How to implement max order size, max loss per day, and circuit breaker logic
- Testing patterns for demonstrating compliance with regulatory standards

Include any whitepapers from broker-dealers or compliance frameworks used in retail 
algo trading platforms.
```

**Why this matters:** Before deploying real capital, we need to ensure our Risk Manager and Trade Executor have proper safeguards that meet regulatory expectations.

---

## 📋 Summary: Key Research Objectives

| Component | Current Issue | Research Focus |
|-----------|---------------|----------------|
| **Forecast Engine** | Shape broadcasting errors | Robust EMA+ATR forward path generation |
| **Risk Manager** | Series-to-numeric conversion | Production pandas handling patterns |
| **Watchlist Scanner** | API dependency | Local-only signal computation |
| **Trade Executor** | Series truth-value ambiguity | Clean scalar math + slippage models |
| **System Architecture** | Module integration | Event-driven design + testing patterns |
| **Continuous Improvement** | Static training | Online learning + drift detection |

---

## 🔍 How to Use These Questions

1. **Copy each query** into Perplexity Pro (Pro version recommended for deeper research and citations)
2. **Review citations** for GitHub repos, papers, and blog posts
3. **Extract code patterns** that solve our specific errors
4. **Validate approaches** by comparing multiple sources (if 3+ sources recommend a pattern, it's likely production-tested)
5. **Prioritize** questions 1-3 first (they address our immediate blockers), then 4-7 for validation, and 8-10 for architecture improvements

---

## 📊 Expected Outcomes

After researching these questions, we should have:
- ✅ Clean numpy array handling patterns for forecast alignment
- ✅ Pandas Series-to-scalar conversion best practices
- ✅ Local-only scanner implementation without external APIs
- ✅ Validated slippage and position sizing models
- ✅ Testing patterns to catch type/shape errors early
- ✅ Production architecture patterns for scalable algo trading
- ✅ Continuous learning framework for model adaptation

---

**Next Steps After Research:**
1. Implement the discovered patterns in our code
2. Add comprehensive unit tests using pytest patterns found
3. Validate metrics against known-good libraries (e.g., compare our Sharpe to quantstats output)
4. Document "secret sauce" decisions in architecture docs
5. Plan continuous training pipeline based on online learning patterns

---

*Generated: 2025-12-01*  
*System: Quantum AI Trader v1.1*  
*Purpose: Production-grade algorithmic trading system research*
