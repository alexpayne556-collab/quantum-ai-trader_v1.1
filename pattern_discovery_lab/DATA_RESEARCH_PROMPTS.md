# DATA ACQUISITION RESEARCH - AI TEAM PROMPTS

## THE PROBLEM

Our Lab V1 adaptive framework computed these diagnostics:

```
n_observations: 503 (2 years daily)
n_effective: 513
n_required: 3,078 (to detect IC=0.05 with 80% power)
sample_deficit: 2,564 observations
min_detectable_effect: 0.124 IC
```

**Translation**: We need ~12+ years of daily data OR use higher frequency data to get enough observations.

## AVAILABLE DATA SOURCES

| API | Rate Limit | Historical Depth | Frequencies |
|-----|------------|------------------|-------------|
| **yfinance** | Unlimited | 30+ years | Daily, 1h (2 years) |
| **Finnhub** | 60/min | 1 year | Intraday |
| **Twelve Data** | 800/day | Variable | 1min-Monthly |
| **Alpha Vantage** | 25/day | 20+ years | Daily, Intraday |
| **EODHD** | 20/day | 30+ years | Daily |
| **FRED** | Unlimited | 50+ years | Daily/Monthly |

---

## RED TEAM (DeepSeek) - PROMPT

```
ROLE: Devil's Advocate / Attack Vector Analyst

CONTEXT:
We need 3,078+ observations to reliably detect IC=0.05 effects.
Our options:
1. Get more historical daily data (10+ years)
2. Use higher frequency data (hourly/minute)
3. Use multiple assets as pseudo-observations
4. Lower our detection threshold (accept only large effects)

QUESTIONS:

1. HISTORICAL DATA QUALITY RISKS
   - What data quality issues exist in stock data before 2015? (survivorship bias, decimal pricing, flash crashes, regime changes)
   - How does structural market change (HFT proliferation, COVID, QE) invalidate older patterns?
   - Should we weight recent data higher than ancient data?

2. HIGHER FREQUENCY TRAPS
   - If we use hourly data (252 days × 6.5 hours = 1,638 obs/year), what microstructure noise gets amplified?
   - At what frequency does transaction cost drag dominate any alpha signal?
   - Are patterns discovered at 1-hour truly different from daily patterns or just noise?

3. CROSS-ASSET POOLING DANGERS
   - If we pool 76 stocks to get N = 76 × 500 = 38,000 obs, is this legitimate?
   - What assumptions break when treating different stocks as exchangeable?
   - How does sector/factor exposure correlation reduce effective N?

4. SURVIVORSHIP BIAS IN HISTORICAL DATA
   - How many S&P 500 constituents from 2010 no longer exist?
   - Do free APIs include delisted tickers?
   - What's the expected bias in backtest returns from survivorship?

5. DATA VENDOR RELIABILITY
   - Which of our APIs have known data quality issues?
   - Are there corporate actions (splits, dividends) adjustment inconsistencies?
   - How do we detect and handle bad data points?

Please identify every way our data collection strategy could give us FALSE CONFIDENCE in patterns that won't work live.
```

---

## BLUE TEAM (Claude) - PROMPT

```
ROLE: Solution Architect / Implementation Expert

CONTEXT:
We need to increase sample size from ~500 to 3,000+ observations.
Available APIs: yfinance (unlimited), Finnhub (60/min), Twelve Data (800/day), Alpha Vantage (25/day), EODHD (20/day)

DESIGN CHALLENGES:

1. OPTIMAL FREQUENCY SELECTION
   Design an algorithm that:
   - Computes optimal data frequency for pattern discovery
   - Balances: sample size vs microstructure noise vs transaction cost drag
   - Input: target effect size, available history, asset volatility
   - Output: recommended frequency + expected N_effective

2. MULTI-ASSET POOLING FRAMEWORK
   Design a framework that:
   - Pools observations across assets legitimately
   - Adjusts N_effective for cross-sectional correlation
   - Uses hierarchical/mixed models to account for asset heterogeneity
   - Provides valid confidence intervals for pooled estimates

3. HISTORICAL DATA AGGREGATOR
   Design a system that:
   - Pulls maximum history from each API
   - Reconciles conflicting data points between sources
   - Handles corporate actions consistently
   - Flags suspicious data points for review
   - Outputs clean, validated dataset with quality scores

4. ADAPTIVE SAMPLE SIZE GATE
   Design a gate that:
   - Given current dataset, computes what effects we CAN reliably detect
   - Warns when sample is insufficient for target effect
   - Suggests: "collect X more observations" or "increase frequency" or "relax effect threshold"

5. WALK-FORWARD DATA REQUIREMENTS
   For our walk-forward validation:
   - Train: 252 bars, Test: 63 bars, Embargo: adaptive
   - How many years of data needed for N splits with 80% power per split?
   - Design formula: years_needed = f(n_splits, effect_size, power, frequency)

Please provide pseudocode or Python implementations for each design.
```

---

## RESEARCH TEAM (Perplexity) - PROMPT

```
ROLE: Academic Research / Citation Provider

CONTEXT:
Pattern discovery in financial time series. We need sample size recommendations backed by peer-reviewed research.

RESEARCH QUESTIONS:

1. SAMPLE SIZE IN FINANCIAL ML
   - What do academic papers say about minimum sample size for:
     a) Detecting Rank IC of 0.03-0.10 reliably?
     b) Training machine learning models on price data?
     c) Walk-forward validation in quantitative finance?
   - Cite specific papers with their recommendations.

2. OPTIMAL DATA FREQUENCY
   - What academic research exists on optimal frequency for alpha discovery?
   - Papers comparing daily vs hourly vs minute patterns?
   - At what frequency does market microstructure noise dominate?
   - Cite: Ait-Sahalia, Hansen, Bandi & Russell, or similar.

3. CROSS-SECTIONAL POOLING VALIDITY
   - Under what conditions can we pool observations across assets?
   - What does Fama-MacBeth regression assume about cross-sectional independence?
   - Papers on "panel data" approaches in quantitative finance?
   - How to adjust standard errors for cross-sectional correlation?

4. SURVIVORSHIP BIAS QUANTIFICATION
   - Academic estimates of survivorship bias magnitude in stock returns?
   - Papers quantifying the bias (e.g., Brown, Goetzmann, Ross)?
   - How much does it inflate backtest Sharpe ratios?

5. DATA QUALITY IN FREE APIs
   - Any academic or practitioner papers comparing free vs premium data quality?
   - Known issues with Yahoo Finance, Alpha Vantage, etc.?
   - Best practices for data validation in quant research?

6. EXPANDING SAMPLE WITH SYNTHETIC DATA
   - Is IAAFT (surrogate data) valid for expanding training sets?
   - Papers on bootstrapping/simulation in financial ML?
   - When does synthetic data help vs hurt?

Please provide specific citations (author, year, journal) for all claims.
```

---

## IMMEDIATE ACTION PLAN

After getting AI team responses, we will:

1. **Implement Blue Team's data aggregator** to pull max history
2. **Validate with Red Team's concerns** - check for survivorship, regime shifts
3. **Use Research citations** to justify our frequency/pooling choices
4. **Update Lab V1** with actual data diagnostics

## DATA TARGETS

| Approach | Target N | Detectable IC | Feasibility |
|----------|----------|---------------|-------------|
| 10yr daily | 2,520 | 0.06 | High (yfinance) |
| 2yr hourly | 3,276 | 0.05 | High (Finnhub) |
| Cross-asset pool (76 stocks × 2yr) | ~38,000 | 0.015 | Medium (needs adjustment) |
| 5yr daily + hourly hybrid | ~4,000 | 0.04 | High |

---

## COPY-PASTE READY

### For DeepSeek (Red Team):
Copy the RED TEAM section above.

### For Claude (Blue Team):  
Copy the BLUE TEAM section above.

### For Perplexity:
Copy the RESEARCH TEAM section above.

---

*Generated by Lab V1 Framework - Data-Driven Pattern Discovery*
