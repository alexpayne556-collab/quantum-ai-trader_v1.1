# AI RESEARCH CONSULTATION
## Context for External AI Systems (Perplexity Pro, Claude, DeepSeek)

**Date:** December 18, 2025  
**Project:** Quantum AI Trader - Heavyweight Quantitative Research Infrastructure

---

## OUR CURRENT SITUATION

### What We're Doing (The Big Picture)

We **rejected lightweight pattern matching** after months of trying approaches that didn't work. We're now building **heavyweight institutional-grade quantitative research** based on the scientific method.

**Core Philosophy:**
> "We don't premake laws, we discover them."

We're building a **financial physics lab** - discovering universal market laws through rigorous hypothesis testing, not assumptions.

---

## THE HEAVYWEIGHT APPROACH

### 1. Complete Universe Acquisition
- **Downloaded:** 10,986 US equities (complete NASDAQ, NYSE, AMEX, NYSE Arca)
- **Time Period:** 2 years (2023-12-18 to 2025-12-17, 504 trading days)
- **Data Source:** Multi-source (yfinance primary, Polygon.io fallback)
- **Current Status:** Downloading on Shadow PC (with GPU), ~12-15 hours to complete
- **Expected Output:** ~1,300 tickers passing quality checks, ~650,000 bars, ~500 MB database

**Why complete universe?**
- Cross-sectional power: 10,986 tickers × 504 days = **5.5 million observations**
- Separates signal from noise (pattern works on 100 tickers but fails on 10k = noise)
- Prevents cherry-picking and survivorship bias

### 2. Scientific Infrastructure Built (~2,350 lines)

**Statistical Testing Framework (600 lines):**
- Tests for momentum (16 variations), mean reversion (12 variations), volatility clustering, cross-sectional correlation
- Multiple testing corrections: Bonferroni, Benjamini-Hochberg FDR, Holm-Bonferroni
- Output: hypothesis_test_results.csv with p-values, effect sizes, significance

**Regime Detection System (500 lines):**
- HMM-based: volatility regimes (low/normal/high), trend regimes (bull/bear/sideways), correlation regimes
- Manual breakpoints for known events (2023 banking crisis, AI boom, etc.)
- **Critical insight:** Test if patterns are universal or regime-dependent
- Most "edges" only work in specific market conditions

**Survivorship Bias Correction (450 lines):**
- Detects delistings and IPOs
- Reconstructs point-in-time universes (monthly/quarterly snapshots)
- **Warning:** Typical backtests inflate returns 3-5% annually by excluding bankrupt stocks

**Factor Analysis (400 lines):**
- PCA on return matrix (find orthogonal drivers)
- Don't assume Fama-French factors exist - discover what ACTUALLY drives returns
- Test factor predictive power through cross-sectional regression
- Build traditional factors from data (market, size, momentum, reversal)

**Cross-Validation Framework (400 lines):**
- Walk-forward split (expanding/rolling window)
- Purged k-fold (López de Prado method with embargo periods)
- Overfit detection (compare in-sample vs out-of-sample Sharpe)
- Monte Carlo permutation tests (shuffle returns, test significance)

### 3. Why This Takes Months (And That's The Moat)

**The competitive advantage IS the barrier to entry:**
- Retail can't do this (no compute, no complete data, no statistics knowledge)
- Most quant shops cut corners (assume Fama-French, don't test regimes, ignore survivorship)
- Fast/easy methods don't work because if they did, everyone would use them (edge arbitraged away)

**We're building nuclear weapons while retail brings pocket knives.**

---

## WHAT WE'VE LEARNED (Critical Insights)

### 1. Your Manual Trades Are Working
- **MU (Micron):** Up 15% after earnings beat
- **KDK:** Up 10% today
- **Pattern:** You're good at earnings plays and picking momentum stocks

**But we don't know:**
- Is this skill or luck?
- What's the actual hit rate over 100+ trades?
- Does it work in all market regimes?
- What's the maximum drawdown we should expect?

**This is EXACTLY why we need the heavyweight research** - to validate what works and quantify the edge.

### 2. Paper Trading Results (Mixed)
- **HUT:** -11.34% (Vol2x+GapUp edge, 85.4% claimed hit rate - in the 14.6% failure zone)
- **RXRX:** -6.93% (same edge, also failing)
- **Lesson:** Even "high hit rate" edges fail sometimes. Need regime filters.

### 3. Cross-Sectional Power Matters More Than Time Series Length
- 10,986 tickers × 2 years > 100 tickers × 10 years
- More tickers = better separation of signal from noise
- Regime changes make old data less relevant anyway

---

## THE PATH FORWARD (Next 48-72 Hours)

### Phase 1: Data Completion (In Progress)
- Shadow PC downloading complete universe (12-15 hours)
- Auto-resumes if interrupted
- Quality checks: no missing days >30%, no price gaps >50%, min 504 days required

### Phase 2: GPU-Accelerated Hypothesis Testing
Once data is complete:

**Run statistical_framework.py (2-4 hours CPU → 5-10 min GPU):**
- Test 28 hypotheses (momentum variations, mean reversion, volatility clustering, etc.)
- Apply multiple testing corrections
- Get p-values and effect sizes
- **Output:** Which patterns are statistically significant after corrections?

**Run regime_detection.py (1 hour CPU → 1-2 min GPU):**
- Detect HMM volatility regimes
- Identify trend regimes
- Find correlation regimes
- **Critical:** Test if patterns work in ALL regimes or only specific ones

**Test regime dependence:**
- Re-run hypothesis tests WITHIN each regime separately
- **Discovery:** Most patterns are regime-dependent, not universal
- Example: Momentum works in trending markets, fails in sideways/choppy

### Phase 3: Factor Discovery
**Run factor_analysis.py (30 min CPU → 30 sec GPU):**
- PCA on 1,300 tickers × 504 days return matrix
- Discover latent factors (what ACTUALLY drives returns)
- Test if traditional Fama-French factors still work in 2023-2025 data
- **Output:** pca_factor_returns.csv, traditional_factors.csv

### Phase 4: Validation
**Run survivorship_bias.py:**
- Measure how much survivorship inflates returns
- Reconstruct point-in-time universes
- Re-test patterns with PIT data

**Run cross_validation.py:**
- Walk-forward validation (no lookahead)
- Compare in-sample vs out-of-sample Sharpe
- Monte Carlo permutation test (is this real or random?)
- **Output:** Only patterns that survive this are REAL edges

### Phase 5: Strategy Implementation
**Only after rigorous validation:**
- Take discovered edges (tested, validated, regime-filtered)
- Build trading rules
- Paper trade for 30-60 days
- If paper trading matches backtest → go live with small capital

---

## QUESTIONS FOR EXTERNAL AI SYSTEMS

### 1. Research Methodology Questions

**For Perplexity Pro (web search capability):**
- What are the latest academic papers (2023-2025) on:
  - Regime-dependent trading strategies?
  - Cross-sectional factor models in modern markets?
  - High-frequency pattern detection with statistical rigor?
- How do Renaissance Technologies, Two Sigma, and Citadel approach hypothesis testing?
- What are the known pitfalls in backtesting that even professionals miss?
- Are there open-source implementations of purged k-fold cross-validation we should review?
- What's the current state of research on:
  - Earnings momentum persistence?
  - Volume-price relationship in gap-up scenarios?
  - Regime detection using hidden Markov models vs machine learning?

**For Claude (deep reasoning):**
- Given our infrastructure (statistical framework, regime detection, factor analysis, cross-validation):
  - What hypothesis tests are we MISSING that we should run?
  - Are there interaction effects between factors we should test?
  - How should we handle the multiple testing problem with 28+ hypotheses?
  - Should we use Bayesian methods instead of/in addition to frequentist tests?
- Critique our approach:
  - What are the weaknesses in our methodology?
  - What assumptions are we making that might be wrong?
  - Where could we be fooling ourselves?
- Factor discovery:
  - Is PCA the right approach or should we use ICA, NMF, or autoencoders?
  - How do we interpret discovered factors without falling into narrative fallacy?
  - Should we use sparse PCA to get more interpretable factors?

**For DeepSeek (coding/implementation):**
- GPU optimization:
  - How can we parallelize the hypothesis testing framework for GPU?
  - What libraries (CuPy, Rapids, PyTorch) are best for our use case?
  - Can we batch-process 1,300 tickers in parallel for 100x speedup?
- Data pipeline optimization:
  - Are we handling missing data correctly in our quality checks?
  - Should we use Dask or Polars instead of Pandas for 500 MB dataset?
  - How do we efficiently compute rolling correlations across 1,300 tickers?
- Statistical implementation:
  - Are we implementing multiple testing corrections correctly?
  - Should we use statsmodels, scipy, or custom implementations?
  - How do we handle edge cases (tickers with <504 days after quality filtering)?

### 2. Specific Technical Questions

**Regime Detection:**
- HMM vs Gaussian Mixture Models vs K-means for regime detection?
- How many regimes should we detect? (We're using 3 volatility states)
- Should we use online/adaptive regime detection or fixed historical regimes?
- How do we handle regime transitions (gradual vs sudden)?

**Multiple Testing Corrections:**
- We're using Bonferroni (conservative), Benjamini-Hochberg (less conservative), Holm-Bonferroni
- Should we also use:
  - False Discovery Rate control?
  - Family-wise error rate (FWER)?
  - Permutation-based corrections?
- How do we choose the right alpha level (0.05 too lenient, 0.01 too strict)?

**Survivorship Bias:**
- Are we correctly identifying delistings vs data gaps?
- Should we weight surviving stocks differently than delisted ones?
- How far back do we need to go to measure true survivorship impact?

**Cross-Validation:**
- Walk-forward window size: 252 days (1 year) vs 126 days (6 months)?
- Embargo period: 21 days vs 63 days to prevent information leakage?
- Should we use combinatorial purged CV (CPCV) from Advances in Financial ML?

### 3. Edge Validation Questions

**Your Manual Trading Success:**
- MU +15% after earnings, KDK +10%
- How do we test if your "earnings beat → momentum" edge is real?
- What statistical test validates "stock selection skill" vs luck?
- Should we build an earnings surprise factor and test it?

**Hypothesis to Test:**
```
H1: Stocks that beat earnings estimates by >10% outperform 
    market in next 5/10/21 days

Variables:
- Earnings surprise % (actual - estimate) / estimate
- Post-earnings drift (days 1-5, 1-10, 1-21)
- Control for: market cap, sector, volume, pre-earnings momentum

Statistical test:
- Cross-sectional regression: return_t+k = α + β(surprise) + controls
- Test significance of β across all earnings events in 2023-2025
- Check if effect varies by regime (bull vs bear markets)
```

Should we build this test into our framework?

### 4. Infrastructure Questions

**GPU Acceleration:**
- We have Shadow PC with GPU but haven't optimized code yet
- What's the best way to convert Pandas/NumPy operations to GPU?
- Should we rewrite in JAX, PyTorch, or use Rapids (cuDF)?
- What's the expected speedup for our workload?

**Data Storage:**
- Currently using SQLite (500 MB)
- Should we switch to:
  - Parquet files (faster columnar reads)?
  - PostgreSQL (better for complex queries)?
  - HDF5 (better for time series)?
  - DuckDB (faster analytics on Parquet)?

**Scalability:**
- What if we want to expand to:
  - All global equities (50,000+ tickers)?
  - Intraday data (1-minute bars = 100x more data)?
  - Options data (millions of contracts)?
- How do we architect for that now?

### 5. Philosophical/Strategic Questions

**Research vs Trading Tradeoff:**
- We could spend 6 months on rigorous research
- Or spend 1 month and start trading with "good enough" validation
- **Question:** At what point is additional research hitting diminishing returns?
- How do Renaissance/Two Sigma decide when research is "done enough"?

**Human Edge vs Systematic Edge:**
- You're clearly good at picking MU and KDK manually
- Should we:
  - A) Fully automate and remove human judgment?
  - B) Build system to AUGMENT your stock picking (AI suggests, you decide)?
  - C) Hybrid: System trades most edges, you trade special situations?

**Capital Allocation:**
- Once we validate edges, how much capital per edge?
- Kelly Criterion vs fixed fractional vs volatility-targeting?
- How do we handle correlation between edges?

**Live Trading Feedback Loop:**
- Should we start paper trading NOW with incomplete research?
- Use live results to guide which hypotheses to test next?
- Or finish all research first, then deploy?

---

## WHAT WE NEED FROM YOU (AI Systems)

### Perplexity Pro:
- Latest academic research on our specific questions
- What are quant funds doing in 2024-2025 that's different from 2010s?
- Real-world examples of regime-dependent strategies
- Benchmark our approach against known best practices

### Claude:
- Deep critique of our methodology
- Identify blind spots and hidden assumptions
- Suggest hypothesis tests we haven't thought of
- Help us think through the philosophy/strategy questions

### DeepSeek:
- Code optimization strategies for GPU
- Implementation reviews of our statistical tests
- Architectural suggestions for scaling
- Catch bugs/edge cases in our frameworks

---

## OUR SPECIFIC CONCERNS

1. **Are we testing the right things?**
   - 28 hypotheses seems like a lot but also arbitrary
   - Are we missing obvious tests?
   - Are we testing things that don't matter?

2. **Are we handling statistics correctly?**
   - Multiple testing corrections seem conservative
   - Are we being TOO conservative and missing real edges?
   - Or not conservative enough and finding false positives?

3. **Is regime detection the right approach?**
   - HMM feels principled but is it practical?
   - Should we just use simple rules (VIX > 20 = high vol regime)?
   - How do we avoid overfitting regimes to historical data?

4. **How do we validate the whole system?**
   - We can validate individual edges
   - But how do we validate the entire research methodology?
   - What if our framework itself has systemic bias?

5. **When do we stop researching and start trading?**
   - Real money is the ultimate test
   - But premature trading burns capital on bad edges
   - How do we know when we're "ready"?

---

## CONTEXT YOU SHOULD KNOW

**Our Background:**
- We've been working on this for months
- Tried many lightweight approaches (pattern matching, ML classifiers, etc.)
- None worked consistently
- User has manual trading skill (MU +15%, KDK +10% prove this)
- But we want systematic, scalable, provable edges

**Our Resources:**
- Shadow PC with GPU (for acceleration)
- Codespace (for development)
- Complete US equity universe (10,986 tickers, 2 years)
- Alpaca paper trading account ($100k virtual)
- Time: willing to spend months on this

**Our Constraints:**
- No institutional data (no Bloomberg terminal, no tick data)
- Limited to free/cheap APIs (yfinance, Polygon.io free tier)
- No options/futures data yet (could add later)
- Starting capital: unknown but probably <$50k

**Our Philosophy:**
- "We don't premake laws, we discover them"
- Real science takes time - that's the moat
- Test everything rigorously
- Assume nothing
- Separate signal from noise through statistics, not intuition

---

## FINAL QUESTION FOR ALL AI SYSTEMS

**Given everything above, what would YOU do if you were us?**

What's the highest-leverage action we should take in the next 48 hours while data downloads?

Should we:
1. Build more hypothesis tests?
2. Optimize GPU code for 100x speedup?
3. Start paper trading NOW with incomplete research?
4. Read specific academic papers?
5. Build visualization tools to explore the data?
6. Test your manual trading edge (earnings surprise momentum)?
7. Something completely different we haven't thought of?

**Be brutally honest. If our approach is flawed, tell us. If we're missing something obvious, tell us. If we're on the right track, confirm it.**

We want the truth, not reassurance.

---

## HOW TO RESPOND

For each AI system, please provide:

1. **Immediate feedback** - What do you think of our approach? Flaws? Strengths?
2. **Specific answers** to your domain's questions above
3. **Highest-leverage action** we should take in next 48 hours
4. **Long-term suggestions** for after initial research is complete
5. **Resources** - Papers, libraries, tools, references we should check out

**Format:** Markdown, be specific, provide code examples if relevant, cite sources if applicable.

**Length:** As long as needed. We'd rather have thorough answers than brief ones.

---

*This document will be provided to Perplexity Pro, Claude.ai, and DeepSeek for consultation on our heavyweight quantitative research approach.*
