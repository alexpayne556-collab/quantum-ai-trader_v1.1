# AI RESEARCH CONSULTATION
## Context for External AI Systems (Perplexity Pro, Claude, DeepSeek)

**Date:** December 18, 2025  
**Project:** Quantum AI Trader - Heavyweight Quantitative Research Infrastructure

---

## ⚠️ CRITICAL: READ THIS FIRST

**We are NOT looking for surface-level advice or generic recommendations.**

**What We Want:**
- Deep, heavyweight quantitative concepts from institutional research
- Brutally honest critique of our methodology (tell us if we're wrong)
- Specific, actionable guidance backed by academic research or real-world quant fund practices
- Technical depth - we can handle the math and statistics

**What We DON'T Want:**
- Generic "diversify your portfolio" type advice
- Disclaimers about risk (we know)
- Simplified explanations for beginners
- Surface-level pattern observations without statistical rigor

**Our Commitment:**
- **Timeframe:** NOT SET - we will work as long as it takes to do this right
- We've already spent MONTHS on lightweight approaches that failed
- We're willing to spend 6-12 months on rigorous research if that's what it takes
- Real science takes time - that's our competitive advantage (the moat)

**Our Work Ethic:**
- We are NOT slouches looking for get-rich-quick schemes
- We will implement whatever methodology is proven to work
- We can code, we can do math, we can read papers
- We want the REAL DEAL, not dumbed-down versions

**If your answer would be different for a PhD quant researcher at Renaissance Technologies vs a retail trader, give us the Renaissance answer.**

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

## READY-TO-COPY PROMPTS FOR EACH AI SYSTEM

### 📋 PROMPT FOR PERPLEXITY PRO (Copy This)

```
I'm building a heavyweight quantitative trading research infrastructure to discover market edges through rigorous scientific method. I need your web search capabilities to find the latest academic research and real-world practices.

CONTEXT:
- Spent months on lightweight pattern matching - all failed
- Now building institutional-grade research: complete US equity universe (10,986 tickers, 2 years data)
- Built infrastructure: statistical testing (600 lines), regime detection (500 lines), survivorship correction, factor analysis, cross-validation
- Philosophy: "We don't premake laws, we discover them" - test everything rigorously
- Timeframe: NOT SET - willing to work 6-12 months if needed for rigorous research

YOUR EXPERTISE NEEDED:

1. **Latest Academic Research (2023-2025):**
   - Regime-dependent trading strategies papers
   - Cross-sectional factor models in modern markets
   - Pattern detection with statistical rigor (not curve-fitting)
   - Earnings momentum persistence research
   - Post-earnings-announcement drift (PEAD) latest findings

2. **How Elite Quant Funds Operate:**
   - How do Renaissance Technologies / Two Sigma / Citadel approach hypothesis testing?
   - What methods do they use for regime detection?
   - How do they handle multiple testing problem?
   - What's their typical research-to-deployment timeline?

3. **Known Pitfalls in Backtesting:**
   - What do even professional quants miss?
   - Latest research on lookahead bias, survivorship bias, data snooping
   - Are there open-source implementations of purged k-fold CV we should review?

4. **Specific Technical Questions:**
   - HMM vs GMM vs K-means for regime detection - which papers show best results?
   - Volume-price relationship in gap-up scenarios - any 2024-2025 research?
   - Best practices for multiple testing corrections in finance (Bonferroni too conservative?)
   - GPU acceleration for financial data analysis - best libraries/approaches?

5. **Validation of Our Specific Edge:**
   - We see earnings surprise momentum (MU +15% after beat, KDK +10%)
   - Latest research on post-earnings drift?
   - How to statistically validate stock selection skill vs luck?
   - Should we build earnings surprise factor? Papers on this?

WHAT I DON'T WANT:
- Generic investment advice
- Risk disclaimers
- Beginner explanations
- Anything you'd tell a retail trader

WHAT I WANT:
- Deep academic papers with citations
- Real quant fund methodologies (if publicly known)
- Specific library/tool recommendations
- Benchmark our approach against institutional best practices

Give me the Renaissance Technologies answer, not the retail trader answer.
```

---

### 📋 PROMPT FOR CLAUDE OPUS (Copy This)

```
I need a deep, brutal critique of my quantitative trading research methodology. Don't hold back - I want to know where I'm fooling myself.

CONTEXT:
After months of failed lightweight approaches, I'm building heavyweight institutional-grade research infrastructure based on scientific method.

Core Philosophy: "We don't premake laws, we discover them"

INFRASTRUCTURE BUILT (~2,350 lines):

1. **Statistical Testing Framework (600 lines):**
   - Tests 28 hypotheses: momentum (16 variations), mean reversion (12), volatility clustering, cross-sectional correlation
   - Multiple testing corrections: Bonferroni, Benjamini-Hochberg FDR, Holm-Bonferroni
   - Output: p-values, effect sizes, significance flags

2. **Regime Detection (500 lines):**
   - HMM-based: volatility regimes (low/normal/high), trend regimes (bull/bear/sideways)
   - Manual breakpoints for known events (2023 banking crisis, AI boom)
   - Test if patterns are universal or regime-dependent

3. **Survivorship Bias Correction (450 lines):**
   - Detects delistings/IPOs, reconstructs point-in-time universes
   - Typical backtests inflate returns 3-5% by excluding bankruptcies

4. **Factor Analysis (400 lines):**
   - PCA on return matrix, don't assume Fama-French factors exist
   - Discover what ACTUALLY drives returns in 2023-2025 data

5. **Cross-Validation (400 lines):**
   - Walk-forward split, purged k-fold (López de Prado method)
   - Overfit detection, Monte Carlo permutation tests

DATA:
- Complete US equity universe: 10,986 tickers (NASDAQ, NYSE, AMEX, NYSE Arca)
- 2 years (504 trading days), 2023-12-18 to 2025-12-17
- Expected: ~1,300 tickers passing quality checks, ~650k bars, 5.5M observations

WHAT I'VE LEARNED:
- Manual trades work: MU +15% after earnings beat, KDK +10%
- Paper trades mixed: HUT -11%, RXRX -7% (claimed 85% hit rate edge failing)
- Cross-sectional power > time series length (10,986 tickers × 2 years > 100 tickers × 10 years)

YOUR DEEP REASONING EXPERTISE NEEDED:

1. **Methodology Critique:**
   - What hypothesis tests am I MISSING?
   - Are there interaction effects between factors I should test?
   - Am I handling multiple testing correctly (28+ hypotheses)?
   - Should I use Bayesian methods instead/in addition to frequentist?
   - What assumptions am I making that might be wrong?
   - Where could I be fooling myself?

2. **Factor Discovery:**
   - Is PCA the right approach or should I use ICA, NMF, autoencoders?
   - How do I interpret discovered factors without narrative fallacy?
   - Sparse PCA for interpretability?
   - How do I avoid "discovering" spurious factors that are just noise?

3. **Regime Detection Philosophy:**
   - Is HMM-based regime detection the right approach?
   - Or should I use simple rules (VIX > 20 = high vol)?
   - How do I avoid overfitting regimes to historical data?
   - How do I handle regime transitions (gradual vs sudden)?

4. **Statistical Rigor:**
   - Multiple testing corrections: Bonferroni too conservative? BH too lenient?
   - Alpha level: 0.05 vs 0.01 vs 0.001?
   - Should I use permutation-based corrections?
   - How do I validate the ENTIRE research framework (not just individual edges)?

5. **Philosophy Questions:**
   - Research vs trading tradeoff: when is research "done enough"?
   - At what point do diminishing returns kick in?
   - Human edge (MU/KDK picks) vs systematic: should I:
     A) Fully automate?
     B) Build system to AUGMENT my picks (AI suggests, I decide)?
     C) Hybrid (system trades most, I trade special situations)?
   - Live trading feedback loop: start paper trading NOW or finish research first?

6. **Earnings Edge Validation:**
   - I see: earnings surprise → momentum (MU +15%, KDK +10%)
   - How do I test if this is skill vs luck statistically?
   - Should I build this hypothesis test:
     ```
     H1: Stocks beating estimates by >10% outperform in next 5/10/21 days
     Variables: earnings surprise %, post-earnings drift
     Controls: market cap, sector, volume, pre-earnings momentum
     Test: Cross-sectional regression across all earnings 2023-2025
     Check: Does effect vary by regime?
     ```

SPECIFIC CONCERNS:
1. Am I testing the right things? (28 hypotheses seems arbitrary)
2. Am I being TOO conservative (missing real edges) or not conservative enough (finding false positives)?
3. Is HMM practical or over-engineered?
4. How do I know when to stop researching and start trading?
5. What if the framework itself has systemic bias?

COMMITMENT:
- Timeframe NOT SET - willing to work 6-12 months for rigor
- NOT looking for get-rich-quick
- Can handle technical depth, math, statistics
- Want institutional-level methodology

Be brutally honest. If I'm wrong, tell me. If I'm missing something obvious, tell me. What would YOU do if you were me?

Give me the Renaissance Technologies answer, not the retail answer.
```

---

### 📋 PROMPT FOR DEEPSEEK (Copy This)

```
I need technical code review and GPU optimization guidance for my quantitative trading research infrastructure. Looking for implementation-level expertise.

CONTEXT:
Building heavyweight quant research (not lightweight pattern matching). Have ~2,350 lines of statistical frameworks ready to run on GPU.

INFRASTRUCTURE:

1. **Statistical Testing Framework (600 lines):**
   - Tests 28 hypotheses across 1,300 tickers × 504 days
   - Currently CPU-bound: 2-4 hours estimated
   - Target: 5-10 min on GPU

2. **Regime Detection (500 lines):**
   - HMM-based volatility/trend/correlation regimes
   - CPU: ~1 hour | Target: 1-2 min GPU

3. **Factor Analysis (400 lines):**
   - PCA on return matrix (1,300 × 504)
   - CPU: ~30 min | Target: 30 sec GPU

DATA:
- SQLite database: ~500 MB, 1,300 tickers, 650k bars
- In-memory operations on Pandas DataFrames
- Shadow PC with GPU available (not yet optimized)

YOUR CODING EXPERTISE NEEDED:

1. **GPU Optimization Strategy:**
   - How to parallelize hypothesis testing framework for GPU?
   - Best libraries: CuPy vs Rapids (cuDF) vs PyTorch vs JAX?
   - Can I batch-process 1,300 tickers in parallel for 100x speedup?
   - Expected realistic speedup for our workload?
   - Code examples for converting Pandas/NumPy to GPU operations?

2. **Data Pipeline Optimization:**
   - Currently using Pandas on 500 MB dataset - fast enough?
   - Should I switch to: Dask? Polars? Vaex?
   - How to efficiently compute rolling correlations across 1,300 tickers?
   - Am I handling missing data correctly in quality checks?
   - Best practices for memory management with large cross-sectional data?

3. **Statistical Implementation Review:**
   - Are we implementing multiple testing corrections correctly?
   - Should we use statsmodels, scipy, or custom implementations?
   - How to handle edge cases (tickers with <504 days after filtering)?
   - Purged k-fold cross-validation: review our implementation approach?
   - HMM fitting: use hmmlearn, statsmodels, or custom?

4. **Database/Storage Architecture:**
   - Currently SQLite (500 MB) - is this optimal?
   - Should we switch to:
     * Parquet files (faster columnar reads)?
     * PostgreSQL (better complex queries)?
     * HDF5 (better time series)?
     * DuckDB (faster analytics on Parquet)?
   - How to architect for future scaling:
     * Global equities (50k+ tickers)?
     * Intraday data (1-min bars = 100x data)?
     * Options data (millions of contracts)?

5. **Code Quality & Edge Cases:**
   - Review our quality checks logic:
     * Missing days threshold: <30% acceptable?
     * Price gap detection: >50% threshold?
     * Zero volume handling?
   - Potential bugs in our frameworks?
   - Race conditions or data leakage in walk-forward CV?
   - Numerical stability issues in PCA/HMM?

6. **Specific Implementation Questions:**

   **Multiple Testing Corrections:**
   ```python
   # We're using:
   from statsmodels.stats.multitest import multipletests
   reject, pvals_corrected, _, _ = multipletests(
       pvals, alpha=0.05, method='fdr_bh'
   )
   # Correct implementation? Better alternatives?
   ```

   **Walk-Forward Validation:**
   ```python
   # Our approach:
   for i in range(n_splits):
       train_end = start_date + timedelta(days=252*i)
       test_start = train_end + timedelta(days=21)  # embargo
       test_end = test_start + timedelta(days=63)   # test window
       # Is 21-day embargo sufficient? 63-day test window optimal?
   ```

   **Rolling Correlations:**
   ```python
   # Current approach:
   returns = prices.pct_change()
   rolling_corr = returns.rolling(window=21).corr()
   # How to optimize for 1,300 tickers × 1,300 correlation matrix?
   ```

7. **Deployment/Production Considerations:**
   - How to structure code for:
     * Research (flexibility, experimentation)
     * Production (speed, reliability)
   - Testing strategy for statistical code?
   - CI/CD for quant research?
   - Version control for data + code?

RESOURCES:
- Shadow PC with GPU (specific model unknown)
- Python environment: NumPy, Pandas, scikit-learn, statsmodels, scipy
- Can install: CuPy, Rapids, PyTorch, JAX, etc.
- Budget: free/open-source preferred

COMMITMENT:
- Willing to rewrite code for 100x speedup
- Can handle technical implementation details
- Want production-grade code quality
- Timeframe: NOT SET - will do it right

What specific code optimizations should I implement in next 48 hours while data downloads?

Give me specific code examples, library recommendations, and catch any bugs/issues you see.
```

---

*Copy the specific prompt above for each AI system: Perplexity Pro for research, Claude Opus for methodology critique, DeepSeek for code optimization.*
