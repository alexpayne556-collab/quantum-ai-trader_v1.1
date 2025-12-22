# RESEARCH AGENT: LITERATURE-GUIDED STRATEGY GENERATOR
**Systematically mine academic literature for testable strategies**

---

## IMMEDIATE RESEARCH TASKS (Run these searches NOW)

### Search 1: Validate Fibonacci Discovery (82.9% hit rate!)
```
arXiv search: "fibonacci retracement"
arXiv search: "technical analysis self-fulfilling"
arXiv search: "support resistance clustering"
Google Scholar: "fibonacci trading profitability" filetype:pdf
```

**Expected findings**:
- Self-fulfilling prophecy research
- Pattern recognition in markets
- Support/resistance clustering studies

**Action**: Document WHY Fibonacci works (not just that it does)

---

### Search 2: Validate Ichimoku Discovery (t=131.57!)
```
arXiv search: "ichimoku cloud"
SSRN search: "multi-timeframe momentum"
Google Scholar: "japanese technical analysis" filetype:pdf
```

**Expected findings**:
- Japanese technical analysis validation
- Multi-timeframe trend following
- Cloud as dynamic support/resistance

**Action**: Understand Ichimoku components interaction

---

### Search 3: Understand Momentum Failure (only 49.8%)
```
arXiv search: "momentum crash"
arXiv search: "momentum reversal"
SSRN search: "when momentum fails"
SSRN search: "2020-2024 momentum performance"
```

**Critical question**: Why did momentum underperform?
- Is 2020-2024 a mean-reverting regime?
- Did we test momentum wrong (too short horizons)?
- Do we need 6-12 month momentum (not 5-60 day)?

**Action**: Test longer-horizon momentum in Part 2

---

### Search 4: Harvey-Liu-Zhu Deep Dive (OUR METHODOLOGY SOURCE!)
```
Direct link: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2249314
Title: "...and the Cross-Section of Expected Returns" (2016)
```

**Must read because**:
- This is WHERE t>3.0 threshold comes from
- They tested 316 factors - how many false discoveries?
- Multiple testing correction methodology
- Bonferroni vs Holm-Bonferroni vs BHY

**Action**: Ensure we're applying their framework correctly

---

### Search 5: Fama-French Factors (Foundation of factor investing)
```
1993 paper: "Common Risk Factors in Returns on Stocks and Bonds"
2015 paper: "A Five-Factor Asset Pricing Model"
AQR website: https://www.aqr.com/Insights/Datasets (FREE factor data!)
```

**Factors to test**:
1. **Mkt-RF** (Market minus risk-free) - baseline
2. **SMB** (Small minus Big) - size factor
3. **HML** (High minus Low B/M) - value factor
4. **RMW** (Robust minus Weak profitability) - quality
5. **CMA** (Conservative minus Aggressive investment) - investment

**Problem**: We don't have fundamental data yet
**Solution**: Part 3 or Part 4 with fundamental data integration

---

### Search 6: Known Anomalies List
```
Green, Hand, Zhang (2017): "The Characteristics that Provide Independent Information about Average U.S. Monthly Stock Returns"
- Tests 94 characteristics
- Provides replication code

McLean & Pontiff (2016): "Does Academic Research Destroy Stock Return Predictability?"
- 97 anomalies tested
- Post-publication decay analysis
```

**Extract anomalies we haven't tested**:
- Post-earnings announcement drift
- Accruals anomaly
- 52-week high effect
- Idiosyncratic volatility puzzle
- Short interest effect
- Analyst revision drift
- Insider trading patterns

**Action**: Add to Part 2-4 strategy lists

---

### Search 7: Volatility Regime Research
```
arXiv search: "GARCH volatility clustering"
arXiv search: "regime switching trading"
arXiv search: "VIX trading strategies"
SSRN search: "volatility regime detection"
```

**Our finding**: 72.4% hit rate for volatility regime strategies
**Need to understand**:
- GARCH(1,1) vs hidden Markov models
- Does low vol → high vol predict DIRECTION or just MAGNITUDE?
- VIX mean reversion thresholds
- Historical vol vs implied vol regimes

---

### Search 8: Market Microstructure
```
arXiv search: "market microstructure"
arXiv search: "bid-ask spread prediction"
arXiv search: "volume price relationships"
SSRN search: "intraday patterns"
```

**Patterns to validate**:
- Opening hour volatility
- Closing hour volume
- Day-of-week effects (Monday/Friday)
- Month-end rebalancing
- Options expiration effects

---

### Search 9: Behavioral Finance
```
Kahneman & Tversky: "Prospect Theory" (1979)
Shiller: "Irrational Exuberance" (2000)
Barber & Odean: "Trading is Hazardous to Your Wealth" (2000)
Thaler: "Mental Accounting Matters" (1999)
```

**Biases to exploit**:
- Disposition effect (sell winners, hold losers)
- Overconfidence (excess trading)
- Anchoring (52-week high)
- Recency bias
- Loss aversion

---

### Search 10: Machine Learning in Finance
```
arXiv search: "deep learning stock prediction"
arXiv search: "LSTM time series forecasting finance"
arXiv search: "XGBoost feature importance trading"
arXiv search: "reinforcement learning portfolio"
```

**For Part 2 ML strategies**:
- LSTM architectures for time-series
- XGBoost hyperparameter tuning
- Ensemble methods (bagging, stacking)
- Attention mechanisms for financial data

---

## EXTRACTION TEMPLATE

For each paper found, extract:

```python
{
    'paper_id': 'arXiv:XXXX.XXXXX or Author_Year',
    'title': 'Full paper title',
    'authors': 'Last names',
    'year': YYYY,
    'source': 'arXiv/SSRN/Journal',
    'url': 'Direct download link',
    'category': 'momentum/value/volatility/etc',
    
    'key_finding': 'One sentence: What did they find?',
    
    'relevant_to_us': 'How does this relate to our Part 1 results?',
    
    'strategies_to_test': [
        'Strategy 1: Specific testable hypothesis',
        'Strategy 2: Another testable hypothesis',
        # ...
    ],
    
    'economic_rationale': 'WHY does this pattern work?',
    
    'expected_t_stat': 'High/Medium/Low based on paper results',
    
    'requires_data': ['fundamentals', 'options', 'sentiment'] or ['OHLCV only'],
    
    'priority': 1-5  # 1=test immediately, 5=later
}
```

---

## STRATEGY GENERATION RULES

### Rule 1: Every strategy needs economic rationale
**Bad**: "Test RSI < 25"
**Good**: "Test RSI < 25 because behavioral finance shows investors overreact to news (DeBondt-Thaler 1985), creating temporary oversold conditions that mean-revert"

### Rule 2: Link to literature
**Bad**: "Fibonacci 61.8% retracement"
**Good**: "Fibonacci 61.8% retracement (golden ratio) - self-fulfilling prophecy where traders cluster orders at known levels (Lo et al 2000)"

### Rule 3: Specify regime dependence
**Bad**: "Momentum works"
**Good**: "Momentum works in low-volatility regimes (Daniel-Moskowitz 2016) but crashes in high-volatility periods"

### Rule 4: Parameter ranges from literature
**Bad**: "Test momentum"
**Good**: "Test 3-12 month momentum (Jegadeesh-Titman 1993) and 6-12 month momentum (Fama-French 1996)"

---

## OUTPUT: LITERATURE_SUGGESTED_STRATEGIES.csv

Generate CSV with columns:
```
strategy_name,category,paper_source,description,hypothesis,parameters,test_priority,tested,result
```

Example rows:
```csv
PostEarningsDrift,fundamental,Bernard1989,"Buy stocks with positive earnings surprises","SUE > 1.5 predicts 2-3% drift over 60 days","SUE threshold: 1.5, hold: 60 days",1,no,
Accruals,fundamental,Sloan1996,"Short high accrual stocks","High accruals signal earnings manipulation","Accruals > 80th percentile, hold: 252 days",2,no,
52WeekHigh,momentum,George2014,"Buy stocks near 52W high","Anchoring bias - breakout continuation","Price > 0.95 * 52W high, hold: 20 days",1,no,
IdioVol,anomaly,Ang2006,"Short high idiosyncratic vol","Puzzle: high IV predicts LOW returns","IV > 80th percentile, hold: 20 days",2,no,
```

---

## INTEGRATION WITH PARTS 2-5

### Part 2 (ML + Ensemble): **Priority: Machine learning papers**
- Search: "XGBoost trading", "LSTM stock prediction", "ensemble methods finance"
- Extract: Model architectures, hyperparameters, feature engineering
- Test: 500-1000 ML strategies

### Part 3 (Multi-Factor): **Priority: Fama-French + factor timing**
- Search: "factor investing", "factor timing", "multi-factor models"
- Extract: Factor definitions, combination methods, regime-switching
- Test: 1000-2000 factor strategies

### Part 4 (Behavioral + Microstructure): **Priority: Anomalies list papers**
- Search: Green-Hand-Zhang, McLean-Pontiff anomaly lists
- Extract: All 97+ documented anomalies
- Test: 1000-2000 anomaly strategies

### Part 5 (Cross-Asset + Macro): **Priority: VIX, carry trades, spillovers**
- Search: "VIX mean reversion", "currency carry", "equity bond correlation"
- Extract: Cross-asset relationships, macro indicators
- Test: 500-1000 cross-asset strategies

---

## EXECUTION PLAN

### Step 1: Download Papers (Today - 2 hours)
- Harvey-Liu-Zhu (2016) - methodology foundation
- Fama-French (1993, 2015) - factor foundation  
- Jegadeesh-Titman (1993) - momentum foundation
- DeBondt-Thaler (1985) - mean reversion foundation
- 10-20 arXiv papers on Fibonacci, Ichimoku, volatility regimes

### Step 2: Read & Extract (Tomorrow - 4 hours)
- Read Harvey-Liu-Zhu thoroughly (understand t>3.0)
- Skim others for key findings
- Build ACADEMIC_RESEARCH_DATABASE.csv (50-100 papers)
- Build LITERATURE_SUGGESTED_STRATEGIES.csv (200-500 strategies)

### Step 3: Prioritize Strategies (Tomorrow - 1 hour)
- Priority 1: Strategies matching our Part 1 discoveries
- Priority 2: Strategies with strong academic support
- Priority 3: Strategies requiring only OHLCV data
- Priority 4: Strategies requiring fundamental data (later)
- Priority 5: Exotic strategies (options, sentiment, etc.)

### Step 4: Integrate into Part 2 Script (Dec 23)
- Add literature-backed strategies to SHADOW_GPU_EXPANSION_PART2.py
- Test ML models on literature-suggested features
- Validate literature findings with our data

---

## SUCCESS METRICS

### Research Phase:
- ✅ 50+ papers downloaded
- ✅ Harvey-Liu-Zhu read and understood
- ✅ 200+ literature-suggested strategies extracted
- ✅ Economic rationale documented for each

### Testing Phase:
- ✅ Test all literature strategies in Parts 2-5
- ✅ Compare our hit rates to published hit rates
- ✅ Document which anomalies replicate (and which don't)
- ✅ Build "anomaly replication report"

### Companion Phase:
- ✅ Every strategy linked to source paper
- ✅ Economic rationale included in companion
- ✅ Decay detection (compare pre/post publication)
- ✅ World-class documentation rivaling institutional research

---

**This is how we build on 70+ years of finance research instead of reinventing the wheel. 🎖️**
